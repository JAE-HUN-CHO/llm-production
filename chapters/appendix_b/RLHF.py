import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
import evaluate
import numpy as np
import torch
import torch.nn as nn

# FORCE_CPU=1 hides the Apple GPU (mps) from torch, e.g. when it is busy
if os.getenv("FORCE_CPU", "0") == "1":
    torch.backends.mps.is_available = lambda: False
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
    AutoModelForSequenceClassification,
    PreTrainedTokenizerBase,
    Trainer,
    pipeline,
)
from transformers.utils import PaddingStrategy

from trl import (
    SFTTrainer,
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
)
from trl.trainer import ConstantLengthDataset
from trl.core import LengthSampler


# ---------------------------------------------------------------------------
# Runtime settings. Defaults reproduce the book (Llama-2-7b on CUDA); override
# with environment variables to run a tiny version on a laptop, e.g.
#   BASE_MODEL=gpt2 SFT_MAX_STEPS=5 N_TRAIN=64 N_VAL=16 RL_STEPS=4 python RLHF.py
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
BASE_MODEL = os.getenv("BASE_MODEL", "meta-llama/Llama-2-7b-chat-hf")
SFT_MAX_STEPS = int(os.getenv("SFT_MAX_STEPS", 5000))
N_TRAIN = int(os.getenv("N_TRAIN", 100000))  # reward / RL samples
N_VAL = int(os.getenv("N_VAL", 50000))  # reward eval samples
SEQ_LEN = int(os.getenv("SEQ_LEN", 1024))
SFT_VAL_SAMPLES = int(os.getenv("SFT_VAL_SAMPLES", 4000))
SHUFFLE_BUFFER = int(os.getenv("SHUFFLE_BUFFER", 5000))
RL_STEPS = int(os.getenv("RL_STEPS", 20000))
RL_BATCH_SIZE = int(os.getenv("RL_BATCH_SIZE", 8))
RL_MINI_BATCH_SIZE = int(os.getenv("RL_MINI_BATCH_SIZE", 1))
RL_GRAD_ACC = int(os.getenv("RL_GRAD_ACC", 8))
MAX_NEW_MIN = int(os.getenv("MAX_NEW_MIN", 32))
MAX_NEW_MAX = int(os.getenv("MAX_NEW_MAX", 128))
NUM_PROC = int(os.getenv("NUM_PROC", 24))
if NUM_PROC <= 1:
    NUM_PROC = None

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
try:
    import bitsandbytes  # noqa: F401

    HAS_BNB = True
except ImportError:
    HAS_BNB = False
# 8-bit loading only works with CUDA + bitsandbytes (not on Mac)
USE_8BIT = DEVICE == "cuda" and HAS_BNB
USE_BF16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()


def take_dataset(data_dir, split_n, skip=0):
    """Materialise the first `split_n` rows of a stack-exchange-paired
    sub-folder using streaming (avoids downloading the whole dataset)."""
    stream = load_dataset(
        "lvwerra/stack-exchange-paired",
        data_dir=data_dir,
        split="train",
        streaming=True,
    )
    if skip:
        stream = stream.skip(skip)
    return Dataset.from_list(list(stream.take(split_n)))


def prepare_sample_text(example):
    "Prepare the text from a sample of the Supervised FineTuning dataset."
    text = (
        f"Question: {example['question']}\n\n"
        f"Answer: {example['response_j']}"
    )
    return text


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the Supervised
    FineTuning dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(
        zip(range(nb_examples), iter(dataset)), total=nb_examples
    ):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the Supervised
    FineTuning model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: "
        f"{all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def create_datasets(tokenizer):
    # Supervised Finetuning Creation of Datasets
    dataset = load_dataset(
        "lvwerra/stack-exchange-paired",
        data_dir="data/finetune",
        split="train",
        streaming=True,
    )

    print("Loading the dataset in streaming mode")
    valid_data = dataset.take(SFT_VAL_SAMPLES)
    train_data = dataset.skip(SFT_VAL_SAMPLES)
    train_data = train_data.shuffle(
        buffer_size=SHUFFLE_BUFFER, seed=8855
    )

    chars_per_token = chars_token_ratio(
        train_data, tokenizer, nb_examples=min(400, SFT_VAL_SAMPLES)
    )
    print(
        "The character to token ratio of the dataset is: "
        f"{chars_per_token:.2f}"
    )

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=SEQ_LEN,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=SEQ_LEN,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


def run_training(train_data, val_data, tokenizer):
    print("Loading the Supervised FineTuning model")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_data.start_iteration = 0

    print("Starting main Supervised FineTuning loop")

    training_args = TrainingArguments(
        output_dir=os.path.join(MODELS_DIR, "RLHF"),
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=SFT_MAX_STEPS,
        eval_steps=max(1, min(1000, SFT_MAX_STEPS)),
        save_steps=max(1, min(1000, SFT_MAX_STEPS)),
        logging_steps=1,
        per_device_train_batch_size=int(os.getenv("SFT_BATCH_SIZE", 4)),
        per_device_eval_batch_size=int(os.getenv("SFT_BATCH_SIZE", 4)),
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_steps=min(100, SFT_MAX_STEPS // 10),
        gradient_accumulation_steps=1,
        gradient_checkpointing=DEVICE == "cuda",
        fp16=False,
        bf16=False,
        weight_decay=0.05,
        run_name="llama2-7b-finetuned",
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    if USE_8BIT:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_8bit=True,
            device_map={"": Accelerator().process_index},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        packing=True,
        tokenizer=tokenizer,
        max_seq_length=SEQ_LEN,
    )

    print_trainable_parameters(trainer.model)

    print("Supervised FineTuning...")
    trainer.train()

    print("Saving last checkpoint of the Supervised FineTuning model")
    trainer.model.save_pretrained(
        os.path.join(MODELS_DIR, "RLHF", "final_SFT_checkpoint")
    )


def compute_metrics(eval_pred):
    # Computing eval metrics for Reward Modeling
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=predictions, references=labels)


class RewardTrainer(Trainer):
    # Define how to compute the reward loss for Reward Modeling.
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(
            input_ids=inputs["input_ids_j"],
            attention_mask=inputs["attention_mask_j"],
        )[0]
        rewards_k = model(
            input_ids=inputs["input_ids_k"],
            attention_mask=inputs["attention_mask_k"],
        )[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


@dataclass
class RewardDataCollatorWithPadding:
    # Data Collator for Reward Modeling
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


def run_reward_modeling(train_data, val_data):
    # Run the Reward modeling training
    model_name_split = "gpt2"
    output_name = (
        f"{model_name_split}_peft_stack-exchange-paired_rmts__100000_2e-5"
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(MODELS_DIR, output_name),
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        weight_decay=0.001,
        evaluation_strategy="steps",
        eval_steps=max(1, min(500, len(train_data) // 4)),
        save_strategy="steps",
        save_steps=max(1, min(500, len(train_data) // 4)),
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        deepspeed=None,
        local_rank=-1,  # Used for Multi-GPU
        remove_unused_columns=False,
        label_names=[],
        bf16=USE_BF16,
        logging_strategy="steps",
        logging_steps=10,
        optim="adamw_hf",
        lr_scheduler_type="linear",
    )

    # Parameter Efficient FineTuning config
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    # Set up Reward Modeling model
    model = AutoModelForSequenceClassification.from_pretrained(
        "gpt2",
        num_labels=1,
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float32,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Set up Reward Modeling Tokenizer
    tokenizer_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # Need to do this for gpt2, because it doesn't have an official pad token.
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = True
    num_proc = NUM_PROC  # Can adjust to be higher if you have more processors.
    original_columns = train_data.column_names

    def preprocess_function(examples):
        # Turn the Reward Modeling dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
        # Then tokenize the Reward Modeling dataset.
        new_examples = {
            "input_ids_j": [],
            "attention_mask_j": [],
            "input_ids_k": [],
            "attention_mask_k": [],
        }
        for question, response_j, response_k in zip(
            examples["question"],
            examples["response_j"],
            examples["response_k"],
        ):
            tokenized_j = tokenizer(
                "Question: " + question + "\n\nAnswer: " + response_j,
                truncation=True,
            )
            tokenized_k = tokenizer(
                "Question: " + question + "\n\nAnswer: " + response_k,
                truncation=True,
            )

            new_examples["input_ids_j"].append(tokenized_j["input_ids"])
            new_examples["attention_mask_j"].append(
                tokenized_j["attention_mask"]
            )
            new_examples["input_ids_k"].append(tokenized_k["input_ids"])
            new_examples["attention_mask_k"].append(
                tokenized_k["attention_mask"]
            )

        return new_examples

    # preprocess the Reward Modeling dataset and filter out QAs that are longer than 512
    train_data = train_data.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    train_data = train_data.filter(
        lambda x: len(x["input_ids_j"]) <= 512
        and len(x["input_ids_k"]) <= 512
    )

    # preprocess the eval dataset
    val_data = val_data.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    val_data = val_data.filter(
        lambda x: len(x["input_ids_j"]) <= 512
        and len(x["input_ids_k"]) <= 512
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer, max_length=512
        ),
    )
    trainer.train()

    print("Saving last checkpoint of the Reward Modeling model")
    model.save_pretrained(
        os.path.join(MODELS_DIR, "RLHF", output_name)
    )


def build_dataset_preprocess_function(examples, tokenizer):
    new_examples = {
        "query": [],
        "input_ids": [],
    }
    for question in examples["question"]:
        query = "Question: " + question + "\n\nAnswer: "
        tokenized_question = tokenizer(query, truncation=True)
        new_examples["query"].append(query)
        new_examples["input_ids"].append(tokenized_question["input_ids"])

    return new_examples


def build_dataset(
    tokenizer,
    dataset_name="lvwerra/stack-exchange-paired",
):
    """
    Build RL dataset for training. This builds the dataset from
    `load_dataset`, one should customize this function to train the model
    on their own dataset.
    """

    # load imdb with datasets
    ds = take_dataset("data/rl", N_TRAIN)
    original_columns = ds.column_names
    num_proc = NUM_PROC

    ds = ds.map(
        build_dataset_preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
        fn_kwargs={"tokenizer": tokenizer},
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False)

    ds.set_format(type="torch")
    return ds


def collator(data):
    # Data Collator for Reinforcement Learning
    return dict((key, [d[key] for d in data]) for key in data[0])


def run_RL():
    # Run Reinforcement Learning once the SFT and Reward training are done
    config = PPOConfig(
        steps=RL_STEPS,
        model_name=os.path.join(
            MODELS_DIR, "RLHF", "final_SFT_checkpoint"
        ),
        learning_rate=1.4e-5,
        batch_size=RL_BATCH_SIZE,
        mini_batch_size=RL_MINI_BATCH_SIZE,
        gradient_accumulation_steps=RL_GRAD_ACC,
        optimize_cuda_cache=DEVICE == "cuda",
        early_stopping=True,
        target_kl=0.1,
        ppo_epochs=4,
        seed=8855,
        init_kl_coef=0.2,
        adap_kl_ctrl=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # GPT-2 tokenizer has a pad token, but it is not eos_token by default.
    # We need to set it to eos_token. only for this model.

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(tokenizer)

    # Now let's build the model, the reference model, and the tokenizer.
    current_device = Accelerator().local_process_index

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if USE_8BIT:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config.model_name,
            load_in_8bit=True,
            device_map={"": current_device},
            peft_config=lora_config,
        )
    else:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config.model_name,
            peft_config=lora_config,
        )

    optimizer = None
    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
    )

    # We then build the sentiment analysis pipeline, passing the model name
    # and the sentiment analysis pipeline arguments. Let's also make sure to
    # set the device to the same device as the PPOTrainer.
    reward_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(
            MODELS_DIR,
            "RLHF",
            "gpt2_peft_stack-exchange-paired_rmts__100000_2e-5",
        ),
        num_labels=1,
    )
    reward_model.config.pad_token_id = reward_tokenizer.eos_token_id
    if USE_8BIT:
        pipe_kwargs = dict(
            device_map={"": current_device},
            model_kwargs={"load_in_8bit": True},
        )
    else:
        pipe_kwargs = dict(device=ppo_trainer.accelerator.device)
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=reward_model,
        tokenizer=reward_tokenizer,
        **pipe_kwargs,
        return_token_type_ids=False,
    )

    # We then define the arguments to pass to the `generate` function. These
    # arguments are passed to the `generate` function of the PPOTrainer,
    # which is a wrapper around the `generate` function of the trained model
    generation_kwargs = {
        # "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": 100_000,
    }
    output_min_length = MAX_NEW_MIN
    output_max_length = MAX_NEW_MAX
    output_length_sampler = LengthSampler(
        output_min_length, output_max_length
    )

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        if epoch >= config.total_ppo_epochs:
            break

        question_tensors = batch["input_ids"]

        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(
            response_tensors, skip_special_tokens=True
        )

        sent_kwargs = {
            "return_all_scores": True,
            "function_to_apply": "none",
            "batch_size": RL_BATCH_SIZE,
            "truncation": True,
        }

        # Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [
            torch.tensor(output[0]["score"] - 0.0)
            for output in pipe_outputs
        ]

        # Run PPO step
        stats = ppo_trainer.step(
            question_tensors, response_tensors, rewards
        )
        ppo_trainer.log_stats(stats, batch, rewards)

        if epoch % 100 == 0:
            ppo_trainer.save_pretrained(
                os.path.join(
                    MODELS_DIR, "RLHF", "RLedHFed", f"step_{epoch}"
                )
            )


def main():
    # STAGES lets you run/resume only parts of the pipeline, e.g.
    # STAGES=reward,rl
    stages = os.getenv("STAGES", "sft,reward,rl").split(",")

    # Supervised FineTuning
    if "sft" in stages:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_dataset, eval_dataset = create_datasets(tokenizer)
        run_training(train_dataset, eval_dataset, tokenizer)

        del train_dataset, eval_dataset

    # Reward Model Training
    if "reward" in stages:
        train_dataset = take_dataset("data/reward", N_TRAIN)
        eval_dataset = take_dataset("data/evaluation", N_VAL)

        run_reward_modeling(train_dataset, eval_dataset)

        del train_dataset, eval_dataset

    # Reinforcement Learning Training
    if "rl" in stages:
        tqdm.pandas()

        # We then define the arguments to pass to the sentiment analysis
        # pipeline. `return_all_scores` gives the score for each label.
        run_RL()


if __name__ == "__main__":
    set_seed(8855)
    os.makedirs(os.path.join(MODELS_DIR, "RLHF"), exist_ok=True)

    main()
