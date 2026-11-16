import os
from transformers import (
    AutoTokenizer,
    SwitchTransformersForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import torch

# Repo-level data folder (independent of the current working directory)
DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"
)
# Set MAX_STEPS / MAX_SAMPLES env vars to shrink the run (e.g. on a laptop)
MAX_STEPS = int(os.getenv("MAX_STEPS", -1))
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", 0))
# float16 + device_map="auto" only make sense on CUDA machines
USE_CUDA = torch.cuda.is_available()
MODEL_KWARGS = (
    {"device_map": "auto", "torch_dtype": torch.float16} if USE_CUDA else {}
)

# Load and format the dataset
dataset = load_dataset(
    "text", data_files=os.path.join(DATA_DIR, "crimeandpunishment.txt")
)
dataset = dataset.filter(lambda sentence: len(sentence["text"]) > 1)
if MAX_SAMPLES:
    dataset["train"] = dataset["train"].select(range(MAX_SAMPLES))
print(f"Dataset 1: {dataset['train'][0]}")

# Create model directory to save to
model_dir = "./models/MoE/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Instantiate our tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")

# Instantiate our model from the config
model = SwitchTransformersForConditionalGeneration.from_pretrained(
    "google/switch-base-8",
    **MODEL_KWARGS,
)


# Create a tokenize function
def tokenize(batch):
    return tokenizer(
        str(batch), padding="max_length", truncation=True, max_length=256
    )


# tokenize our whole dataset (so we never have to do it again)
tokenized_dataset = dataset.map(tokenize, batched=False)
print(f"Tokenized: {tokenized_dataset['train'][0]}")

# Create a data collator to format the data for training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, mlm_probability=0.0
)  # Causal Language Modeling - Does not use mask


# Establish training arguments
train_args = TrainingArguments(
    output_dir=model_dir,
    num_train_epochs=1,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=8,
    save_steps=5000,
    save_total_limit=2,
    report_to="none",
    use_cpu=os.getenv("USE_CPU") == "1",  # force CPU (skip MPS/CUDA)
)

# Instantiate the Trainer
trainer = Trainer(
    model=model,
    args=train_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
)

# Train and save the model
trainer.train()
trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)

# Load the saved model
model = SwitchTransformersForConditionalGeneration.from_pretrained(
    model_dir,
    **MODEL_KWARGS,
)
if not USE_CUDA:
    model = model.to(trainer.args.device)

# Test the saved model
input = "To be or not <extra_id_0> <extra_id_0>"
tokenized_inputs = tokenizer(input, return_tensors="pt")
out = model.generate(
    input_ids=tokenized_inputs["input_ids"].to(model.device),
    attention_mask=tokenized_inputs["attention_mask"].to(model.device),
    max_length=256,
    num_beams=5,
    temperature=0.7,
    top_k=50,
    top_p=0.90,
    no_repeat_ngram_size=2,
)
print(f"To be or not {tokenizer.decode(out[0], skip_special_tokens=True)}")

# To be or not sss
