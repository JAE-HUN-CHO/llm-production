from trl import SFTTrainer
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
import os

import torch
from datasets import load_dataset

# Jamba (52B) needs large CUDA GPUs. Without CUDA (e.g. Apple Silicon)
# we fall back to a tiny model so the same fine-tuning code still runs.
USE_JAMBA = torch.cuda.is_available()
model_id = "ai21labs/Jamba-v0.1" if USE_JAMBA else "distilgpt2"
target_modules = (
    ["embed_tokens", "x_proj", "in_proj", "out_proj"]
    if USE_JAMBA
    else ["c_attn"]
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id, **({"device_map": "auto"} if USE_JAMBA else {})
)

dataset = load_dataset("Abirate/english_quotes", split="train")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3 if USE_JAMBA else 1,
    max_steps=-1 if USE_JAMBA else int(os.environ.get("MAX_STEPS", 20)),
    use_cpu=not USE_JAMBA,  # tiny model: CPU is fast enough and stable
    per_device_train_batch_size=4,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=2e-3,
)
lora_config = LoraConfig(
    r=8,
    target_modules=target_modules,
    task_type="CAUSAL_LM",
    bias="none",
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset,
    dataset_text_field="quote",
)

trainer.train()

tokenizer.save_pretrained("./JAMBA/")
# trainer.model holds the LoRA-wrapped model (adapter + config)
trainer.model.save_pretrained("./JAMBA/")

del model

if USE_JAMBA:
    from transformers import BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True, llm_int8_skip_modules=["mamba"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config,
    )
else:
    from peft import AutoPeftModelForCausalLM

    model = AutoPeftModelForCausalLM.from_pretrained("./JAMBA/")
input_ids = tokenizer(
    "In the recent Super Bowl LVIII,", return_tensors="pt"
).to(model.device)["input_ids"]

outputs = model.generate(input_ids, max_new_tokens=216)

print(tokenizer.batch_decode(outputs))
