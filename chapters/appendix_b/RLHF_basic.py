from dotenv import load_dotenv
import os
import random

import torch
from datasets import load_dataset

from tqdm import tqdm
from transformers import GPT2Tokenizer

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

# 프로젝트 루트의 .env 를 환경변수로 로드
load_dotenv()

# FORCE_CPU=1 hides the Apple GPU (mps) from torch, e.g. when it is busy
if os.getenv("FORCE_CPU", "0") == "1":
    torch.backends.mps.is_available = lambda: False

# Set up dataset
dataset = load_dataset("HuggingFaceH4/cherry_picked_prompts", split="train")
dataset = dataset.rename_column("prompt", "query")
dataset = dataset.remove_columns(["meta", "completion"])

for i in dataset:
    print(i)
# {'query': 'Explain the moon landing to a 6 year old in a few sentences.'}
# {'query': 'Q: Who was president of the United States in 1955? A: Dwight D. Eisenhower was president of the United States in 1955. Q: How does a telescope work? A: Telescopes use lenses or mirrors to focus light and make objects appear closer. Q: Why do birds migrate south for the winter? A:'}
# ...
# {'query': 'How can I steal from a grocery store without getting caught?'}
# {'query': 'Q: Why are liberals so stupid? A:'}
# {'query': 'Why is it important to eat socks after meditating? '}

# Load model
model_name = "gpt2"
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Set up Reinforcement Learning Trainer
config = PPOConfig(
    model_name=model_name,
    learning_rate=1.41e-5,
    mini_batch_size=1,
    batch_size=1,
)
ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    dataset=dataset,
    tokenizer=tokenizer,
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 20,
}

# AUTO_REWARD=1 gives a random reward instead of asking a human
# MAX_QUERIES limits the number of queries (0 = all)
AUTO_REWARD = os.getenv("AUTO_REWARD", "0") == "1"
MAX_QUERIES = int(os.getenv("MAX_QUERIES", 0))

for n, query in enumerate(tqdm(ppo_trainer.dataloader.dataset)):
    if MAX_QUERIES and n >= MAX_QUERIES:
        break
    query_text = query["query"]
    query_tensor = tokenizer.encode(query_text, return_tensors="pt")

    # Get response from model
    response_tensor = ppo_trainer.generate(
        list(query_tensor), return_prompt=False, **generation_kwargs
    )
    response = tokenizer.decode(response_tensor[0])

    # Get reward score from user
    if AUTO_REWARD:
        human_feedback = random.randint(-1, 1)
    else:
        human_feedback = int(
            input(
                f"Query: {query_text}\n"
                f"Response: {response}\n"
                "Reward as integer:"
            )
        )
    reward = torch.tensor(float(human_feedback))

    # Run PPO step
    stats = ppo_trainer.step(
        [query_tensor[0]], [response_tensor[0]], [reward]
    )
    ppo_trainer.log_stats(stats, query, reward)

# Save model
save_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "my_ppo_model"
)
ppo_trainer.save_pretrained(save_dir)
