from dotenv import load_dotenv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# 프로젝트 루트의 .env 를 환경변수로 로드
load_dotenv()

# full "bigscience/bloom" is 176B params; override via env var (e.g. bigscience/bloom-3b)
MODEL_NAME = os.environ.get("BLOOM_MODEL", "bigscience/bloom")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

prompt = "Hello world! This is my first time running an LLM!"

input_tokens = tokenizer.encode(prompt, return_tensors="pt", padding=True)
generated_tokens = model.generate(input_tokens, max_new_tokens=20)
generated_text = tokenizer.batch_decode(
    generated_tokens, skip_special_tokens=True
)
print(generated_text)
