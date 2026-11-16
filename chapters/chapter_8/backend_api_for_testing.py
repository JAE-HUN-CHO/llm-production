import argparse
import os
import asyncio
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import uvicorn

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)
from threading import Thread

# device 자동 선택: cuda > mps > cpu
if os.getenv("DEVICE"):  # 강제 지정: DEVICE=cpu|mps|cuda
    device = torch.device(os.environ["DEVICE"])
elif torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load tokenizer, model, and streamer into memory
# 기본값은 원본 모델. 작은 모델로 테스트하려면 BACKEND_MODEL=gpt2 처럼 지정
MODEL_NAME = os.getenv(
    "BACKEND_MODEL", "togethercomputer/RedPajama-INCITE-Chat-3B-v1"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
)
model = model.to(device)
streamer = TextIteratorStreamer(
    tokenizer, timeout=float(os.getenv("STREAMER_TIMEOUT", "60")), skip_prompt=True, skip_special_tokens=True
)


class StopOnTokens(StoppingCriteria):
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs
    ) -> bool:
        stop_ids = [29, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


async def stream_results() -> AsyncGenerator[bytes, None]:
    for response in streamer:
        await asyncio.sleep(0.05)  # slow things down to see streaming
        # It's typical to return streamed responses byte encoded
        print(response)
        yield (response + "\n").encode("utf-8")


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate LLM Response

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")

    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "1024")),
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
    )

    # Start a seperate thread to generate results
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    return StreamingResponse(stream_results())


if __name__ == "__main__":
    # Start Service - Defaults to localhost on port 8000
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="debug")
