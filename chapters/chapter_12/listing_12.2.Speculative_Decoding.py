from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)
import torch
from datasets import load_dataset

from time import perf_counter
from tqdm import tqdm

from evaluate import load

if torch.cuda.is_available():
    device = "cuda:0"
else:
    # mps lacks some ops used by Whisper generation (e.g. aten::isin)
    device = "cpu"
print(f"Device: {device}")
attention = "sdpa"
print(attention)
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

if torch.cuda.is_available():
    model_id = "openai/whisper-large-v3"
    assistant_model_id = "distil-whisper/distil-large-v3"
    base_kwargs = {"language": "en", "task": "transcribe"}
else:
    # Smaller (English-only) pair so it runs on Apple Silicon / CPU
    model_id = "openai/whisper-small.en"
    assistant_model_id = "distil-whisper/distil-small.en"
    base_kwargs = {}
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    low_cpu_mem_usage=False,
    use_safetensors=True,
    attn_implementation=attention,
    torch_dtype=torch_dtype,
).to(device)
processor = AutoProcessor.from_pretrained(model_id)
assistant_model = AutoModelForCausalLM.from_pretrained(
    assistant_model_id,
    low_cpu_mem_usage=False,
    use_safetensors=True,
    attn_implementation=attention,
    torch_dtype=torch_dtype,
).to(device)

dataset = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy",
    "clean",
    split="validation",
    trust_remote_code=True,
)
if not torch.cuda.is_available():
    # keep the CPU run short
    dataset = dataset.select(range(min(5, len(dataset))))
wer = load("wer")

generate_kwargs_1 = {**base_kwargs}
generate_kwargs_2 = {**base_kwargs, "assistant_model": assistant_model}

spec_decoding = False
for i, generate_kwargs in enumerate([generate_kwargs_1, generate_kwargs_2]):
    all_time = 0
    predictions = []
    references = []
    for sample in tqdm(dataset):
        audio = sample["audio"]
        inputs = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        )
        inputs = inputs.to(device=device, dtype=torch_dtype)
        start_time = perf_counter()
        output = model.generate(
            **inputs,
            **generate_kwargs,
        )
        gen_time = perf_counter() - start_time
        all_time += gen_time
        predictions.append(
            processor.batch_decode(
                output, skip_special_tokens=True, normalize=True
            )[0]
        )
        references.append(processor.tokenizer.normalize(sample["text"]))
    score = wer.compute(predictions=predictions, references=references)
    if i > 0:
        spec_decoding = True
    print(f"Speculative Decoding: {spec_decoding}")
    print(f"Time: {all_time}")
    print(f"Word Error Rate: {score}")
