"""Download small local copies of the datasets used by the notebook.

Creates (in ../../data, skipped if they already exist):
  CombinedTinyStories-train.txt / CombinedTinyStories-valid.txt
  Lima-train.jsonl / Lima-test.jsonl
Set N_TRAIN / N_VALID to change how many TinyStories lines are written.
"""
from dotenv import load_dotenv
import json
import os

from datasets import load_dataset
from huggingface_hub import hf_hub_download

# 프로젝트 루트의 .env 를 환경변수로 로드
load_dotenv()

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"
)
N_TRAIN = int(os.getenv("N_TRAIN", 20000))
N_VALID = int(os.getenv("N_VALID", 2000))


def tinystories():
    train = os.path.join(DATA_DIR, "CombinedTinyStories-train.txt")
    valid = os.path.join(DATA_DIR, "CombinedTinyStories-valid.txt")
    if os.path.exists(train) and os.path.exists(valid):
        return
    ds = load_dataset(
        "IMJONEZZ/CombinedTinyStories", split="train", streaming=True
    )
    with open(train, "w") as ft, open(valid, "w") as fv:
        for i, ex in enumerate(ds):
            if i >= N_TRAIN + N_VALID:
                break
            line = ex["text"].replace("\n", " ").strip()
            (ft if i < N_TRAIN else fv).write(line + "\n")


def lima():
    train = os.path.join(DATA_DIR, "Lima-train.jsonl")
    test = os.path.join(DATA_DIR, "Lima-test.jsonl")
    if os.path.exists(train) and os.path.exists(test):
        return
    # GAIR/lima is gated; the vicuna-format mirror is open.
    path = hf_hub_download(
        "64bits/lima_vicuna_format",
        "lima_vicuna_format.json",
        repo_type="dataset",
    )
    with open(path) as f:
        rows = json.load(f)
    n_test = max(1, len(rows) // 10)
    for fname, part in ((train, rows[n_test:]), (test, rows[:n_test])):
        with open(fname, "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    tinystories()
    lima()
    print("data ready in", os.path.abspath(DATA_DIR))
