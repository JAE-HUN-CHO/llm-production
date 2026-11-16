import argparse
from openai import OpenAI

import os
import numpy as np
import pandas as pd
import time
from urllib import request
import tarfile

from utils import crop



# 루트 .env 의 OPENAI_BASE_URL / OPENAI_API_KEY / OLLAMA_MODEL 을 읽는다
def _load_env():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"
    )
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env()

# 로컬 Ollama(OpenAI 호환) 엔드포인트 또는 원하는 모델을 사용
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)
choices = ["A", "B", "C", "D"]


# Download and extract MMLU Dataset
def download_and_extract_mmlu(dir):
    mmlu = os.path.join(dir, "data.tar")
    request.urlretrieve(
        "https://people.eecs.berkeley.edu/~hendrycks/data.tar", mmlu
    )
    mmlu_tar = tarfile.open(mmlu)
    mmlu_tar.extractall(dir)
    mmlu_tar.close()


# Helper functions
def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator / denominator
    return softmax


def format_subject(subject):
    entries = subject.split("_")
    s = ""
    for entry in entries:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = (
        "The following are multiple choice questions "
        "(with answers) about {}.\n\n".format(format_subject(subject))
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


# Here we will evaluate the model against the dataset
def eval(args, subject, engine, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # Generate prompt from dataset
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1] - 1]

        # Prompt your model
        # 로그확률(logprobs+echo)을 지원하지 않는 로컬 모델(Ollama 등)은
        # chat 응답에서 A/B/C/D 를 파싱하는 방식으로 대체한다.
        lprobs = None
        for _ in range(3):
            try:
                c = client.completions.create(
                    model=engine,
                    prompt=prompt,
                    max_tokens=1,
                    logprobs=100,
                    temperature=0,
                    echo=True,
                )  # Use your model here!
                lprobs = []
                for ans in answers:
                    lprobs.append(
                        c.choices[0].logprobs.top_logprobs[-1].get(
                            " {}".format(ans), -100
                        )
                    )
                break
            except Exception:
                time.sleep(1)
        if lprobs is None:
            resp = client.chat.completions.create(
                model=engine,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                        + " (Reply with only the letter of the answer.)",
                    }
                ],
                max_tokens=2048,  # 추론 모델은 reasoning 에 토큰을 소비한다
                temperature=0,
            )
            msg = resp.choices[0].message
            text = (msg.content or "").strip()
            found = [a for a in answers if text[:3].upper().startswith(a)]
            if not found:
                import re

                m = re.findall(r"\b([A-D])\b", text)
                found = [m[-1]] if m and m[-1] in answers else []
            lprobs = [0 if a in found else -100 for a in answers]

        # Evaluate models response to questions answer
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
        probs = softmax(np.array(lprobs))

        # Record results
        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    # Summarize results
    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


# Run Evaluations
def main(args):
    engines = args.engine

    # Download and prepare dataset subjects
    download_and_extract_mmlu(args.data_dir)
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "data/test"))
            if "_test.csv" in f
        ]
    )

    # Create directories to save results to if they don't exist
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    for engine in engines:
        if not os.path.exists(
            os.path.join(args.save_dir, "results_{}".format(engine.replace("/", "_").replace(":", "_")))
        ):
            os.mkdir(
                os.path.join(args.save_dir, "results_{}".format(engine.replace("/", "_").replace(":", "_")))
            )

    if args.max_subjects:
        subjects = subjects[: args.max_subjects]
    print(subjects)
    print(args)

    # Run evaluations for each engine
    for engine in engines:
        print(engine)
        all_cors = []

        for subject in subjects:
            # Prepare examples for prompting
            dev_df = pd.read_csv(
                os.path.join(
                    args.data_dir, "data/dev", subject + "_dev.csv"
                ),
                header=None,
            )[: args.ntrain]
            # Prepare actual test questions
            test_df = pd.read_csv(
                os.path.join(
                    args.data_dir, "data/test", subject + "_test.csv"
                ),
                header=None,
            )
            if args.max_questions:
                test_df = test_df[: args.max_questions].copy()

            # Run evaluations
            cors, acc, probs = eval(args, subject, engine, dev_df, test_df)
            all_cors.append(cors)

            # Save results
            test_df["{}_correct".format(engine)] = cors
            for j in range(probs.shape[1]):
                choice = choices[j]
                test_df["{}_choice{}_probs".format(engine, choice)] = probs[
                    :, j
                ]
            test_df.to_csv(
                os.path.join(
                    args.save_dir,
                    "results_{}".format(engine.replace("/", "_").replace(":", "_")),
                    "{}.csv".format(subject),
                ),
                index=None,
            )

        weighted_acc = np.mean(np.concatenate(all_cors))
        print("Average accuracy: {:.3f}".format(weighted_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ntrain",
        "-k",
        type=int,
        default=5,
        help="Number of samples to include for n-shot prompting",
    )
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument(
        "--engine",
        "-e",
        default=[
            os.environ.get(
                "OLLAMA_MODEL", "davinci"
            )
        ],
        nargs="+",
        help="OpenAI model(s) you would like to evaluate",
    )
    parser.add_argument(
        "--max_subjects", type=int, default=None, help="평가할 과목 수 제한"
    )
    parser.add_argument(
        "--max_questions", type=int, default=None, help="과목당 문항 수 제한"
    )
    args = parser.parse_args()
    main(args)
