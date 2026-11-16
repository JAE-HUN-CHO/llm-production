#!/bin/bash
# 모든 챕터를 실행할 수 있는 단일 conda 환경(llm-production)을 만든다.

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

if ! command -v conda &>/dev/null; then
  echo "conda not found. Please install Miniconda or Anaconda."
  exit 1
fi

if ! command -v git-lfs &> /dev/null; then
  echo "please install git-lfs, e.g. 'brew install git-lfs' or visit https://git-lfs.com/ for your OS instructions"
  exit 1
fi

ENV_NAME=llm-production

if conda info --envs | grep -q "^$ENV_NAME "; then
  echo "Environment '$ENV_NAME' already exists. Skipping environment creation."
else
  conda create -n $ENV_NAME python=3.10 --yes
fi

conda activate $ENV_NAME

# deepeval 은 feast 와 protobuf 요구 버전이 충돌하므로 먼저 설치하고, 나머지가 protobuf 를 낮춘다
pip install deepeval==1.1.0
pip install -r requirements-all.txt
pip install --no-deps langchain-pinecone==0.1.1

conda deactivate

echo -e "\nYou are now set up.\n\nPlease run 'conda activate $ENV_NAME' to begin."
