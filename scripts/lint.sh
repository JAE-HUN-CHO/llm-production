#!/bin/bash

# If black/ruff are not on PATH, try the 'llmbook' conda env (see setup.sh)
if ! command -v black &>/dev/null || ! command -v ruff &>/dev/null; then
  for base in "${CONDA_PREFIX_1:-}" "$(conda info --base 2>/dev/null)"; do
    if [ -n "$base" ] && [ -x "$base/envs/llmbook/bin/black" ]; then
      export PATH="$base/envs/llmbook/bin:$PATH"
      break
    fi
  done
fi

# Check if black and ruff are installed
if ! command -v black &>/dev/null; then
  echo "'black' is not installed. Please install black with 'pip install black'."
  exit 1
fi

if ! command -v ruff &>/dev/null; then
  echo "'ruff' is not installed. Please install ruff with 'pip install ruff'"
  exit 1
fi

# Set your source code folder
src_folder="chapters"

# Run black for formatting
echo "Running black on the $src_folder:"
black $src_folder

# Run ruff for linting
echo "Running ruff on the $src_folder:"
ruff check --fix $src_folder
