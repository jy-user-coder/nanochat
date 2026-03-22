#!/bin/bash

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a blank 10xA40 GPU node with bf16 + Flash Attention 2.

# 1) Example launch (simplest):
# bash runs/speedrun_10A40.sh
# 2) Example launch in a screen session:
# screen -L -Logfile runs/speedrun_10A40.log -S speedrun_10A40 bash runs/speedrun_10A40.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun_10A40 screen -L -Logfile runs/speedrun_10A40.log -S speedrun_10A40 bash runs/speedrun_10A40.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-bfloat16}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
NPROC_PER_NODE="${NPROC_PER_NODE:-10}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-983040}"
WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"
mkdir -p "$NANOCHAT_BASE_DIR"

WORLD_TOKENS=$((NPROC_PER_NODE * DEVICE_BATCH_SIZE * MAX_SEQ_LEN))
if [ $((TOTAL_BATCH_SIZE % WORLD_TOKENS)) -ne 0 ]; then
    echo "TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE must be divisible by world micro-batch $WORLD_TOKENS" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# make sure Python 3.12 is available so the prebuilt FA2 wheel can be installed
uv python install "$PYTHON_VERSION"
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv --python "$PYTHON_VERSION"
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate
# fail fast if the existing venv is on the wrong Python for the prebuilt FA2 wheel
python - <<'PY'
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        "This script expects .venv to use Python 3.12 for the prebuilt flash-attn wheel. "
        "Recreate .venv with `uv venv --python 3.12` and rerun."
    )
PY
# install the repo dependencies
uv sync --extra gpu
# install the prebuilt FA2 wheel if it did not come in via dependency resolution
python -c "import flash_attn" >/dev/null 2>&1 || uv pip install --python .venv/bin/python --no-build-isolation "$FLASH_ATTN_WHEEL_URL"

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=dummy bash runs/speedrun_10A40.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Download the first ~2B characters of pretraining dataset
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# look at dev/repackage_data_reference.py for details on how this data was prepared
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# Approximately 150 shards are needed for GPT-2 capability pretraining, add 20 for padding.
# The maximum total number of shards available in the entire dataset is 6542.
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 32770 (rounded up from 32768 to divide by 10) on ~2B characters of data
python -m scripts.tok_train --vocab-size=32770
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# d24 model on 10xA40, explicitly bf16 with FA2-friendly SSSL attention
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=$DEVICE_BATCH_SIZE --total-batch-size=$TOTAL_BATCH_SIZE --max-seq-len=$MAX_SEQ_LEN --window-pattern=$WINDOW_PATTERN --run=$WANDB_RUN
# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- --device-batch-size=$DEVICE_BATCH_SIZE

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run SFT and eval the model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --device-batch-size=$DEVICE_BATCH_SIZE --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
