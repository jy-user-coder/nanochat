#!/bin/bash

# FA2 + SSSL path tuned for a 10xA40 node.
# This keeps the repo's d24 GPT-2-grade setup, but swaps the Hopper-specific
# assumptions for Ampere-friendly ones:
# - bf16 compute
# - Flash Attention 2 (installed into the local venv if missing)
# - SSSL attention windows
# - 10 GPUs with a total batch size divisible by the world micro-batch
#
# Notes:
# - The default 10xA40 micro-batch here is 10 * 8 * 2048 = 163,840 tokens.
# - TOTAL_BATCH_SIZE must be divisible by that number or base_train will assert.
# - If you lower DEVICE_BATCH_SIZE to avoid OOM, also lower TOTAL_BATCH_SIZE.

set -euo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-bfloat16}"
mkdir -p "$NANOCHAT_BASE_DIR"

NPROC_PER_NODE="${NPROC_PER_NODE:-10}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-26}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-532480}"
WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"
WANDB_RUN="${WANDB_RUN:-D12-DATA4}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"

source .venv/bin/activate

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
    --depth=12 \
    --target-param-data-ratio=4 \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --total-batch-size=$TOTAL_BATCH_SIZE \
    --max-seq-len=$MAX_SEQ_LEN \
    --window-pattern=$WINDOW_PATTERN \
    --run=$WANDB_RUN
