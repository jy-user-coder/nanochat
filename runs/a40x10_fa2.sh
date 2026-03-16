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
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-1146880}"
WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"
WANDB_RUN="${WANDB_RUN:-dummy}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"

WORLD_TOKENS=$((NPROC_PER_NODE * DEVICE_BATCH_SIZE * MAX_SEQ_LEN))
if [ $((TOTAL_BATCH_SIZE % WORLD_TOKENS)) -ne 0 ]; then
    echo "TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE must be divisible by world micro-batch $WORLD_TOKENS" >&2
    exit 1
fi

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu

if ! .venv/bin/python -c "import flash_attn" >/dev/null 2>&1; then
    uv pip install --python .venv/bin/python --no-build-isolation "$FLASH_ATTN_WHEEL_URL"
fi

source .venv/bin/activate

python -m nanochat.report reset

python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
    --depth=24 \
    --target-param-data-ratio=8 \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --total-batch-size=$TOTAL_BATCH_SIZE \
    --max-seq-len=$MAX_SEQ_LEN \
    --window-pattern=$WINDOW_PATTERN \
    --run=$WANDB_RUN

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- \
    --device-batch-size=$DEVICE_BATCH_SIZE

curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --run=$WANDB_RUN

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

python -m nanochat.report generate
