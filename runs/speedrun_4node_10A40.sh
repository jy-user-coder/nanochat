#!/bin/bash

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# across 4 nodes of 10xA40 GPUs using bf16 + Flash Attention 2.
#
# Assumptions:
# - You launch this script on all 4 nodes at roughly the same time.
# - All nodes share the same NANOCHAT_BASE_DIR so rank 0 can prepare tokenizer/data once.
# - Each node may keep its own local .venv; this script sets that up independently per node.
#
# Example launch on node ranks 0..3:
# MASTER_ADDR=node0 MASTER_PORT=29500 NODE_RANK=0 WANDB_RUN=speedrun_4node bash runs/speedrun_4node_10A40.sh
# MASTER_ADDR=node0 MASTER_PORT=29500 NODE_RANK=1 WANDB_RUN=speedrun_4node bash runs/speedrun_4node_10A40.sh
# MASTER_ADDR=node0 MASTER_PORT=29500 NODE_RANK=2 WANDB_RUN=speedrun_4node bash runs/speedrun_4node_10A40.sh
# MASTER_ADDR=node0 MASTER_PORT=29500 NODE_RANK=3 WANDB_RUN=speedrun_4node bash runs/speedrun_4node_10A40.sh

set -euo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-bfloat16}"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
NNODES="${NNODES:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-10}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-1310720}"
WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"
MASTER_PORT="${MASTER_PORT:-29500}"
WANDB_RUN="${WANDB_RUN:-dummy}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"

if [ -z "${MASTER_ADDR:-}" ]; then
    echo "Set MASTER_ADDR to the hostname or IP of node rank 0." >&2
    exit 1
fi
if [ -z "${NODE_RANK:-}" ]; then
    echo "Set NODE_RANK to 0..$((NNODES - 1))." >&2
    exit 1
fi
if [ "$NODE_RANK" -lt 0 ] || [ "$NODE_RANK" -ge "$NNODES" ]; then
    echo "NODE_RANK=$NODE_RANK is out of range for NNODES=$NNODES." >&2
    exit 1
fi

mkdir -p "$NANOCHAT_BASE_DIR"

WORLD_TOKENS=$((NNODES * NPROC_PER_NODE * DEVICE_BATCH_SIZE * MAX_SEQ_LEN))
if [ $((TOTAL_BATCH_SIZE % WORLD_TOKENS)) -ne 0 ]; then
    echo "TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE must be divisible by global micro-batch $WORLD_TOKENS" >&2
    exit 1
fi

MARKER_DIR="$NANOCHAT_BASE_DIR/run_markers/${WANDB_RUN}_4node_10A40"
PREP_DONE_MARKER="$MARKER_DIR/pretrain_prep.done"
IDENTITY_DONE_MARKER="$MARKER_DIR/identity.done"

wait_for_marker() {
    local marker_path="$1"
    local label="$2"
    echo "Node $NODE_RANK waiting for $label..."
    while [ ! -f "$marker_path" ]; do
        sleep 5
    done
}

run_torchrun() {
    .venv/bin/torchrun \
        --nnodes="$NNODES" \
        --nproc_per_node="$NPROC_PER_NODE" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        "$@"
}

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install "$PYTHON_VERSION"
[ -d ".venv" ] || uv venv --python "$PYTHON_VERSION"
source .venv/bin/activate

python - <<'PY'
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        "This script expects .venv to use Python 3.12 for the prebuilt flash-attn wheel. "
        "Recreate .venv with `uv venv --python 3.12` and rerun."
    )
PY

uv sync --extra gpu
python -c "import flash_attn" >/dev/null 2>&1 || uv pip install --python .venv/bin/python --no-build-isolation "$FLASH_ATTN_WHEEL_URL"

# -----------------------------------------------------------------------------
# One-time shared prep on node rank 0

if [ "$NODE_RANK" = "0" ]; then
    rm -rf "$MARKER_DIR"
    mkdir -p "$MARKER_DIR"

    python -m nanochat.report reset

    # Tokenizer/data prep
    python -m nanochat.dataset -n 8
    python -m nanochat.dataset -n 170 &
    DATASET_DOWNLOAD_PID=$!
    python -m scripts.tok_train --vocab-size=32770
    python -m scripts.tok_eval

    echo "Node 0 waiting for shared dataset download to complete..."
    wait $DATASET_DOWNLOAD_PID
    touch "$PREP_DONE_MARKER"
else
    wait_for_marker "$PREP_DONE_MARKER" "rank 0 tokenizer/data preparation"
fi

# -----------------------------------------------------------------------------
# Base model (pretraining)

run_torchrun -m scripts.base_train -- \
    --depth=24 \
    --target-param-data-ratio=8 \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --max-seq-len="$MAX_SEQ_LEN" \
    --window-pattern="$WINDOW_PATTERN" \
    --run="$WANDB_RUN"

run_torchrun -m scripts.base_eval -- \
    --device-batch-size="$DEVICE_BATCH_SIZE"

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

if [ "$NODE_RANK" = "0" ]; then
    curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
    touch "$IDENTITY_DONE_MARKER"
else
    wait_for_marker "$IDENTITY_DONE_MARKER" "rank 0 identity data download"
fi

run_torchrun -m scripts.chat_sft -- \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --run="$WANDB_RUN"

run_torchrun -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
# Generate the full report on node 0 only

if [ "$NODE_RANK" = "0" ]; then
    python -m nanochat.report generate
fi
