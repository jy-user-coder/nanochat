#!/bin/bash

# Sweep Polar Express iteration counts 1-7, with and without NorMuon variance reduction,
# using the same d12/data4 training setup as runs/fast.sh.

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
SAVE_EVERY="${SAVE_EVERY:-200}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-D12-DATA4-polar-express}"
MODEL_TAG_PREFIX="${MODEL_TAG_PREFIX:-d12-data4-polar-express}"
RESULTS_DIR="${RESULTS_DIR:-$NANOCHAT_BASE_DIR/${MODEL_TAG_PREFIX}_logs}"
DRY_RUN="${DRY_RUN:-0}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"

mkdir -p "$RESULTS_DIR"
source .venv/bin/activate

run_experiment() {
    local polar_iters="$1"
    local use_normuon="$2"
    local normuon_tag="off"
    if [ "$use_normuon" -eq 1 ]; then
        normuon_tag="on"
    fi

    local suffix="pe${polar_iters}_normuon-${normuon_tag}"
    local run_name="${WANDB_RUN_PREFIX}-${suffix}"
    local model_tag="${MODEL_TAG_PREFIX}-${suffix}"
    local log_file="$RESULTS_DIR/${model_tag}.log"

    local -a cmd=(
        torchrun
        --standalone
        --nproc_per_node="$NPROC_PER_NODE"
        -m scripts.base_train
        --
        --depth=12
        --target-param-data-ratio=4
        --device-batch-size="$DEVICE_BATCH_SIZE"
        --total-batch-size="$TOTAL_BATCH_SIZE"
        --max-seq-len="$MAX_SEQ_LEN"
        --window-pattern="$WINDOW_PATTERN"
        --run="$run_name"
        --model-tag="$model_tag"
        --polar-express-iters="$polar_iters"
        --use-normuon="$use_normuon"
        --save-every="$SAVE_EVERY"
    )

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${suffix}"
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '  %q' "${cmd[@]}"
        printf '\n'
        return
    fi

    "${cmd[@]}" 2>&1 | tee "$log_file"
}

for polar_iters in 1 2 3 4 5 6 7; do
    run_experiment "$polar_iters" 0
    run_experiment "$polar_iters" 1
done
