#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/qwen3_5_sft_pyc}"
unset PYTHONDONTWRITEBYTECODE

IFS=',' read -r -a CUDA_DEVICE_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE="${#CUDA_DEVICE_ARRAY[@]}"

# Prewarm heavy imports once so DDP workers avoid pycache write contention.
uv run python -c "import sympy, torch, transformers"

uv run torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes=1 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${SCRIPT_DIR}/train.py"
