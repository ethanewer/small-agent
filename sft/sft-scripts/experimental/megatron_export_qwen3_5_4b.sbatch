#!/bin/bash

# Usage:
#   cd sft
#   sbatch sft-scripts/megatron_export_qwen3_5_4b.sbatch

#SBATCH -J qwen35-4b-export
#SBATCH -p h200
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=192
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH -o logs-20260302/export_qwen35_%j.out
#SBATCH -e logs-20260302/export_qwen35_%j.err

set -euo pipefail
set -x

IMAGE="${IMAGE:-/wbl-fast/usrs/shl/enroot/modelscope-torch2.8.sqsh}"
EXPORT_NAME="${EXPORT_NAME:-Qwen3.5-4B-mcore}"

JOB_COMMAND="$(scontrol show job "${SLURM_JOB_ID}" | sed -n "s/.* Command=\([^ ]*\).*/\1/p")"
if [[ "${JOB_COMMAND}" = /* ]]; then
  SCRIPT_PATH="${JOB_COMMAND}"
else
  SCRIPT_PATH="${SLURM_SUBMIT_DIR}/${JOB_COMMAND}"
fi
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
RUN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${RUN_ROOT}/logs-20260302"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
MCORE_OUTPUT_DIR="${CHECKPOINT_DIR}/${EXPORT_NAME}"
mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"

CONTAINER_WORKDIR="/workspace/run"
LOCAL_MODEL_DIR="${RUN_ROOT}/checkpoints/Qwen3.5-4B"
CONTAINER_MODEL_DIR="${CONTAINER_WORKDIR}/checkpoints/Qwen3.5-4B"
CONTAINER_MCORE_OUTPUT_DIR="${CONTAINER_WORKDIR}/checkpoints/${EXPORT_NAME}"
MODEL_ID="${MODEL_ID:-${CONTAINER_MODEL_DIR}}"
export CONTAINER_WORKDIR CONTAINER_MODEL_DIR MODEL_ID CONTAINER_MCORE_OUTPUT_DIR

if [[ ! -d "${LOCAL_MODEL_DIR}" ]]; then
  echo "[preflight] local model directory missing: ${LOCAL_MODEL_DIR}" >&2
  exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0
export NCCL_NET=OFI
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet"
export NCCL_TIMEOUT=120
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1

echo "[export] run root: ${RUN_ROOT}"
echo "[export] model: ${MODEL_ID}"
echo "[export] output: ${MCORE_OUTPUT_DIR}"

srun --mpi=pmix \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-task=8 \
  --cpus-per-task=192 \
  --container-image="${IMAGE}" \
  --container-mounts="${RUN_ROOT}:${CONTAINER_WORKDIR},/opt/amazon/efa:/opt/amazon/efa,/opt/amazon/ofi-nccl:/opt/amazon/ofi-nccl,/wbl-fast/usrs/dongwon/efa_overlay:/workspace/efa_overlay,/wbl:/wbl,/wbl-fast:/wbl-fast,/dev/shm:/dev/shm" \
  --container-workdir="${CONTAINER_WORKDIR}" \
  --container-writable \
  bash -lc '
    set -euo pipefail
    set -x

    export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/efa/lib64:/opt/amazon/ofi-nccl/lib:/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/workspace/efa_overlay:${LD_LIBRARY_PATH:-}
    export PATH=/opt/amazon/efa/bin:${PATH}
    export NCCL_IB_DISABLE=0
    export NCCL_NET=OFI
    export FI_PROVIDER=efa
    export FI_EFA_USE_DEVICE_RDMA=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export TE_DISABLE_UB=1
    export NCCL_NVLS_ENABLE=0
    export MODELSCOPE_CACHE="${CONTAINER_WORKDIR}/.modelscope_cache"

    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export NPROC_PER_NODE=8
    export NNODES=1

    mkdir -p "${CONTAINER_WORKDIR}/logs-20260302" "${CONTAINER_WORKDIR}/checkpoints" "${MODELSCOPE_CACHE}"

    if ! command -v swift >/dev/null 2>&1; then
      echo "[preflight] swift CLI not found in container PATH" >&2
      exit 3
    fi

    if [[ ! -d "${MODEL_ID}" ]]; then
      echo "[preflight] model directory missing in container: ${MODEL_ID}" >&2
      exit 4
    fi

    python -c "import torch; print(f\"[preflight] torch={torch.__version__}, ngpu={torch.cuda.device_count()}\")"

    # Qwen3.5 config uses model_type=qwen3_5, which needs newer Transformers.
    if ! python -c "from transformers import AutoConfig; AutoConfig.from_pretrained(\"${MODEL_ID}\", trust_remote_code=True)" >/dev/null 2>&1; then
      echo "[preflight] upgrading transformers for qwen3_5 support"
      python -m pip install -U "transformers @ git+https://github.com/huggingface/transformers.git"
      python -m pip install -U "peft @ git+https://github.com/huggingface/peft.git"
      python -c "from transformers import AutoConfig; AutoConfig.from_pretrained(\"${MODEL_ID}\", trust_remote_code=True); print(\"[preflight] transformers supports qwen3_5\")"
    fi

    # Ensure swift import path is compatible with the currently installed transformers/peft.
    python -c "import swift; from swift.cli.main import cli_main; print(\"[preflight] swift import ok\")"

    rm -rf "${CONTAINER_MCORE_OUTPUT_DIR}"
    swift export \
      --model "${MODEL_ID}" \
      --output_dir "${CONTAINER_MCORE_OUTPUT_DIR}" \
      --to_mcore true \
      --test_convert_precision true
  '

echo "[export] finished ${SLURM_JOB_ID}"
echo "[export] output dir: ${MCORE_OUTPUT_DIR}"
