#!/bin/bash

# Usage:
#   mkdir -p logs outputs checkpoints
#   sbatch training-scripts/train_locooperator_balanced_65k_b200_megatron_export.sbatch

#SBATCH -J ee-locoop-4b-mega-export
#SBATCH -p b200
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=192
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err

set -euo pipefail
set -x

IMAGE="/wbl-fast/usrs/shl/enroot/modelscope-torch2.8.sqsh"
RUN_ID="locooperator4b_balanced_65k_megatron_export_${SLURM_JOB_ID}"
export RUN_ID

JOB_COMMAND="$(scontrol show job "${SLURM_JOB_ID}" | sed -n "s/.* Command=\([^ ]*\).*/\1/p")"
if [[ "${JOB_COMMAND}" = /* ]]; then
  SCRIPT_PATH="${JOB_COMMAND}"
else
  SCRIPT_PATH="${SLURM_SUBMIT_DIR}/${JOB_COMMAND}"
fi
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
RUN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${RUN_ROOT}/logs"
OUTPUT_BASE="${RUN_ROOT}/outputs"
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_ID}"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
mkdir -p "${LOG_DIR}" "${OUTPUT_BASE}" "${OUTPUT_DIR}" "${CHECKPOINT_DIR}"

CONTAINER_WORKDIR="/workspace/run"
MODEL_ID="${CONTAINER_WORKDIR}/checkpoints/LocoreMind-LocoOperator-4B"
MCORE_OUTPUT_DIR="${CONTAINER_WORKDIR}/checkpoints/LocoreMind-LocoOperator-4B-mcore"

export CONTAINER_WORKDIR MODEL_ID MCORE_OUTPUT_DIR

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
echo "[export] mcore output: ${MCORE_OUTPUT_DIR}"

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

    export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/efa/lib64:/opt/amazon/ofi-nccl/lib:/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/workspace/efa_overlay
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

    mkdir -p "${CONTAINER_WORKDIR}/logs" "${CONTAINER_WORKDIR}/outputs" "${CONTAINER_WORKDIR}/checkpoints" "${MODELSCOPE_CACHE}"

    if [[ ! -d "${MODEL_ID}" ]]; then
      echo "[preflight] source model directory missing: ${MODEL_ID}" >&2
      exit 2
    fi

    if ! command -v swift >/dev/null 2>&1; then
      echo "[preflight] swift CLI not found in container PATH" >&2
      exit 3
    fi

    python -c "import torch; print(f\"[preflight] torch={torch.__version__}, ngpu={torch.cuda.device_count()}\")"

    rm -rf "${MCORE_OUTPUT_DIR}"
    swift export \
      --model "${MODEL_ID}" \
      --output_dir "${MCORE_OUTPUT_DIR}" \
      --to_mcore true \
      --test_convert_precision true
  '

echo "[export] finished ${SLURM_JOB_ID}"
echo "[export] logs dir: ${LOG_DIR}"
echo "[export] output dir: ${OUTPUT_DIR}"
