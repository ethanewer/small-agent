#!/bin/bash

# Usage:
#   mkdir -p logs outputs checkpoints
#   sbatch training-scripts/train_locooperator_balanced_65k_b200_megatron_train.sbatch

#SBATCH -J ee-locoop-4b-mega-train
#SBATCH -p b200
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=192
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err

set -euo pipefail
set -x

IMAGE="/wbl-fast/usrs/shl/enroot/modelscope-torch2.8.sqsh"
RUN_ID="locooperator4b_balanced_65k_megatron_train_${SLURM_JOB_ID}"
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
BASE_MODEL_DIR="${CONTAINER_WORKDIR}/checkpoints/LocoreMind-LocoOperator-4B"
MCORE_MODEL_DIR="${CONTAINER_WORKDIR}/checkpoints/LocoreMind-LocoOperator-4B-mcore"
DATASET_PATH="${CONTAINER_WORKDIR}/data/balanced-dataset/dataset_shuffled.jsonl"
CONTEXT_LENGTH="65536"
MAX_STEPS="100000"
SAVE_STEPS="2000"
MICRO_BATCH_SIZE="1"
GLOBAL_BATCH_SIZE="8"
LEARNING_RATE="1e-6"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"

export CONTAINER_WORKDIR BASE_MODEL_DIR MCORE_MODEL_DIR DATASET_PATH CONTEXT_LENGTH
export MAX_STEPS SAVE_STEPS MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE LEARNING_RATE
export DATASET_NUM_PROC

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

echo "[train] run root: ${RUN_ROOT}"
echo "[train] base model: ${BASE_MODEL_DIR}"
echo "[train] mcore model: ${MCORE_MODEL_DIR}"
echo "[train] dataset: ${DATASET_PATH}"
echo "[train] context length: ${CONTEXT_LENGTH}"
echo "[train] output dir: ${OUTPUT_DIR}"

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

    mkdir -p "${CONTAINER_WORKDIR}/logs" "${CONTAINER_WORKDIR}/outputs" "${MODELSCOPE_CACHE}"

    if [[ ! -d "${BASE_MODEL_DIR}" ]]; then
      echo "[preflight] base model directory missing: ${BASE_MODEL_DIR}" >&2
      exit 2
    fi
    if [[ ! -d "${MCORE_MODEL_DIR}" ]]; then
      echo "[preflight] exported mcore model missing: ${MCORE_MODEL_DIR}" >&2
      exit 3
    fi
    if [[ ! -f "${DATASET_PATH}" ]]; then
      echo "[preflight] dataset missing: ${DATASET_PATH}" >&2
      exit 4
    fi
    if ! command -v megatron >/dev/null 2>&1; then
      echo "[preflight] megatron CLI not found in container PATH" >&2
      exit 5
    fi

    python -c "import torch; print(f\"[preflight] torch={torch.__version__}, ngpu={torch.cuda.device_count()}\")"

    megatron sft \
      --model "${BASE_MODEL_DIR}" \
      --load "${MCORE_MODEL_DIR}" \
      --dataset "${DATASET_PATH}" \
      --load_from_cache_file true \
      --tensor_model_parallel_size 1 \
      --pipeline_model_parallel_size 1 \
      --sequence_parallel false \
      --micro_batch_size "${MICRO_BATCH_SIZE}" \
      --global_batch_size "${GLOBAL_BATCH_SIZE}" \
      --packing true \
      --recompute_granularity full \
      --recompute_method uniform \
      --recompute_num_layers 1 \
      --finetune true \
      --cross_entropy_loss_fusion true \
      --lr "${LEARNING_RATE}" \
      --lr_warmup_fraction 0.05 \
      --min_lr 1e-7 \
      --train_iters "${MAX_STEPS}" \
      --log_interval 10 \
      --save "${CONTAINER_WORKDIR}/outputs/${RUN_ID}" \
      --save_interval "${SAVE_STEPS}" \
      --max_length "${CONTEXT_LENGTH}" \
      --dataset_num_proc "${DATASET_NUM_PROC}" \
      --no_save_optim true \
      --no_save_rng true \
      --attention_backend flash
  '

echo "[train] submitted ${SLURM_JOB_ID}"
echo "[train] logs dir: ${LOG_DIR}"
echo "[train] output dir: ${OUTPUT_DIR}"
