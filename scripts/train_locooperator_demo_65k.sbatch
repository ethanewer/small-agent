#!/bin/bash

# NOTE: Ensure these exist before submit so Slurm can open stdout/stderr:
#   mkdir -p logs outputs
#   sbatch scripts/train_locooperator_full_65k_b200.sbatch

#SBATCH -J ee-locoop-4b-demo-65k-ddp8
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
RUN_ID="locooperator4b_demo_65k_ddp8_${SLURM_JOB_ID}"
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
mkdir -p "${LOG_DIR}" "${OUTPUT_BASE}" "${OUTPUT_DIR}"

MODEL_ID="/workspace/run/checkpoints/LocoreMind-LocoOperator-4B"
DATASET_REL_PATH="data/demo-dataset/train_ddp_debug.jsonl"
CONTEXT_LENGTH="65536"
NUM_TRAIN_EPOCHS="1"
MAX_STEPS="10"
SAVE_STEPS="5"
PER_DEVICE_TRAIN_BATCH_SIZE="1"
GRADIENT_ACCUMULATION_STEPS="1"
LEARNING_RATE="1e-6"
DEEPSPEED_CONFIG=""

CONTAINER_WORKDIR="/workspace/run"
CONTAINER_DATASET_PATH="${CONTAINER_WORKDIR}/${DATASET_REL_PATH}"
CONTAINER_OUTPUT_DIR="${CONTAINER_WORKDIR}/outputs/${RUN_ID}"
export MODEL_ID DATASET_REL_PATH CONTEXT_LENGTH NUM_TRAIN_EPOCHS MAX_STEPS SAVE_STEPS
export PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS LEARNING_RATE
export DEEPSPEED_CONFIG CONTAINER_WORKDIR CONTAINER_DATASET_PATH CONTAINER_OUTPUT_DIR

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_TIMEOUT=120
export NCCL_NET=OFI
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet"
export PYTHONUNBUFFERED=1

echo "[train] run root: ${RUN_ROOT}"
echo "[train] output dir: ${OUTPUT_DIR}"
echo "[train] model: ${MODEL_ID}"
echo "[train] dataset: ${DATASET_REL_PATH}"
echo "[train] context length: ${CONTEXT_LENGTH}"

if [[ ! -f "${RUN_ROOT}/${DATASET_REL_PATH}" ]]; then
  echo "[train] dataset file missing: ${RUN_ROOT}/${DATASET_REL_PATH}" >&2
  exit 1
fi

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

    mkdir -p "${CONTAINER_WORKDIR}/logs" "${CONTAINER_WORKDIR}/outputs" "${CONTAINER_OUTPUT_DIR}" "${MODELSCOPE_CACHE}"

    if [[ ! -f "${CONTAINER_DATASET_PATH}" ]]; then
      echo "[preflight] dataset missing in container: ${CONTAINER_DATASET_PATH}" >&2
      exit 2
    fi

    if ! command -v swift >/dev/null 2>&1; then
      echo "[preflight] swift CLI not found in container PATH" >&2
      exit 3
    fi

    swift sft \
      --model "${MODEL_ID}" \
      --dataset "${CONTAINER_DATASET_PATH}" \
      --train_type full \
      --attn_impl flash_attention_2 \
      --torch_dtype bfloat16 \
      --max_length "${CONTEXT_LENGTH}" \
      --packing true \
      --packing_length "${CONTEXT_LENGTH}" \
      --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
      --max_steps "${MAX_STEPS}" \
      --learning_rate "${LEARNING_RATE}" \
      --warmup_steps 1000 \
      --lr_scheduler_type constant_with_warmup \
      --logging_steps 1 \
      --save_steps "${SAVE_STEPS}" \
      --save_total_limit 2 \
      --dataloader_num_workers 4 \
      --output_dir "${CONTAINER_OUTPUT_DIR}"
  '

echo "[train] submitted ${SLURM_JOB_ID}"
echo "[train] logs dir: ${LOG_DIR}"
echo "[train] output dir: ${OUTPUT_DIR}"
