#!/bin/bash

# Usage:
#   mkdir -p logs outputs
#   sbatch training-scripts/train_locooperator_balanced_65k_b200_fast_ddp.sbatch

#SBATCH -J ee-locoop-4b-bal65k-fast-ddp
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
RUN_ID="locooperator4b_balanced_65k_fast_ddp_${SLURM_JOB_ID}"
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
DATASET_REL_PATH="data/balanced-dataset/dataset_shuffled.jsonl"
CONTEXT_LENGTH="65536"
NUM_TRAIN_EPOCHS="1"
MAX_STEPS="100000"
SAVE_STEPS="2000"
PER_DEVICE_TRAIN_BATCH_SIZE="1"
GRADIENT_ACCUMULATION_STEPS="1"
LEARNING_RATE="1e-6"

# Throughput knobs
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-false}"
ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-false}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"

CONTAINER_WORKDIR="/workspace/run"
CONTAINER_DATASET_PATH="${CONTAINER_WORKDIR}/${DATASET_REL_PATH}"
CONTAINER_OUTPUT_DIR="${CONTAINER_WORKDIR}/outputs/${RUN_ID}"

export MODEL_ID DATASET_REL_PATH CONTEXT_LENGTH NUM_TRAIN_EPOCHS MAX_STEPS SAVE_STEPS
export PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS LEARNING_RATE
export ATTN_IMPL USE_LIGER_KERNEL ENABLE_TORCH_COMPILE LOGGING_STEPS DATALOADER_NUM_WORKERS DATASET_NUM_PROC SAVE_ONLY_MODEL
export CONTAINER_WORKDIR CONTAINER_DATASET_PATH CONTAINER_OUTPUT_DIR

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
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export NVIDIA_TF32_OVERRIDE=0
export TORCHDYNAMO_DISABLE=1

echo "[train] run root: ${RUN_ROOT}"
echo "[train] output dir: ${OUTPUT_DIR}"
echo "[train] model: ${MODEL_ID}"
echo "[train] dataset: ${DATASET_REL_PATH}"
echo "[train] context length: ${CONTEXT_LENGTH}"
echo "[train] attn_impl: ${ATTN_IMPL}"
echo "[train] use_liger_kernel: ${USE_LIGER_KERNEL}"
echo "[train] torch_compile: ${ENABLE_TORCH_COMPILE}"

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
    export NODE_RANK=0
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=${MASTER_PORT:-29517}
    export OMP_NUM_THREADS=24
    export TORCHDYNAMO_DISABLE=1

    mkdir -p "${CONTAINER_WORKDIR}/logs" "${CONTAINER_WORKDIR}/outputs" "${CONTAINER_OUTPUT_DIR}" "${MODELSCOPE_CACHE}"

    if [[ ! -f "${CONTAINER_DATASET_PATH}" ]]; then
      echo "[preflight] dataset missing in container: ${CONTAINER_DATASET_PATH}" >&2
      exit 2
    fi

    if ! command -v swift >/dev/null 2>&1; then
      echo "[preflight] swift CLI not found in container PATH" >&2
      exit 3
    fi

    python -c "import torch; print(f\"[preflight] torch={torch.__version__}, ngpu={torch.cuda.device_count()}\")"
    echo "[preflight] WORLD_SIZE=${NPROC_PER_NODE}, MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

    torchrun \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --nnodes="${NNODES}" \
      --node_rank="${NODE_RANK}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${MASTER_PORT}" \
      -m swift.cli.sft \
      --model "${MODEL_ID}" \
      --dataset "${CONTAINER_DATASET_PATH}" \
      --train_type full \
      --attn_impl "${ATTN_IMPL}" \
      --use_liger_kernel "${USE_LIGER_KERNEL}" \
      --torch_compile "${ENABLE_TORCH_COMPILE}" \
      --torch_compile_backend inductor \
      --torch_compile_mode reduce-overhead \
      --torch_dtype bfloat16 \
      --max_length "${CONTEXT_LENGTH}" \
      --packing true \
      --packing_length "${CONTEXT_LENGTH}" \
      --load_from_cache_file true \
      --dataset_num_proc "${DATASET_NUM_PROC}" \
      --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
      --max_steps "${MAX_STEPS}" \
      --learning_rate "${LEARNING_RATE}" \
      --warmup_steps 1000 \
      --lr_scheduler_type constant_with_warmup \
      --logging_steps "${LOGGING_STEPS}" \
      --save_steps "${SAVE_STEPS}" \
      --save_total_limit 20 \
      --save_only_model "${SAVE_ONLY_MODEL}" \
      --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
      --dataloader_persistent_workers true \
      --use_logits_to_keep false \
      --output_dir "${CONTAINER_OUTPUT_DIR}"
  '

echo "[train] submitted ${SLURM_JOB_ID}"
echo "[train] logs dir: ${LOG_DIR}"
echo "[train] output dir: ${OUTPUT_DIR}"
