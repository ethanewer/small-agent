#!/bin/bash

# Usage:
#   mkdir -p logs outputs checkpoints
#   sbatch training-scripts/train_balanced_32k_b200_tp2_2node.sbatch

#SBATCH -J ee-balanced-32k-4b-tp2-2n
#SBATCH -p b200
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=192
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH -o logs-20260301/train_%j.out
#SBATCH -e logs-20260301/train_%j.err

set -euo pipefail
set -x

IMAGE="${IMAGE:-/wbl-fast/usrs/shl/enroot/modelscope-torch2.8.sqsh}"
MASTER_PORT="${MASTER_PORT:-29500}"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
NNODES="${SLURM_NNODES}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WORLD_SIZE="$((NNODES * NPROC_PER_NODE))"
TP_SIZE=2
PP_SIZE=1
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8
CONTEXT_LENGTH=32768
RUN_ID="locooperator4b_balanced_32k_megatron_tp2_2node_${SLURM_JOB_ID}"
export MASTER_PORT MASTER_ADDR NNODES NPROC_PER_NODE WORLD_SIZE RUN_ID

JOB_COMMAND="$(scontrol show job "${SLURM_JOB_ID}" | sed -n "s/.* Command=\([^ ]*\).*/\1/p")"
if [[ "${JOB_COMMAND}" = /* ]]; then
  SCRIPT_PATH="${JOB_COMMAND}"
else
  SCRIPT_PATH="${SLURM_SUBMIT_DIR}/${JOB_COMMAND}"
fi
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
RUN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${RUN_ROOT}/logs"
OUTPUT_BASE="${RUN_ROOT}/outputs-20260301"
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_ID}"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
mkdir -p "${LOG_DIR}" "${OUTPUT_BASE}" "${OUTPUT_DIR}" "${CHECKPOINT_DIR}"

CONTAINER_WORKDIR="/workspace/run"
CONTAINER_OUTPUT_BASE="${CONTAINER_WORKDIR}/outputs-20260301"
BASE_MODEL_DIR="${CONTAINER_WORKDIR}/checkpoints/LocoreMind-LocoOperator-4B"
MCORE_MODEL_DIR="${CONTAINER_WORKDIR}/checkpoints/LocoreMind-LocoOperator-4B-mcore"
DATASET_PATH="${CONTAINER_WORKDIR}/data/balanced-dataset/dataset_shuffled.jsonl"
MAX_STEPS="${MAX_STEPS:-100000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"

export CONTAINER_WORKDIR CONTAINER_OUTPUT_BASE BASE_MODEL_DIR MCORE_MODEL_DIR DATASET_PATH
export MAX_STEPS SAVE_STEPS LEARNING_RATE DATASET_NUM_PROC NUM_WORKERS
export TP_SIZE PP_SIZE MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE CONTEXT_LENGTH

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
echo "[train] run id: ${RUN_ID}"
echo "[train] nodes: ${NNODES}, gpus/node: ${NPROC_PER_NODE}, world_size: ${WORLD_SIZE}"
echo "[train] tp size: ${TP_SIZE}, pp size: ${PP_SIZE}"
echo "[train] context length: ${CONTEXT_LENGTH}"
echo "[train] global batch size: ${GLOBAL_BATCH_SIZE}"
echo "[train] output dir: ${OUTPUT_DIR}"

srun --mpi=pmix \
  --nodes="${NNODES}" \
  --ntasks-per-node=1 \
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
    export PYTHONUNBUFFERED=1
    export MODELSCOPE_CACHE="${CONTAINER_WORKDIR}/.modelscope_cache"

    mkdir -p "${CONTAINER_WORKDIR}/logs" "${CONTAINER_OUTPUT_BASE}" "${MODELSCOPE_CACHE}"

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
    if [[ ! -f "${MCORE_MODEL_DIR}/latest_checkpointed_iteration.txt" ]]; then
      echo "[preflight] mcore checkpoint marker missing: ${MCORE_MODEL_DIR}/latest_checkpointed_iteration.txt" >&2
      exit 5
    fi
    if ! command -v python >/dev/null 2>&1; then
      echo "[preflight] python not found in container PATH" >&2
      exit 6
    fi

    python -c "import torch; print(f\"[preflight] torch={torch.__version__}, ngpu={torch.cuda.device_count()}\")"

    if [[ "${SLURM_LOCALID}" == "0" ]]; then
      python -m torch.distributed.run \
        --nnodes="${NNODES}" \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --node_rank="${SLURM_NODEID}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        --rdzv_id="${RUN_ID}" \
        /usr/local/lib/python3.11/site-packages/swift/cli/_megatron/sft.py \
          --model "${BASE_MODEL_DIR}" \
          --load "${MCORE_MODEL_DIR}" \
          --dataset "${DATASET_PATH}" \
          --load_from_cache_file true \
          --tensor_model_parallel_size "${TP_SIZE}" \
          --pipeline_model_parallel_size "${PP_SIZE}" \
          --sequence_parallel true \
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
          --save "${CONTAINER_OUTPUT_BASE}/${RUN_ID}" \
          --save_interval "${SAVE_STEPS}" \
          --max_length "${CONTEXT_LENGTH}" \
          --dataset_num_proc "${DATASET_NUM_PROC}" \
          --num_workers "${NUM_WORKERS}" \
          --no_save_optim true \
          --no_save_rng true \
          --attention_backend flash \
          --log_throughput true
    else
      exit 0
    fi
  '

echo "[train] submitted ${SLURM_JOB_ID}"
echo "[train] logs dir: ${LOG_DIR}"
echo "[train] output dir: ${OUTPUT_DIR}"
