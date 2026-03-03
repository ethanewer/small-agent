#!/bin/bash

#SBATCH -J qwen35-4b-sft
#SBATCH -p h200
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=192
#SBATCH --time=7-00:00:00
#SBATCH --exclusive
#SBATCH -o logs-20260303/train_qwen35_tf_%j.out
#SBATCH -e logs-20260303/train_qwen35_tf_%j.err

set -euo pipefail
set -x

IMAGE="${IMAGE:-/wbl-fast/usrs/shl/enroot/modelscope-torch2.8.sqsh}"
MASTER_PORT="${MASTER_PORT:-29500}"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
NNODES="${SLURM_NNODES}"

JOB_COMMAND="$(scontrol show job "${SLURM_JOB_ID}" | sed -n "s/.* Command=\([^ ]*\).*/\1/p")"
if [[ "${JOB_COMMAND}" = /* ]]; then
  SCRIPT_PATH="${JOB_COMMAND}"
else
  SCRIPT_PATH="${SLURM_SUBMIT_DIR}/${JOB_COMMAND}"
fi

SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
RUN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${RUN_ROOT}/logs-20260303"
mkdir -p "${LOG_DIR}"

CONTAINER_WORKDIR="/workspace/run"
MODEL_DIR="/workspace/run/checkpoints/Qwen3.5-4B"
DATASET_PATH="/workspace/run/data/full-dataset-extended/dataset_shuffled.jsonl"
OUTPUT_BASE="/workspace/run/outputs-20260303"
RUN_ID="qwen35_4b_full_tf_${SLURM_JOB_ID}"
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_ID}"

NPROC_PER_NODE="8"
WORLD_SIZE="$((NNODES * NPROC_PER_NODE))"
MAX_STEPS="20000"
LEARNING_RATE="2e-7"
WARMUP_RATIO="0.05"
GROUP_BY_LENGTH="true"
MODEL_TYPE="${MODEL_TYPE:-qwen3}"
TEMPLATE="${TEMPLATE:-qwen3}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
PADDING_FREE="${PADDING_FREE:-true}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
TOKENS_PER_BATCH="$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * WORLD_SIZE * MAX_LENGTH))"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE WORLD_SIZE
export MODEL_DIR DATASET_PATH OUTPUT_BASE RUN_ID OUTPUT_DIR
export MASTER_PORT MASTER_ADDR NNODES
export CONTAINER_WORKDIR
export PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS MAX_LENGTH TOKENS_PER_BATCH
export MAX_STEPS LEARNING_RATE WARMUP_RATIO GROUP_BY_LENGTH PADDING_FREE ATTN_IMPL MODEL_TYPE TEMPLATE

echo "[train] run root: ${RUN_ROOT}"
echo "[train] run id: ${RUN_ID}"
echo "[train] model: ${MODEL_DIR}"
echo "[train] dataset: ${DATASET_PATH}"
echo "[train] output: ${OUTPUT_DIR}"
echo "[train] nodes: ${NNODES}, gpus/node: ${NPROC_PER_NODE}, world_size: ${WORLD_SIZE}"
echo "[train] per_device_train_batch_size: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "[train] gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS}"
echo "[train] max_length: ${MAX_LENGTH}"
echo "[train] approx tokens per optimizer step: ${TOKENS_PER_BATCH}"
echo "[train] max_steps: ${MAX_STEPS}"
echo "[train] learning_rate: ${LEARNING_RATE}"
echo "[train] warmup_ratio: ${WARMUP_RATIO}"
echo "[train] group_by_length: ${GROUP_BY_LENGTH}"
echo "[train] padding_free: ${PADDING_FREE}"
echo "[train] attn_impl: ${ATTN_IMPL}"
echo "[train] model_type: ${MODEL_TYPE}"
echo "[train] template: ${TEMPLATE}"

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

    mkdir -p "${OUTPUT_BASE}" "${OUTPUT_DIR}" "${MODELSCOPE_CACHE}" "${CONTAINER_WORKDIR}/logs-20260303"

    if ! command -v swift >/dev/null 2>&1; then
      echo "[preflight] swift CLI not found in container PATH" >&2
      exit 2
    fi
    if [[ ! -d "${MODEL_DIR}" ]]; then
      echo "[preflight] model directory missing: ${MODEL_DIR}" >&2
      exit 3
    fi
    if [[ ! -f "${DATASET_PATH}" ]]; then
      echo "[preflight] dataset file missing: ${DATASET_PATH}" >&2
      exit 4
    fi

    python -c "import torch; print(f\"[preflight] torch={torch.__version__}, ngpu={torch.cuda.device_count()}\")"

    # Qwen3.5 config requires bleeding-edge Transformers model registry.
    if ! python -c "from transformers import AutoConfig; AutoConfig.from_pretrained(\"${MODEL_DIR}\", trust_remote_code=True)" >/dev/null 2>&1; then
      echo "[preflight] upgrading transformers/peft for qwen3_5 support"
      python -m pip install -U "transformers @ git+https://github.com/huggingface/transformers.git"
      python -m pip install -U "peft @ git+https://github.com/huggingface/peft.git"
      python -c "from transformers import AutoConfig; AutoConfig.from_pretrained(\"${MODEL_DIR}\", trust_remote_code=True); print(\"[preflight] transformers supports qwen3_5\")"
    fi

    cat > "${CONTAINER_WORKDIR}/swift_sft_entry.py" <<'"'"'PY'"'"'
import sys
from transformers import TrainingArguments, Seq2SeqTrainingArguments

# transformers main removed past_index; ms-swift trainer still references it.
if not hasattr(TrainingArguments, "past_index"):
    TrainingArguments.past_index = -1
if not hasattr(Seq2SeqTrainingArguments, "past_index"):
    Seq2SeqTrainingArguments.past_index = -1

from swift.llm.train.sft import sft_main

if __name__ == "__main__":
    sft_main()
PY

    if [[ "${SLURM_LOCALID}" == "0" ]]; then
      python -m torch.distributed.run \
          --nnodes "${NNODES}" \
          --nproc_per_node "${NPROC_PER_NODE}" \
          --node_rank "${SLURM_NODEID}" \
          --master_addr "${MASTER_ADDR}" \
          --master_port "${MASTER_PORT}" \
          "${CONTAINER_WORKDIR}/swift_sft_entry.py" \
          --model_type "${MODEL_TYPE}" \
          --template "${TEMPLATE}" \
          --model "${MODEL_DIR}" \
          --train_type full \
          --dataset "${DATASET_PATH}" \
          --load_from_cache_file true \
          --split_dataset_ratio "${SPLIT_DATASET_RATIO:-0}" \
          --torch_dtype bfloat16 \
          --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
          --max_steps "${MAX_STEPS}" \
          --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
          --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}" \
          --learning_rate "${LEARNING_RATE}" \
          --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
          --output_dir "${OUTPUT_DIR}" \
          --report_to tensorboard \
          --save_steps 500 \
          --logging_steps 5 \
          --max_length "${MAX_LENGTH}" \
          --warmup_ratio "${WARMUP_RATIO}" \
          --padding_free "${PADDING_FREE}" \
          --attn_impl "${ATTN_IMPL}" \
          --dataset_num_proc "${DATASET_NUM_PROC:-4}" \
          --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
          --deepspeed "${DEEPSPEED_STAGE:-zero3}"
    else
      exit 0
    fi
  '

echo "[train] finished ${SLURM_JOB_ID}"
echo "[train] output dir: ${OUTPUT_DIR}"
