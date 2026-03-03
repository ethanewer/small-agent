#!/bin/bash

# Usage:
#   cd sft
#   sbatch sft-scripts/megatron_export_4500.sbatch

#SBATCH -J ee-export-ckpt4500-hf
#SBATCH -p l40s-1gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH -o logs-20260303/export_hf_%j.out
#SBATCH -e logs-20260303/export_hf_%j.err

set -euo pipefail
set -x

IMAGE="${IMAGE:-/wbl-fast/usrs/shl/enroot/modelscope-torch2.8.sqsh}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-/wbl-fast/usrs/ethan/small-agent/sft/outputs-20260302/ee_train_full_65k_megatron_tp2_9754/v0-20260302-175728}"
SOURCE_STEP="${SOURCE_STEP:-4500}"
SOURCE_ITER_DIR="$(printf "iter_%07d" "${SOURCE_STEP}")"
TARGET_ROOT_DIR="${TARGET_ROOT_DIR:-/wbl-fast/usrs/ethan/small-agent/sft/checkpoints}"
TARGET_MCORE_DIR="${TARGET_ROOT_DIR}/ee_train_full_65k_megatron_tp2_9754_step${SOURCE_STEP}"
TARGET_HF_DIR="${TARGET_ROOT_DIR}/ee_train_full_65k_megatron_tp2_9754_step${SOURCE_STEP}_hf"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${TARGET_ROOT_DIR}/LocoreMind-LocoOperator-4B}"
TEST_CONVERT_PRECISION="${TEST_CONVERT_PRECISION:-false}"

echo "[export] source run dir: ${SOURCE_RUN_DIR}"
echo "[export] source iter dir: ${SOURCE_ITER_DIR}"
echo "[export] target mcore dir: ${TARGET_MCORE_DIR}"
echo "[export] target hf dir: ${TARGET_HF_DIR}"
echo "[export] base model dir: ${BASE_MODEL_DIR}"

if [[ ! -d "${SOURCE_RUN_DIR}" ]]; then
  echo "[preflight] source run directory missing: ${SOURCE_RUN_DIR}" >&2
  exit 2
fi

if [[ ! -f "${SOURCE_RUN_DIR}/args.json" ]]; then
  echo "[preflight] source args.json missing: ${SOURCE_RUN_DIR}/args.json" >&2
  exit 3
fi

if [[ ! -d "${SOURCE_RUN_DIR}/${SOURCE_ITER_DIR}" ]]; then
  echo "[preflight] source checkpoint directory missing: ${SOURCE_RUN_DIR}/${SOURCE_ITER_DIR}" >&2
  exit 4
fi

if [[ ! -d "${BASE_MODEL_DIR}" ]]; then
  echo "[preflight] base model directory missing: ${BASE_MODEL_DIR}" >&2
  exit 5
fi

rm -rf "${TARGET_MCORE_DIR}" "${TARGET_HF_DIR}"
mkdir -p "${TARGET_MCORE_DIR}"

rsync -a "${SOURCE_RUN_DIR}/args.json" "${TARGET_MCORE_DIR}/args.json"
rsync -a "${SOURCE_RUN_DIR}/${SOURCE_ITER_DIR}/" "${TARGET_MCORE_DIR}/${SOURCE_ITER_DIR}/"
printf "%d\n" "${SOURCE_STEP}" > "${TARGET_MCORE_DIR}/latest_checkpointed_iteration.txt"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SRUN_PREFIX=(
    srun
    --jobid="${SLURM_JOB_ID}"
    --overlap
    --container-image="${IMAGE}"
    --container-mounts="/wbl-fast/usrs/ethan/small-agent/sft:/workspace/run,/wbl:/wbl,/wbl-fast:/wbl-fast,/dev/shm:/dev/shm"
    --container-workdir="/workspace/run"
    --container-writable
  )
else
  SRUN_PREFIX=(
    srun
    --partition=l40s-1gpu
    --nodes=1
    --ntasks=1
    --gpus-per-task=1
    --cpus-per-task=32
    --time=02:00:00
    --container-image="${IMAGE}"
    --container-mounts="/wbl-fast/usrs/ethan/small-agent/sft:/workspace/run,/wbl:/wbl,/wbl-fast:/wbl-fast,/dev/shm:/dev/shm"
    --container-workdir="/workspace/run"
    --container-writable
  )
fi

"${SRUN_PREFIX[@]}" bash -lc '
  set -euo pipefail
  set -x

  export PYTHONUNBUFFERED=1
  export MODELSCOPE_CACHE="/workspace/run/.modelscope_cache"
  mkdir -p "${MODELSCOPE_CACHE}"

  if ! command -v swift >/dev/null 2>&1; then
    echo "[preflight] swift CLI not found in container PATH" >&2
    exit 6
  fi

  python -c "import torch; print(f\"[preflight] torch={torch.__version__}, ngpu={torch.cuda.device_count()}\")"

  swift export \
    --model "/workspace/run/checkpoints/LocoreMind-LocoOperator-4B" \
    --mcore_model "/workspace/run/checkpoints/ee_train_full_65k_megatron_tp2_9754_step'"${SOURCE_STEP}"'" \
    --output_dir "/workspace/run/checkpoints/ee_train_full_65k_megatron_tp2_9754_step'"${SOURCE_STEP}"'_hf" \
    --to_hf true \
    --test_convert_precision '"${TEST_CONVERT_PRECISION}"'
'

echo "[export] done"
echo "[export] target mcore dir: ${TARGET_MCORE_DIR}"
echo "[export] target hf dir: ${TARGET_HF_DIR}"
