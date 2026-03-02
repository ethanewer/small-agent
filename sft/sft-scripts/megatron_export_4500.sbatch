#!/bin/bash

# Usage:
#   cd sft
#   sbatch sft-scripts/megatron_export_4500.sbatch

#SBATCH -J ee-export-ckpt4500
#SBATCH -p h200
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH -o logs-20260302/export_%j.out
#SBATCH -e logs-20260302/export_%j.err

set -euo pipefail
set -x

SOURCE_RUN_DIR="/wbl-fast/usrs/ethan/small-agent/sft/outputs-20260302/ee_train_full_65k_megatron_tp2_9754/v0-20260302-175728"
SOURCE_STEP=4500
SOURCE_ITER_DIR="$(printf "iter_%07d" "${SOURCE_STEP}")"
TARGET_ROOT_DIR="/wbl-fast/usrs/ethan/small-agent/sft/checkpoints"
TARGET_EXPORT_DIR="${TARGET_ROOT_DIR}/ee_train_full_65k_megatron_tp2_9754_step4500"

echo "[export] source run dir: ${SOURCE_RUN_DIR}"
echo "[export] source iter dir: ${SOURCE_ITER_DIR}"
echo "[export] target export dir: ${TARGET_EXPORT_DIR}"

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

mkdir -p "${TARGET_EXPORT_DIR}"

rsync -a "${SOURCE_RUN_DIR}/args.json" "${TARGET_EXPORT_DIR}/args.json"
rsync -a "${SOURCE_RUN_DIR}/${SOURCE_ITER_DIR}/" "${TARGET_EXPORT_DIR}/${SOURCE_ITER_DIR}/"
printf "%d\n" "${SOURCE_STEP}" > "${TARGET_EXPORT_DIR}/latest_checkpointed_iteration.txt"

echo "[export] done"
echo "[export] target export dir: ${TARGET_EXPORT_DIR}"
