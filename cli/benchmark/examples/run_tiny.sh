#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

usage() {
  echo "Usage: $0 [--model <model_key>] [--agent <agent_key>] [--concurrency <n>] [--run-id <run_id>]" >&2
}

MODEL_KEY=""
AGENT_KEY=""
CONCURRENCY="4"
RUN_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_KEY="${2:-}"
      shift 2
      ;;
    --agent)
      AGENT_KEY="${2:-}"
      shift 2
      ;;
    --concurrency)
      CONCURRENCY="${2:-}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${MODEL_KEY}" || -z "${AGENT_KEY}" ]]; then
  load_default_model_and_agent
  MODEL_KEY="${MODEL_KEY:-${DEFAULT_MODEL_KEY}}"
  AGENT_KEY="${AGENT_KEY:-${DEFAULT_AGENT_KEY}}"
fi

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="tiny-${AGENT_KEY}-${MODEL_KEY}-$(date +%Y%m%d-%H%M%S)"
fi

run_tb_for_model \
  "${MODEL_KEY}" \
  "${AGENT_KEY}" \
  "${RUN_ID}" \
  --task-id configure-git-webserver \
  --task-id fix-git \
  --task-id count-dataset-tokens \
  --task-id sqlite-db-truncate \
  --task-id nginx-request-logging \
  --n-attempts 1 \
  --n-concurrent "${CONCURRENCY}"
