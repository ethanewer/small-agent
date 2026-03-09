#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET_REF="terminal-bench@2.0"
N_CONCURRENT=5

DEV_TASKS=(
  adaptive-rejection-sampler
  build-cython-ext
  constraints-scheduling
  extract-elf
  git-leak-recovery
  hf-model-inference
  kv-store-grpc
  modernize-scientific-stack
  nginx-request-logging
  regex-log
)

main() {
  if ! parse_common_args "$@"; then
    cat <<'EOF'
Usage: ./harbor/run_dev_benchmark.sh [--model <key>] [--agent <key>] [--dry-run]
Runs 10 medium tasks disjoint from the eval benchmark.
EOF
    usage_common
    exit 0
  fi
  resolve_model_and_agent
  build_harbor_dataset_command "${DATASET_REF}" "${N_CONCURRENT}"
  append_task_name_filters "${DEV_TASKS[@]}"
  run_or_echo
}

main "$@"
