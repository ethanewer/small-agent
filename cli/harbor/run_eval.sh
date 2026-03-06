#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET_REF="terminal-bench@2.0"
N_CONCURRENT=6

EVAL_TASKS=(
  adaptive-rejection-sampler
  break-filter-js-from-html
  build-pmars
  cancel-async-tasks
  cobol-modernization
  constraints-scheduling
  crack-7z-hash
  db-wal-recovery
  extract-elf
  fix-git
  git-leak-recovery
  hf-model-inference
  kv-store-grpc
  largest-eigenval
  modernize-scientific-stack
  nginx-request-logging
  openssl-selfsigned-cert
  overfull-hbox
  prove-plus-comm
  pypi-server
)

main() {
  if ! parse_common_args "$@"; then
    cat <<'EOF'
Usage: ./harbor/run_eval.sh [--model <key>] [--agent <key>] [--dry-run]
Runs 20 held-out eval tasks from terminal-bench@2.0 (no overlap with the dev sample).
Used by agent_evolve to measure generalization on tasks the inner agent has not seen.
EOF
    usage_common
    exit 0
  fi
  resolve_model_and_agent
  build_harbor_dataset_command "${DATASET_REF}" "${N_CONCURRENT}"
  append_task_name_filters "${EVAL_TASKS[@]}"
  run_or_echo
}

main "$@"
