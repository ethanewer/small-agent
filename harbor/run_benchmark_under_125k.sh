#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET_REF="terminal-bench@2.0"
N_CONCURRENT=5

BENCHMARK_TASKS=(
  cancel-async-tasks
  constraints-scheduling
  fix-git
  git-leak-recovery
  git-multibranch
  headless-terminal
  hf-model-inference
  kv-store-grpc
  log-summary-date-ranges
  modernize-scientific-stack
  mteb-retrieve
  multi-source-data-merger
  nginx-request-logging
  openssl-selfsigned-cert
  polyglot-c-py
  polyglot-rust-c
  prove-plus-comm
  pypi-server
  sqlite-with-gcov
  vulnerable-secret
)

main() {
  if ! parse_common_args "$@"; then
    cat <<'EOF'
Usage: ./harbor/run_short_benchmark.sh [--model <key>] [--agent <key>] [--dry-run]
Runs 20 short tasks from terminal-bench@2.0.
EOF
    usage_common
    exit 0
  fi
  resolve_model_and_agent
  build_harbor_dataset_command "${DATASET_REF}" "${N_CONCURRENT}"
  append_task_name_filters "${BENCHMARK_TASKS[@]}"
  run_or_echo
}

main "$@"
