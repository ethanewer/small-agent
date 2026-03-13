#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET_REF="terminal-bench@2.0"
N_CONCURRENT=8

BENCHMARK_TASKS=(
  polyglot-c-py
  polyglot-rust-c
  headless-terminal
  regex-log
  git-multibranch
  configure-git-webserver
  vulnerable-secret
  sqlite-with-gcov
  cancel-async-tasks
  pypi-server
  multi-source-data-merger
  git-leak-recovery
  fix-git
  log-summary-date-ranges
  modernize-scientific-stack
  openssl-selfsigned-cert
  kv-store-grpc
  constraints-scheduling
  nginx-request-logging
  prove-plus-comm
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
