#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET_REF="terminal-bench@2.0"
N_CONCURRENT=16

BENCHMARK_TASKS=(
  bn-fit-modify
  cancel-async-tasks
  configure-git-webserver
  count-dataset-tokens
  feal-differential-cryptanalysis
  fix-git
  gcode-to-text
  git-multibranch
  gpt2-codegolf
  headless-terminal
  kv-store-grpc
  log-summary-date-ranges
  model-extraction-relu-logits
  nginx-request-logging
  openssl-selfsigned-cert
  overfull-hbox
  pypi-server
  pytorch-model-cli
  regex-chess
  regex-log
  reshard-c4-data
  sanitize-git-repo
  sqlite-db-truncate
  sqlite-with-gcov
  vulnerable-secret
)

main() {
  if ! parse_common_args "$@"; then
    cat <<'EOF'
Usage: ./harbor/run_short_benchmark.sh [--model <key>] [--agent <key>] [--dry-run]
Runs 25 tasks from terminal-bench@2.0.
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
