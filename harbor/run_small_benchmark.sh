#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET_REF="terminal-bench@2.0"
N_CONCURRENT=16

BENCHMARK_TASKS=(
  circuit-fibsqrt
  portfolio-optimization
  winning-avg-corewars
  feal-differential-cryptanalysis
  rstan-to-pystan
  path-tracing
  feal-linear-cryptanalysis
  caffe-cifar-10
  financial-document-processor
  code-from-image
  gpt2-codegolf
  build-cython-ext
  torch-tensor-parallelism
  crack-7z-hash
  sqlite-with-gcov
  write-compressor
  regex-log
  nginx-request-logging
  configure-git-webserver
  sanitize-git-repo
  fix-git
  cancel-async-tasks
  gcode-to-text
  kv-store-grpc
  overfull-hbox
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
