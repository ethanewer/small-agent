#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET_REF="terminal-bench@2.0"
N_CONCURRENT=5

BENCHMARK_TASKS=(
  sanitize-git-repo
  custom-memory-heap-crash
  torch-pipeline-parallelism
  raman-fitting
  tune-mjcf
  largest-eigenval
  financial-document-processor
  adaptive-rejection-sampler
  build-cython-ext
  chess-best-move
  torch-tensor-parallelism
  count-dataset-tokens
  pytorch-model-recovery
  qemu-startup
  fix-code-vulnerability
  extract-elf
  caffe-cifar-10
  build-pmars
  large-scale-text-editing
  code-from-image
)

main() {
  if ! parse_common_args "$@"; then
    cat <<'EOF'
Usage: ./harbor/run_long_benchmark.sh [--model <key>] [--agent <key>] [--dry-run]
Runs 20 long tasks from terminal-bench@2.0.
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
