#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET_REF="terminal-bench@2.0"
N_CONCURRENT=5

BENCHMARK_TASKS=(
  # hard (1)
  cancel-async-tasks
  # medium (10)
  custom-memory-heap-crash
  distribution-search
  dna-insert
  filter-js-from-html
  financial-document-processor
  gcode-to-text
  git-multibranch
  headless-terminal
  large-scale-text-editing
  log-summary-date-ranges
  # easy (4)
  cobol-modernization
  fix-git
  overfull-hbox
  prove-plus-comm
)

main() {
  if ! parse_common_args "$@"; then
    cat <<'EOF'
Usage: ./harbor/run_small_benchmark.sh [--model <key>] [--agent <key>] [--dry-run]
Runs 15 tasks (4 easy, 10 medium, 1 hard) disjoint from all run_debug splits.
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
