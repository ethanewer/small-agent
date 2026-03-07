#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/harbor/common.sh"

DATASET_REF="terminal-bench@2.0"
N_CONCURRENT=2

BENCHMARK_TASKS=(
  cancel-async-tasks
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
  cobol-modernization
  fix-git
  overfull-hbox
  prove-plus-comm
)

main() {
  if ! parse_common_args "$@"; then
    echo "Usage: run_baseline_eval.sh [--model <key>] [--agent <key>]"
    exit 0
  fi
  resolve_model_and_agent
  build_harbor_dataset_command "${DATASET_REF}" "${N_CONCURRENT}"
  append_task_name_filters "${BENCHMARK_TASKS[@]}"
  run_or_echo
}

main "$@"
