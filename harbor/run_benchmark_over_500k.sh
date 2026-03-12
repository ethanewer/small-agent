#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET_REF="terminal-bench@2.0"
N_CONCURRENT=5

BENCHMARK_TASKS=(
  build-pov-ray
  circuit-fibsqrt
  compile-compcert
  db-wal-recovery
  dna-insert
  extract-moves-from-video
  fix-ocaml-gc
  gcode-to-text
  mailman
  make-doom-for-mips
  make-mips-interpreter
  mteb-leaderboard
  path-tracing
  path-tracing-reverse
  regex-chess
  rstan-to-pystan
  sam-cell-seg
  schemelike-metacircular-eval
  train-fasttext
  winning-avg-corewars
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
