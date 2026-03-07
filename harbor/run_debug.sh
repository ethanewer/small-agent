#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DATASET_REF="terminal-bench@2.0"
N_CONCURRENT=5

SPLIT_1=(
  adaptive-rejection-sampler
  break-filter-js-from-html
  constraints-scheduling
  crack-7z-hash
  regex-log
)

SPLIT_2=(
  db-wal-recovery
  extract-elf
  git-leak-recovery
  hf-model-inference
  kv-store-grpc
)

SPLIT_3=(
  largest-eigenval
  modernize-scientific-stack
  nginx-request-logging
  openssl-selfsigned-cert
  pypi-server
)

SPLIT_4=(
  build-cython-ext
  caffe-cifar-10
  chess-best-move
  code-from-image
  compile-compcert
)

SPLIT=""

parse_split_arg() {
  local remaining=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --split)
        if [[ $# -lt 2 ]]; then
          echo "Missing value for --split" >&2
          exit 2
        fi
        SPLIT="$2"
        shift 2
        ;;
      *)
        remaining+=("$1")
        shift
        ;;
    esac
  done
  set -- "${remaining[@]+"${remaining[@]}"}"
  REMAINING_ARGS=("$@")
}

main() {
  parse_split_arg "$@"
  set -- "${REMAINING_ARGS[@]+"${REMAINING_ARGS[@]}"}"

  if ! parse_common_args "$@"; then
    cat <<'EOF'
Usage: ./harbor/run_debug.sh --split <1|2|3|4> [--model <key>] [--agent <key>] [--dry-run]
Runs 5 medium tasks from the selected split (4 splits available).
EOF
    usage_common
    exit 0
  fi

  if [[ -z "${SPLIT}" ]]; then
    echo "Missing required --split argument (1, 2, 3, or 4)" >&2
    exit 2
  fi

  local tasks=()
  case "${SPLIT}" in
    1) tasks=("${SPLIT_1[@]}") ;;
    2) tasks=("${SPLIT_2[@]}") ;;
    3) tasks=("${SPLIT_3[@]}") ;;
    4) tasks=("${SPLIT_4[@]}") ;;
    *)
      echo "Invalid split '${SPLIT}'. Must be 1, 2, 3, or 4." >&2
      exit 2
      ;;
  esac

  resolve_model_and_agent
  build_harbor_dataset_command "${DATASET_REF}" "${N_CONCURRENT}"
  append_task_name_filters "${tasks[@]}"
  run_or_echo
}

main "$@"
