#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Fixed official public smoke dataset.
DATASET_REF="terminal-bench-sample@2.0"

main() {
  if ! parse_common_args "$@"; then
    cat <<'EOF'
Usage: ./harbor/run_smoke.sh [--model <key>] [--agent <key>] [--dry-run]
Runs a smoke Harbor evaluation on an official public dataset.
EOF
    usage_common
    exit 0
  fi
  resolve_model_and_agent
  build_harbor_dataset_command "${DATASET_REF}"
  run_or_echo
}

main "$@"
