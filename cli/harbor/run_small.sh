#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Fixed official public small dataset.
DATASET_REF="terminal-bench-sample@2.0"

main() {
  if ! parse_common_args "$@"; then
    cat <<'EOF'
Usage: ./harbor/run_small.sh [--model <key>] [--agent <key>] [--dry-run] [-- <extra harbor args>]
Runs Harbor on a fixed official public small dataset.
EOF
    usage_common
    exit 0
  fi
  resolve_model_and_agent
  build_harbor_dataset_command "${DATASET_REF}"
  run_or_echo
}

main "$@"
