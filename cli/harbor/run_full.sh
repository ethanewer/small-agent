#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Fixed official public full dataset.
DATASET_REF="terminal-bench@2.0"

main() {
  if ! parse_common_args "$@"; then
    cat <<'EOF'
Usage: ./harbor/run_full.sh [--model <key>] [--agent <key>] [--dry-run]
Runs Harbor on the full fixed official public dataset.
EOF
    usage_common
    exit 0
  fi
  resolve_model_and_agent
  build_harbor_dataset_command "${DATASET_REF}"
  run_or_echo
}

main "$@"
