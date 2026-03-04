#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CLI_DIR}/config.json"
AGENT_IMPORT_PATH="agent:SmallAgentHarborAgent"

MODEL_OVERRIDE=""
AGENT_OVERRIDE=""
DRY_RUN=0
POSITIONAL_ARGS=()
HARBOR_CMD=()

usage_common() {
  cat <<'EOF'
Options:
  --model <key>   Model key from cli/config.json models
  --agent <key>   Agent key from cli/config.json agents
  --dry-run       Print resolved harbor command and exit
  -h, --help      Show help
EOF
}

parse_common_args() {
  MODEL_OVERRIDE=""
  AGENT_OVERRIDE=""
  DRY_RUN=0
  POSITIONAL_ARGS=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)
        if [[ $# -lt 2 ]]; then
          echo "Missing value for --model" >&2
          exit 2
        fi
        MODEL_OVERRIDE="$2"
        shift 2
        ;;
      --agent)
        if [[ $# -lt 2 ]]; then
          echo "Missing value for --agent" >&2
          exit 2
        fi
        AGENT_OVERRIDE="$2"
        shift 2
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      --)
        shift
        while [[ $# -gt 0 ]]; do
          POSITIONAL_ARGS+=("$1")
          shift
        done
        ;;
      -h|--help)
        return 64
        ;;
      *)
        POSITIONAL_ARGS+=("$1")
        shift
        ;;
    esac
  done
}

resolve_harbor_command() {
  if command -v harbor >/dev/null 2>&1; then
    HARBOR_CMD=(harbor)
    return
  fi
  if command -v uvx >/dev/null 2>&1; then
    # Fall back to uvx and inject truststore so TLS uses system trust roots.
    HARBOR_CMD=(
      uvx
      --from harbor
      --with truststore
      --with pexpect
      --with rich
      --with litellm
      python
      -c
      "import truststore; truststore.inject_into_ssl(); from harbor.cli.main import app; app()"
    )
    return
  fi

  echo "Neither 'harbor' nor 'uvx' is available on PATH." >&2
  echo "Install Harbor CLI or uv (uvx) to run these scripts." >&2
  exit 127
}

resolve_model_and_agent() {
  local json_output
  json_output="$(
    python3 - "${CONFIG_PATH}" "${MODEL_OVERRIDE}" "${AGENT_OVERRIDE}" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
model_override = sys.argv[2].strip()
agent_override = sys.argv[3].strip()
data = json.loads(config_path.read_text(encoding="utf-8"))
models = data.get("models")
agents = data.get("agents")
if not isinstance(models, dict) or not models:
    raise SystemExit("config.json 'models' must be a non-empty object")
if not isinstance(agents, dict) or not agents:
    raise SystemExit("config.json 'agents' must be a non-empty object")

default_model = str(data.get("default_model", "")).strip()
default_agent = str(data.get("default_agent", "")).strip()
if default_model not in models:
    raise SystemExit("default_model must match a key in models")
if default_agent not in agents:
    raise SystemExit("default_agent must match a key in agents")

selected_model = model_override or default_model
selected_agent = agent_override or default_agent
if selected_model not in models:
    known = ", ".join(models.keys())
    raise SystemExit(f"Unknown model key '{selected_model}'. Available: {known}")
if selected_agent not in agents:
    known = ", ".join(agents.keys())
    raise SystemExit(f"Unknown agent key '{selected_agent}'. Available: {known}")

print(selected_model)
print(selected_agent)
PY
  )"

  RESOLVED_MODEL="$(printf '%s\n' "${json_output}" | sed -n '1p')"
  RESOLVED_AGENT="$(printf '%s\n' "${json_output}" | sed -n '2p')"
}

build_harbor_dataset_command() {
  resolve_harbor_command
  local dataset_ref="$1"
  shift
  local extra_args=("$@")
  HARBOR_COMMAND=(
    "${HARBOR_CMD[@]}"
    run
    -d "${dataset_ref}"
    --agent-import-path "${AGENT_IMPORT_PATH}"
  )
  if [[ -n "${RESOLVED_MODEL}" ]]; then
    HARBOR_COMMAND+=(--agent-env "SMALL_AGENT_HARBOR_MODEL=${RESOLVED_MODEL}")
  fi
  if [[ -n "${RESOLVED_AGENT}" ]]; then
    HARBOR_COMMAND+=(--agent-env "SMALL_AGENT_HARBOR_AGENT=${RESOLVED_AGENT}")
  fi
  if [[ "${#extra_args[@]}" -gt 0 ]]; then
    HARBOR_COMMAND+=("${extra_args[@]}")
  fi
  if [[ "${#POSITIONAL_ARGS[@]}" -gt 0 ]]; then
    HARBOR_COMMAND+=("${POSITIONAL_ARGS[@]}")
  fi
}

build_harbor_path_command() {
  resolve_harbor_command
  local task_or_dataset_path="$1"
  shift
  local extra_args=("$@")
  HARBOR_COMMAND=(
    "${HARBOR_CMD[@]}"
    run
    --path "${task_or_dataset_path}"
    --agent-import-path "${AGENT_IMPORT_PATH}"
  )
  if [[ -n "${RESOLVED_MODEL}" ]]; then
    HARBOR_COMMAND+=(--agent-env "SMALL_AGENT_HARBOR_MODEL=${RESOLVED_MODEL}")
  fi
  if [[ -n "${RESOLVED_AGENT}" ]]; then
    HARBOR_COMMAND+=(--agent-env "SMALL_AGENT_HARBOR_AGENT=${RESOLVED_AGENT}")
  fi
  if [[ "${#extra_args[@]}" -gt 0 ]]; then
    HARBOR_COMMAND+=("${extra_args[@]}")
  fi
  if [[ "${#POSITIONAL_ARGS[@]}" -gt 0 ]]; then
    HARBOR_COMMAND+=("${POSITIONAL_ARGS[@]}")
  fi
}

print_harbor_command() {
  printf 'Resolved model=%s agent=%s\n' "${RESOLVED_MODEL}" "${RESOLVED_AGENT}"
  printf 'Command:'
  local token
  for token in "${HARBOR_COMMAND[@]}"; do
    printf ' %q' "${token}"
  done
  printf '\n'
}

run_or_echo() {
  print_harbor_command
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi

  (
    cd "${SCRIPT_DIR}"
    "${HARBOR_COMMAND[@]}"
  )
}
