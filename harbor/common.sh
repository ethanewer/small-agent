#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CLI_DIR}/config.json"
AGENT_IMPORT_PATH="agent:SmallAgentHarborAgent"

MODEL_OVERRIDE=""
AGENT_OVERRIDE=""
DRY_RUN=0
HARBOR_CMD=()
JOBS_ROOT="${SCRIPT_DIR}/jobs"
RESOLVED_JOBS_DIR=""
RUN_ID=""
RESOLVED_N_CONCURRENT=""

usage_common() {
  cat <<'EOF'
Options:
  --model <key>   Model key from config.json models
  --agent <key>   Agent key from config.json agents
  --dry-run       Print resolved harbor command and exit
  -h, --help      Show help

Safety policy:
  Only fixed benchmark options are allowed.
  Extra Harbor arguments are rejected by design.
EOF
}

parse_common_args() {
  MODEL_OVERRIDE=""
  AGENT_OVERRIDE=""
  DRY_RUN=0

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
      -h|--help)
        return 64
        ;;
      *)
        echo "Unknown argument: $1" >&2
        echo "These benchmark scripts intentionally reject extra Harbor arguments." >&2
        echo "Use --help to see supported options." >&2
        exit 2
        ;;
    esac
  done
}

resolve_safe_jobs_dir() {
  mkdir -p "${JOBS_ROOT}"
  RUN_ID="$(date -u +"%Y-%m-%d__%H-%M-%S")__$$"
  RESOLVED_JOBS_DIR="${JOBS_ROOT}/${RUN_ID}"
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
      --with rich
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

append_agent_env_passthrough() {
  resolve_from_shell() {
    local env_name="$1"
    if [[ -n "${!env_name:-}" ]]; then
      printf '%s' "${!env_name}"
      return 0
    fi

    if command -v zsh >/dev/null 2>&1; then
      local resolved
      resolved="$(
        zsh -ic "source ~/.zshrc >/dev/null 2>&1; printf %s \"\${${env_name}}\"" 2>/dev/null || true
      )"
      if [[ -n "${resolved}" ]]; then
        printf '%s' "${resolved}"
        return 0
      fi
    fi

    return 1
  }

  local env_name
  for env_name in OPENROUTER_API_KEY OPENAI_API_KEY OPENAI_BASE_URL SMALL_AGENT_CA_BUNDLE; do
    local value
    if value="$(resolve_from_shell "${env_name}")"; then
      HARBOR_COMMAND+=(--agent-env "${env_name}=${value}")
    fi
  done
}

append_task_name_filters() {
  local task_name
  for task_name in "$@"; do
    HARBOR_COMMAND+=(--task-name "${task_name}")
  done
}

build_harbor_dataset_command() {
  resolve_harbor_command
  resolve_safe_jobs_dir
  local dataset_ref="$1"
  local n_concurrent="${2:-}"
  RESOLVED_N_CONCURRENT="${n_concurrent}"
  HARBOR_COMMAND=(
    "${HARBOR_CMD[@]}"
    run
    --jobs-dir "${RESOLVED_JOBS_DIR}"
    --n-concurrent "${n_concurrent}"
    --env docker
    --delete
    --no-force-build
    -d "${dataset_ref}"
    --agent-import-path "${AGENT_IMPORT_PATH}"
  )
  if [[ -n "${RESOLVED_MODEL}" ]]; then
    HARBOR_COMMAND+=(--agent-env "SMALL_AGENT_HARBOR_MODEL=${RESOLVED_MODEL}")
  fi
  if [[ -n "${RESOLVED_AGENT}" ]]; then
    HARBOR_COMMAND+=(--agent-env "SMALL_AGENT_HARBOR_AGENT=${RESOLVED_AGENT}")
  fi
  append_agent_env_passthrough
}

build_harbor_path_command() {
  resolve_harbor_command
  resolve_safe_jobs_dir
  local task_or_dataset_path="$1"
  local n_concurrent="${2:-}"
  RESOLVED_N_CONCURRENT="${n_concurrent}"
  HARBOR_COMMAND=(
    "${HARBOR_CMD[@]}"
    run
    --jobs-dir "${RESOLVED_JOBS_DIR}"
    --n-concurrent "${n_concurrent}"
    --env docker
    --delete
    --no-force-build
    --path "${task_or_dataset_path}"
    --agent-import-path "${AGENT_IMPORT_PATH}"
  )
  if [[ -n "${RESOLVED_MODEL}" ]]; then
    HARBOR_COMMAND+=(--agent-env "SMALL_AGENT_HARBOR_MODEL=${RESOLVED_MODEL}")
  fi
  if [[ -n "${RESOLVED_AGENT}" ]]; then
    HARBOR_COMMAND+=(--agent-env "SMALL_AGENT_HARBOR_AGENT=${RESOLVED_AGENT}")
  fi
  append_agent_env_passthrough
}

print_harbor_command() {
  printf 'Resolved model=%s agent=%s n_concurrent=%s jobs_dir=%s\n' "${RESOLVED_MODEL}" "${RESOLVED_AGENT}" "${RESOLVED_N_CONCURRENT}" "${RESOLVED_JOBS_DIR}"
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

  echo "Pruning stale Docker networks before benchmark run..."
  docker network prune -f 2>/dev/null || true

  (
    cd "${SCRIPT_DIR}"
    "${HARBOR_COMMAND[@]}"
  )
}
