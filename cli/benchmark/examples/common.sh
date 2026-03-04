#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TB_DATASET="${TB_DATASET:-terminal-bench-core==0.1.1}"
TB_OUTPUT_PATH="${TB_OUTPUT_PATH:-.benchmark-artifacts/tb2-runs}"
TB_AGENT_IMPORT_PATH="${TB_AGENT_IMPORT_PATH:-benchmark.harbor_bridge:HarborTB2DefaultAgent}"
TB_CONFIG_PATH="${TB_CONFIG_PATH:-./config.json}"
TB_CMD=(uvx --with pexpect --with rich --from terminal-bench tb run)

run_tb() {
  (
    cd "${CLI_DIR}"
    "${TB_CMD[@]}" "$@"
  )
}

run_tb_for_model() {
  local model_key="$1"
  local agent_key="$2"
  local run_id="$3"
  shift 3

  run_tb \
    --dataset "${TB_DATASET}" \
    --agent-import-path "${TB_AGENT_IMPORT_PATH}" \
    --agent-kwarg "config_path=${TB_CONFIG_PATH}" \
    --agent-kwarg "agent_key=${agent_key}" \
    --agent-kwarg "model_key=${model_key}" \
    "$@" \
    --output-path "${TB_OUTPUT_PATH}" \
    --run-id "${run_id}"
}

load_default_model_and_agent() {
  local defaults
  defaults="$(
    cd "${CLI_DIR}" && TB_CONFIG_PATH="${TB_CONFIG_PATH}" uv run python - <<'PY'
import json
import os
from pathlib import Path

cfg_path = Path(os.environ["TB_CONFIG_PATH"])
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
print(cfg.get("default_model", ""))
print(cfg.get("default_agent", ""))
PY
  )"

  DEFAULT_MODEL_KEY="$(printf '%s\n' "${defaults}" | sed -n '1p')"
  DEFAULT_AGENT_KEY="$(printf '%s\n' "${defaults}" | sed -n '2p')"

  if [[ -z "${DEFAULT_MODEL_KEY}" ]]; then
    echo "Unable to resolve default_model from ${TB_CONFIG_PATH}" >&2
    exit 1
  fi

  if [[ -z "${DEFAULT_AGENT_KEY}" ]]; then
    echo "Unable to resolve default_agent from ${TB_CONFIG_PATH}" >&2
    exit 1
  fi
}
