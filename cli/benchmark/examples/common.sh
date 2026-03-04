#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TB_DATASET="${TB_DATASET:-terminal-bench-core==0.1.1}"
TB_DATASET_PATH="${TB_DATASET_PATH:-}"
TB_OUTPUT_PATH="${TB_OUTPUT_PATH:-${TMPDIR:-/tmp}/small-agent-tb2-runs}"
TB_AGENT_IMPORT_PATH="${TB_AGENT_IMPORT_PATH:-benchmark.harbor_bridge:HarborTB2DefaultAgent}"
TB_CONFIG_PATH="${TB_CONFIG_PATH:-./config.json}"
TB_LOCAL_REGISTRY_PATH="${TB_LOCAL_REGISTRY_PATH:-}"
TB_USE_DATASET_CACHE="${TB_USE_DATASET_CACHE:-1}"
TB_CMD=(uvx --with pexpect --with rich --from terminal-bench tb run)

resolve_cli_path() {
  local candidate="$1"
  if [[ "${candidate}" = /* ]]; then
    printf '%s\n' "${candidate}"
    return
  fi

  local normalized="${candidate#./}"
  printf '%s\n' "${CLI_DIR}/${normalized}"
}

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
  local -a dataset_args=()
  local -a run_args=()
  local config_path_abs
  local dataset_name
  local dataset_version
  local cached_dataset_path

  config_path_abs="$(resolve_cli_path "${TB_CONFIG_PATH}")"

  if [[ -n "${TB_DATASET_PATH}" ]]; then
    dataset_args=(--dataset-path "${TB_DATASET_PATH}")
  elif [[ "${TB_USE_DATASET_CACHE}" == "1" && "${TB_DATASET}" == *"=="* ]]; then
    dataset_name="${TB_DATASET%%==*}"
    dataset_version="${TB_DATASET##*==}"
    cached_dataset_path="${HOME}/.cache/terminal-bench/${dataset_name}/${dataset_version}"
    if [[ -d "${cached_dataset_path}" ]]; then
      dataset_args=(--dataset-path "${cached_dataset_path}")
    else
      dataset_args=(--dataset "${TB_DATASET}")
    fi
  else
    dataset_args=(--dataset "${TB_DATASET}")
  fi

  run_args=("${dataset_args[@]}")
  if [[ -n "${TB_LOCAL_REGISTRY_PATH}" ]]; then
    run_args+=(--local-registry-path "${TB_LOCAL_REGISTRY_PATH}")
  fi
  run_args+=(
    --agent-import-path "${TB_AGENT_IMPORT_PATH}"
    --agent-kwarg "config_path=${config_path_abs}"
    --agent-kwarg "agent_key=${agent_key}"
    --agent-kwarg "model_key=${model_key}"
  )
  run_args+=("$@")
  run_args+=(
    --output-path "${TB_OUTPUT_PATH}"
    --run-id "${run_id}"
  )

  run_tb "${run_args[@]}"
}

load_default_model_and_agent() {
  local defaults
  local config_path_abs

  config_path_abs="$(resolve_cli_path "${TB_CONFIG_PATH}")"
  defaults="$(
    cd "${CLI_DIR}" && TB_CONFIG_PATH="${config_path_abs}" uv run python - <<'PY'
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
