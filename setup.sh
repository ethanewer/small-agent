#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI_DIR="${ROOT_DIR}/cli"

log() {
  printf '[setup] %s\n' "$*"
}

warn() {
  printf '[setup][warn] %s\n' "$*" >&2
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

ensure_python() {
  if ! have_cmd python3; then
    warn "python3 is required but was not found."
    exit 1
  fi
}

ensure_pipx() {
  if have_cmd pipx; then
    return
  fi

  log "pipx not found; installing with python3 --user."
  python3 -m pip install --user pipx

  if [ -d "${HOME}/.local/bin" ]; then
    export PATH="${HOME}/.local/bin:${PATH}"
  fi

  if ! have_cmd pipx; then
    warn "pipx install completed but pipx is still not on PATH."
    warn "Add ~/.local/bin to PATH, then re-run setup.sh."
    exit 1
  fi
}

ensure_uv() {
  if have_cmd uv; then
    return
  fi

  log "Installing uv via pipx."
  pipx install uv
}

install_or_upgrade_pipx_pkg() {
  local package="$1"
  if pipx list --short 2>/dev/null | awk '{print $1}' | rg -x --quiet "${package}"; then
    log "Upgrading pipx package: ${package}"
    pipx upgrade "${package}" || true
  else
    log "Installing pipx package: ${package}"
    pipx install "${package}"
  fi
}

install_local_cli() {
  if [ ! -d "${CLI_DIR}" ]; then
    warn "Expected CLI project at ${CLI_DIR}, but directory was not found."
    exit 1
  fi

  log "Installing local terminus2-cli from ${CLI_DIR}"
  pipx install --force "${CLI_DIR}"
}

install_qwen_cli() {
  log "Attempting pipx install for qwen code agent CLI."
  if pipx install qwen-code >/dev/null 2>&1; then
    return
  fi

  warn "pipx package 'qwen-code' not available; falling back to npm package @qwen-code/qwen-code."
  if ! have_cmd npm; then
    warn "npm is required for qwen CLI fallback install."
    exit 1
  fi
  npm install -g @qwen-code/qwen-code
}

install_codex_cli() {
  log "Attempting pipx install for codex agent CLI."
  if pipx install codex >/dev/null 2>&1; then
    return
  fi

  warn "pipx package 'codex' not available; falling back to npm package @openai/codex."
  if ! have_cmd npm; then
    warn "npm is required for codex CLI fallback install."
    exit 1
  fi
  npm install -g @openai/codex
}

verify_clis() {
  local missing=0
  for cmd in terminus2-cli qwen codex; do
    if have_cmd "${cmd}"; then
      log "Found CLI: ${cmd}"
    else
      warn "Missing CLI after setup: ${cmd}"
      missing=1
    fi
  done

  if [ "${missing}" -ne 0 ]; then
    exit 1
  fi
}

print_next_steps() {
  cat <<'EOF'
[setup] Install complete.

[setup] Ensure these env vars are set for model providers:
  - OPENAI_API_KEY
  - OPENROUTER_API_KEY
  - OPENAI_BASE_URL (optional; needed for custom OpenAI-compatible endpoints)

[setup] Quick checks:
  terminus2-cli --help
  qwen --help
  codex --help
EOF
}

main() {
  ensure_python
  ensure_pipx
  ensure_uv
  install_local_cli
  install_qwen_cli
  install_codex_cli
  verify_clis
  print_next_steps
}

main "$@"
