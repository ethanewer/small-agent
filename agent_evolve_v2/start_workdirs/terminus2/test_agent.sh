#!/bin/sh
set -eu

workspace_root=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
repo_root="$workspace_root"
while [ ! -f "$repo_root/agent_evolve_v2/runs.json" ]; do
  parent=$(dirname "$repo_root")
  if [ "$parent" = "$repo_root" ]; then
    echo "Unable to locate repo root containing agent_evolve_v2/runs.json" >&2
    exit 1
  fi
  repo_root="$parent"
done

model_key=${1:-}
if [ -z "$model_key" ]; then
  echo "usage: ./test_agent.sh <model-key>" >&2
  exit 2
fi

cd "$repo_root"
uv run python -m agent_evolve_v2.service_cli validate \
  --workspace "$workspace_root" \
  --model-key "$model_key"
