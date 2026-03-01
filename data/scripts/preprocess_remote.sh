#!/usr/bin/env bash
set -euo pipefail

# Optional helper: preprocess dataset shards on available remote nodes
# and sync partial JSONL outputs back locally.

KEY_PATH="${1:-krafton-dld-public.pem}"
REMOTE_DIR="${2:-/home/ubuntu/qwen-preprocess}"
LOCAL_ROOT="${3:-/wbl-fast/usrs/ethan/small-agent}"

SSH_OPTS=(-i "$KEY_PATH" -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new)

NODES=(
  "ubuntu@ec2-18-116-64-18.us-east-2.compute.amazonaws.com"
  "ubuntu@ec2-3-142-90-94.us-east-2.compute.amazonaws.com"
  "ubuntu@ec2-18-117-243-14.us-east-2.compute.amazonaws.com"
  "ubuntu@ec2-3-141-11-64.us-east-2.compute.amazonaws.com"
)

SPECS=(
  "nvidia/Nemotron-Terminal-Corpus|skill_based_easy|conversations"
  "nvidia/Nemotron-Terminal-Corpus|skill_based_medium|conversations"
  "nvidia/Nemotron-Terminal-Corpus|skill_based_mixed|conversations"
  "Nanbeige/ToolMind-Web-QA|test|conversations"
  "SWE-Factory/DeepSWE-Agent-Kimi-K2-Trajectories-2.8K|default|messages"
)

mkdir -p "$LOCAL_ROOT/data/remote-partials"

AVAILABLE_HOSTS=()
echo "Checking available remote nodes..."
for host in "${NODES[@]}"; do
  if ssh "${SSH_OPTS[@]}" "$host" "echo ok" >/dev/null 2>&1; then
    AVAILABLE_HOSTS+=("$host")
    echo "[FREE] $host"
  else
    echo "[BUSY/UNREACHABLE] $host"
  fi
done

if [[ ${#AVAILABLE_HOSTS[@]} -eq 0 ]]; then
  echo "No available hosts found."
  exit 1
fi

echo "Syncing project to available nodes..."
for host in "${AVAILABLE_HOSTS[@]}"; do
  ssh "${SSH_OPTS[@]}" "$host" "mkdir -p '$REMOTE_DIR'"
  rsync -az --delete --exclude '.venv' --exclude 'data' "$LOCAL_ROOT/" "$host:$REMOTE_DIR/"
done

run_spec_job() {
  local host="$1"
  local spec="$2"
  local dataset_name="${spec%%|*}"
  local rest="${spec#*|}"
  local dataset_config="${rest%%|*}"
  local messages_key="${spec##*|}"

  ssh "${SSH_OPTS[@]}" "$host" "cd '$REMOTE_DIR' && uv sync && uv run python - <<'PY'
from pathlib import Path
import orjson
from datasets import load_dataset
from data.pipeline.build_qwen_dataset import normalize_messages, _shuffle_key

dataset_name = ${dataset_name@Q}
dataset_config = ${dataset_config@Q}
messages_key = ${messages_key@Q}
seed = 42

out_dir = Path('data/remote-partials')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / (dataset_name.replace('/', '__') + '__' + dataset_config + '.jsonl')

if dataset_name == 'Nanbeige/ToolMind-Web-QA':
    ds = load_dataset(
        'json',
        data_files=[
            'hf://datasets/Nanbeige/ToolMind-Web-QA/open-wiki-traj.jsonl',
            'hf://datasets/Nanbeige/ToolMind-Web-QA/syn_wikiqa.jsonl',
        ],
        split='train',
        streaming=True,
    )
else:
    ds = load_dataset(dataset_name, dataset_config, split='train', streaming=True)

with out_path.open('wb') as f:
    for row_id, row in enumerate(ds):
        messages = normalize_messages(row.get(messages_key))
        if not messages:
            continue
        out = {
            'source_dataset': dataset_name,
            'source_config': dataset_config,
            'source_split': 'train',
            'row_id': f'{dataset_name}:{dataset_config}:{row_id}',
            'shuffle_key': _shuffle_key(seed, dataset_name, row_id),
            'messages': messages,
        }
        f.write(orjson.dumps(out))
        f.write(b'\\n')
print(f'Wrote {out_path}')
PY"
}

echo "Launching remote preprocessing jobs..."
host_count=${#AVAILABLE_HOSTS[@]}
job_count=0
for i in "${!SPECS[@]}"; do
  host="${AVAILABLE_HOSTS[$((i % host_count))]}"
  spec="${SPECS[$i]}"
  run_spec_job "$host" "$spec" &
  job_count=$((job_count + 1))
  if (( job_count % host_count == 0 )); then
    wait
  fi
done
wait

echo "Pulling partial shards back to login node..."
for host in "${AVAILABLE_HOSTS[@]}"; do
  if ssh "${SSH_OPTS[@]}" "$host" "test -d '$REMOTE_DIR/data/remote-partials'" >/dev/null 2>&1; then
    rsync -az "$host:$REMOTE_DIR/data/remote-partials/" "$LOCAL_ROOT/data/remote-partials/"
  fi
done

echo "Done. Merge partials locally with your chosen build policy."
