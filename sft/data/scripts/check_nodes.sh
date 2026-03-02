#!/usr/bin/env bash
set -euo pipefail

KEY_PATH="${1:-krafton-dld-public.pem}"
SSH_OPTS=(-i "$KEY_PATH" -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new)

NODES=(
  "ubuntu@ec2-18-116-64-18.us-east-2.compute.amazonaws.com"
  "ubuntu@ec2-3-142-90-94.us-east-2.compute.amazonaws.com"
  "ubuntu@ec2-18-117-243-14.us-east-2.compute.amazonaws.com"
  "ubuntu@ec2-3-141-11-64.us-east-2.compute.amazonaws.com"
)

echo "Checking node availability with key: $KEY_PATH"
for host in "${NODES[@]}"; do
  if ssh "${SSH_OPTS[@]}" "$host" "echo ok" >/dev/null 2>&1; then
    echo "[FREE] $host"
  else
    echo "[BUSY/UNREACHABLE] $host"
  fi
done
