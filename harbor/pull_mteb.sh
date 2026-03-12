#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="/wbl-fast/usrs/ethan/small-agent/harbor/docker-cache"
NODE_STORE="/wbl-fast/usrs/ethan/docker-data/$(hostname)"

echo "Redirecting Docker+containerd storage to ${NODE_STORE} ..."
sudo mkdir -p "${NODE_STORE}/docker" "${NODE_STORE}/containerd"

sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
    "data-root": "${NODE_STORE}/docker",
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
EOF

sudo sed -i "s|^#root = .*|root = \"${NODE_STORE}/containerd\"|; s|^root = .*|root = \"${NODE_STORE}/containerd\"|" /etc/containerd/config.toml

sudo systemctl restart containerd
sudo systemctl restart docker
sleep 3

echo "Docker info:"
docker info 2>&1 | grep "Docker Root Dir"

echo "Pulling mteb-leaderboard..."
docker pull alexgshaw/mteb-leaderboard:20251031

echo "Saving to tar..."
docker save alexgshaw/mteb-leaderboard:20251031 -o "${CACHE_DIR}/mteb-leaderboard-20251031.tar"
echo "DONE: $(ls -lh "${CACHE_DIR}/mteb-leaderboard-20251031.tar")"
