#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TIMESTAMP="$(date -u +"%Y-%m-%d__%H-%M-%S")"
EXPERIMENT_DIR="${SCRIPT_DIR}/jobs/long_context_experiment_${TIMESTAMP}"
LOG_DIR="${EXPERIMENT_DIR}/slurm_logs"
DOCKER_CACHE_DIR="${SCRIPT_DIR}/docker-cache"

mkdir -p "${LOG_DIR}"

MODELS=(
  "qwen3.5-9b"
  "qwen3.5-flash"
  "qwen3.5-plus"
)

CONTEXT_LENGTHS=(
  "16k"
  "32k"
  "65k"
  "131k"
  "262k"
)

echo "=== Long Context Length Experiment ==="
echo "Experiment dir: ${EXPERIMENT_DIR}"
echo "Models: ${MODELS[*]}"
echo "Context lengths: ${CONTEXT_LENGTHS[*]}"
echo "Benchmark: run_long_benchmark.sh (20 tasks, 5 concurrent)"
echo "Docker cache: ${DOCKER_CACHE_DIR}"
echo ""

# Cancel all existing m7i-cpu jobs
echo "Cancelling existing m7i-cpu jobs..."
squeue -u "$(whoami)" -p m7i-cpu --format="%.10i" --noheader | xargs -r scancel 2>/dev/null || true
echo "Done."
echo ""

ALL_JOB_IDS=()

for model in "${MODELS[@]}"; do
  for ctx in "${CONTEXT_LENGTHS[@]}"; do
    MODEL_KEY="${model}-${ctx}"

    SHORT_MODEL="${model#qwen3.5-}"
    JOB_NAME="long-${SHORT_MODEL}-${ctx}"

    JOB_SCRIPT="${EXPERIMENT_DIR}/job_${MODEL_KEY}.sh"

    cat > "${JOB_SCRIPT}" <<JOBEOF
#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT}"

set -a
source "${PROJECT_ROOT}/.env"
set +a

echo "========================================"
echo "Long benchmark: model=${MODEL_KEY}"
echo "========================================"

# Load pre-cached Docker images so Harbor doesn't hit Docker Hub rate limits.
# mteb-leaderboard is excluded (image too large for local disk).

# Load pre-cached Docker images from shared filesystem
echo "[docker] Loading cached images from ${DOCKER_CACHE_DIR} ..."
LOADED=0
for tarfile in "${DOCKER_CACHE_DIR}"/*.tar; do
  if [ -f "\${tarfile}" ]; then
    docker load -i "\${tarfile}" 2>&1 | tail -1
    LOADED=\$((LOADED + 1))
  fi
done
echo "[docker] Loaded \${LOADED} images."
echo ""

"${SCRIPT_DIR}/run_long_benchmark.sh" --model "${MODEL_KEY}"

echo ""
echo "========================================"
echo "Benchmark complete: model=${MODEL_KEY}"
echo "========================================"
JOBEOF

    chmod +x "${JOB_SCRIPT}"

    SLURM_JOB_ID="$(sbatch \
      --parsable \
      --partition=m7i-cpu \
      --nodes=1 \
      --cpus-per-task=16 \
      --time=1-00:00:00 \
      --job-name="${JOB_NAME}" \
      --output="${LOG_DIR}/${MODEL_KEY}_%j.out" \
      --error="${LOG_DIR}/${MODEL_KEY}_%j.err" \
      "${JOB_SCRIPT}")"

    ALL_JOB_IDS+=("${SLURM_JOB_ID}")
    echo "Submitted job ${SLURM_JOB_ID}: ${JOB_NAME} (model=${MODEL_KEY})"
  done
done

echo ""
echo "=== All ${#ALL_JOB_IDS[@]} benchmark jobs submitted ==="
echo "Job IDs: ${ALL_JOB_IDS[*]}"
echo ""

# Submit 5 keepalive jobs that start after all benchmarks finish
DEPENDENCY_LIST="$(IFS=:; echo "${ALL_JOB_IDS[*]}")"

echo "Submitting 5 keepalive jobs (depend on all benchmark jobs)..."
for i in $(seq 1 5); do
  KA_ID="$(sbatch \
    --parsable \
    --partition=m7i-cpu \
    --nodes=1 \
    --cpus-per-task=16 \
    --time=2-00:00:00 \
    --job-name="keepalive-long-${i}" \
    --dependency="afterany:${DEPENDENCY_LIST}" \
    --wrap="sleep infinity")"
  echo "  keepalive-long-${i}: job ${KA_ID}"
done

echo ""
echo "=== Experiment fully submitted ==="
echo "Experiment dir: ${EXPERIMENT_DIR}"
echo "Monitor with: squeue -u \$(whoami) -p m7i-cpu"
