#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TIMESTAMP="$(date -u +"%Y-%m-%d__%H-%M-%S")"
EXPERIMENT_DIR="${SCRIPT_DIR}/jobs/context_experiment_${TIMESTAMP}"
LOG_DIR="${EXPERIMENT_DIR}/slurm_logs"

mkdir -p "${LOG_DIR}"

MODELS=(
  "qwen3.5-9b"
  "qwen3.5-flash"
  "qwen3.5-plus"
)

CONTEXT_LENGTHS=(
  "16k"
  "65k"
  "262k"
)

INFRA_EXCEPTION_PATTERNS="DockerError|ConnectionError|AgentSetupError|EnvironmentError|SetupError|BuildError"

echo "=== Context Length Experiment ==="
echo "Experiment dir: ${EXPERIMENT_DIR}"
echo "Models: ${MODELS[*]}"
echo "Context lengths: ${CONTEXT_LENGTHS[*]}"
echo ""

for model in "${MODELS[@]}"; do
  JOB_SCRIPT="${EXPERIMENT_DIR}/job_${model}.sh"

  cat > "${JOB_SCRIPT}" <<JOBEOF
#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT}"

# Export API key from .env
set -a
source "${PROJECT_ROOT}/.env"
set +a

HARBOR_JOBS_ROOT="${SCRIPT_DIR}/jobs"

echo "========================================"
echo "Context experiment: model=${model}"
echo "========================================"

# --- Smoke test ---
SMOKE_MODEL="${model}-16k"
echo ""
echo "[smoke] Running smoke test with model=\${SMOKE_MODEL} ..."

# Capture smoke output to parse the jobs_dir from the "Resolved ... jobs_dir=" line
SMOKE_LOG="\$(mktemp)"
"${SCRIPT_DIR}/run_smoke.sh" --model "\${SMOKE_MODEL}" 2>&1 | tee "\${SMOKE_LOG}" || true

# Extract the jobs_dir from the "Resolved ... jobs_dir=..." line printed by common.sh
SMOKE_JOBS_DIR="\$(grep -oP 'jobs_dir=\K\S+' "\${SMOKE_LOG}" | head -1)"
rm -f "\${SMOKE_LOG}"
if [[ -z "\${SMOKE_JOBS_DIR}" ]]; then
  echo "[smoke] FATAL: Could not parse jobs_dir from smoke output." >&2
  exit 1
fi

# Find the run-level result.json (not per-trial) inside the smoke jobs dir
RESULT_JSON="\$(find "\${SMOKE_JOBS_DIR}" -maxdepth 2 -name 'result.json' -not -path '*/fix-git*/result.json' | head -1)"
if [[ -z "\${RESULT_JSON}" || ! -f "\${RESULT_JSON}" ]]; then
  echo "[smoke] FATAL: No result.json found in \${SMOKE_JOBS_DIR}" >&2
  exit 1
fi

echo "[smoke] Checking result: \${RESULT_JSON}"

# Validate: check for infra errors (not model mistakes)
SMOKE_OK=1
N_ERRORS="\$(python3 -c "
import json, sys
data = json.load(open(sys.argv[1]))
print(data.get('stats', {}).get('n_errors', 0))
" "\${RESULT_JSON}")"

if [[ "\${N_ERRORS}" -gt 0 ]]; then
  EXCEPTION_TYPES="\$(python3 -c "
import json, sys
data = json.load(open(sys.argv[1]))
evals = data.get('stats', {}).get('evals', {})
types = set()
for ev in evals.values():
    for exc_type in ev.get('exception_stats', {}).keys():
        types.add(exc_type)
print(' '.join(sorted(types)))
" "\${RESULT_JSON}")"

  echo "[smoke] Errors found (n_errors=\${N_ERRORS}). Exception types: \${EXCEPTION_TYPES}"

  for exc_type in \${EXCEPTION_TYPES}; do
    if echo "\${exc_type}" | grep -qE "${INFRA_EXCEPTION_PATTERNS}"; then
      echo "[smoke] FATAL: Infrastructure error detected: \${exc_type}" >&2
      SMOKE_OK=0
    fi
  done
fi

if [[ "\${SMOKE_OK}" -eq 0 ]]; then
  echo "[smoke] Aborting: infrastructure errors would invalidate benchmark results." >&2
  exit 1
fi

echo "[smoke] Smoke test validated (n_errors=\${N_ERRORS}, no infra exceptions)."

# --- Benchmark runs ---
for ctx in ${CONTEXT_LENGTHS[*]}; do
  MODEL_KEY="${model}-\${ctx}"
  echo ""
  echo "========================================"
  echo "[benchmark] model=\${MODEL_KEY}"
  echo "========================================"
  "${SCRIPT_DIR}/run_small_benchmark.sh" --model "\${MODEL_KEY}"
  echo "[benchmark] Completed: \${MODEL_KEY}"
done

echo ""
echo "========================================"
echo "All benchmarks complete for model=${model}"
echo "========================================"
JOBEOF

  chmod +x "${JOB_SCRIPT}"

  SLURM_JOB_ID="$(sbatch \
    --parsable \
    --partition=m7i-cpu \
    --nodes=1 \
    --cpus-per-task=16 \
    --time=1-00:00:00 \
    --job-name="ctx-${model}" \
    --output="${LOG_DIR}/${model}_%j.out" \
    --error="${LOG_DIR}/${model}_%j.err" \
    "${JOB_SCRIPT}")"

  echo "Submitted Slurm job ${SLURM_JOB_ID} for model=${model}"
done

echo ""
echo "=== All jobs submitted ==="
echo "Experiment dir: ${EXPERIMENT_DIR}"
echo "Monitor with: squeue -u \$(whoami) -n ctx-qwen3.5-9b,ctx-qwen3.5-flash,ctx-qwen3.5-plus"
