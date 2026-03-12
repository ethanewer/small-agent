#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENT_DIR="${SCRIPT_DIR}/jobs/long_context_experiment_2026-03-12__05-46-04"
LOG_DIR="${EXPERIMENT_DIR}/slurm_logs"
DOCKER_CACHE_DIR="${SCRIPT_DIR}/docker-cache"
RETRY_DIR="${EXPERIMENT_DIR}/retries"
MONITOR_LOG="${EXPERIMENT_DIR}/monitor.log"

TIMEOUT_MULTIPLIER=3

mkdir -p "${RETRY_DIR}"

set -a
source "${PROJECT_ROOT}/.env"
set +a

log() {
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MONITOR_LOG}"
}

get_failed_tasks() {
  local jobs_dir="$1"
  python3 - "${jobs_dir}" <<'PY'
import json, glob, os, sys

jobs_dir = sys.argv[1]
trial_results = glob.glob(os.path.join(jobs_dir, "*", "*__*", "result.json"))

failed_tasks = []
for tr in trial_results:
    data = json.load(open(tr))
    trial_name = tr.split("/")[-2]
    task_name = trial_name.rsplit("__", 1)[0]
    reward_path = os.path.join(os.path.dirname(tr), "verifier", "reward.txt")

    exc_info = data.get("exception_info")
    if exc_info:
        exc_type = exc_info.get("exception_type", "")
        # Only retry timeouts and transient runtime errors, not mteb-leaderboard docker pull failures
        exc_msg = exc_info.get("exception_message", "")
        if "rate limit" in exc_msg.lower() or "mteb-leaderboard" in exc_msg.lower():
            continue
        if exc_type in ("AgentTimeoutError", "RuntimeError", "EnvironmentError"):
            failed_tasks.append(task_name)

for t in sorted(set(failed_tasks)):
    print(t)
PY
}

get_jobs_dir() {
  local out_file="$1"
  grep 'jobs_dir=' "${out_file}" | grep -oP 'jobs_dir=\K\S+' | head -1
}

is_job_done() {
  local out_file="$1"
  grep -q "Benchmark complete" "${out_file}" 2>/dev/null
}

run_retry() {
  local model_key="$1"
  shift
  local tasks=("$@")

  if [ ${#tasks[@]} -eq 0 ]; then
    return
  fi

  log "RETRY ${model_key}: ${#tasks[@]} tasks with timeout_multiplier=${TIMEOUT_MULTIPLIER}"
  log "  Tasks: ${tasks[*]}"

  local retry_script="${RETRY_DIR}/retry_${model_key}.sh"
  local task_flags=""
  for t in "${tasks[@]}"; do
    task_flags="${task_flags} --task-name ${t}"
  done

  cat > "${retry_script}" <<RETRYEOF
#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT}"
set -a
source "${PROJECT_ROOT}/.env"
set +a

echo "========================================"
echo "RETRY: model=${model_key} (timeout_multiplier=${TIMEOUT_MULTIPLIER})"
echo "========================================"

# Load cached Docker images
for tarfile in "${DOCKER_CACHE_DIR}"/*.tar; do
  if [ -f "\${tarfile}" ]; then
    docker load -i "\${tarfile}" 2>&1 | tail -1
  fi
done

cd "${SCRIPT_DIR}"

HARBOR_CMD=()
if command -v harbor >/dev/null 2>&1; then
  HARBOR_CMD=(harbor)
elif command -v uvx >/dev/null 2>&1; then
  HARBOR_CMD=(
    uvx --from harbor --with truststore --with rich python -c
    "import truststore; truststore.inject_into_ssl(); from harbor.cli.main import app; app()"
  )
fi

JOBS_DIR="${RETRY_DIR}/jobs_${model_key}_\$(date -u +%Y%m%d_%H%M%S)"

docker network prune -f 2>/dev/null || true

"\${HARBOR_CMD[@]}" run \\
  --jobs-dir "\${JOBS_DIR}" \\
  --n-concurrent 5 \\
  --env docker \\
  --delete \\
  --no-force-build \\
  -d "terminal-bench@2.0" \\
  --agent-import-path "agent:SmallAgentHarborAgent" \\
  --agent-env "SMALL_AGENT_HARBOR_MODEL=${model_key}" \\
  --agent-env "SMALL_AGENT_HARBOR_AGENT=terminus-2" \\
  --agent-env "OPENROUTER_API_KEY=\${OPENROUTER_API_KEY}" \\
  --timeout-multiplier ${TIMEOUT_MULTIPLIER} \\
  ${task_flags}

echo ""
echo "========================================"
echo "RETRY complete: model=${model_key}"
echo "========================================"
RETRYEOF

  chmod +x "${retry_script}"

  local short_model="${model_key#qwen3.5-}"
  local slurm_id
  slurm_id="$(sbatch \
    --parsable \
    --partition=m7i-cpu \
    --nodes=1 \
    --cpus-per-task=16 \
    --time=1-00:00:00 \
    --job-name="retry-${short_model}" \
    --output="${RETRY_DIR}/${model_key}_retry_%j.out" \
    --error="${RETRY_DIR}/${model_key}_retry_%j.err" \
    "${retry_script}")"

  log "  Submitted retry job ${slurm_id} for ${model_key}"
}

report_progress() {
  log "=== PROGRESS REPORT ==="
  python3 - "${LOG_DIR}" "${RETRY_DIR}" <<'PY'
import json, glob, os, sys

log_dir = sys.argv[1]
retry_dir = sys.argv[2]

def analyze_jobs_dir(jobs_dir):
    if not jobs_dir or not os.path.isdir(jobs_dir):
        return 0, 0, 0, 0, 0.0
    reward_files = glob.glob(os.path.join(jobs_dir, "*", "*__*", "verifier", "reward.txt"))
    passed = sum(1 for f in reward_files if open(f).read().strip() == "1")
    failed = len(reward_files) - passed
    trial_results = glob.glob(os.path.join(jobs_dir, "*", "*__*", "result.json"))
    errors = 0
    for tr in trial_results:
        trial_dir = os.path.dirname(tr)
        reward_path = os.path.join(trial_dir, "verifier", "reward.txt")
        if not os.path.exists(reward_path):
            data = json.load(open(tr))
            if data.get("exception_info"):
                errors += 1
    total_c = 0; ct = 0
    for tr in trial_results:
        try:
            data = json.load(open(tr))
            stdout = data.get("agent_result", {}) or {}
            stdout = stdout.get("metadata", {}).get("small_agent_result", {}).get("stdout", "")
            for line in stdout.splitlines():
                if line.startswith("Compactions:"):
                    parts = line.split("(")[1].rstrip(")")
                    for part in parts.split(","):
                        k, v = part.strip().split("=")
                        if k in ("proactive", "reactive"):
                            total_c += int(v)
                    ct += 1
                    break
        except Exception:
            pass
    avg_c = total_c / ct if ct else 0
    return len(reward_files) + errors, passed, failed, errors, avg_c

print(f"{'Model':<25} {'Done':>7} {'Pass':>5} {'Fail':>5} {'Err':>5} {'AvgC':>6} {'Status'}")
print("-" * 65)

for out_file in sorted(glob.glob(os.path.join(log_dir, "*.out"))):
    bn = os.path.basename(out_file)
    model_key = bn.rsplit("_", 1)[0]
    with open(out_file) as f:
        content = f.read()
    jobs_dir = None
    for line in content.splitlines():
        if "jobs_dir=" in line:
            parts = line.split("jobs_dir=")
            if len(parts) > 1:
                jobs_dir = parts[1].strip()
                break
    done, p, f_, e, avg_c = analyze_jobs_dir(jobs_dir)
    complete = "Benchmark complete" in content
    status = "done" if complete else "running"
    rate = f"{p}/{done - e}" if (done - e) > 0 else "-"
    print(f"{model_key:<25} {done:>4}/20 {p:>5} {f_:>5} {e:>5} {avg_c:>6.2f} {status}")
PY
  log "========================"
}

# --- Main monitoring loop ---
log "Monitor started. Experiment: ${EXPERIMENT_DIR}"
log "Checking every hour. Will retry errored tasks with timeout_multiplier=${TIMEOUT_MULTIPLIER}."

RETRIED_MODELS=""

while true; do
  report_progress

  # Check for completed jobs and retry errored tasks
  STILL_RUNNING=0
  for out_file in "${LOG_DIR}"/*.out; do
    model_key="$(basename "${out_file}" | sed 's/_[0-9]*\.out$//')"

    if ! is_job_done "${out_file}"; then
      STILL_RUNNING=$((STILL_RUNNING + 1))
      continue
    fi

    # Skip if already retried
    if echo "${RETRIED_MODELS}" | grep -q "${model_key}"; then
      continue
    fi

    jobs_dir="$(get_jobs_dir "${out_file}")"
    if [ -z "${jobs_dir}" ]; then
      continue
    fi

    failed_tasks="$(get_failed_tasks "${jobs_dir}")"
    if [ -n "${failed_tasks}" ]; then
      mapfile -t task_array <<< "${failed_tasks}"
      run_retry "${model_key}" "${task_array[@]}"
      RETRIED_MODELS="${RETRIED_MODELS} ${model_key}"
    else
      log "OK ${model_key}: no retryable errors"
    fi
  done

  # Also check retry jobs
  RETRY_RUNNING=0
  for retry_out in "${RETRY_DIR}"/*_retry_*.out 2>/dev/null; do
    [ -f "${retry_out}" ] || continue
    if ! grep -q "RETRY complete" "${retry_out}" 2>/dev/null; then
      RETRY_RUNNING=$((RETRY_RUNNING + 1))
    fi
  done

  log "Status: ${STILL_RUNNING} main jobs running, ${RETRY_RUNNING} retry jobs running"

  if [ "${STILL_RUNNING}" -eq 0 ] && [ "${RETRY_RUNNING}" -eq 0 ]; then
    log "All jobs and retries complete. Final report:"
    report_progress
    log "Monitor exiting."
    break
  fi

  log "Sleeping 1 hour..."
  sleep 3600
done
