# Long Context Length Experiment

## Goal

Study how context length impacts terminal agent performance on long-running tasks.

## Parameters

| Parameter | Value |
|---|---|
| Models | `qwen/qwen3.5-9b`, `qwen/qwen3.5-flash-02-23`, `qwen/qwen3.5-plus-02-15` |
| Context lengths | 16k, 32k, 65k, 131k, 262k |
| Benchmark | `harbor/run_long_benchmark.sh` (20 tasks from terminal-bench@2.0, 5 concurrent) |
| Total jobs | 15 (3 models x 5 context lengths) |
| Infrastructure | Slurm `m7i-cpu` partition, 1 node per job, 16 CPUs, 24h time limit |
| Reasoning | Disabled for all models (`"reasoning": {"enabled": false}`) |

## Config entries

Each (model, context_length) pair has a dedicated key in `config.json` following the pattern `qwen3.5-{9b,flash,plus}-{16k,32k,65k,131k,262k}`. For example:

```json
"qwen3.5-9b-16k": {
  "model": "qwen/qwen3.5-9b",
  "api_base": "https://openrouter.ai/api/v1",
  "api_key": "OPENROUTER_API_KEY",
  "context_length": 16384,
  "extra_params": { "reasoning": { "enabled": false } }
}
```

Context length values: 16384, 32768, 65536, 131072, 262144.

## Benchmark tasks (20)

build-pov-ray, circuit-fibsqrt, compile-compcert, db-wal-recovery, dna-insert,
extract-moves-from-video, fix-ocaml-gc, gcode-to-text, mailman, make-doom-for-mips,
make-mips-interpreter, mteb-leaderboard, path-tracing, path-tracing-reverse,
regex-chess, rstan-to-pystan, sam-cell-seg, schemelike-metacircular-eval,
train-fasttext, winning-avg-corewars.

## Docker image caching

Running 15 parallel jobs that all pull Docker images from Docker Hub triggers
anonymous rate limiting. To avoid this, all 20 task images were pre-pulled and
saved as tars to a shared filesystem location:

```
/wbl-fast/usrs/ethan/small-agent/harbor/docker-cache/*.tar
```

Each Slurm job loads these tars via `docker load` before running the benchmark.
This avoids any Docker Hub pulls at runtime.

**Exception:** The `mteb-leaderboard` image (`alexgshaw/mteb-leaderboard:20251031`)
is ~8.4 GB compressed and exceeds the local disk capacity of the m7i-cpu nodes
(44 GB root, ~8 GB free). This image could not be cached. The `mteb-leaderboard`
task will fail with a Docker pull error on each run and is effectively excluded
from results. The remaining 19 tasks run normally.

Attempts to redirect Docker/containerd storage to `/wbl-fast` (a Lustre/FSx
filesystem) failed because overlayfs whiteout operations are not supported on
network filesystems.

### Cached images (19 of 20)

| Image | Tar size |
|---|---|
| alexgshaw/fix-ocaml-gc:20251031 | 733 MB |
| alexgshaw/make-mips-interpreter:20251031 | 497 MB |
| alexgshaw/train-fasttext:20251031 | 488 MB |
| alexgshaw/sam-cell-seg:20251031 | 395 MB |
| alexgshaw/path-tracing:20251031 | 383 MB |
| alexgshaw/winning-avg-corewars:20251031 | 282 MB |
| alexgshaw/make-doom-for-mips:20251031 | 182 MB |
| alexgshaw/db-wal-recovery:20251031 | 168 MB |
| alexgshaw/path-tracing-reverse:20251031 | 164 MB |
| alexgshaw/build-pov-ray:20251031 | 157 MB |
| alexgshaw/mailman:20251031 | 155 MB |
| alexgshaw/circuit-fibsqrt:20251031 | 143 MB |
| alexgshaw/regex-chess:20251031 | 54 MB |
| alexgshaw/rstan-to-pystan:20251031 | 49 MB |
| alexgshaw/gcode-to-text:20251031 | 44 MB |
| alexgshaw/schemelike-metacircular-eval:20251031 | 43 MB |
| alexgshaw/compile-compcert:20251031 | 29 MB |
| alexgshaw/dna-insert:20251031 | 29 MB |
| alexgshaw/extract-moves-from-video:20251031 | 29 MB |
| **Total** | **4.0 GB** |

## Orchestration script

`harbor/run_long_context_experiment.sh` performs the following:

1. Cancels any existing m7i-cpu Slurm jobs.
2. For each of the 15 (model, context_length) combinations, generates a job
   script that:
   - Sources `.env` for the `OPENROUTER_API_KEY`.
   - Loads all 19 cached Docker image tars via `docker load`.
   - Runs `harbor/run_long_benchmark.sh --model <model_key>`.
3. Submits each job via `sbatch` with `--partition=m7i-cpu --nodes=1
   --cpus-per-task=16 --time=1-00:00:00`.
4. Submits 5 keepalive jobs (`sleep infinity`, 2-day limit) with
   `--dependency=afterany:<all benchmark job IDs>` to retain nodes after
   completion.

All 15 jobs run fully in parallel on separate nodes.

## Issues encountered and resolved

### 1. Task cache race condition (`FileExistsError`)

When multiple jobs start simultaneously, they race to download task definitions
to the shared `~/.cache/harbor/tasks/` directory. Harbor's `_copy_task_source_to_target`
uses non-atomic `rmtree` + `copytree`, causing `FileExistsError` when two jobs
try to create the same directory.

**Resolution:** The pre-cached Docker images eliminate most of the startup
contention. Jobs that still hit this error on the task cache are resubmitted
manually after the cache is populated by the first successful job.

### 2. Docker Hub anonymous rate limiting

With 15 jobs pulling 20 Docker images each, Docker Hub's anonymous pull rate
limit was immediately exhausted, causing all trials to fail with `RuntimeError`.

**Resolution:** Pre-pulled all images on the login node, saved them as tars to
`/wbl-fast` (shared filesystem), and added `docker load` to each job script.

### 3. Login node disk space

The login node has only 44 GB root disk. After pulling ~12 images, it ran out
of space. Images were saved to `/wbl-fast` incrementally (pull, save, remove)
to stay within disk limits.

### 4. Compute node disk space (mteb-leaderboard)

The `mteb-leaderboard` image is 8.4 GB compressed and cannot fit on the compute
nodes' local disk alongside the other 19 images. Attempts to redirect
Docker/containerd storage to `/wbl-fast` failed due to Lustre not supporting
overlayfs whiteout operations.

**Resolution:** Accepted that `mteb-leaderboard` will fail on all runs. Results
are collected for the remaining 19 tasks.

## Compaction logging

The agent code (`agents/terminus2/core_agent.py` and `agents/terminus2/agent.py`)
was previously instrumented to log context compaction events. Each run prints a
summary line:

```
Compactions: N (proactive=P, reactive=R)
```

This data is captured in each trial's `result.json` under
`agent_result.metadata.small_agent_result.stdout`.

## Monitoring

```bash
# Check job status
squeue -u $(whoami) -p m7i-cpu

# Check a specific job's output
tail -f harbor/jobs/long_context_experiment_<timestamp>/slurm_logs/<model_key>_<jobid>.out

# Experiment directory
harbor/jobs/long_context_experiment_2026-03-12__05-46-04/
```

## Current run

| Job ID | Name | Model key |
|---|---|---|
| 12066 | long-9b-16k | qwen3.5-9b-16k |
| 12067 | long-9b-32k | qwen3.5-9b-32k |
| 12068 | long-9b-65k | qwen3.5-9b-65k |
| 12069 | long-9b-131k | qwen3.5-9b-131k |
| 12070 | long-9b-262k | qwen3.5-9b-262k |
| 12071 | long-flash-16k | qwen3.5-flash-16k |
| 12072 | long-flash-32k | qwen3.5-flash-32k |
| 12073 | long-flash-65k | qwen3.5-flash-65k |
| 12074 | long-flash-131k | qwen3.5-flash-131k |
| 12075 | long-flash-262k | qwen3.5-flash-262k |
| 12076 | long-plus-16k | qwen3.5-plus-16k |
| 12077 | long-plus-32k | qwen3.5-plus-32k |
| 12078 | long-plus-65k | qwen3.5-plus-65k |
| 12079 | long-plus-131k | qwen3.5-plus-131k |
| 12080 | long-plus-262k | qwen3.5-plus-262k |
| 12081-12085 | keepalive-long-{1..5} | (pending, depend on above) |
