# Long Context Experiment Benchmark Report
## Experiment: long_context_experiment_2026-03-12__05-46-04

---

## A) SUMMARY TABLE

| Model | Trials | Errors | Mean | RuntimeErrors | AgentTimeoutErrors | Passes |
|-------|--------|--------|------|---------------|---------------------|--------|
| qwen3.5-9b-16k | 17 | 13 | 0.000 | 3 | 10 | 0 |
| qwen3.5-9b-32k | 17 | 13 | 0.050 | 3 | 10 | 1 |
| qwen3.5-9b-65k | 17 | 11 | 0.000 | 3 | 8 | 0 |
| qwen3.5-9b-131k | 17 | 11 | 0.000 | 3 | 8 | 0 |
| qwen3.5-9b-262k | 17 | 11 | 0.050 | 3 | 8 | 1 |
| qwen3.5-flash-16k | 17 | 10 | 0.000 | 3 | 7 | 0 |
| qwen3.5-flash-32k | 17 | 10 | 0.050 | 3 | 7 | 1 |
| qwen3.5-flash-65k | 17 | 10 | 0.050 | 3 | 7 | 1 |
| qwen3.5-flash-131k | 17 | 10 | 0.050 | 3 | 7 | 1 |
| qwen3.5-flash-262k | 17 | 12 | 0.100 | 3 | 9 | 2 |
| qwen3.5-plus-16k | 17 | 15 | 0.050 | 3 | 12 | 1 |
| qwen3.5-plus-32k | 17 | 13 | 0.000 | 3 | 10 | 0 |
| qwen3.5-plus-65k | 17 | 14 | 0.050 | 3 | 11 | 1 |
| qwen3.5-plus-131k | 15 | 12 | 0.050 | 5 | 7 | 1 |
| qwen3.5-plus-262k | 16 | 13 | 0.000 | 3 | 10 | 0 |

**Note:** Trials = 17 (not 20) because `mteb-leaderboard` is excluded—Docker image too large for compute node disk. All 15 models have 3 RuntimeErrors from infrastructure (mailman, winning-avg-corewars, mteb-leaderboard).

---

## B) ERRORED TASKS BY MODEL

### qwen3.5-9b-16k
- compile-compcert: AgentTimeoutError (Agent execution timed out after 2400.0 seconds)
- db-wal-recovery: AgentTimeoutError (Agent execution timed out after 900.0 seconds)
- dna-insert: AgentTimeoutError (Agent execution timed out after 1800.0 seconds)
- extract-moves-from-video: AgentTimeoutError (Agent execution timed out after 1800.0 seconds)
- mailman: RuntimeError (Harbor setup/bootstrap failure)
- make-doom-for-mips: AgentTimeoutError (Agent execution timed out after 900.0 seconds)
- make-mips-interpreter: AgentTimeoutError (Agent execution timed out after 1800.0 seconds)
- mteb-leaderboard: RuntimeError (mteb-leaderboard Docker pull - image too large)
- path-tracing: AgentTimeoutError (Agent execution timed out after 1800.0 seconds)
- path-tracing-reverse: AgentTimeoutError (Agent execution timed out after 1800.0 seconds)
- regex-chess: AgentTimeoutError (Agent execution timed out after 3600.0 seconds)
- rstan-to-pystan: AgentTimeoutError (Agent execution timed out after 1800.0 seconds)
- train-fasttext: AgentTimeoutError (Agent execution timed out after 3600.0 seconds)
- winning-avg-corewars: RuntimeError (Harbor setup/bootstrap failure)

### qwen3.5-9b-32k
- compile-compcert, dna-insert, extract-moves-from-video, gcode-to-text, mailman, make-doom-for-mips, make-mips-interpreter, mteb-leaderboard, path-tracing, path-tracing-reverse, regex-chess, rstan-to-pystan, winning-avg-corewars

### qwen3.5-9b-65k
- extract-moves-from-video, mailman, make-doom-for-mips, make-mips-interpreter, mteb-leaderboard, path-tracing, path-tracing-reverse, regex-chess, rstan-to-pystan, train-fasttext, winning-avg-corewars

### qwen3.5-9b-131k
- compile-compcert, extract-moves-from-video, mailman, make-doom-for-mips, make-mips-interpreter, mteb-leaderboard, path-tracing, path-tracing-reverse, rstan-to-pystan, train-fasttext, winning-avg-corewars

### qwen3.5-9b-262k
- circuit-fibsqrt, compile-compcert, extract-moves-from-video, mailman, make-doom-for-mips, mteb-leaderboard, path-tracing, path-tracing-reverse, schemelike-metacircular-eval, train-fasttext, winning-avg-corewars

### qwen3.5-flash-16k
- db-wal-recovery, dna-insert, extract-moves-from-video, mailman, make-doom-for-mips, mteb-leaderboard, path-tracing, path-tracing-reverse, rstan-to-pystan, winning-avg-corewars

### qwen3.5-flash-32k
- compile-compcert, db-wal-recovery, dna-insert, extract-moves-from-video, mailman, make-doom-for-mips, make-mips-interpreter, mteb-leaderboard, path-tracing, path-tracing-reverse, train-fasttext, winning-avg-corewars

### qwen3.5-flash-65k
- circuit-fibsqrt, compile-compcert, extract-moves-from-video, mailman, make-doom-for-mips, make-mips-interpreter, mteb-leaderboard, path-tracing, rstan-to-pystan, winning-avg-corewars

### qwen3.5-flash-131k
- compile-compcert, extract-moves-from-video, fix-ocaml-gc, mailman, make-doom-for-mips, mteb-leaderboard, path-tracing, path-tracing-reverse, train-fasttext, winning-avg-corewars

### qwen3.5-flash-262k
- circuit-fibsqrt, compile-compcert, extract-moves-from-video, mailman, make-doom-for-mips, make-mips-interpreter, mteb-leaderboard, path-tracing, path-tracing-reverse, regex-chess, schemelike-metacircular-eval, winning-avg-corewars

### qwen3.5-plus-16k
- compile-compcert, db-wal-recovery, dna-insert, extract-moves-from-video, gcode-to-text, mailman, make-doom-for-mips, make-mips-interpreter, mteb-leaderboard, path-tracing, path-tracing-reverse, regex-chess, rstan-to-pystan, train-fasttext, winning-avg-corewars

### qwen3.5-plus-32k
- circuit-fibsqrt, compile-compcert, extract-moves-from-video, gcode-to-text, mailman, make-doom-for-mips, make-mips-interpreter, mteb-leaderboard, path-tracing, regex-chess, rstan-to-pystan, train-fasttext, winning-avg-corewars

### qwen3.5-plus-65k
- circuit-fibsqrt, compile-compcert, extract-moves-from-video, mailman, make-doom-for-mips, make-mips-interpreter, mteb-leaderboard, path-tracing, path-tracing-reverse, regex-chess, rstan-to-pystan, schemelike-metacircular-eval, train-fasttext, winning-avg-corewars

### qwen3.5-plus-131k
- build-pov-ray: RuntimeError (Harbor setup/bootstrap failure)
- compile-compcert, extract-moves-from-video, fix-ocaml-gc, mailman, make-doom-for-mips, make-mips-interpreter, mteb-leaderboard, path-tracing, regex-chess, schemelike-metacircular-eval: RuntimeError (Harbor setup/bootstrap failure) for schemelike-metacircular-eval
- winning-avg-corewars: RuntimeError (Harbor setup/bootstrap failure)

### qwen3.5-plus-262k
- compile-compcert, extract-moves-from-video, fix-ocaml-gc, mailman, make-doom-for-mips, make-mips-interpreter, mteb-leaderboard, path-tracing, regex-chess, rstan-to-pystan, schemelike-metacircular-eval, train-fasttext, winning-avg-corewars

---

## C) TASKS THAT ERROR ON ALL MODELS (likely infrastructure)

These tasks fail on every one of the 15 model/config runs. They are likely infrastructure issues that cannot be fixed by retrying:

1. **mailman** – RuntimeError (Harbor setup/bootstrap failure – typing-extensions pip install conflict)
2. **make-doom-for-mips** – AgentTimeoutError (900s timeout)
3. **mteb-leaderboard** – RuntimeError (Docker image too large for compute node disk; documented exclusion)
4. **path-tracing** – AgentTimeoutError (1800s timeout)
5. **winning-avg-corewars** – RuntimeError (Harbor setup/bootstrap failure – typing-extensions pip install conflict)

---

## D) TASKS THAT ERROR ON SOME MODELS ONLY (worth retrying)

| Task | Fails on | Models |
|------|----------|--------|
| build-pov-ray | 1/15 | qwen3.5-plus-131k |
| circuit-fibsqrt | 5/15 | qwen3.5-9b-262k, qwen3.5-flash-262k, qwen3.5-flash-65k, qwen3.5-plus-32k, qwen3.5-plus-65k |
| compile-compcert | 12/15 | Most models except 9b-16k, 9b-65k, flash-16k |
| db-wal-recovery | 4/15 | qwen3.5-9b-16k, qwen3.5-flash-16k, qwen3.5-flash-32k, qwen3.5-plus-16k |
| dna-insert | 5/15 | qwen3.5-9b-16k, qwen3.5-9b-32k, qwen3.5-flash-16k, qwen3.5-flash-32k, qwen3.5-plus-16k |
| extract-moves-from-video | 14/15 | All except qwen3.5-flash-32k |
| fix-ocaml-gc | 3/15 | qwen3.5-flash-131k, qwen3.5-plus-131k, qwen3.5-plus-262k |
| gcode-to-text | 3/15 | qwen3.5-9b-32k, qwen3.5-plus-16k, qwen3.5-plus-32k |
| make-mips-interpreter | 11/15 | Most models |
| path-tracing-reverse | 10/15 | 9b variants, flash-131k/16k/262k, plus-16k/65k |
| regex-chess | 10/15 | Various |
| rstan-to-pystan | 10/15 | Various |
| schemelike-metacircular-eval | 5/15 | qwen3.5-9b-262k, qwen3.5-flash-262k, qwen3.5-plus-131k/262k/65k |
| train-fasttext | 10/15 | Various |

---

## Key Findings

1. **Infrastructure RuntimeErrors (3 per run):** `mailman`, `winning-avg-corewars`, and `mteb-leaderboard` fail on every model due to:
   - **mailman / winning-avg-corewars:** Harbor setup bootstrap fails (typing-extensions pip install conflict in Docker)
   - **mteb-leaderboard:** Image too large for compute node disk (documented exclusion)

2. **AgentTimeoutError dominates:** Most failures are timeouts. `extract-moves-from-video` times out on 14/15 models and is a strong retry candidate.

3. **Best performer:** qwen3.5-flash-262k (Mean 0.100, 2 passes) with fewer timeouts than other configs.

4. **Retry priority:** `extract-moves-from-video`, `compile-compcert`, `make-mips-interpreter`, `path-tracing-reverse`, `regex-chess`, `rstan-to-pystan`, `train-fasttext` fail on many models and are good candidates for retries with increased timeout.
