You are responsible for improving this agent implementation. **Your real objective is maximizing performance on the full terminal-bench benchmark (89 tasks), not on the small dev set you can see.** The dev set is only a diagnostic tool. You can use it to find and fix general problems, but your score is measured on the full benchmark which includes many tasks you have never seen.

Current dev score: **{dev_score}** (mean reward across {dev_trials} tasks)
Model context length: **{context_length} tokens**

Workspace root: `{workdir_root}`
Latest dev eval summary: `{eval_summary_path}`
Latest dev eval artifacts root: `{eval_artifacts_path}`

{scoreboard}

{snapshot_index}

## How the workspace agent is deployed

The workspace `agent.py` IS the code that runs during every benchmark. Before each benchmark run, `run_recorded_benchmark.py` automatically deploys your workspace `agent.py` as the active agent, runs the benchmark, then restores the original. You do NOT need to manually copy or "deploy" your code anywhere — editing `agent.py` in the workspace is sufficient. **NEVER modify files outside the workspace.**

## Critical constraint: generalize, do not overfit

Every change you make must be justified by asking: "will this help on an arbitrary unseen task?" The dev set results are diagnostic — use them to understand *categories* of failure, not to write fixes for individual tasks.

**Forbidden patterns (these will hurt held-out performance):**
- Adding task-specific tips, hints, or solutions to the system prompt (e.g. regex patterns, polyglot tricks, chess strategies, specific API replacements).
- Detecting task names or keywords to branch behavior.
- Hard-coding answers, file paths, or domain knowledge for specific tasks.
- Adding example commands that only help one task category.

**If a fix only helps one or two specific tasks, it is the wrong fix.** Find the general principle behind the failure and address that instead.

## Improvement priorities

There are two tiers of improvement. **You must work on both**, but Tier 1 is where the largest gains come from. Do not spend all your time on Tier 2 robustness fixes — those have diminishing returns. The biggest performance jumps come from rethinking the agent's architecture.

### Tier 1 — Agent architecture (primary focus)

These are high-leverage structural changes that can unlock entirely new capabilities. The current baseline is a simple loop: the model receives terminal output, emits a JSON blob with raw keystrokes, and the harness executes them. This architecture has fundamental limitations. Explore alternatives:

1. **Structured tool use instead of raw keystrokes.** Instead of `{{"keystrokes": "ls -la\\n", "duration": 3}}`, define higher-level tools like `run_command(cmd, timeout)`, `read_file(path)`, `write_file(path, content)`, `search(pattern, path)`. The model calls tools via a structured JSON schema. This separates *intent* from terminal mechanics — the model says "write this file" instead of wrestling with heredocs and quoting. The harness translates tool calls into terminal actions.

2. **Explicit planning phase.** Enforce a dedicated planning turn before execution begins. Turn 1 produces only a structured plan (no commands). Subsequent turns execute one plan step at a time, with the full plan visible in every prompt. If a step fails, the agent revises the plan before continuing. This prevents the common failure mode of jumping into commands without understanding the task.

3. **Persistent scratchpad / todo list.** The model maintains a `scratchpad` field in its JSON response that persists across turns. It contains the current plan, completed steps, key observations, and remaining work. This is prepended to every prompt. Unlike chat history (which gets trimmed and compressed), the scratchpad is a compact, structured state that the model actively maintains. This prevents context loss on long tasks.

4. **Subagent delegation.** The main agent can emit a `delegate` action that spawns a fresh model call with a focused sub-instruction and returns the result to the main agent. Useful for: writing complex files (give a subagent the full file spec and let it produce the content), debugging errors (give a subagent the error + context and let it diagnose), or research (let a subagent explore a directory tree without polluting the main context). This is especially valuable given the {context_length}-token context window — long conversations degrade quality.

5. **Multi-phase execution.** Instead of one monolithic turn loop, structure the agent as distinct phases — explore, plan, execute, verify — each with its own prompt template and context window. The explore phase reads the task and environment. The plan phase produces a structured plan. The execute phase runs commands against the plan. The verify phase checks the work against the original requirements. Each phase can have a fresh or summarized context.

6. **Context summarization.** Instead of just trimming old messages or keeping the first N + last M, have the model periodically produce a structured summary of its progress. This summary replaces the full history, giving the model a compact but accurate picture of what has been done and what remains. This is strictly better than mechanical trimming which loses important details.

### Tier 2 — Robustness and reliability (secondary focus)

These are important but have diminishing returns once the basics are covered. Do not stop at Tier 2 — if you have only made robustness fixes, you have not finished.

- **Shell interaction robustness**: Handle pexpect EOF (shell dies), long-running commands, background processes, and command timeouts gracefully.
- **Error recovery**: When commands fail, diagnose the root cause and retry with a different approach — do not loop on the same failing command.
- **Verification before marking complete**: Re-read the original task requirements and verify the work satisfies them before setting `task_complete: true`.
- **Efficient use of turns**: Avoid spending many turns on a single approach that isn't working. Recognize when to pivot.
- **Safe file editing**: When modifying source files, verify the result (e.g. syntax check after editing Python files).
- **Context management**: Keep the original instruction visible. Manage conversation length so the model doesn't run out of context.

## Execution rules

- Your workspace is `{workdir_root}`. ALL file reads and writes MUST stay inside this directory.
- **NEVER read, write, or modify files outside `{workdir_root}`.**
- The only file you should be editing is `agent.py` (and optionally `NOTES.md`) inside the workspace.
- Your workspace `agent.py` is automatically deployed for benchmarks — you do not need to copy it anywhere.
- Use `run_recorded_benchmark.py` (in the workspace) for benchmark runs so artifacts are captured.
- Do not add new dependencies or import modules that are not already available in the workdir/runtime.
- Keep `agent.py` importable in a clean `uv run` environment used by `test_interface_contract`.
- If a change causes import/test failures, revert or fix before finishing.

## Benchmark budget and discipline

The dev benchmark (`run_dev_benchmark.sh`) runs 10 medium-difficulty tasks. Each benchmark run is expensive — treat every run as precious.

**Before running the dev benchmark:**
1. Make sure you have a clear hypothesis about what your change improves.
2. Verify the code is syntactically valid and passes interface tests first.

**After a benchmark run completes:**
1. Read every failed task's logs carefully (`verifier/test-stdout.txt`, `agent/trajectory.json`).
2. Categorize failures by root cause before making more changes.
3. Do not immediately re-run. Investigate first, change second, then run again.

To run the dev benchmark:
```
uv run python run_recorded_benchmark.py --iteration {iteration} --runner harbor/run_dev_benchmark.sh
```

## Task flow

1. Read `README.md` to understand the workdir files and workflow.
2. Read `NOTES.md` and continue documenting observations as you work.
3. Inspect the latest dev benchmark summary/logs from `{eval_summary_path}` and `{eval_artifacts_path}`.
4. **Categorize failures into three buckets** (not by task name):
   - **Agent design limitation**: The architecture prevents the model from succeeding. Examples: "no way to maintain a plan across turns", "context gets too long and model loses track", "no verification step before marking complete", "model wastes turns on file-writing mechanics instead of problem-solving".
   - **Robustness bug**: The harness crashes or misbehaves. Examples: "shell EOF crash", "encoding error on binary output", "API error not retried".
   - **Model ability**: The model lacks the knowledge or reasoning to solve the task regardless of architecture. Examples: "model doesn't know chess", "model can't crack a hash".
5. Prioritize **agent design limitations** first — these are where architectural changes yield the biggest gains. Then address robustness bugs. Only classify a failure as "model ability" after you have tried at least one architectural change that could help.
6. Implement focused improvements in `agent.py`.
7. Validate interface compatibility before benchmarking:
   `uv run python -m unittest test_interface_contract`
8. Run the dev benchmark to validate your changes:
   `uv run python run_recorded_benchmark.py --iteration {iteration} --runner harbor/run_dev_benchmark.sh`
9. **Investigate results thoroughly** before making more changes.
10. Compare new results against prior run(s). Note whether performance improved, regressed, or stayed flat.
11. Update `NOTES.md` with:
    - failure categories observed (design limitation / robustness bug / model ability)
    - the architectural or robustness hypothesis
    - changes made
    - outcome after re-eval
    - whether you expect the change to help on unseen tasks and why

## Stop condition

Stop when ALL of the following are true:

1. **You have tried at least two fundamentally different agent architectures.** Examples of "fundamentally different": raw keystroke loop vs. tool-use agent, single-agent vs. multi-agent with delegation, flat chat history vs. structured scratchpad/memory, monolithic loop vs. multi-phase (plan/execute/verify). Tweaking the system prompt or adding error handling within the same architecture does NOT count.
2. **The best-performing architecture has been hardened for robustness** (shell crashes, API errors, encoding issues, etc. are handled).
3. **Remaining failures are attributable to the model's raw ability** (reasoning, knowledge, skill) rather than agent design or infrastructure — and you have documented why no architectural change would help.

Before stopping, document in `NOTES.md`:
- Which architectures you tried and their comparative results.
- Which failures you believe are model-ability-limited and why no design change would help.
- What robustness improvements you made and their impact.
- Confirmation that interface tests pass.

A change that improves the dev set by adding task-specific hacks is **not** progress — it will regress on the held-out eval.

## Harbor artifact structure

Each trial directory under the job dir has:
- `result.json` -- trial result with reward, duration, errors
- `verifier/reward.txt` -- numeric reward (0.0 or 1.0)
- `verifier/test-stdout.txt` -- test script output (why it passed/failed)
- `agent/trajectory.json` -- full agent trajectory (tool calls + observations)

To quickly see which tasks failed: look for trials where `reward.txt` contains `0`.
