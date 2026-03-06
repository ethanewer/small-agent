You are responsible for improving this agent implementation. **Your real objective is maximizing performance on the full terminal-bench benchmark (89 tasks), not on the small dev set you can see.** The dev set is only a diagnostic tool. You can use it to find and fix general problems with the harness, but your score is measured on the full benchmark which includes many tasks you have never seen.

Current dev score: **{dev_score}** (mean reward across {dev_trials} tasks)
Model context length: **{context_length} tokens**

Workspace root: `{run_root}`
Workdir root: `{workdir_root}`
Latest dev eval summary: `{eval_summary_path}`
Latest dev eval artifacts root: `{eval_artifacts_path}`

## Critical constraint: generalize, do not overfit

Every change you make must be justified by asking: "will this help on an arbitrary unseen task?" The dev set results are diagnostic — use them to understand *categories* of failure, not to write fixes for individual tasks.

**Forbidden patterns (these will hurt held-out performance):**
- Adding task-specific tips, hints, or solutions to the system prompt (e.g. regex patterns, polyglot tricks, chess strategies, specific API replacements).
- Detecting task names or keywords to branch behavior.
- Hard-coding answers, file paths, or domain knowledge for specific tasks.
- Adding example commands that only help one task category.

**If a fix only helps one or two specific tasks, it is the wrong fix.** Find the general principle behind the failure and address that instead.

## What good improvements look like

Focus on **general agent capabilities** that help across all tasks:

1. **Planning before acting**: The agent should read the task description carefully, form a plan, and reason about what success looks like before executing commands.
2. **Verification before marking complete**: The agent should re-read the original task requirements and verify its work actually satisfies them before setting `task_complete: true`. Check outputs, re-run tests, inspect files.
3. **Error recovery**: When commands fail, builds break, or syntax errors occur, the agent should diagnose the root cause and retry — not loop on the same failing approach.
4. **Context management**: Keep the original instruction visible in follow-up prompts so the agent doesn't lose track of what it's solving. Manage conversation length so the model doesn't run out of context.
5. **Shell interaction robustness**: Handle pexpect EOF (shell dies), long-running commands, background processes, and command timeouts gracefully.
6. **Efficient use of turns**: Avoid spending many turns on a single approach that isn't working. Recognize when to pivot.
7. **Safe file editing**: When the agent modifies source files, it should verify the result (e.g. syntax check after editing Python files).
8. **Task decomposition**: The agent could maintain an internal todo list — break the task into discrete steps, track which are done, and use it to stay on track across turns.
9. **Subagents for context isolation**: Consider spawning subagents (separate model calls) for self-contained subtasks so the main agent's context stays uncluttered. This is especially important given the model's context window ({context_length} tokens) — long conversations degrade quality.
10. **Context-length awareness**: The agent should be aware of how much context it has consumed and proactively summarize or trim conversation history before hitting the limit. Losing context mid-task is a common failure mode.

## Execution rules

- Treat all relative paths in this prompt as relative to workspace root (`{run_root}`).
- Keep code edits scoped to `agent_evolve/` unless explicitly required.
- Do not read or write files outside this workspace run directory.
- Use `agent_evolve/run_recorded_benchmark.py` for benchmark runs so artifacts are captured.
- Do not add new dependencies or import modules that are not already available in the workdir/runtime.
- Keep `agent_evolve/agent.py` importable in a clean `uv run` environment used by `agent_evolve.test_interface_contract`.
- If a change causes import/test failures, revert or fix before finishing.

## Benchmark budget and discipline

The dev benchmark (`run_debug.sh`) has 4 splits of 5 tasks each. **You must run only one split at a time.** Each benchmark run is expensive — treat every run as precious.

**Before running a benchmark split:**
1. Make sure you have a clear hypothesis about what your change improves.
2. Verify the code is syntactically valid and passes interface tests first.

**After a benchmark run completes:**
1. Read every failed task's logs carefully (`verifier/test-stdout.txt`, `agent/trajectory.json`).
2. Categorize failures by root cause before making more changes.
3. Do not immediately re-run. Investigate first, change second, then run again.

**Do not run multiple splits back-to-back without investigating results in between.** The goal is to extract maximum signal from each 5-task run, not to burn through all 4 splits quickly.

To run a single split:
```
uv run python agent_evolve/run_recorded_benchmark.py --iteration {iteration} --runner harbor/run_debug.sh --runner-args "--split <N>"
```
where `<N>` is 1, 2, 3, or 4.

## Task flow

1. Read `agent_evolve/README.md` to understand the workdir files and workflow.
2. Read `agent_evolve/NOTES.md` and continue documenting observations as you work.
3. Inspect the latest dev benchmark summary/logs from `{eval_summary_path}` and `{eval_artifacts_path}`.
4. **Categorize failures by root cause** (not by task name). Examples of good categories:
   - "Agent marks complete without verifying output"
   - "Agent loses track of the original instruction after many turns"
   - "Agent doesn't recover from build/syntax errors"
   - "Agent doesn't plan before acting"
   - "Shell interaction crashes on EOF"
5. Choose 1-2 **general** improvements that address the most common failure categories.
6. Implement focused improvements in `agent_evolve/agent.py` or related existing workdir files.
7. Validate interface compatibility before benchmarking:
   `uv run python -m unittest agent_evolve.test_interface_contract`
8. Run **one** debug split to validate your changes:
   `uv run python agent_evolve/run_recorded_benchmark.py --iteration {iteration} --runner harbor/run_debug.sh --runner-args "--split <N>"`
9. **Investigate results thoroughly** before running another split or making more changes.
10. Compare new results against prior run(s). Note whether performance improved, regressed, or stayed flat.
11. Update `agent_evolve/NOTES.md` with:
    - failure categories observed (not task-specific fixes)
    - the general improvement hypothesis
    - changes made
    - outcome after re-eval
    - whether you expect the change to help on unseen tasks and why

## Stop condition

Stop when **every remaining failure is attributable to the model's raw ability** (reasoning, knowledge, skill) rather than a harness/infrastructure problem. In other words, keep iterating as long as there are failures you can fix by improving the harness — better prompting structure, shell robustness, error recovery, context management, verification logic, etc. Once the only failures left are ones where the model simply wasn't smart enough to solve the task (and no harness change would help), you are done.

Before stopping, document in `agent_evolve/NOTES.md`:
- Which failures you believe are model-ability-limited and why.
- What harness improvements you made and their impact.
- Confirmation that interface tests pass.

A change that improves the dev set by adding task-specific hacks is **not** progress — it will regress on the held-out eval.

## Harbor artifact structure

Each trial directory under the job dir has:
- `result.json` -- trial result with reward, duration, errors
- `verifier/reward.txt` -- numeric reward (0.0 or 1.0)
- `verifier/test-stdout.txt` -- test script output (why it passed/failed)
- `agent/trajectory.json` -- full agent trajectory (tool calls + observations)

To quickly see which tasks failed: look for trials where `reward.txt` contains `0`.
