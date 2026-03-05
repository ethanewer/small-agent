You are responsible for improving this agent implementation to perform better on terminal benchmark tasks.

Workspace root: `{run_root}`
Workdir root: `{workdir_root}`
Latest eval summary: `{eval_summary_path}`
Latest eval artifacts root: `{eval_artifacts_path}`

Execution rules:
- Treat all relative paths in this prompt as relative to workspace root (`{run_root}`).
- Keep code edits scoped to `agent_evolve/` unless explicitly required.
- Do not read or write files outside this workspace run directory.
- Use `agent_evolve/run_recorded_benchmark.py` for benchmark runs so artifacts are captured.

Task flow:
1. Read `agent_evolve/README.md` to understand the workdir files and workflow.
2. Read `agent_evolve/NOTES.md` and continue documenting observations as you work.
3. Inspect the latest benchmark summary/logs from `{eval_summary_path}` and `{eval_artifacts_path}`.
4. Identify 1-2 concrete weaknesses and choose a focused change with expected impact.
5. Implement focused improvements in `agent_evolve/agent.py` or related existing workdir files.
6. Re-run benchmark (at most twice in this iteration):
   `uv run python agent_evolve/run_recorded_benchmark.py --iteration {iteration} --runner cli/harbor/run_small.sh`
7. Validate interface compatibility:
   `uv run python -m unittest agent_evolve.test_interface_contract`
8. Compare new results against prior run(s) and note whether performance improved, regressed, or stayed flat.
9. Update `agent_evolve/NOTES.md` with:
   - key observations from eval/logs
   - hypotheses
   - changes made
   - outcome after re-eval

Success and stop conditions:
- Stop after no more than 2 benchmark reruns for this iteration.
- Stop early once tests pass and you have either:
  - a measurable improvement, or
  - a documented no-improvement/regression result with a clear next-step hypothesis.
