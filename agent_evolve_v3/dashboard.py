# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false, reportCallIssue=false

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from agent_evolve_v3.state.planner_context import (
    classify_state_status,
    parent_iteration_for_state,
    plan_summary,
)
from agent_evolve_v3.state.types import AgentState

app = FastAPI()

OUTPUTS_ROOT = Path(__file__).resolve().parent / "outputs"


# ---------------------------------------------------------------------------
# Trial outcome classification
# ---------------------------------------------------------------------------

OUTCOME_PASS = "pass"
OUTCOME_FAILURE = "failure"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_MAX_ITERS = "max_iters"
OUTCOME_OTHER_ERROR = "other_error"
OUTCOME_PENDING = "pending"


def _classify_trial(trial: dict[str, Any]) -> str:
    exc = trial.get("exception_info")
    if exc and isinstance(exc, dict):
        exc_type = exc.get("exception_type", "")
        exc_msg = exc.get("exception_message", "")
        if exc_type == "AgentTimeoutError":
            return OUTCOME_TIMEOUT
        if exc_type == "RuntimeError":
            if "exit_code=1" in str(exc_msg):
                return OUTCOME_MAX_ITERS
            return OUTCOME_OTHER_ERROR

        return OUTCOME_OTHER_ERROR

    verifier = trial.get("verifier_result")
    if verifier and isinstance(verifier, dict):
        rewards = verifier.get("rewards", {})
        if isinstance(rewards, dict):
            reward = rewards.get("reward")
            if reward == 1.0:
                return OUTCOME_PASS
            if reward is not None:
                return OUTCOME_FAILURE

    return OUTCOME_PENDING


def _task_name_from_trial_name(trial_name: str) -> str:
    parts = trial_name.rsplit("__", 1)
    return parts[0] if len(parts) == 2 else trial_name


# ---------------------------------------------------------------------------
# Harbor job dir scanning
# ---------------------------------------------------------------------------


def _find_harbor_job_dir(*, artifacts_dir: Path) -> Path | None:
    official = artifacts_dir / "official_benchmark" / "harbor_jobs"
    if not official.is_dir():
        return None
    subdirs = sorted(official.iterdir())
    return subdirs[-1] if subdirs else None


def _load_aggregate_result(*, harbor_job_dir: Path) -> dict[str, Any] | None:
    result_path = harbor_job_dir / "result.json"
    if not result_path.exists():
        return None
    try:
        data: object = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _load_trial_results(*, harbor_job_dir: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not harbor_job_dir.is_dir():
        return results
    for child in harbor_job_dir.iterdir():
        if not child.is_dir() or "__" not in child.name:
            continue
        result_path = child / "result.json"
        if not result_path.exists():
            continue
        try:
            data: object = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict):
            results[child.name] = data

    return results


# ---------------------------------------------------------------------------
# Iteration-level metrics
# ---------------------------------------------------------------------------


def _compute_iteration_metrics(
    *, trial_results: dict[str, dict[str, Any]], benchmark_tasks: list[str]
) -> dict[str, Any]:
    task_outcomes: dict[str, str] = {}
    for trial_name, trial_data in trial_results.items():
        task = _task_name_from_trial_name(trial_name)
        task_outcomes[task] = _classify_trial(trial_data)

    total = len(benchmark_tasks) if benchmark_tasks else max(len(task_outcomes), 1)
    counts = {
        OUTCOME_PASS: 0,
        OUTCOME_FAILURE: 0,
        OUTCOME_TIMEOUT: 0,
        OUTCOME_MAX_ITERS: 0,
        OUTCOME_OTHER_ERROR: 0,
    }
    for outcome in task_outcomes.values():
        if outcome in counts:
            counts[outcome] += 1

    return {
        "completion_rate": counts[OUTCOME_PASS] / total,
        "failure_rate": counts[OUTCOME_FAILURE] / total,
        "timeout_rate": counts[OUTCOME_TIMEOUT] / total,
        "max_iters_rate": counts[OUTCOME_MAX_ITERS] / total,
        "other_error_rate": counts[OUTCOME_OTHER_ERROR] / total,
        "timeout_or_turn_limit_rate": (
            counts[OUTCOME_TIMEOUT] + counts[OUTCOME_MAX_ITERS]
        )
        / total,
        "tasks": task_outcomes,
    }


# ---------------------------------------------------------------------------
# Stage detection and timing
# ---------------------------------------------------------------------------

_STAGE_FILES = [
    ("planner_step.json", "planning"),
    ("implementation_step.json", "implementing"),
    ("validation_step.json", "validating"),
    ("benchmark_step.json", "benchmarking"),
]


def _detect_current_stage(*, artifacts_dir: Path, is_bootstrap: bool = False) -> str:
    if not artifacts_dir.exists():
        return "waiting"

    if is_bootstrap:
        harbor_dir = _find_harbor_job_dir(artifacts_dir=artifacts_dir)
        if harbor_dir is not None:
            agg = _load_aggregate_result(harbor_job_dir=harbor_dir)
            if agg is not None and agg.get("finished_at") is not None:
                return "completed"
            return "benchmarking"
        return "waiting"

    last_completed = "waiting"
    for filename, stage_name in _STAGE_FILES:
        if (artifacts_dir / filename).exists():
            last_completed = stage_name
        else:
            return stage_name

    return "completed" if last_completed == "benchmarking" else last_completed


def _get_benchmark_progress(*, artifacts_dir: Path) -> dict[str, int] | None:
    harbor_dir = _find_harbor_job_dir(artifacts_dir=artifacts_dir)
    if harbor_dir is None:
        return None
    agg = _load_aggregate_result(harbor_job_dir=harbor_dir)
    if agg is None:
        return None
    n_total = agg.get("n_total_trials", 0)
    n_done = agg.get("stats", {}).get("n_trials", 0)
    return {"completed": n_done, "total": n_total}


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------


def _load_manifest(*, run_dir: Path) -> dict[str, Any] | None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        data: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _load_states(*, run_dir: Path) -> list[AgentState]:
    states_dir = run_dir / "states"
    if not states_dir.is_dir():
        return []
    state_files = sorted(states_dir.glob("iteration-*.json"))
    states: list[AgentState] = []
    for sf in state_files:
        try:
            states.append(AgentState.load(path=sf))
        except (json.JSONDecodeError, ValueError, KeyError, OSError):
            continue

    return states


def _load_single_run(*, run_dir: Path) -> dict[str, Any] | None:
    manifest = _load_manifest(run_dir=run_dir)
    if manifest is None:
        return None

    states = _load_states(run_dir=run_dir)
    benchmark_tasks: list[str] = manifest.get("benchmark_tasks", [])
    max_iterations: int = manifest.get("iterations", 25)

    iterations: list[dict[str, Any]] = []
    best_reward: float | None = None

    for state in states:
        status = classify_state_status(state=state, run_root=run_dir)
        parent_iter = parent_iteration_for_state(state=state)
        ps = plan_summary(plan=state.plan)

        iter_data: dict[str, Any] = {
            "iteration": state.iteration,
            "status": status,
            "created_at": state.created_at_utc,
            "parent_iteration": parent_iter,
            "plan_summary": ps,
            "reward_mean": None,
            "completion_rate": None,
            "failure_rate": None,
            "timeout_rate": None,
            "max_iters_rate": None,
            "other_error_rate": None,
            "timeout_or_turn_limit_rate": None,
            "tasks": {},
        }

        artifacts_dir = run_dir / "artifacts" / f"iteration-{state.iteration:04d}"
        harbor_dir = _find_harbor_job_dir(artifacts_dir=artifacts_dir)

        if harbor_dir is not None:
            trial_results = _load_trial_results(harbor_job_dir=harbor_dir)
            if trial_results:
                metrics = _compute_iteration_metrics(
                    trial_results=trial_results,
                    benchmark_tasks=benchmark_tasks,
                )
                iter_data.update(metrics)

        if state.result and state.result.reward_mean is not None:
            iter_data["reward_mean"] = state.result.reward_mean
            if best_reward is None or state.result.reward_mean > best_reward:
                best_reward = state.result.reward_mean

        iterations.append(iter_data)

    current_iteration = states[-1].iteration if states else 0
    latest_artifacts = run_dir / "artifacts" / f"iteration-{current_iteration:04d}"
    current_stage = _detect_current_stage(
        artifacts_dir=latest_artifacts,
        is_bootstrap=(current_iteration == 0),
    )

    current_artifacts = latest_artifacts
    if current_stage == "completed" and current_iteration < max_iterations:
        next_artifacts = (
            run_dir / "artifacts" / f"iteration-{current_iteration + 1:04d}"
        )
        if next_artifacts.exists():
            current_stage = _detect_current_stage(artifacts_dir=next_artifacts)
            current_iteration = current_iteration + 1
            current_artifacts = next_artifacts
        else:
            current_stage = "planning"

    benchmark_progress = None
    if current_stage == "benchmarking":
        benchmark_progress = _get_benchmark_progress(
            artifacts_dir=current_artifacts,
        )

    task_aggregate = _build_task_aggregate(
        iterations=iterations,
        benchmark_tasks=benchmark_tasks,
    )

    return {
        "run_dir": run_dir.name,
        "name": manifest.get("name", run_dir.name),
        "model_key": manifest.get("model_key", ""),
        "cursor_model": manifest.get("cursor_model", ""),
        "baseline": manifest.get("baseline", ""),
        "max_iterations": max_iterations,
        "current_iteration": current_iteration,
        "current_stage": current_stage,
        "benchmark_progress": benchmark_progress,
        "best_reward": best_reward,
        "iterations": iterations,
        "task_aggregate": task_aggregate,
    }


def _build_task_aggregate(
    *,
    iterations: list[dict[str, Any]],
    benchmark_tasks: list[str],
) -> dict[str, dict[str, Any]]:
    task_outcomes: dict[str, list[str]] = {t: [] for t in benchmark_tasks}

    for it in iterations:
        tasks = it.get("tasks", {})
        if not isinstance(tasks, dict):
            continue
        for task_name, outcome in tasks.items():
            if task_name not in task_outcomes:
                task_outcomes[task_name] = []
            task_outcomes[task_name].append(outcome)

    aggregate: dict[str, dict[str, Any]] = {}
    for task_name, outcomes in task_outcomes.items():
        n = len(outcomes) if outcomes else 1
        aggregate[task_name] = {
            "completion_rate": outcomes.count(OUTCOME_PASS) / n if outcomes else 0,
            "failure_rate": outcomes.count(OUTCOME_FAILURE) / n if outcomes else 0,
            "timeout_rate": outcomes.count(OUTCOME_TIMEOUT) / n if outcomes else 0,
            "max_iters_rate": outcomes.count(OUTCOME_MAX_ITERS) / n if outcomes else 0,
            "other_error_rate": (
                outcomes.count(OUTCOME_OTHER_ERROR) / n if outcomes else 0
            ),
            "timeout_or_turn_limit_rate": (
                (outcomes.count(OUTCOME_TIMEOUT) + outcomes.count(OUTCOME_MAX_ITERS))
                / n
                if outcomes
                else 0
            ),
            "n_iterations": len(outcomes),
        }

    return aggregate


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/api/runs")
def api_runs() -> JSONResponse:
    runs: list[dict[str, Any]] = []
    if OUTPUTS_ROOT.is_dir():
        for child in sorted(OUTPUTS_ROOT.iterdir()):
            if not child.is_dir():
                continue
            run_data = _load_single_run(run_dir=child)
            if run_data is not None:
                runs.append(run_data)

    return JSONResponse(content=runs)


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(content=_DASHBOARD_HTML)


# ---------------------------------------------------------------------------
# HTML / JS / CSS
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Evolve V3 Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text-muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922; --orange: #db6d28;
    --purple: #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
    padding: 24px; max-width: 1400px; margin: 0 auto;
  }
  h1 { font-size: 24px; margin-bottom: 4px; }
  .header { display: flex; justify-content: space-between; align-items: baseline;
            margin-bottom: 24px; border-bottom: 1px solid var(--border); padding-bottom: 16px; }
  .header-right { font-size: 13px; color: var(--text-muted); }
  .run-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 20px; margin-bottom: 24px;
  }
  .run-header { display: flex; justify-content: space-between; align-items: center;
                margin-bottom: 16px; flex-wrap: wrap; gap: 8px; }
  .run-name { font-size: 18px; font-weight: 600; }
  .run-meta { font-size: 13px; color: var(--text-muted); }
  .run-meta span { margin-right: 16px; }
  .progress-section { margin-bottom: 20px; }
  .progress-bar-container {
    background: var(--border); border-radius: 4px; height: 8px;
    margin: 8px 0; overflow: hidden;
  }
  .progress-bar-fill {
    height: 100%; background: var(--accent); border-radius: 4px;
    transition: width 0.3s ease;
  }
  .progress-text { font-size: 13px; color: var(--text-muted); display: flex;
                   justify-content: space-between; }
  .stage-badge {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 12px; font-weight: 600; text-transform: uppercase;
  }
  .stage-planning { background: #1f2d5c; color: var(--accent); }
  .stage-implementing { background: #2a1f3d; color: var(--purple); }
  .stage-validating { background: #2d2a0f; color: var(--yellow); }
  .stage-benchmarking { background: #2a1a0f; color: var(--orange); }
  .stage-completed { background: #1a2f1a; color: var(--green); }
  .stage-waiting { background: #1a1a1a; color: var(--text-muted); }
  .best-reward { font-size: 28px; font-weight: 700; color: var(--green); }
  .best-reward-label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; }
  .stats-row { display: flex; gap: 24px; margin-bottom: 16px; flex-wrap: wrap; }
  .stat-box { text-align: center; }
  .stat-value { font-size: 20px; font-weight: 600; }
  .stat-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; }
  .section-title {
    font-size: 14px; font-weight: 600; text-transform: uppercase;
    color: var(--text-muted); margin: 20px 0 12px; letter-spacing: 0.5px;
  }
  .chart-controls { margin-bottom: 12px; }
  .chart-controls select {
    background: var(--bg); color: var(--text); border: 1px solid var(--border);
    padding: 6px 12px; border-radius: 6px; font-size: 13px; cursor: pointer;
  }
  .chart-container { position: relative; height: 300px; margin-bottom: 20px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--text-muted); font-weight: 600; font-size: 11px;
       text-transform: uppercase; letter-spacing: 0.5px; position: sticky; top: 0;
       background: var(--surface); }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .table-scroll { max-height: 400px; overflow-y: auto; border: 1px solid var(--border);
                  border-radius: 6px; }
  .cell-pass { color: var(--green); }
  .cell-fail { color: var(--red); }
  .cell-warn { color: var(--yellow); }
  .cell-muted { color: var(--text-muted); }
  .empty-state { text-align: center; padding: 60px 20px; color: var(--text-muted); }
  .benchmark-progress { font-size: 13px; color: var(--text-muted); margin-top: 4px; }
  .tabs { display: flex; gap: 0; margin-bottom: 0; border-bottom: 1px solid var(--border); }
  .tab {
    padding: 8px 16px; font-size: 13px; cursor: pointer; border: none;
    background: none; color: var(--text-muted); border-bottom: 2px solid transparent;
    transition: all 0.15s;
  }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>Agent Evolve V3 Dashboard</h1>
  </div>
  <div class="header-right">
    <span id="last-updated">Loading...</span>
  </div>
</div>
<div id="runs-container">
  <div class="empty-state">Loading runs...</div>
</div>

<script>
const METRIC_LABELS = {
  completion_rate: 'Completion Rate',
  failure_rate: 'Failure Rate',
  timeout_or_turn_limit_rate: 'Timeout / Turn Limit Rate',
  timeout_rate: 'Timeout Rate',
  max_iters_rate: 'Max Iters Hit Rate',
  other_error_rate: 'Other Runtime Error Rate',
};
const METRIC_COLORS = {
  completion_rate: '#3fb950',
  failure_rate: '#f85149',
  timeout_or_turn_limit_rate: '#d29922',
  timeout_rate: '#db6d28',
  max_iters_rate: '#bc8cff',
  other_error_rate: '#8b949e',
};

let charts = {};
let currentMetrics = {};

function destroyAllCharts() {
  for (const [key, chart] of Object.entries(charts)) {
    chart.destroy();
  }
  charts = {};
}

function stageClass(stage) {
  const map = {planning:'stage-planning', implementing:'stage-implementing',
    validating:'stage-validating', benchmarking:'stage-benchmarking',
    completed:'stage-completed'};
  return map[stage] || 'stage-waiting';
}

function pct(v) {
  if (v == null) return '-';
  return (v * 100).toFixed(1) + '%';
}

function cellClass(val, isGood) {
  if (val == null) return 'cell-muted';
  if (isGood) return val >= 0.7 ? 'cell-pass' : val >= 0.3 ? 'cell-warn' : 'cell-fail';
  return val <= 0.05 ? 'cell-pass' : val <= 0.2 ? 'cell-warn' : 'cell-fail';
}

function renderRun(run, idx) {
  const id = `run-${idx}`;
  const pctDone = run.max_iterations > 0
    ? ((run.iterations.length - 1) / run.max_iterations * 100).toFixed(0)
    : 0;
  const completedIters = run.iterations.filter(i => i.status === 'completed').length;

  let benchHtml = '';
  if (run.current_stage === 'benchmarking' && run.benchmark_progress) {
    const bp = run.benchmark_progress;
    benchHtml = `<div class="benchmark-progress">${bp.completed}/${bp.total} tasks completed</div>`;
  }

  let iterTableRows = '';
  for (const it of run.iterations) {
    const rw = it.reward_mean != null ? it.reward_mean.toFixed(3) : '-';
    const parent = it.parent_iteration != null ? it.parent_iteration : 'root';
    iterTableRows += `<tr>
      <td class="num">${it.iteration}</td>
      <td><span class="stage-badge ${stageClass(it.status)}">${it.status}</span></td>
      <td class="num">${rw}</td>
      <td class="num ${cellClass(it.completion_rate, true)}">${pct(it.completion_rate)}</td>
      <td class="num ${cellClass(it.failure_rate, false)}">${pct(it.failure_rate)}</td>
      <td class="num ${cellClass(it.timeout_or_turn_limit_rate, false)}">${pct(it.timeout_or_turn_limit_rate)}</td>
      <td class="num ${cellClass(it.other_error_rate, false)}">${pct(it.other_error_rate)}</td>
      <td class="num">${parent}</td>
      <td title="${(it.plan_summary||'').replace(/"/g,'&quot;')}">${truncate(it.plan_summary || '-', 60)}</td>
    </tr>`;
  }

  const tasks = Object.entries(run.task_aggregate || {}).sort((a,b) => a[0].localeCompare(b[0]));
  let taskRows = '';
  for (const [name, m] of tasks) {
    taskRows += `<tr>
      <td>${name}</td>
      <td class="num ${cellClass(m.completion_rate, true)}">${pct(m.completion_rate)}</td>
      <td class="num ${cellClass(m.failure_rate, false)}">${pct(m.failure_rate)}</td>
      <td class="num ${cellClass(m.timeout_or_turn_limit_rate, false)}">${pct(m.timeout_or_turn_limit_rate)}</td>
      <td class="num ${cellClass(m.timeout_rate, false)}">${pct(m.timeout_rate)}</td>
      <td class="num ${cellClass(m.max_iters_rate, false)}">${pct(m.max_iters_rate)}</td>
      <td class="num ${cellClass(m.other_error_rate, false)}">${pct(m.other_error_rate)}</td>
      <td class="num">${m.n_iterations}</td>
    </tr>`;
  }

  return `
  <div class="run-card" id="${id}">
    <div class="run-header">
      <div>
        <div class="run-name">${run.name}</div>
        <div class="run-meta">
          <span>Model: ${run.model_key}</span>
          <span>Planner: ${run.cursor_model}</span>
          <span>Baseline: ${run.baseline}</span>
        </div>
      </div>
      <div style="text-align:right">
        <div class="best-reward-label">Best Reward</div>
        <div class="best-reward">${run.best_reward != null ? run.best_reward.toFixed(3) : '-'}</div>
      </div>
    </div>

    <div class="progress-section">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">
        <span class="stage-badge ${stageClass(run.current_stage)}">${run.current_stage}</span>
        ${benchHtml}
      </div>
      <div class="progress-bar-container">
        <div class="progress-bar-fill" style="width:${pctDone}%"></div>
      </div>
      <div class="progress-text">
        <span>Iteration ${run.iterations.length - 1} of ${run.max_iterations}</span>
        <span>${completedIters} completed</span>
      </div>
    </div>

    <div class="tabs">
      <button class="tab active" onclick="switchTab('${id}','chart')">Chart</button>
      <button class="tab" onclick="switchTab('${id}','iterations')">Iterations</button>
      <button class="tab" onclick="switchTab('${id}','tasks')">Tasks</button>
    </div>

    <div class="tab-content active" data-tab="chart" data-run="${id}">
      <div class="section-title">Metrics Over Iterations</div>
      <div class="chart-controls">
        <select onchange="changeMetric('${id}', this.value)" id="metric-select-${id}">
          ${Object.entries(METRIC_LABELS).map(([k,v]) =>
            `<option value="${k}" ${k === 'completion_rate' ? 'selected' : ''}>${v}</option>`
          ).join('')}
        </select>
      </div>
      <div class="chart-container">
        <canvas id="chart-${id}"></canvas>
      </div>
    </div>

    <div class="tab-content" data-tab="iterations" data-run="${id}">
      <div class="section-title">Iteration History</div>
      <div class="table-scroll">
        <table>
          <thead><tr>
            <th>Iter</th><th>Status</th><th>Reward</th><th>Comp%</th>
            <th>Fail%</th><th>TO/TL%</th><th>Err%</th><th>Parent</th><th>Plan</th>
          </tr></thead>
          <tbody>${iterTableRows}</tbody>
        </table>
      </div>
    </div>

    <div class="tab-content" data-tab="tasks" data-run="${id}">
      <div class="section-title">Per-Task Aggregate</div>
      <div class="table-scroll">
        <table>
          <thead><tr>
            <th>Task</th><th>Comp%</th><th>Fail%</th><th>TO/TL%</th>
            <th>TO%</th><th>MaxIter%</th><th>Err%</th><th>N</th>
          </tr></thead>
          <tbody>${taskRows}</tbody>
        </table>
      </div>
    </div>
  </div>`;
}

function truncate(s, n) {
  return s.length > n ? s.slice(0, n) + '...' : s;
}

function switchTab(runId, tabName) {
  const card = document.getElementById(runId);
  card.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  card.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  const tabs = card.querySelectorAll('.tab');
  const contents = card.querySelectorAll('.tab-content');
  for (let i = 0; i < contents.length; i++) {
    if (contents[i].dataset.tab === tabName) {
      contents[i].classList.add('active');
      tabs[i].classList.add('active');
    }
  }
}

function buildChart(runId, run, metric) {
  const canvas = document.getElementById(`chart-${runId}`);
  if (!canvas) return;

  if (charts[runId]) {
    charts[runId].destroy();
    delete charts[runId];
  }

  const iters = run.iterations.filter(i => i.status === 'completed' && i[metric] != null);

  charts[runId] = new Chart(canvas, {
    type: 'scatter',
    data: {
      datasets: [{
        label: METRIC_LABELS[metric],
        data: iters.map(i => ({x: i.iteration, y: i[metric]})),
        backgroundColor: METRIC_COLORS[metric] + '99',
        borderColor: METRIC_COLORS[metric],
        pointRadius: 6,
        pointHoverRadius: 8,
        showLine: true,
        tension: 0.1,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: {display: false},
        tooltip: {
          callbacks: {
            label: (pt) => `Iter ${pt.raw.x}: ${(pt.raw.y * 100).toFixed(1)}%`
          }
        }
      },
      scales: {
        x: {
          title: {display: true, text: 'Iteration', color: '#8b949e'},
          ticks: {color: '#8b949e', stepSize: 1},
          grid: {color: '#30363d44'},
        },
        y: {
          title: {display: true, text: METRIC_LABELS[metric], color: '#8b949e'},
          ticks: {color: '#8b949e', callback: v => (v*100).toFixed(0)+'%'},
          grid: {color: '#30363d44'},
          min: 0, max: 1,
        }
      }
    }
  });
}

function changeMetric(runId, metric) {
  currentMetrics[runId] = metric;
  if (lastData) {
    const idx = parseInt(runId.replace('run-',''));
    const run = lastData[idx];
    if (run) buildChart(runId, run, metric);
  }
}

let lastData = null;

async function fetchRuns() {
  try {
    const resp = await fetch('/api/runs');
    const runs = await resp.json();
    lastData = runs;
    const container = document.getElementById('runs-container');

    if (!runs.length) {
      container.innerHTML = '<div class="empty-state">No runs found in outputs/</div>';
      return;
    }

    // Preserve active tabs and metrics before destroying DOM
    const activeTabs = {};
    document.querySelectorAll('.run-card').forEach(card => {
      const activeContent = card.querySelector('.tab-content.active');
      if (activeContent) activeTabs[card.id] = activeContent.dataset.tab;
    });

    destroyAllCharts();
    container.innerHTML = runs.map((r, i) => renderRun(r, i)).join('');

    // Restore tabs
    for (const [cardId, tab] of Object.entries(activeTabs)) {
      if (document.getElementById(cardId)) switchTab(cardId, tab);
    }

    // Build/update charts
    runs.forEach((run, i) => {
      const id = `run-${i}`;
      const metric = currentMetrics[id] || 'completion_rate';
      const select = document.getElementById(`metric-select-${id}`);
      if (select) select.value = metric;
      buildChart(id, run, metric);
    });

    document.getElementById('last-updated').textContent =
      'Updated ' + new Date().toLocaleTimeString();
  } catch (e) {
    console.error('Fetch error:', e);
  }
}

fetchRuns();
setInterval(fetchRuns, 5000);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app="agent_evolve_v3.dashboard:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
