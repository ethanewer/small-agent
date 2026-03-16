# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false, reportUnknownParameterType=false, reportMissingParameterType=false

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
import os
from pathlib import Path
import shutil
import subprocess

from agent_evolve_v2.runtime import run_command
from agent_evolve_v2.service_runtime import discover_repo_root, resolve_env_value
from agent_evolve_v2.critic import summarize_job
from agent_evolve_v2.state import BenchmarkSummary, OfficialBenchmarkRun
from agent_evolve_v2.workspace_support.benchmark_cache import (
    CanonicalRunRecord,
    compute_benchmark_fingerprint,
    load_cached_run,
    next_canonical_run_dir,
    project_visible_run,
    write_canonical_run,
)

DEFAULT_BENCHMARK_TASKS = [
    "polyglot-c-py",
    "polyglot-rust-c",
    "headless-terminal",
    "regex-log",
    "git-multibranch",
    "configure-git-webserver",
    "vulnerable-secret",
    "sqlite-with-gcov",
    "cancel-async-tasks",
    "pypi-server",
    "multi-source-data-merger",
    "git-leak-recovery",
    "fix-git",
    "log-summary-date-ranges",
    "modernize-scientific-stack",
    "openssl-selfsigned-cert",
    "kv-store-grpc",
    "constraints-scheduling",
    "nginx-request-logging",
    "prove-plus-comm",
]


def run_workspace_benchmark(
    *,
    workspace_path: Path,
    model_key: str,
    result_json_out: Path,
    record_visible: bool,
    request_label: str,
) -> subprocess.CompletedProcess[str]:
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "agent_evolve_v2.service_cli",
        "benchmark",
        "--workspace",
        str(workspace_path),
        "--model-key",
        model_key,
        "--request-label",
        request_label,
        "--result-json-out",
        str(result_json_out),
    ]
    if not record_visible:
        command.append("--no-record-visible")
    return run_command(
        command=command,
        cwd=discover_repo_root(start_path=workspace_path),
    )


def load_benchmark_summary(*, summary_path: Path) -> BenchmarkSummary:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return BenchmarkSummary(**data)


def load_workspace_benchmark_result(
    *,
    result_json_path: Path,
) -> tuple[OfficialBenchmarkRun, BenchmarkSummary]:
    payload = json.loads(result_json_path.read_text(encoding="utf-8"))
    official_run = OfficialBenchmarkRun(
        fingerprint=str(payload["fingerprint"]),
        model_key=str(payload["model_key"]),
        canonical_run_dir=str(payload["canonical_run_dir"]),
        canonical_manifest_path=str(payload["canonical_manifest_path"]),
        aggregate_result_path=str(payload["aggregate_result_path"]),
        harbor_job_dir=str(payload["harbor_job_dir"]),
        benchmark_summary_path=str(payload["benchmark_summary_path"]),
        benchmark_stdout_path=str(payload["benchmark_stdout_path"]),
        benchmark_stderr_path=str(payload["benchmark_stderr_path"]),
        included_files=[str(item) for item in payload.get("included_files", [])],
        cache_hit=bool(payload.get("cache_hit", False)),
        created_at_utc=str(payload["created_at_utc"]),
    )
    summary = load_benchmark_summary(
        summary_path=Path(official_run.benchmark_summary_path),
    )
    return official_run, summary


def execute_workspace_benchmark(
    *,
    workspace_path: Path,
    model_key: str,
    result_json_out: Path | None,
    record_visible: bool,
    request_label: str,
) -> int:
    workspace_root = workspace_path.resolve()
    cache_root = resolve_benchmark_cache_root(workspace_root=workspace_root)
    task_names = resolve_workspace_benchmark_tasks(workspace_root=workspace_root)
    fingerprint_bundle = compute_benchmark_fingerprint(
        workspace_root=workspace_root,
        model_key=model_key,
        task_names=task_names,
    )
    cached_run = load_cached_run(
        cache_root=cache_root,
        fingerprint=fingerprint_bundle.fingerprint,
    )
    if cached_run is not None:
        return _emit_result(
            canonical_run=cached_run,
            cache_hit=True,
            result_json_out=result_json_out,
            record_visible=record_visible,
            request_label=request_label,
            workspace_root=workspace_root,
        )
    return _execute_and_emit_benchmark(
        cache_root=cache_root,
        fingerprint_bundle=fingerprint_bundle,
        model_key=model_key,
        result_json_out=result_json_out,
        record_visible=record_visible,
        request_label=request_label,
        task_names=task_names,
        workspace_root=workspace_root,
    )


def resolve_benchmark_cache_root(*, workspace_root: Path) -> Path:
    for candidate in (workspace_root, *workspace_root.parents):
        cache_root = candidate / "benchmark_cache"
        if cache_root.exists():
            return cache_root.resolve()
    repo_root = discover_repo_root(start_path=workspace_root)
    cache_root = repo_root / "agent_evolve_v2" / "outputs" / "manual_benchmark_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root.resolve()


def _execute_and_emit_benchmark(
    *,
    cache_root: Path,
    fingerprint_bundle,
    model_key: str,
    result_json_out: Path | None,
    record_visible: bool,
    request_label: str,
    task_names: list[str] | None,
    workspace_root: Path,
) -> int:
    canonical_run_dir = next_canonical_run_dir(
        cache_root=cache_root,
        fingerprint=fingerprint_bundle.fingerprint,
    )
    canonical_run_dir.mkdir(parents=True, exist_ok=False)
    jobs_dir = canonical_run_dir / "harbor_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    command = build_harbor_command(
        workspace_root=workspace_root,
        jobs_dir=jobs_dir,
        model_key=model_key,
        task_names=task_names,
    )
    completed = subprocess.run(
        command,
        cwd=discover_repo_root(start_path=workspace_root),
        text=True,
        capture_output=True,
        check=False,
        env=os.environ.copy(),
    )
    stdout_path = canonical_run_dir / "benchmark_stdout.log"
    stderr_path = canonical_run_dir / "benchmark_stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        failed_run = CanonicalRunRecord(
            run_id=canonical_run_dir.name,
            fingerprint=fingerprint_bundle.fingerprint,
            model_key=model_key,
            cache_root=str(cache_root),
            canonical_run_dir=str(canonical_run_dir),
            aggregate_result_path="",
            harbor_job_dir="",
            benchmark_summary_path="",
            benchmark_stdout_path=str(stdout_path),
            benchmark_stderr_path=str(stderr_path),
            created_at_utc=datetime.now(UTC).isoformat(),
            status="failed",
            included_files=[item.relative_path for item in fingerprint_bundle.inputs],
            included_file_hashes=list(fingerprint_bundle.inputs),
        )
        write_canonical_run(cache_root=cache_root, record=failed_run)
        print(completed.stdout)
        print(completed.stderr)
        return completed.returncode
    aggregate_result_path = resolve_aggregate_result_path(jobs_dir=jobs_dir)
    harbor_job_dir = aggregate_result_path.parent
    benchmark_summary, _critic = summarize_job(
        harbor_job_dir=harbor_job_dir,
        max_failure_items=0,
        max_success_items=0,
    )
    payload = asdict(benchmark_summary)
    payload["aggregate_result_path"] = str(aggregate_result_path)
    payload["harbor_job_dir"] = str(harbor_job_dir)
    payload["created_at_utc"] = datetime.now(UTC).isoformat()
    summary_path = canonical_run_dir / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (canonical_run_dir / "fingerprint_inputs.json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint_bundle.fingerprint,
                "model_key": model_key,
                "included_files": [asdict(item) for item in fingerprint_bundle.inputs],
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    canonical_run = CanonicalRunRecord(
        run_id=canonical_run_dir.name,
        fingerprint=fingerprint_bundle.fingerprint,
        model_key=model_key,
        cache_root=str(cache_root),
        canonical_run_dir=str(canonical_run_dir),
        aggregate_result_path=str(aggregate_result_path),
        harbor_job_dir=str(harbor_job_dir),
        benchmark_summary_path=str(summary_path),
        benchmark_stdout_path=str(stdout_path),
        benchmark_stderr_path=str(stderr_path),
        created_at_utc=datetime.now(UTC).isoformat(),
        status="completed",
        included_files=[item.relative_path for item in fingerprint_bundle.inputs],
        included_file_hashes=list(fingerprint_bundle.inputs),
    )
    write_canonical_run(cache_root=cache_root, record=canonical_run)
    return _emit_result(
        canonical_run=canonical_run,
        cache_hit=False,
        result_json_out=result_json_out,
        record_visible=record_visible,
        request_label=request_label,
        workspace_root=workspace_root,
    )


def _emit_result(
    *,
    canonical_run: CanonicalRunRecord,
    cache_hit: bool,
    result_json_out: Path | None,
    record_visible: bool,
    request_label: str,
    workspace_root: Path,
) -> int:
    visible_run_dir = None
    if record_visible:
        visible_run_dir = str(
            project_visible_run(
                workspace_root=workspace_root,
                canonical_run=canonical_run,
                cache_hit=cache_hit,
                request_label=request_label,
            )
        )
    payload = {
        "cache_hit": cache_hit,
        "fingerprint": canonical_run.fingerprint,
        "model_key": canonical_run.model_key,
        "canonical_manifest_path": canonical_run.canonical_manifest_path,
        "canonical_run_dir": canonical_run.canonical_run_dir,
        "aggregate_result_path": canonical_run.aggregate_result_path,
        "harbor_job_dir": canonical_run.harbor_job_dir,
        "benchmark_summary_path": canonical_run.benchmark_summary_path,
        "benchmark_stdout_path": canonical_run.benchmark_stdout_path,
        "benchmark_stderr_path": canonical_run.benchmark_stderr_path,
        "included_files": canonical_run.included_files,
        "visible_run_dir": visible_run_dir,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    if result_json_out is not None:
        result_json_out.parent.mkdir(parents=True, exist_ok=True)
        result_json_out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=True))
    return 0


def build_harbor_command(
    *,
    workspace_root: Path,
    jobs_dir: Path,
    model_key: str,
    task_names: list[str] | None = None,
) -> list[str]:
    repo_root = discover_repo_root(start_path=workspace_root)
    base_command = resolve_harbor_command()
    command = [
        *base_command,
        "run",
        "--jobs-dir",
        str(jobs_dir),
        "--n-concurrent",
        "8",
        "--env",
        "docker",
        "--delete",
        "--no-force-build",
        "-d",
        "terminal-bench@2.0",
        "--agent-import-path",
        "agent_evolve_v2.workspace_support.workspace_harbor_agent:WorkspaceHarborAgent",
        "--model",
        model_key,
        "--agent-env",
        f"AGENT_EVOLVE_V2_WORKSPACE_PATH={workspace_root}",
        "--agent-env",
        f"AGENT_EVOLVE_V2_REPO_ROOT={repo_root}",
    ]
    for env_name in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "SMALL_AGENT_CA_BUNDLE",
    ):
        value = resolve_env_value(env_name=env_name)
        if value:
            command.extend(["--agent-env", f"{env_name}={value}"])
    selected_tasks = task_names or DEFAULT_BENCHMARK_TASKS
    for task_name in selected_tasks:
        command.extend(["--task-name", task_name])
    return command


def resolve_workspace_benchmark_tasks(*, workspace_root: Path) -> list[str] | None:
    manifest_path = _discover_run_manifest(start_path=workspace_root)
    if manifest_path is None:
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    raw_task_names = payload.get("benchmark_tasks")
    if not isinstance(raw_task_names, list):
        return None
    task_names = [str(task_name).strip() for task_name in raw_task_names]
    task_names = [task_name for task_name in task_names if task_name]
    return task_names or None


def _discover_run_manifest(*, start_path: Path) -> Path | None:
    for candidate_root in (start_path, *start_path.parents):
        manifest_path = candidate_root / "run_manifest.json"
        if manifest_path.exists():
            return manifest_path
    return None


def resolve_harbor_command() -> list[str]:
    if shutil.which("harbor"):
        return ["harbor"]
    if shutil.which("uvx"):
        return [
            "uvx",
            "--from",
            "harbor",
            "--with",
            "truststore",
            "--with",
            "rich",
            "python",
            "-c",
            "import truststore; truststore.inject_into_ssl(); from harbor.cli.main import app; app()",
        ]
    raise RuntimeError("Neither 'harbor' nor 'uvx' is available on PATH.")


def resolve_aggregate_result_path(*, jobs_dir: Path) -> Path:
    candidates = [
        path for path in jobs_dir.rglob("result.json") if path.parent.parent == jobs_dir
    ]
    if not candidates:
        raise FileNotFoundError("Unable to locate Harbor aggregate result.json")
    return sorted(candidates)[-1]
