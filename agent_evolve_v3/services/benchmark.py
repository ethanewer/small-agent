# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false, reportUnknownParameterType=false, reportMissingParameterType=false

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Literal

from agent_evolve_v3.services.runtime import run_command
from agent_evolve_v3.services.workspace import discover_repo_root, resolve_env_value
from agent_evolve_v3.state.types import BenchmarkSummary, OfficialBenchmarkRun

BenchmarkPreset = Literal["official", "smoke"]

OFFICIAL_BENCHMARK_SCRIPT_RELATIVE_PATH = Path("harbor") / "run_small_benchmark.sh"
SMOKE_BENCHMARK_SCRIPT_RELATIVE_PATH = Path("harbor") / "run_smoke.sh"
VISIBLE_RUNS_DIR = Path("outputs") / "benchmark_runs"
LATEST_VISIBLE_RUN_PATH = Path("outputs") / "latest_run.json"
ARTIFACTS_RUNS_DIR = Path("benchmark-artifacts")
HARBOR_TASK_CACHE_DIR = Path("~/.cache/harbor/tasks").expanduser()


@dataclass(frozen=True)
class BenchmarkSpec:
    dataset_ref: str
    n_concurrent: int
    task_names: tuple[str, ...]


def prepull_task_images(
    *,
    task_names: tuple[str, ...],
    repo_root: Path,
    benchmark_preset: BenchmarkPreset = "official",
) -> None:
    """Pull Docker images for benchmark tasks that are not already cached locally."""
    effective_tasks = task_names
    if not effective_tasks:
        spec = load_benchmark_spec(
            repo_root=repo_root,
            benchmark_preset=benchmark_preset,
        )
        effective_tasks = spec.task_names

    images = _resolve_task_docker_images(task_names=effective_tasks)
    if not images:
        return

    local_images = _list_local_docker_images()
    missing = [img for img in images if img not in local_images]
    if not missing:
        print(f"All {len(images)} task images already cached locally.")
        return

    print(f"Pre-pulling {len(missing)}/{len(images)} missing Docker images...")
    for image in missing:
        print(f"  Pulling {image} ...")
        result = subprocess.run(
            ["docker", "pull", image],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(f"  WARNING: failed to pull {image}: {result.stderr.strip()}")
        else:
            print(f"  OK: {image}")

    print("Pre-pull complete.")


def _resolve_task_docker_images(*, task_names: tuple[str, ...]) -> list[str]:
    """Read docker_image from task.toml files in the harbor task cache."""
    images: list[str] = []
    if not HARBOR_TASK_CACHE_DIR.is_dir():
        return images

    task_dirs = {
        path.parent.name: path.parent
        for path in HARBOR_TASK_CACHE_DIR.rglob("task.toml")
    }
    for task_name in task_names:
        task_dir = task_dirs.get(task_name)
        if task_dir is None:
            continue
        toml_path = task_dir / "task.toml"
        match = re.search(
            r'^docker_image\s*=\s*"([^"]+)"',
            toml_path.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        if match:
            images.append(match.group(1))

    return images


def _list_local_docker_images() -> set[str]:
    """Return a set of 'repository:tag' strings for locally available images."""
    result = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return set()
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and line.strip() != "<none>:<none>"
    }


def run_workspace_benchmark(
    *,
    workspace_path: Path,
    model_key: str,
    result_json_out: Path,
    record_visible: bool,
    request_label: str,
    artifacts_dir: Path | None = None,
    benchmark_preset: BenchmarkPreset = "official",
) -> subprocess.CompletedProcess[str]:
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "agent_evolve_v3.services.cli",
        "benchmark",
        "--workspace",
        str(workspace_path),
        "--model-key",
        model_key,
        "--request-label",
        request_label,
        "--benchmark-preset",
        benchmark_preset,
        "--result-json-out",
        str(result_json_out),
    ]
    if artifacts_dir is not None:
        command.extend(["--artifacts-dir", str(artifacts_dir)])
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
        model_key=str(payload["model_key"]),
        aggregate_result_path=str(payload["aggregate_result_path"]),
        harbor_job_dir=str(payload["harbor_job_dir"]),
        benchmark_summary_path=str(payload["benchmark_summary_path"]),
        benchmark_stdout_path=str(payload["benchmark_stdout_path"]),
        benchmark_stderr_path=str(payload["benchmark_stderr_path"]),
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
    artifacts_dir: Path | None = None,
    benchmark_preset: BenchmarkPreset = "official",
) -> int:
    workspace_root = workspace_path.resolve()
    repo_root = discover_repo_root(start_path=workspace_root)
    benchmark_spec = load_benchmark_spec(
        repo_root=repo_root,
        benchmark_preset=benchmark_preset,
    )
    configured_tasks = (
        resolve_workspace_benchmark_tasks(workspace_root=workspace_root)
        if benchmark_preset == "official"
        else None
    )
    task_names = configured_tasks or list(benchmark_spec.task_names)
    artifact_root, visible_run_dir = _prepare_benchmark_output_dirs(
        workspace_root=workspace_root,
        artifacts_dir=artifacts_dir,
        record_visible=record_visible,
        request_label=request_label,
    )
    jobs_dir = artifact_root / "harbor_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    command = build_harbor_command(
        workspace_root=workspace_root,
        jobs_dir=jobs_dir,
        model_key=model_key,
        task_names=task_names,
        benchmark_preset=benchmark_preset,
    )
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
        env=os.environ.copy(),
    )
    stdout_path = artifact_root / "benchmark_stdout.log"
    stderr_path = artifact_root / "benchmark_stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        return completed.returncode

    aggregate_result_path = resolve_aggregate_result_path(jobs_dir=jobs_dir)
    harbor_job_dir = aggregate_result_path.parent
    benchmark_summary = summarize_benchmark_job(harbor_job_dir=harbor_job_dir)
    summary_payload = asdict(benchmark_summary)
    summary_payload["aggregate_result_path"] = str(aggregate_result_path)
    summary_payload["harbor_job_dir"] = str(harbor_job_dir)
    created_at_utc = datetime.now(UTC).isoformat()
    summary_payload["created_at_utc"] = created_at_utc
    summary_path = artifact_root / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    official_run = OfficialBenchmarkRun(
        model_key=model_key,
        aggregate_result_path=str(aggregate_result_path),
        harbor_job_dir=str(harbor_job_dir),
        benchmark_summary_path=str(summary_path),
        benchmark_stdout_path=str(stdout_path),
        benchmark_stderr_path=str(stderr_path),
        created_at_utc=created_at_utc,
    )
    if visible_run_dir is not None:
        _write_visible_run_manifest(
            workspace_root=workspace_root,
            visible_run_dir=visible_run_dir,
            request_label=request_label,
            benchmark=official_run,
        )
    payload = {
        "model_key": official_run.model_key,
        "aggregate_result_path": official_run.aggregate_result_path,
        "harbor_job_dir": official_run.harbor_job_dir,
        "benchmark_summary_path": official_run.benchmark_summary_path,
        "benchmark_stdout_path": official_run.benchmark_stdout_path,
        "benchmark_stderr_path": official_run.benchmark_stderr_path,
        "visible_run_dir": str(visible_run_dir) if visible_run_dir else None,
        "created_at_utc": official_run.created_at_utc,
    }
    if result_json_out is not None:
        result_json_out.parent.mkdir(parents=True, exist_ok=True)
        result_json_out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=True))
    return 0


def reset_visible_benchmark_outputs(
    *,
    workspace_root: Path,
    benchmark: OfficialBenchmarkRun,
    request_label: str,
) -> Path:
    outputs_root = workspace_root / "outputs"
    if outputs_root.exists():
        shutil.rmtree(path=outputs_root)
    visible_run_dir = _create_visible_run_dir(
        workspace_root=workspace_root,
        request_label=request_label,
    )
    _copy_benchmark_artifacts(
        benchmark=benchmark,
        destination_root=visible_run_dir,
    )
    _write_visible_run_manifest(
        workspace_root=workspace_root,
        visible_run_dir=visible_run_dir,
        request_label=request_label,
        benchmark=benchmark,
    )
    return visible_run_dir


@lru_cache(maxsize=None)
def load_benchmark_spec(
    *,
    repo_root: Path,
    benchmark_preset: BenchmarkPreset,
) -> BenchmarkSpec:
    script_path = (repo_root / _script_relative_path(benchmark_preset)).resolve()
    content = script_path.read_text(encoding="utf-8")
    dataset_ref = _extract_script_setting(
        content=content,
        name="DATASET_REF",
        script_path=script_path,
    )
    n_concurrent_text = _extract_script_setting(
        content=content,
        name="N_CONCURRENT",
        script_path=script_path,
    )
    try:
        n_concurrent = int(n_concurrent_text)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid N_CONCURRENT value in {script_path}: {n_concurrent_text!r}"
        ) from exc
    task_names = tuple(
        _extract_benchmark_tasks(content=content, script_path=script_path)
    )
    return BenchmarkSpec(
        dataset_ref=dataset_ref,
        n_concurrent=n_concurrent,
        task_names=task_names,
    )


def build_harbor_command(
    *,
    workspace_root: Path,
    jobs_dir: Path,
    model_key: str,
    task_names: list[str] | None = None,
    benchmark_preset: BenchmarkPreset = "official",
) -> list[str]:
    repo_root = discover_repo_root(start_path=workspace_root)
    benchmark_spec = load_benchmark_spec(
        repo_root=repo_root.resolve(),
        benchmark_preset=benchmark_preset,
    )
    base_command = resolve_harbor_command()
    command = [
        *base_command,
        "run",
        "--jobs-dir",
        str(jobs_dir),
        "--n-concurrent",
        str(benchmark_spec.n_concurrent),
        "--env",
        "docker",
        "--delete",
        "--no-force-build",
        "-d",
        benchmark_spec.dataset_ref,
        "--agent-import-path",
        "agent_evolve_v3.workspace_support.workspace_harbor_agent:WorkspaceHarborAgent",
        "--model",
        model_key,
        "--agent-env",
        f"AGENT_EVOLVE_V3_WORKSPACE_PATH={workspace_root}",
        "--agent-env",
        f"AGENT_EVOLVE_V3_REPO_ROOT={repo_root}",
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
    selected_tasks = task_names or list(benchmark_spec.task_names)
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


def _prepare_benchmark_output_dirs(
    *,
    workspace_root: Path,
    artifacts_dir: Path | None,
    record_visible: bool,
    request_label: str,
) -> tuple[Path, Path | None]:
    if artifacts_dir is not None:
        artifact_root = artifacts_dir.resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        return artifact_root, None
    if record_visible:
        visible_run_dir = _create_visible_run_dir(
            workspace_root=workspace_root,
            request_label=request_label,
        )
        return visible_run_dir, visible_run_dir
    artifact_root = _create_artifacts_dir(
        workspace_root=workspace_root,
        request_label=request_label,
    )
    return artifact_root, None


def _create_artifacts_dir(*, workspace_root: Path, request_label: str) -> Path:
    artifacts_root = workspace_root / ARTIFACTS_RUNS_DIR
    artifacts_root.mkdir(parents=True, exist_ok=True)
    artifact_root = artifacts_root / _next_run_id(
        runs_root=artifacts_root,
        request_label=request_label,
    )
    artifact_root.mkdir(parents=True, exist_ok=False)
    return artifact_root


def _create_visible_run_dir(*, workspace_root: Path, request_label: str) -> Path:
    runs_root = workspace_root / VISIBLE_RUNS_DIR
    runs_root.mkdir(parents=True, exist_ok=True)
    visible_run_dir = runs_root / _next_run_id(
        runs_root=runs_root,
        request_label=request_label,
    )
    visible_run_dir.mkdir(parents=True, exist_ok=False)
    return visible_run_dir


def _next_run_id(*, runs_root: Path, request_label: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = f"{timestamp}-{request_label}"
    candidate = base
    suffix = 1
    while (runs_root / candidate).exists():
        suffix += 1
        candidate = f"{base}-{suffix:02d}"
    return candidate


def _write_visible_run_manifest(
    *,
    workspace_root: Path,
    visible_run_dir: Path,
    request_label: str,
    benchmark: OfficialBenchmarkRun,
) -> None:
    payload = {
        "request_label": request_label,
        "visible_run_dir": str(visible_run_dir),
        "visible_created_at_utc": datetime.now(UTC).isoformat(),
        "official_benchmark": asdict(benchmark),
    }
    (visible_run_dir / "run_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    latest_path = workspace_root / LATEST_VISIBLE_RUN_PATH
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _copy_benchmark_artifacts(
    *,
    benchmark: OfficialBenchmarkRun,
    destination_root: Path,
) -> None:
    copies = {
        Path(benchmark.benchmark_summary_path): destination_root
        / "benchmark_summary.json",
        Path(benchmark.benchmark_stdout_path): destination_root
        / "benchmark_stdout.log",
        Path(benchmark.benchmark_stderr_path): destination_root
        / "benchmark_stderr.log",
    }
    aggregate_result_path = Path(benchmark.aggregate_result_path)
    if aggregate_result_path.exists():
        copies[aggregate_result_path] = destination_root / "aggregate_result.json"
    for src, dst in copies.items():
        if src.exists():
            shutil.copy2(src=src, dst=dst)
    harbor_job_src = Path(benchmark.harbor_job_dir)
    harbor_job_dst = destination_root / "harbor_job"
    if harbor_job_src.exists():
        shutil.copytree(
            src=harbor_job_src,
            dst=harbor_job_dst,
            dirs_exist_ok=True,
        )


def _script_relative_path(benchmark_preset: BenchmarkPreset) -> Path:
    if benchmark_preset == "smoke":
        return SMOKE_BENCHMARK_SCRIPT_RELATIVE_PATH
    return OFFICIAL_BENCHMARK_SCRIPT_RELATIVE_PATH


def _extract_script_setting(
    *,
    content: str,
    name: str,
    script_path: Path,
) -> str:
    pattern = re.compile(
        rf'^{name}=(?:"(?P<quoted>[^"]+)"|(?P<bare>\S+))$',
        flags=re.MULTILINE,
    )
    match = pattern.search(content)
    if match is None:
        raise RuntimeError(f"Missing {name} in {script_path}.")
    value = match.group("quoted") or match.group("bare") or ""
    if not value:
        raise RuntimeError(f"Empty {name} in {script_path}.")
    return value


def _extract_benchmark_tasks(*, content: str, script_path: Path) -> list[str]:
    in_tasks = False
    task_names: list[str] = []
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not in_tasks:
            if stripped.endswith("_TASKS=("):
                in_tasks = True
            continue
        if stripped == ")":
            break
        if not stripped or stripped.startswith("#"):
            continue
        task_names.append(stripped)
    if not task_names:
        raise RuntimeError(f"Unable to parse task list from {script_path}.")
    return task_names


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


def summarize_benchmark_job(*, harbor_job_dir: Path) -> BenchmarkSummary:
    aggregate_result_path = harbor_job_dir / "result.json"
    aggregate = _load_harbor_json(path=aggregate_result_path)
    stats = _as_dict(value=aggregate.get("stats"))
    evals = _as_dict(value=stats.get("evals"))
    eval_payload = next(iter(evals.values()), {})
    eval_stats = _as_dict(value=eval_payload)
    reward_stats = _as_dict(value=eval_stats.get("reward_stats"))
    reward_buckets = _as_dict(value=reward_stats.get("reward"))
    passed_trials = _string_list(value=reward_buckets.get("1.0"))
    failed_trials = _string_list(value=reward_buckets.get("0.0"))
    exception_types = {
        key: _string_list(value=value)
        for key, value in _as_dict(value=eval_stats.get("exception_stats")).items()
    }

    reward_mean = _extract_reward_mean(eval_stats=eval_stats)
    n_trials = _coerce_int(
        value=eval_stats.get("n_trials", stats.get("n_trials", 0)),
    )
    error_count = _coerce_int(
        value=eval_stats.get("n_errors", stats.get("n_errors", 0)),
    )

    return BenchmarkSummary(
        created_at_utc=datetime.now(UTC).isoformat(),
        aggregate_result_path=str(aggregate_result_path),
        harbor_job_dir=str(harbor_job_dir),
        reward_mean=reward_mean,
        n_trials=n_trials,
        pass_count=len(passed_trials),
        failure_count=len(failed_trials),
        error_count=error_count,
        passed_trials=passed_trials,
        failed_trials=failed_trials,
        exception_types=exception_types,
    )


def _extract_reward_mean(*, eval_stats: dict[str, object]) -> float | None:
    metrics = eval_stats.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        return None
    first_metric = metrics[0]
    if not isinstance(first_metric, dict):
        return None
    mean = first_metric.get("mean")
    if isinstance(mean, (int, float)):
        return float(mean)
    return None


def _load_harbor_json(*, path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _as_dict(*, value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _string_list(*, value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _coerce_int(*, value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return 0
