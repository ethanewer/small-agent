# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnusedCallResult=false

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil

IGNORED_FINGERPRINT_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".basedpyright",
    ".venv",
    ".local",
    "outputs",
    "benchmark-artifacts",
}
WORKSPACE_AGENT_DIR = "agent"
PROMPT_GLOB = "**/templates/*.md"
VISIBLE_RUNS_DIR = Path("outputs") / "benchmark_runs"
LATEST_VISIBLE_RUN_PATH = Path("outputs") / "latest_run.json"


@dataclass(frozen=True)
class FingerprintInput:
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class BenchmarkFingerprint:
    fingerprint: str
    model_key: str
    inputs: list[FingerprintInput] = field(default_factory=list)


@dataclass(frozen=True)
class CanonicalRunRecord:
    run_id: str
    fingerprint: str
    model_key: str
    cache_root: str
    canonical_run_dir: str
    aggregate_result_path: str
    harbor_job_dir: str
    benchmark_summary_path: str
    benchmark_stdout_path: str
    benchmark_stderr_path: str
    created_at_utc: str
    status: str
    included_files: list[str] = field(default_factory=list)
    included_file_hashes: list[FingerprintInput] = field(default_factory=list)

    @property
    def canonical_manifest_path(self) -> str:
        return str(Path(self.canonical_run_dir) / "run_manifest.json")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=True)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CanonicalRunRecord":
        hashes = []
        raw_hashes = payload.get("included_file_hashes", [])
        if isinstance(raw_hashes, list):
            for item in raw_hashes:
                if isinstance(item, dict):
                    hashes.append(
                        FingerprintInput(
                            relative_path=str(item.get("relative_path", "")),
                            sha256=str(item.get("sha256", "")),
                        )
                    )
        raw_included_files = payload.get("included_files", [])
        included_files = []
        if isinstance(raw_included_files, list):
            included_files = [str(item) for item in raw_included_files]
        return cls(
            run_id=str(payload["run_id"]),
            fingerprint=str(payload["fingerprint"]),
            model_key=str(payload["model_key"]),
            cache_root=str(payload["cache_root"]),
            canonical_run_dir=str(payload["canonical_run_dir"]),
            aggregate_result_path=str(payload["aggregate_result_path"]),
            harbor_job_dir=str(payload["harbor_job_dir"]),
            benchmark_summary_path=str(payload["benchmark_summary_path"]),
            benchmark_stdout_path=str(payload["benchmark_stdout_path"]),
            benchmark_stderr_path=str(payload["benchmark_stderr_path"]),
            created_at_utc=str(payload["created_at_utc"]),
            status=str(payload["status"]),
            included_files=included_files,
            included_file_hashes=hashes,
        )


def compute_benchmark_fingerprint(
    *,
    workspace_root: Path,
    model_key: str,
) -> BenchmarkFingerprint:
    inputs = _collect_fingerprint_inputs(workspace_root=workspace_root)
    digest = hashlib.sha256()
    digest.update(b"model_key\0")
    digest.update(model_key.encode("utf-8"))
    digest.update(b"\0")
    for item in inputs:
        digest.update(item.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.sha256.encode("ascii"))
        digest.update(b"\0")
    return BenchmarkFingerprint(
        fingerprint=digest.hexdigest(),
        model_key=model_key,
        inputs=inputs,
    )


def load_cached_run(
    *,
    cache_root: Path,
    fingerprint: str,
) -> CanonicalRunRecord | None:
    index_path = cache_index_path(cache_root=cache_root, fingerprint=fingerprint)
    if not index_path.exists():
        return None
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    record = CanonicalRunRecord.from_dict(payload)
    if record.status != "completed":
        return None
    manifest_path = Path(record.canonical_manifest_path)
    summary_path = Path(record.benchmark_summary_path)
    if not manifest_path.exists() or not summary_path.exists():
        return None
    return record


def write_canonical_run(
    *,
    cache_root: Path,
    record: CanonicalRunRecord,
) -> Path:
    run_dir = Path(record.canonical_run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(record.to_json() + "\n", encoding="utf-8")
    if record.status == "completed":
        index_path = cache_index_path(
            cache_root=cache_root,
            fingerprint=record.fingerprint,
        )
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(record.to_json() + "\n", encoding="utf-8")
    return manifest_path


def next_canonical_run_dir(*, cache_root: Path, fingerprint: str) -> Path:
    runs_root = cache_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    prefix = fingerprint[:12]
    candidate = runs_root / f"{timestamp}-{prefix}"
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = runs_root / f"{timestamp}-{prefix}-{suffix:02d}"
    return candidate


def project_visible_run(
    *,
    workspace_root: Path,
    canonical_run: CanonicalRunRecord,
    cache_hit: bool,
    request_label: str,
) -> Path:
    runs_root = workspace_root / VISIBLE_RUNS_DIR
    runs_root.mkdir(parents=True, exist_ok=True)
    visible_run_dir = runs_root / _next_visible_run_id(
        runs_root=runs_root,
        request_label=request_label,
        canonical_run=canonical_run,
    )
    visible_run_dir.mkdir(parents=True, exist_ok=False)
    _copy_visible_artifacts(
        visible_run_dir=visible_run_dir,
        canonical_run=canonical_run,
    )
    visible_manifest = {
        "request_label": request_label,
        "cache_hit": cache_hit,
        "visible_run_dir": str(visible_run_dir),
        "visible_created_at_utc": datetime.now(UTC).isoformat(),
        "canonical_run": json.loads(canonical_run.to_json()),
    }
    (visible_run_dir / "run_manifest.json").write_text(
        json.dumps(visible_manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    latest_path = workspace_root / LATEST_VISIBLE_RUN_PATH
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(
        json.dumps(visible_manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return visible_run_dir


def reset_visible_outputs(
    *,
    workspace_root: Path,
    canonical_run: CanonicalRunRecord,
    request_label: str,
) -> Path:
    outputs_root = workspace_root / "outputs"
    if outputs_root.exists():
        shutil.rmtree(outputs_root)
    return project_visible_run(
        workspace_root=workspace_root,
        canonical_run=canonical_run,
        cache_hit=True,
        request_label=request_label,
    )


def resolve_latest_visible_run_dir(*, workspace_root: Path) -> Path:
    latest_path = workspace_root / LATEST_VISIBLE_RUN_PATH
    if latest_path.exists():
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            visible_run_dir = payload.get("visible_run_dir")
            if isinstance(visible_run_dir, str) and visible_run_dir:
                return Path(visible_run_dir)
    runs_root = workspace_root / VISIBLE_RUNS_DIR
    candidates = (
        [path for path in runs_root.iterdir() if path.is_dir()]
        if runs_root.exists()
        else []
    )
    if not candidates:
        raise FileNotFoundError(
            "Unable to locate a visible benchmark run under outputs/."
        )
    return sorted(candidates)[-1]


def cache_index_path(*, cache_root: Path, fingerprint: str) -> Path:
    return cache_root / "by_fingerprint" / f"{fingerprint}.json"


def _collect_fingerprint_inputs(*, workspace_root: Path) -> list[FingerprintInput]:
    agent_root = workspace_root / WORKSPACE_AGENT_DIR
    if not agent_root.exists():
        raise FileNotFoundError(f"Workspace agent directory not found: {agent_root}")
    inputs: list[FingerprintInput] = []
    for path in sorted(agent_root.rglob("*.py")):
        if _is_ignored_path(workspace_root=workspace_root, path=path):
            continue
        inputs.append(_build_input(workspace_root=workspace_root, path=path))
    for path in sorted(agent_root.glob(PROMPT_GLOB)):
        if path.is_file() and not _is_ignored_path(
            workspace_root=workspace_root, path=path
        ):
            inputs.append(_build_input(workspace_root=workspace_root, path=path))
    inputs.sort(key=lambda item: item.relative_path)
    return inputs


def _build_input(*, workspace_root: Path, path: Path) -> FingerprintInput:
    relative_path = path.relative_to(workspace_root).as_posix()
    return FingerprintInput(
        relative_path=relative_path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _is_ignored_path(*, workspace_root: Path, path: Path) -> bool:
    relative_parts = path.relative_to(workspace_root).parts
    return any(part in IGNORED_FINGERPRINT_DIRS for part in relative_parts[:-1])


def _next_visible_run_id(
    *,
    runs_root: Path,
    request_label: str,
    canonical_run: CanonicalRunRecord,
) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = f"{timestamp}-{request_label}-{canonical_run.run_id}"
    candidate = base
    suffix = 1
    while (runs_root / candidate).exists():
        suffix += 1
        candidate = f"{base}-{suffix:02d}"
    return candidate


def _copy_visible_artifacts(
    *,
    visible_run_dir: Path,
    canonical_run: CanonicalRunRecord,
) -> None:
    copies = {
        Path(canonical_run.benchmark_summary_path): visible_run_dir
        / "benchmark_summary.json",
        Path(canonical_run.benchmark_stdout_path): visible_run_dir
        / "benchmark_stdout.log",
        Path(canonical_run.benchmark_stderr_path): visible_run_dir
        / "benchmark_stderr.log",
        Path(canonical_run.canonical_manifest_path): visible_run_dir
        / "canonical_run_manifest.json",
    }
    for src, dst in copies.items():
        if src.exists():
            shutil.copy2(src=src, dst=dst)
