# pyright: reportAny=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import shutil
from pathlib import Path
from typing import Protocol

from agent_evolve_v2.config import RunSpec
from agent_evolve_v2.state import AgentState
from agent_evolve_v2.workspace_support.benchmark_cache import (
    CanonicalRunRecord,
    reset_visible_outputs,
)

COPY_IGNORES = shutil.ignore_patterns(
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".basedpyright",
    ".venv",
    ".local",
    "benchmark-artifacts",
    "outputs",
    "*.pyc",
    "*.log",
)
OFFICIAL_VISIBLE_REQUEST_LABEL = "official"


@dataclass(frozen=True)
class ParentSample:
    state: AgentState
    score_sample: float
    alpha: float
    beta: float


class BetaSampler(Protocol):
    def betavariate(self, alpha: float, beta: float) -> float: ...


class StateManager:
    def __init__(
        self,
        *,
        repo_root: Path,
        run_root: Path,
        run_spec: RunSpec,
    ) -> None:
        self.repo_root = repo_root
        self.run_root = run_root
        self.run_spec = run_spec
        self.states_dir = run_root / "states"
        self.workspaces_dir = run_root / "workspaces"
        self.critic_workspaces_dir = run_root / "critic_workspaces"
        self.benchmark_cache_dir = run_root / "benchmark_cache"
        self.states_dir.mkdir(parents=True, exist_ok=True)
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)
        self.critic_workspaces_dir.mkdir(parents=True, exist_ok=True)

    @property
    def states(self) -> list[AgentState]:
        loaded = [
            AgentState.load(path=path)
            for path in sorted(self.states_dir.glob("iteration-*.json"))
        ]
        loaded.sort(key=lambda state: state.iteration)
        return loaded

    def bootstrap_root_state(self) -> AgentState:
        existing = self.states
        if existing:
            return existing[0]

        workspace_path = self.workspaces_dir / "iteration-0000"
        self._materialize_seed_workspace(
            workspace_path=workspace_path,
            baseline=self.run_spec.baseline,
        )
        state = AgentState(
            prev_path=None,
            path=str((self.states_dir / "iteration-0000.json").resolve()),
            refiner_workspace_path=str(workspace_path.resolve()),
            iteration=0,
            baseline=self.run_spec.baseline,
            notes="Seed workspace created from the configured baseline harness.",
        )
        state.save()
        return state

    def create_child_state(
        self,
        *,
        parent_state: AgentState,
        iteration: int,
        plan: str,
    ) -> AgentState:
        workspace_path = self.workspaces_dir / f"iteration-{iteration:04d}"
        shutil.copytree(
            src=Path(parent_state.refiner_workspace_path),
            dst=workspace_path,
            ignore=COPY_IGNORES,
            dirs_exist_ok=False,
        )
        self._sync_workspace_docs(workspace_path=workspace_path)
        self._seed_workspace_outputs(
            workspace_path=workspace_path,
            state=parent_state,
            request_label=OFFICIAL_VISIBLE_REQUEST_LABEL,
        )
        state = AgentState(
            prev_path=parent_state.path,
            path=str((self.states_dir / f"iteration-{iteration:04d}.json").resolve()),
            refiner_workspace_path=str(workspace_path.resolve()),
            iteration=iteration,
            baseline=parent_state.baseline,
            plan=plan,
        )
        state.save()
        return state

    def create_critic_workspace(self, *, state: AgentState) -> Path:
        workspace_path = self.critic_workspaces_dir / f"iteration-{state.iteration:04d}"
        if workspace_path.exists():
            shutil.rmtree(workspace_path)
        shutil.copytree(
            src=Path(state.refiner_workspace_path),
            dst=workspace_path,
            ignore=COPY_IGNORES,
            dirs_exist_ok=False,
        )
        self._sync_workspace_docs(workspace_path=workspace_path)
        self._seed_workspace_outputs(
            workspace_path=workspace_path,
            state=state,
            request_label=OFFICIAL_VISIBLE_REQUEST_LABEL,
        )
        state.critic_workspace_path = str(workspace_path.resolve())
        return workspace_path

    def sample_parent_state(self, *, rng: BetaSampler) -> ParentSample:
        candidates = [state for state in self.states if state.result is not None]
        if not candidates:
            root_state = self.bootstrap_root_state()
            alpha, beta = _state_posterior(state=root_state)
            return ParentSample(
                state=root_state,
                score_sample=rng.betavariate(alpha, beta),
                alpha=alpha,
                beta=beta,
            )

        best_sample: ParentSample | None = None
        for state in candidates:
            alpha, beta = _state_posterior(state=state)
            score_sample = rng.betavariate(alpha, beta)
            candidate = ParentSample(
                state=state,
                score_sample=score_sample,
                alpha=alpha,
                beta=beta,
            )
            if best_sample is None or candidate.score_sample > best_sample.score_sample:
                best_sample = candidate

        assert best_sample is not None
        return best_sample

    def write_run_manifest(self) -> Path:
        payload = {
            "baseline": self.run_spec.baseline,
            "model_key": self.run_spec.model_key,
            "cursor_model": self.run_spec.cursor_model,
            "iterations": self.run_spec.iterations,
            "max_critic_failures": self.run_spec.max_critic_failures,
            "max_critic_successes": self.run_spec.max_critic_successes,
            "random_seed": self.run_spec.random_seed,
        }
        path = self.run_root / "run_manifest.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        return path

    def _materialize_seed_workspace(
        self,
        *,
        workspace_path: Path,
        baseline: str,
    ) -> None:
        template_root = self.repo_root / "agent_evolve_v2" / "start_workdirs" / baseline
        shutil.copytree(
            src=template_root,
            dst=workspace_path,
            ignore=COPY_IGNORES,
            dirs_exist_ok=False,
        )

        for template_path in (
            workspace_path / "README.md",
            workspace_path / "NOTES.md",
        ):
            if template_path.exists():
                template_path.write_text(
                    template_path.read_text(encoding="utf-8").format(
                        baseline=baseline,
                        model_key=self.run_spec.model_key,
                    ),
                    encoding="utf-8",
                )
        self._sync_workspace_docs(workspace_path=workspace_path)

    def _sync_workspace_docs(self, *, workspace_path: Path) -> None:
        test_command = f"./test_agent.sh {self.run_spec.model_key}"
        benchmark_command = f"./run_benchmark.sh {self.run_spec.model_key}"
        test_pattern = re.compile(r"\./test_agent\.sh \S+")
        benchmark_pattern = re.compile(r"\./run_benchmark\.sh \S+")
        for relative_path in ("README.md", "NOTES.md"):
            doc_path = workspace_path / relative_path
            if not doc_path.exists():
                continue
            content = doc_path.read_text(encoding="utf-8")
            updated = test_pattern.sub(test_command, content)
            updated = benchmark_pattern.sub(benchmark_command, updated)
            doc_path.write_text(updated, encoding="utf-8")

    def seed_refiner_outputs(self, *, state: AgentState) -> None:
        self._seed_workspace_outputs(
            workspace_path=Path(state.refiner_workspace_path),
            state=state,
            request_label=OFFICIAL_VISIBLE_REQUEST_LABEL,
        )

    def _seed_workspace_outputs(
        self,
        *,
        workspace_path: Path,
        state: AgentState,
        request_label: str,
    ) -> None:
        canonical_run = self._canonical_run_from_state(state=state)
        if canonical_run is None:
            return
        reset_visible_outputs(
            workspace_root=workspace_path,
            canonical_run=canonical_run,
            request_label=request_label,
        )

    def _canonical_run_from_state(
        self,
        *,
        state: AgentState,
    ) -> CanonicalRunRecord | None:
        if state.official_benchmark is None:
            return None
        cache_root = Path(state.official_benchmark.canonical_run_dir).parents[1]
        payload: dict[str, object] = {
            "run_id": Path(state.official_benchmark.canonical_run_dir).name,
            "fingerprint": state.official_benchmark.fingerprint,
            "model_key": state.official_benchmark.model_key,
            "cache_root": str(cache_root),
            "canonical_run_dir": state.official_benchmark.canonical_run_dir,
            "aggregate_result_path": state.official_benchmark.aggregate_result_path,
            "harbor_job_dir": state.official_benchmark.harbor_job_dir,
            "benchmark_summary_path": state.official_benchmark.benchmark_summary_path,
            "benchmark_stdout_path": state.official_benchmark.benchmark_stdout_path,
            "benchmark_stderr_path": state.official_benchmark.benchmark_stderr_path,
            "created_at_utc": state.official_benchmark.created_at_utc,
            "status": "completed",
            "included_files": list(state.official_benchmark.included_files),
            "included_file_hashes": [],
        }
        return CanonicalRunRecord.from_dict(payload)


def _state_posterior(*, state: AgentState) -> tuple[float, float]:
    if state.result is None:
        return 1.0, 1.0
    alpha = 1.0 + max(0, state.result.pass_count)
    beta = 1.0 + max(0, state.result.failure_count)
    return alpha, beta
