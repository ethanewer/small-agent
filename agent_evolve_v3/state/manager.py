# pyright: reportAny=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
import re
import shutil
import tempfile
from pathlib import Path

from agent_evolve_v3.config import RunSpec
from agent_evolve_v3.services.benchmark import reset_visible_benchmark_outputs
from agent_evolve_v3.state.planner_context import PLANNER_NOTES_FILE_NAME
from agent_evolve_v3.state.types import (
    AgentState,
    agent_state_json_schema,
    planning_output_json_schema,
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
        self.states_dir.mkdir(parents=True, exist_ok=True)
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)

    @property
    def states(self) -> list[AgentState]:
        loaded = [
            AgentState.load(path=path)
            for path in sorted(self.states_dir.glob("iteration-*.json"))
        ]
        loaded.sort(key=lambda state: state.iteration)
        return loaded

    @property
    def completed_states(self) -> list[AgentState]:
        return [state for state in self.states if state.result is not None]

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
        )
        state.save()
        return state

    def create_child_state(
        self,
        *,
        parent_state: AgentState,
        iteration: int,
        plan: str,
        planner_selected_state_index: int,
        planner_prompt_artifact_path: Path,
        planner_output_artifact_path: Path,
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
            planner_selected_state_index=planner_selected_state_index,
            planner_selected_iteration=parent_state.iteration,
            planner_prompt_artifact_path=str(planner_prompt_artifact_path.resolve()),
            planner_output_artifact_path=str(planner_output_artifact_path.resolve()),
            planner_summary=plan.splitlines()[0].strip(),
        )
        state.save()
        return state

    @contextmanager
    def planning_environment(
        self,
        *,
        planner_notes_path: Path,
    ) -> Iterator[tuple[Path, list[AgentState]]]:
        candidate_states = self.completed_states
        with tempfile.TemporaryDirectory(prefix="agent-evolve-v3-planning-") as tmpdir:
            planning_root = Path(tmpdir)
            (planning_root / "states.json").write_text(
                json.dumps(
                    [state.to_dict() for state in candidate_states],
                    indent=2,
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (planning_root / "state-schema.json").write_text(
                json.dumps(
                    agent_state_json_schema(),
                    indent=2,
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (planning_root / "output-schema.json").write_text(
                json.dumps(
                    planning_output_json_schema(),
                    indent=2,
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            planner_notes_text = (
                planner_notes_path.read_text(encoding="utf-8")
                if planner_notes_path.exists()
                else ""
            )
            (planning_root / PLANNER_NOTES_FILE_NAME).write_text(
                planner_notes_text,
                encoding="utf-8",
            )
            yield planning_root.resolve(), candidate_states

    def _materialize_seed_workspace(
        self,
        *,
        workspace_path: Path,
        baseline: str,
    ) -> None:
        template_root = self.repo_root / "agent_evolve_v3" / "start_workdirs" / baseline
        shutil.copytree(
            src=template_root,
            dst=workspace_path,
            ignore=COPY_IGNORES,
            dirs_exist_ok=False,
        )

        readme_path = workspace_path / "README.md"
        if readme_path.exists():
            readme_path.write_text(
                readme_path.read_text(encoding="utf-8").format(
                    baseline=baseline,
                    model_key=self.run_spec.model_key,
                ),
                encoding="utf-8",
            )
        self._sync_workspace_docs(workspace_path=workspace_path)

    def _sync_workspace_docs(self, *, workspace_path: Path) -> None:
        test_command = f"./test_agent.sh {self.run_spec.model_key}"
        smoke_benchmark_command = f"./run_smoke_benchmark.sh {self.run_spec.model_key}"
        test_pattern = re.compile(r"\./test_agent\.sh \S+")
        smoke_benchmark_pattern = re.compile(r"\./run_smoke_benchmark\.sh \S+")
        readme_path = workspace_path / "README.md"
        if not readme_path.exists():
            return

        content = readme_path.read_text(encoding="utf-8")
        updated = test_pattern.sub(test_command, content)
        updated = smoke_benchmark_pattern.sub(smoke_benchmark_command, updated)
        readme_path.write_text(updated, encoding="utf-8")

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
        if state.official_benchmark is None:
            return
        reset_visible_benchmark_outputs(
            workspace_root=workspace_path,
            benchmark=state.official_benchmark,
            request_label=request_label,
        )
