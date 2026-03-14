# pyright: reportUnusedCallResult=false

from __future__ import annotations

import json
import shutil
from pathlib import Path

BASELINE_SUPPORT_DIRS = {
    "liteforge": "liteforge_support",
    "terminus2": "terminus2_support",
}
BASELINE_SOURCE_DIRS = {
    "liteforge": "liteforge",
    "terminus2": "terminus2",
}
BASELINE_AGENT_TEMPLATES = {
    "liteforge": "liteforge_agent.py",
    "terminus2": "terminus2_agent.py",
}

IGNORE_NAMES = (
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".basedpyright",
    ".venv",
    ".local",
    "benchmark-artifacts",
    "outputs",
)


def refresh_start_workdir(*, repo_root: Path, baseline: str) -> Path:
    destination = repo_root / "agent_evolve_v2" / "start_workdirs" / baseline
    readme_template = (destination / "README.md").read_text(encoding="utf-8")
    notes_template = (destination / "NOTES.md").read_text(encoding="utf-8")
    pyproject_template = (
        repo_root / "agent_evolve_v2" / "workspace_support" / "pyproject.toml"
    ).read_text(encoding="utf-8")

    if destination.exists():
        shutil.rmtree(path=destination)

    support_root = repo_root / "agent_evolve_v2" / "workspace_support"
    baseline_src = repo_root / "agents" / BASELINE_SOURCE_DIRS[baseline]
    support_package = BASELINE_SUPPORT_DIRS[baseline]
    agent_template = (
        repo_root
        / "agent_evolve_v2"
        / "workspace_agent_templates"
        / BASELINE_AGENT_TEMPLATES[baseline]
    )

    shutil.copytree(
        src=support_root,
        dst=destination,
        ignore=shutil.ignore_patterns(*IGNORE_NAMES),
    )

    support_dst = destination / support_package
    shutil.copytree(
        src=baseline_src,
        dst=support_dst,
        ignore=shutil.ignore_patterns(*IGNORE_NAMES),
    )
    _rewrite_support_imports(
        support_root=support_dst,
        source_prefix=f"agents.{baseline}",
        destination_prefix=support_package,
    )
    _prune_support_entrypoints(
        baseline=baseline,
        support_root=support_dst,
    )

    agents_dir = destination / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src=agent_template, dst=agents_dir / "agent.py")
    shutil.copy2(src=repo_root / "config.json", dst=destination / "model_catalog.json")
    shutil.copy2(
        src=repo_root / "agent_evolve_v2" / "critic.py",
        dst=destination / "critic_tools.py",
    )
    shutil.copy2(
        src=repo_root / "agent_evolve_v2" / "state.py",
        dst=destination / "state_types.py",
    )

    (destination / "workspace_metadata.json").write_text(
        json.dumps(
            {
                "baseline": baseline,
                "created_from": str(baseline_src),
                "benchmark_cache_root": "",
                "workspace_role": "template",
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (destination / "README.md").write_text(readme_template, encoding="utf-8")
    (destination / "NOTES.md").write_text(notes_template, encoding="utf-8")
    (destination / "pyproject.toml").write_text(
        pyproject_template.format(support_package=support_package),
        encoding="utf-8",
    )
    return destination


def _rewrite_support_imports(
    *,
    support_root: Path,
    source_prefix: str,
    destination_prefix: str,
) -> None:
    replacements = (
        (f"from {source_prefix}.", f"from {destination_prefix}."),
        (f"import {source_prefix}.", f"import {destination_prefix}."),
        (f'"{source_prefix}.', f'"{destination_prefix}.'),
        (f"'{source_prefix}.", f"'{destination_prefix}."),
    )
    for path in support_root.rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        updated = content
        for old, new in replacements:
            updated = updated.replace(old, new)
        path.write_text(updated, encoding="utf-8")


def _prune_support_entrypoints(*, baseline: str, support_root: Path) -> None:
    if baseline == "liteforge":
        stale_entrypoint = support_root / "runtime_agent.py"
    else:
        stale_entrypoint = support_root / "agent.py"
    if stale_entrypoint.exists():
        stale_entrypoint.unlink()
    (support_root / "__init__.py").write_text(
        f'"""Workspace-local {baseline} support package."""\n',
        encoding="utf-8",
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    for baseline in BASELINE_SUPPORT_DIRS:
        refreshed = refresh_start_workdir(repo_root=repo_root, baseline=baseline)
        print(refreshed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
