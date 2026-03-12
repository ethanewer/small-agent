from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from harbor_config import resolve_api_key

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = PROJECT_ROOT / "cli.py"

pytestmark = [pytest.mark.requires_api_key]


def test_liteforge_live_if_api_key_available() -> None:
    strict_mode = os.getenv("SMALL_AGENT_STRICT_TESTS") == "1"
    if not resolve_api_key(config_api_key="OPENROUTER_API_KEY"):
        if strict_mode:
            pytest.fail("OPENROUTER_API_KEY not set")

        pytest.skip("OPENROUTER_API_KEY not set")

    proc = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--agent",
            "liteforge",
            "--model",
            "qwen3-coder-next",
            "Reply with exactly: OK",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    combined_output = f"{proc.stdout}\n{proc.stderr}"
    if proc.returncode != 0:
        lowered = combined_output.lower()
        network_failure_markers = (
            "certificate_verify_failed",
            "self-signed certificate",
            "connection error",
            "api error",
            "econnreset",
            "etimedout",
            "enotfound",
        )
        if any(marker in lowered for marker in network_failure_markers):
            if strict_mode:
                pytest.fail(combined_output)

            pytest.skip(
                "liteforge live call failed due to network/API transport issues"
            )

    assert proc.returncode == 0, combined_output
    assert "Agent: liteforge" in combined_output
