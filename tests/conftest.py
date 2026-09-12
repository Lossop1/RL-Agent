"""Global test isolation for console state and remote access."""
from __future__ import annotations

import os
from pathlib import Path
import shutil


# Tests must never inherit a developer shell's real-source setting. Individual
# unit tests that exercise RealDataSource use explicit settings and mock every
# external boundary.
os.environ["LOCOMOTION_CONSOLE_SOURCE"] = "fake"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_TEST_ROOT = _REPO_ROOT / ".pytest-tmp"
os.environ["LOCOMOTION_CONSOLE_STATE_ROOT"] = str(
    _TEST_ROOT / f"console-state-{os.getpid()}"
)


def pytest_sessionfinish(session, exitstatus):
    """Keep pytest state and all temporary paths inside, then remove them."""
    del session, exitstatus
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)
