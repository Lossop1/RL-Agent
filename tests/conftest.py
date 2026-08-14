"""Global test isolation for console state and remote access."""
from __future__ import annotations

import os
import tempfile


# Tests must never inherit a developer shell's real-source setting. Individual
# unit tests that exercise RealDataSource use explicit settings and mock every
# external boundary.
os.environ["LOCOMOTION_CONSOLE_SOURCE"] = "fake"
os.environ["LOCOMOTION_CONSOLE_STATE_ROOT"] = tempfile.mkdtemp(
    prefix="locomotion_console_pytest_"
)
