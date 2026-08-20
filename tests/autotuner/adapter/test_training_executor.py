from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from autotuner.adapter.remote_executors import VersionedTrainingStartExecutor


@dataclass(frozen=True)
class _Plan:
    run_id: str = "run-1"
    run_dir: str = "/runs/run-1"
    tmux_session: str = "train-1"
    process_pattern: str = "trainer-module"

    def prepare_command(self) -> str:
        return f"mkdir -p {self.run_dir}"

    def terminal_command(self) -> str:
        return "python -m trainer-module"

    def marker_command(self) -> str:
        return f"touch {self.run_dir}/console_start.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "tmux_session": self.tmux_session,
            "process_pattern": self.process_pattern,
            "argv": ["python", "-m", "trainer-module"],
            "environment": {},
        }


class _SSH:
    def __init__(self, *, existing_session: bool = False, process_visible: bool = True) -> None:
        self.existing_session = existing_session
        self.process_visible = process_visible
        self.commands: list[str] = []
        self.closed = False

    def exec(self, command: str, timeout: int = 30) -> tuple[str, int]:
        self.commands.append(command)
        if command.startswith("tmux has-session"):
            return "", 0 if self.existing_session else 1
        if command.startswith("tmux new-session"):
            self.existing_session = True
            return "", 0
        if command.startswith("for i in"):
            return "", 0 if self.process_visible else 1
        return "", 0

    def put(self, local: str, remote: str) -> None:
        raise AssertionError("training start must not upload artifacts")

    def close(self) -> None:
        self.closed = True


def test_versioned_training_executor_starts_without_killing_existing_sessions(monkeypatch) -> None:
    ssh = _SSH()
    monkeypatch.setattr(
        "autotuner.adapter.remote_deploy.from_ssh_json",
        lambda path: ssh,
    )

    result = VersionedTrainingStartExecutor("fake-ssh.json").start(_Plan())

    assert result.status == "started"
    assert result.handle_ref == "tmux:train-1"
    assert ssh.closed
    assert not any("kill-session" in command for command in ssh.commands)


@pytest.mark.parametrize(
    ("existing_session", "process_visible", "message"),
    [
        (True, True, "already exists"),
        (False, False, "did not become visible"),
    ],
)
def test_versioned_training_executor_returns_failed_result_without_kill(
    monkeypatch,
    existing_session: bool,
    process_visible: bool,
    message: str,
) -> None:
    ssh = _SSH(existing_session=existing_session, process_visible=process_visible)
    monkeypatch.setattr(
        "autotuner.adapter.remote_deploy.from_ssh_json",
        lambda path: ssh,
    )

    result = VersionedTrainingStartExecutor().start(_Plan())

    assert result.status == "failed"
    assert message in result.error
    assert ssh.closed
    assert not any("kill-session" in command for command in ssh.commands)


def test_versioned_training_executor_converts_ssh_setup_exception_to_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "autotuner.adapter.remote_deploy.from_ssh_json",
        lambda path: (_ for _ in ()).throw(RuntimeError("no credentials")),
    )

    result = VersionedTrainingStartExecutor().start(_Plan())

    assert result.status == "failed"
    assert "ssh setup failed" in result.error
