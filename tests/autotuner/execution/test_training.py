from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from autotuner.execution.training import VersionedRemoteTrainingStarter


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
        }


class _Remote:
    def __init__(self, *, existing_session: bool = False, process_visible: bool = True) -> None:
        self.existing_session = existing_session
        self.process_visible = process_visible
        self.commands: list[str] = []

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
        raise AssertionError("training starter must not upload artifacts")


def test_starter_rejects_existing_tmux_session_without_killing_it() -> None:
    remote = _Remote(existing_session=True)

    result = VersionedRemoteTrainingStarter(remote).start(_Plan())

    assert result.status == "failed"
    assert "already exists" in result.error
    assert not any(command.startswith("tmux new-session") for command in remote.commands)


def test_starter_records_marker_and_verifies_process() -> None:
    remote = _Remote()

    result = VersionedRemoteTrainingStarter(remote).start(_Plan())

    assert result.status == "started"
    assert result.handle_ref == "tmux:train-1"
    assert [item["event"] for item in result.trace] == [
        "prepare",
        "session",
        "command",
        "marker",
        "process",
    ]
    assert any("console_start.json" in command for command in remote.commands)


def test_starter_does_not_treat_legacy_stderr_tuple_as_success() -> None:
    class LegacyRemote(_Remote):
        def exec(self, command: str, timeout: int = 30):
            self.commands.append(command)
            return "output", "warning on stderr"

    result = VersionedRemoteTrainingStarter(LegacyRemote()).start(_Plan())

    assert result.status == "failed"
    assert "return_code" in result.error
