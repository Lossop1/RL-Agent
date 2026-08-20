"""产品无关的远程训练启动边界。

启动器只消费产品层已经确定性生成的启动计划协议，不接受自由 shell、LLM 文本或
产品源码路径。已有 tmux 会话默认拒绝，避免一次候选启动隐式终止正在运行的训练。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import shlex
from typing import Any, Literal, Mapping, Protocol

from .deployment import RemoteTransport


TRAINING_START_SCHEMA = "rl-agent.training-start/v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class TrainingStartPlan(Protocol):
    run_id: str
    run_dir: str
    tmux_session: str
    process_pattern: str

    def prepare_command(self) -> str: ...

    def terminal_command(self) -> str: ...

    def marker_command(self) -> str: ...

    def to_dict(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class TrainingStartResult:
    status: Literal["started", "failed"]
    run_id: str
    run_dir: str
    handle_ref: str = ""
    process_pattern: str = ""
    trace: tuple[Mapping[str, Any], ...] = ()
    error: str = ""
    schema_version: str = TRAINING_START_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["trace"] = [dict(item) for item in self.trace]
        return value


def _exec(remote: RemoteTransport, command: str, *, timeout: int = 30) -> tuple[str, int]:
    result = remote.exec(command, timeout=timeout)
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("remote training transport returned an invalid result")
    output, status = result
    if isinstance(status, bool) or not isinstance(status, int):
        raise RuntimeError(
            "remote training transport must return (output, return_code); "
            "wrap legacy SSH transports with RemoteSSHTransportAdapter"
        )
    return str(output), status


def _validate_plan(plan: TrainingStartPlan) -> None:
    if not _SAFE_ID.fullmatch(str(plan.run_id or "")):
        raise ValueError(f"unsafe training run id: {plan.run_id!r}")
    if not _SAFE_ID.fullmatch(str(plan.tmux_session or "")):
        raise ValueError(f"unsafe training tmux session: {plan.tmux_session!r}")
    if not str(plan.run_dir or "").startswith("/") or ".." in str(plan.run_dir).split("/"):
        raise ValueError(f"unsafe training run directory: {plan.run_dir!r}")
    pattern = str(plan.process_pattern or "")
    if not pattern or any(char in pattern for char in ("\x00", "\r", "\n")):
        raise ValueError("training process pattern is empty or contains control characters")
    data = plan.to_dict()
    if str(data.get("run_id") or "") != plan.run_id:
        raise ValueError("training plan mapping changes run identity")


class VersionedRemoteTrainingStarter:
    """在远程 tmux 中启动一个已验证计划，并返回可持久化回执。"""

    def __init__(self, remote: RemoteTransport) -> None:
        self.remote = remote

    def start(self, plan: TrainingStartPlan) -> TrainingStartResult:
        _validate_plan(plan)
        trace: list[Mapping[str, Any]] = []
        session = shlex.quote(plan.tmux_session)
        try:
            output, status = _exec(self.remote, plan.prepare_command(), timeout=30)
            trace.append({"event": "prepare", "status": status, "output": output[-500:]})
            if status != 0:
                raise RuntimeError("remote run directory preparation failed")
            _, status = _exec(
                self.remote,
                f"tmux has-session -t {session} 2>/dev/null",
                timeout=10,
            )
            if status == 0:
                raise RuntimeError(
                    f"training tmux session already exists: {plan.tmux_session}"
                )
            output, status = _exec(
                self.remote,
                f"tmux new-session -d -s {session}",
                timeout=15,
            )
            trace.append({"event": "session", "status": status, "output": output[-500:]})
            if status != 0:
                raise RuntimeError("training tmux session creation failed")
            command = shlex.quote(plan.terminal_command())
            output, status = _exec(
                self.remote,
                f"tmux send-keys -t {session} {command} Enter",
                timeout=15,
            )
            trace.append({"event": "command", "status": status, "output": output[-500:]})
            if status != 0:
                raise RuntimeError("training command injection failed")
            output, status = _exec(self.remote, plan.marker_command(), timeout=15)
            trace.append({"event": "marker", "status": status, "output": output[-500:]})
            if status != 0:
                raise RuntimeError("training start marker could not be written")
            pattern = shlex.quote(plan.process_pattern)
            probe = (
                "for i in 1 2 3 4 5; do "
                f"pgrep -f -- {pattern} >/dev/null && exit 0; "
                "sleep 1; done; exit 1"
            )
            output, status = _exec(self.remote, probe, timeout=10)
            trace.append({"event": "process", "status": status, "output": output[-500:]})
            if status != 0:
                raise RuntimeError("training process did not become visible after launch")
            return TrainingStartResult(
                status="started",
                run_id=plan.run_id,
                run_dir=plan.run_dir,
                handle_ref=f"tmux:{plan.tmux_session}",
                process_pattern=plan.process_pattern,
                trace=tuple(trace),
            )
        except Exception as exc:
            return TrainingStartResult(
                status="failed",
                run_id=plan.run_id,
                run_dir=plan.run_dir,
                handle_ref=f"tmux:{plan.tmux_session}",
                process_pattern=plan.process_pattern,
                trace=tuple(trace),
                error=f"{type(exc).__name__}: {exc}",
            )


__all__ = [
    "TRAINING_START_SCHEMA",
    "TrainingStartPlan",
    "TrainingStartResult",
    "VersionedRemoteTrainingStarter",
]
