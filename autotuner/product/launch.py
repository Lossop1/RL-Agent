"""产品无关的训练启动计划。

本模块把产品合同和一次启动请求编译为可审计的 argv、环境变量与远程
操作。它不连接 SSH，也不理解奖励、课程或机器人结构；控制台和其他
执行器只负责执行计划，因此不会再各自拼接某个产品的 shell 命令。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
import shlex
from typing import Any, Mapping

from .adapter import ProductAdapterSpec, resolve_product_adapter


LAUNCH_PLAN_SCHEMA = "rl-agent.training-launch/v1"
SUPPORTED_LAUNCHER_PROTOCOL = "rl-agent.payload-train/v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]*$")
_SAFE_ENV = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SAFE_PATH = re.compile(r"/[A-Za-z0-9_.:/-]+")


class TrainingLaunchError(ValueError):
    """训练启动请求或产品启动声明不满足安全合同。"""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TrainingLaunchError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _positive(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TrainingLaunchError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise TrainingLaunchError(f"{name} must be a positive integer")
    return result


def _safe_path(value: Any, name: str, *, required: bool = True) -> str:
    result = str(value or "").strip().rstrip("/")
    if not result and not required:
        return ""
    if not result.startswith("/") or ".." in result.split("/") or not _SAFE_PATH.fullmatch(result):
        raise TrainingLaunchError(f"{name} must be a safe absolute POSIX path: {result!r}")
    return result


def _safe_id(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not _SAFE_ID.fullmatch(result):
        raise TrainingLaunchError(f"{name} is not a safe id: {result!r}")
    return result


def _safe_token(value: Any, name: str) -> str:
    result = str(value)
    if "\x00" in result or "\n" in result or "\r" in result:
        raise TrainingLaunchError(f"{name} contains a control character")
    return result


def _safe_ref(value: Any, name: str, *, required: bool = False) -> str:
    result = str(value or "").strip()
    if not result and not required:
        return ""
    if not _SAFE_REF.fullmatch(result) or ".." in result.split("/"):
        raise TrainingLaunchError(f"{name} is not a safe reference: {result!r}")
    return result


@dataclass(frozen=True)
class TrainingLaunchRequest:
    """一次训练启动所需的运行态输入；不含产品实现细节。"""

    payload_root: str
    run_id: str
    checkpoint: str = ""
    checkpoint_asset_ref: str = ""
    source_run: str = ""
    remote_boot_id: str = ""
    resume: bool = False
    total_steps: int | None = None
    telemetry_interval: int | None = None
    num_envs: int | None = None
    resume_state: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingLaunchRequest":
        data = _mapping(value, "launch_request")
        return cls(
            payload_root=str(data.get("payload_root") or ""),
            run_id=str(data.get("run_id") or ""),
            checkpoint=str(data.get("checkpoint") or ""),
            checkpoint_asset_ref=str(data.get("checkpoint_asset_ref") or ""),
            source_run=str(data.get("source_run") or ""),
            remote_boot_id=str(data.get("remote_boot_id") or ""),
            resume=bool(data.get("resume", False)),
            total_steps=data.get("total_steps"),
            telemetry_interval=data.get("telemetry_interval"),
            num_envs=data.get("num_envs"),
            resume_state=_mapping(data.get("resume_state"), "launch_request.resume_state"),
        )


@dataclass(frozen=True)
class TrainingLaunchPlan:
    """可验证、可记录、但自身无副作用的远程启动计划。"""

    product_id: str
    product_contract_digest: str
    payload_root: str
    run_id: str
    run_dir: str
    source_run: str
    checkpoint: str
    resume: bool
    tmux_session: str
    process_pattern: str
    environment: Mapping[str, str]
    argv: tuple[str, ...]
    remote_boot_id: str = ""
    checkpoint_asset_ref: str = ""
    schema_version: str = LAUNCH_PLAN_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["argv"] = list(self.argv)
        return data

    def prepare_command(self) -> str:
        return f"mkdir -p {shlex.quote(self.run_dir)}"

    def training_command(self) -> str:
        exports = " ".join(
            f"export {key}={shlex.quote(value)};" for key, value in sorted(self.environment.items())
        )
        boot_marker = ""
        if self.remote_boot_id:
            marker = f"{self.run_dir}/remote_boot_id.txt"
            boot_marker = (
                f"printf '%s\\n' {shlex.quote(self.remote_boot_id)} > {shlex.quote(marker)}; "
            )
        command = " ".join(shlex.quote(item) for item in self.argv)
        return (
            "set -euo pipefail; "
            f"cd {shlex.quote(self.payload_root)}; "
            f"{exports} "
            f"{boot_marker}"
            f"exec {command}"
        )

    def terminal_command(self) -> str:
        inner = self.training_command()
        return (
            f"bash -lc {shlex.quote(inner)}; "
            "rc=$?; "
            "printf '\\n[rl-agent] training command exited with rc=%s.\\n' \"$rc\"; "
            "printf '[rl-agent] Full stdout/stderr is in the run console log.\\n'; "
            "printf '[rl-agent] Start a new run from the control plane or exit this shell manually.\\n'; "
            "exec bash -l"
        )

    def start_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "product_id": self.product_id,
            "product_contract_digest": self.product_contract_digest,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "payload": self.payload_root,
            "source_run": self.source_run,
            "checkpoint": self.checkpoint,
            "checkpoint_asset_ref": self.checkpoint_asset_ref,
            "tmux_session": self.tmux_session,
            "remote_boot_id": self.remote_boot_id,
            "resume": self.resume,
            "environment": dict(self.environment),
            "argv": list(self.argv),
        }

    def marker_command(self) -> str:
        payload = json.dumps(self.start_record(), ensure_ascii=False, sort_keys=True)
        target = f"{self.run_dir}/console_start.json"
        return f"printf '%s\\n' {shlex.quote(payload)} > {shlex.quote(target)}"

def _format_train_args(values: Any, *, num_envs: int, run_id: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise TrainingLaunchError("deployment.training_launch.train_args must be a list")
    context = {"num_envs": str(num_envs), "run_id": run_id}
    result: list[str] = []
    for index, value in enumerate(values):
        token = _safe_token(value, f"deployment.training_launch.train_args[{index}]")
        try:
            result.append(token.format_map(context))
        except KeyError as exc:
            raise TrainingLaunchError(f"unknown training argument placeholder: {exc.args[0]}") from exc
    return tuple(result)


def _environment(
    launch: Mapping[str, Any],
    request: TrainingLaunchRequest,
) -> dict[str, str]:
    declared = _mapping(launch.get("environment"), "deployment.training_launch.environment")
    result: dict[str, str] = {}
    for key, value in declared.items():
        if not _SAFE_ENV.fullmatch(key):
            raise TrainingLaunchError(f"unsafe environment variable name: {key!r}")
        result[key] = _safe_token(value, f"environment.{key}")

    state_environment = _mapping(
        launch.get("resume_state_environment"),
        "deployment.training_launch.resume_state_environment",
    )
    unknown = sorted(set(request.resume_state) - set(state_environment))
    if unknown:
        raise TrainingLaunchError("resume state has no declared environment mapping: " + ", ".join(unknown))
    for state_name, value in request.resume_state.items():
        env_name = str(state_environment[state_name])
        if not _SAFE_ENV.fullmatch(env_name):
            raise TrainingLaunchError(f"unsafe resume environment variable name: {env_name!r}")
        result[env_name] = _safe_token(value, f"resume_state.{state_name}")
    return result


def build_training_launch_plan(
    contract: Any,
    request: TrainingLaunchRequest | Mapping[str, Any],
) -> TrainingLaunchPlan:
    """把产品合同与运行输入编译为标准 payload 启动计划。"""
    adapter: ProductAdapterSpec = resolve_product_adapter(contract)
    launch_request = (
        request if isinstance(request, TrainingLaunchRequest) else TrainingLaunchRequest.from_mapping(request)
    )
    launch = _mapping(adapter.deployment.get("training_launch"), "deployment.training_launch")
    protocol = str(adapter.deployment.get("launcher_protocol") or "").strip()
    if protocol != SUPPORTED_LAUNCHER_PROTOCOL:
        raise TrainingLaunchError(f"unsupported payload launcher protocol: {protocol!r}")

    payload_root = _safe_path(launch_request.payload_root, "launch_request.payload_root")
    run_id = _safe_id(launch_request.run_id, "launch_request.run_id")
    checkpoint = _safe_path(launch_request.checkpoint, "launch_request.checkpoint", required=False)
    checkpoint_asset_ref = _safe_ref(
        launch_request.checkpoint_asset_ref,
        "launch_request.checkpoint_asset_ref",
    )
    source_run = _safe_path(launch_request.source_run, "launch_request.source_run", required=False)
    if launch_request.resume and not checkpoint:
        raise TrainingLaunchError("resume launch requires a checkpoint")
    if not launch_request.resume and checkpoint:
        raise TrainingLaunchError("fresh launch cannot carry a checkpoint")
    if checkpoint_asset_ref and not checkpoint:
        raise TrainingLaunchError("checkpoint_asset_ref requires a checkpoint")

    total_steps = _positive(
        launch_request.total_steps if launch_request.total_steps is not None else launch.get("total_steps"),
        "total_steps",
    )
    telemetry_interval = _positive(
        launch_request.telemetry_interval
        if launch_request.telemetry_interval is not None
        else launch.get("telemetry_interval"),
        "telemetry_interval",
    )
    num_envs = _positive(
        launch_request.num_envs if launch_request.num_envs is not None else launch.get("num_envs"),
        "num_envs",
    )
    run_dir = f"{adapter.runs_root}/{run_id}"
    launcher_module = f"{adapter.runtime_package}.{adapter.payload_file('launcher')[:-3]}"
    argv = [
        adapter.runtime_python,
        "-m",
        launcher_module,
        "--python",
        adapter.runtime_python,
        "--data-root",
        adapter.data_root,
        "--run-root",
        adapter.runs_root,
        "--run-id",
        run_id,
        "--total-steps",
        str(total_steps),
        "--telemetry-interval",
        str(telemetry_interval),
    ]
    task_id = str(launch.get("task_id") or "").strip()
    if task_id:
        argv.extend(("--task", _safe_token(task_id, "deployment.training_launch.task_id")))
    if checkpoint:
        argv.extend(("--checkpoint", checkpoint))
    argv.append("--headless" if bool(launch.get("headless", True)) else "--no-headless")
    train_args = _format_train_args(launch.get("train_args"), num_envs=num_envs, run_id=run_id)
    if train_args:
        argv.extend(("--", *train_args))

    session = _safe_id(launch.get("tmux_session", "rl_train"), "deployment.training_launch.tmux_session")
    return TrainingLaunchPlan(
        product_id=adapter.product_id,
        product_contract_digest=adapter.contract_digest,
        payload_root=payload_root,
        run_id=run_id,
        run_dir=run_dir,
        source_run=source_run,
        checkpoint=checkpoint,
        resume=bool(launch_request.resume),
        tmux_session=session,
        process_pattern=adapter.process_pattern(),
        environment=_environment(launch, launch_request),
        argv=tuple(argv),
        remote_boot_id=_safe_token(launch_request.remote_boot_id, "launch_request.remote_boot_id"),
        checkpoint_asset_ref=checkpoint_asset_ref,
    )


def build_training_kill_command(contract: Any) -> tuple[str, str, str]:
    """返回产品 id、tmux 会话和只匹配该产品训练进程的停止命令。"""
    adapter = resolve_product_adapter(contract)
    launch = _mapping(adapter.deployment.get("training_launch"), "deployment.training_launch")
    session = _safe_id(
        launch.get("tmux_session", "rl_train"),
        "deployment.training_launch.tmux_session",
    )
    pattern = adapter.process_pattern()
    command = "bash -lc " + shlex.quote(
        f"tmux kill-session -t {shlex.quote(session)} 2>/dev/null || true; "
        f"pkill -f {shlex.quote(pattern)} 2>/dev/null || true; "
        "echo ok"
    )
    return adapter.product_id, session, command


__all__ = [
    "LAUNCH_PLAN_SCHEMA",
    "SUPPORTED_LAUNCHER_PROTOCOL",
    "TrainingLaunchError",
    "TrainingLaunchPlan",
    "TrainingLaunchRequest",
    "build_training_kill_command",
    "build_training_launch_plan",
]
