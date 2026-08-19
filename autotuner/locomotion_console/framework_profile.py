"""产品合同投影出的框架运行档案。

框架档案描述“如何找到并运行一类训练”，但不拥有任何机器人名称、路径
或任务实现。具体值来自产品清单的 ``framework.profiles``；本模块只负责
把它转换成控制台需要的不可变对象，并保留旧 API 作为兼容入口。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Literal, Mapping

from autotuner.product import ProductManifestError, resolve_product_runtime


FrameworkStatus = Literal["draft", "validated", "reference", "legacy"]


@dataclass(frozen=True)
class FrameworkProfile:
    product_id: str
    id: str
    label: str
    status: FrameworkStatus
    experiment: str
    task_id: str
    diagnostic_task: str
    run_globs: tuple[str, ...]
    checkpoint_roots: tuple[str, ...]
    note: str = ""
    required_robot_capabilities: tuple[str, ...] = field(default_factory=tuple)
    # 每个档案可以覆盖远程动作命令；命令中的占位符由调用方替换。
    physeval_cmd: str = ""
    physeval_log: str = ""
    resume_cmd: str = ""
    train_running_probe: str = ""
    kill_cmd: str = ""


def _tuple_strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _profile_from_mapping(product_id: str, identifier: str, data: Mapping[str, Any]) -> FrameworkProfile:
    commands = data.get("commands") if isinstance(data.get("commands"), Mapping) else {}
    status = str(data.get("status") or "draft").strip().lower()
    if status not in {"draft", "validated", "reference", "legacy"}:
        status = "draft"
    return FrameworkProfile(
        product_id=product_id,
        id=identifier,
        label=str(data.get("label") or identifier),
        status=status,  # type: ignore[arg-type]
        experiment=str(data.get("experiment") or ""),
        task_id=str(data.get("task_id") or ""),
        diagnostic_task=str(data.get("diagnostic_task") or data.get("task_id") or ""),
        run_globs=_tuple_strings(data.get("run_globs")),
        checkpoint_roots=_tuple_strings(data.get("checkpoint_roots")),
        note=str(data.get("note") or ""),
        required_robot_capabilities=_tuple_strings(data.get("required_robot_capabilities")),
        physeval_cmd=str(commands.get("physeval") or data.get("physeval_cmd") or ""),
        physeval_log=str(commands.get("physeval_log") or data.get("physeval_log") or ""),
        resume_cmd=str(commands.get("resume") or data.get("resume_cmd") or ""),
        train_running_probe=str(commands.get("train_running_probe") or data.get("train_running_probe") or ""),
        kill_cmd=str(commands.get("kill") or data.get("kill_cmd") or ""),
    )


def _runtime(product_id: str | None = None):
    return resolve_product_runtime(product_id, check_files=False)


def default_framework_id(product_id: str | None = None) -> str:
    """取得产品声明的默认框架；没有声明时明确返回空字符串。"""
    explicit = os.environ.get("LOCOMOTION_CONSOLE_FRAMEWORK", "").strip()
    if explicit:
        return explicit
    return _runtime(product_id).default_framework_id()


# 兼容旧的 import；新代码应调用 default_framework_id()，避免模块导入时冻结产品选择。
DEFAULT_FRAMEWORK_ID = os.environ.get("LOCOMOTION_CONSOLE_FRAMEWORK", "").strip()


def list_framework_profiles(product_id: str | None = None) -> list[FrameworkProfile]:
    view = _runtime(product_id)
    return [
        _profile_from_mapping(view.product_id, identifier, data)
        for identifier, data in view.framework_profiles().items()
    ]


def get_framework_profile(
    framework_id: str | None = None,
    *,
    product_id: str | None = None,
) -> FrameworkProfile:
    view = _runtime(product_id)
    key = (framework_id or "").strip() or view.default_framework_id()
    profiles = view.framework_profiles()
    if not key:
        raise ProductManifestError(
            f"product {view.product_id!r} does not declare a default framework profile"
        )
    try:
        return _profile_from_mapping(view.product_id, key, profiles[key])
    except KeyError as exc:
        known = ", ".join(sorted(profiles)) or "<none>"
        raise ValueError(
            f"Unknown framework profile {key!r} for product {view.product_id!r}. Known: {known}"
        ) from exc


def merge_box_profile(profile, framework: FrameworkProfile):
    """将发现到的远程实例与合同中的框架结构合并。

    BoxProfile 只保存机器上发现的动态细节；产品合同决定任务、运行目录和
    动作命令，避免探测结果把 teacher 或另一产品误认成当前任务。
    """
    update = {
        "experiment": framework.experiment or profile.experiment,
        "runs_glob": framework.run_globs[0] if framework.run_globs else profile.runs_glob,
        "task_id": framework.task_id or profile.task_id,
    }
    if framework.physeval_cmd:
        update["physeval_cmd"] = framework.physeval_cmd
    if framework.physeval_log:
        update["physeval_log"] = framework.physeval_log
    if framework.resume_cmd:
        update["resume_cmd"] = framework.resume_cmd
    if framework.train_running_probe:
        update["train_running_probe"] = framework.train_running_probe
    if framework.kill_cmd:
        update["kill_cmd"] = framework.kill_cmd
    return profile.model_copy(update=update)


__all__ = [
    "DEFAULT_FRAMEWORK_ID",
    "FrameworkProfile",
    "default_framework_id",
    "get_framework_profile",
    "list_framework_profiles",
    "merge_box_profile",
]
