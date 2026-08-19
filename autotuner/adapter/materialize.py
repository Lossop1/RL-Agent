"""把适配结果写入产品声明的本地配置副本。"""
from __future__ import annotations

import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Edit:
    file: str
    field: str
    old: str
    new: str
    changed: bool
    kind: str
    applied: bool = True


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _fmt(value: Any) -> str:
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(_fmt(item) for item in value) + ")"
    if isinstance(value, float):
        return repr(round(value, 6))
    return str(value)


def _nums(value: Any) -> tuple[float, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(float(item) for item in value)
    return tuple(float(item) for item in re.findall(r"-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?", str(value)))


def _numeric_changed(old_value: str, new_value: Any) -> bool:
    old_numbers, new_numbers = _nums(old_value), _nums(new_value)
    if len(old_numbers) != len(new_numbers):
        return True
    return any(abs(old - new) > 1e-6 for old, new in zip(old_numbers, new_numbers))


def _edit_field(text: str, field: str, new_value: str) -> tuple[str, str | None]:
    pattern = re.compile(
        rf"^(?P<pre>\s*{re.escape(field)}\s*(:[^=\n]*)?=\s*)(?P<value>[^#\n]*?)(?P<post>\s*(#.*)?)$",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if match is None:
        return text, None
    old = match.group("value").strip()
    separator = " " if not match.group("post").startswith(" ") else ""
    result = text[: match.start("value")] + new_value + separator + text[match.end("value") :]
    return result, old


def _edit_asset_dict_value(
    text: str,
    block: str,
    joint_pattern: str,
    new_value: str,
) -> tuple[str, str | None]:
    block_match = re.search(rf"{re.escape(block)}\s*=\s*\{{(.*?)\}}", text, re.DOTALL)
    if block_match is None:
        return text, None
    body = block_match.group(1)
    joint_match = re.search(
        rf'("{re.escape(joint_pattern)}"\s*:\s*)(-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)',
        body,
    )
    if joint_match is None:
        return text, None
    old = joint_match.group(2)
    updated = body[: joint_match.start(2)] + new_value + body[joint_match.end(2) :]
    return text[: block_match.start(1)] + updated + text[block_match.end(1) :], old


def materialize(
    adapted: Mapping[str, Any],
    env_cfg_src: str,
    asset_src: str,
    work_dir: str,
    *,
    spec: Mapping[str, Any],
) -> list[Edit]:
    """按产品字段映射写入副本，返回包含未命中项的完整审计记录。"""
    env_fields = _mapping(spec.get("env_fields"), "materialization.env_fields")
    actuator_blocks = _mapping(spec.get("actuator_blocks"), "materialization.actuator_blocks")
    joint_patterns = _mapping(
        spec.get("actuator_joint_patterns"),
        "materialization.actuator_joint_patterns",
    )
    baseline = _mapping(spec.get("advisory_baseline", {}), "materialization.advisory_baseline")
    advisory_roles = spec.get("advisory_roles", ())
    if not isinstance(advisory_roles, (list, tuple)):
        raise ValueError("materialization.advisory_roles must be a list")

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    env_destination = work / Path(env_cfg_src).name
    asset_destination = work / Path(asset_src).name
    shutil.copy2(env_cfg_src, env_destination)
    shutil.copy2(asset_src, asset_destination)
    edits: list[Edit] = []

    reward_thresholds = _mapping(adapted.get("reward_thresholds"), "adapted.reward_thresholds")
    env_text = env_destination.read_text(encoding="utf-8")
    for key, raw_field in env_fields.items():
        if key not in reward_thresholds:
            continue
        field = str(raw_field)
        value = reward_thresholds[key]
        rendered = _fmt(value)
        env_text, old = _edit_field(env_text, field, rendered)
        if old is None:
            edits.append(Edit(env_destination.name, field, "(not found)", rendered, False, "field", False))
            continue
        edits.append(
            Edit(
                env_destination.name,
                field,
                old,
                rendered,
                _numeric_changed(old, value),
                "tuple" if isinstance(value, (tuple, list)) else "scalar",
            )
        )
    env_destination.write_text(env_text, encoding="utf-8")

    actuator = _mapping(adapted.get("actuator"), "adapted.actuator")
    asset_text = asset_destination.read_text(encoding="utf-8")
    for block, source_name in actuator_blocks.items():
        source = _mapping(actuator.get(str(source_name)), f"adapted.actuator.{source_name}")
        for role, raw_pattern in joint_patterns.items():
            if role not in source:
                continue
            value = source[role]
            rendered = _fmt(value)
            asset_text, old = _edit_asset_dict_value(asset_text, str(block), str(raw_pattern), rendered)
            if old is None:
                edits.append(
                    Edit(asset_destination.name, f"{block}[{role}]", "(not found)", rendered, False, "asset_dict", False)
                )
                continue
            edits.append(
                Edit(
                    asset_destination.name,
                    f"{block}[{role}]",
                    old,
                    rendered,
                    _numeric_changed(old, value),
                    "asset_dict",
                )
            )
    asset_destination.write_text(asset_text, encoding="utf-8")

    for role in (str(item) for item in advisory_roles):
        for field in ("Kp", "Kd"):
            values = _mapping(actuator.get(field), f"adapted.actuator.{field}")
            if role not in values:
                continue
            edits.append(
                Edit(
                    asset_destination.name,
                    f"{field}[{role}]",
                    str(baseline.get(field, "undeclared")),
                    str(values[role]),
                    True,
                    "flagged_advisory",
                    False,
                )
            )
    return edits


def roundtrip_verify(
    adapted: Mapping[str, Any],
    env_materialized: str,
    *,
    spec: Mapping[str, Any],
) -> list[str]:
    """重读物化副本，确认合同声明的奖励字段已准确写入。"""
    env_fields = _mapping(spec.get("env_fields"), "materialization.env_fields")
    thresholds = _mapping(adapted.get("reward_thresholds"), "adapted.reward_thresholds")
    text = Path(env_materialized).read_text(encoding="utf-8")
    bad: list[str] = []
    for key, raw_field in env_fields.items():
        if key not in thresholds:
            continue
        field = str(raw_field)
        match = re.search(rf"^\s*{re.escape(field)}\s*(:[^=\n]*)?=\s*([^#\n]*)", text, re.MULTILINE)
        actual = re.sub(r"\s", "", match.group(2)) if match else "(missing)"
        expected = re.sub(r"\s", "", _fmt(thresholds[key]))
        if actual != expected:
            bad.append(f"{field}: got {actual} want {expected}")
    return bad


__all__ = ["Edit", "materialize", "roundtrip_verify"]
