"""从产品清单解析机器人知识源描述符。

推导器只消费 ``RobotSources`` 接口；奖励、资产和课程文件的位置由产品
清单声明，未知产品不会静默套用另一个机器人的描述。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from autotuner.product import ProductManifestError, get_product


@dataclass(frozen=True)
class RobotSources:
    """一个机器人的知识源:去哪些文件、找哪些符号,才能推导它的奖励/本体。"""
    robot_id: str
    # 奖励域
    reward_file: str          # 奖励函数所在文件(仓库相对路径,须在 code_knowledge allowlist 内)
    reward_func: str          # 构建 comp 字典的函数名
    reward_cfg_class: str     # 权重默认所在的 dataclass 名
    reward_gate_const: str    # 门控 key 清单的模块级常量名
    reward_group_const: str   # 奖励项分组表的模块级常量名(用于覆盖对账)
    env_reward_file: str      # 环境专属奖励项的次要覆盖源
    # 本体域
    asset_file: str           # 机器人资产 cfg 文件
    # 机制域(课程/门控)
    curriculum_file: str      # 阶段门控阈值等课程常量所在文件
    asset_cfg_call: str = "ArticulationCfg"
    # 资产表达式引用的纯常量源码：(资产中的命名空间别名, 仓库相对路径)。
    asset_constant_files: Tuple[Tuple[str, str], ...] = ()


def active_robot_id() -> str:
    """返回产品注册表当前选择的机器人 id。"""
    return get_product().robot.id


def _knowledge_value(product, key: str, *source_keys: str) -> str:
    value = product.knowledge.get(key)
    if value not in (None, ""):
        return str(value)
    for source_key in source_keys:
        value = product.sources.get(source_key)
        if value not in (None, ""):
            return str(value)
    return ""


def _constant_files(product) -> Tuple[Tuple[str, str], ...]:
    raw = product.knowledge.get("asset_constant_files", {})
    if isinstance(raw, dict):
        return tuple((str(alias), str(path)) for alias, path in raw.items())
    if isinstance(raw, (list, tuple)):
        result: list[Tuple[str, str]] = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                result.append((str(item[0]), str(item[1])))
        return tuple(result)
    return ()


def _sources_from_product(product) -> RobotSources:
    knowledge = {
        "reward_file": _knowledge_value(product, "reward_file", "reward"),
        "reward_func": _knowledge_value(product, "reward_func"),
        "reward_cfg_class": _knowledge_value(product, "reward_cfg_class"),
        "reward_gate_const": _knowledge_value(product, "reward_gate_const"),
        "reward_group_const": _knowledge_value(product, "reward_group_const"),
        "env_reward_file": _knowledge_value(product, "env_reward_file", "task_env"),
        "asset_file": _knowledge_value(product, "asset_file", "asset_config"),
        "curriculum_file": _knowledge_value(product, "curriculum_file", "task_config"),
        "asset_cfg_call": _knowledge_value(product, "asset_cfg_call") or "ArticulationCfg",
    }
    missing = [key for key, value in knowledge.items() if key != "asset_cfg_call" and not value]
    if missing:
        raise ProductManifestError(
            f"product {product.product_id!r} has no knowledge declarations: {', '.join(missing)}"
        )
    return RobotSources(
        robot_id=product.robot.id,
        reward_file=knowledge["reward_file"],
        reward_func=knowledge["reward_func"],
        reward_cfg_class=knowledge["reward_cfg_class"],
        reward_gate_const=knowledge["reward_gate_const"],
        reward_group_const=knowledge["reward_group_const"],
        env_reward_file=knowledge["env_reward_file"],
        asset_file=knowledge["asset_file"],
        curriculum_file=knowledge["curriculum_file"],
        asset_cfg_call=knowledge["asset_cfg_call"],
        asset_constant_files=_constant_files(product),
    )


def get_robot_sources(robot_id: Optional[str] = None) -> RobotSources:
    product = get_product(robot_id)
    return _sources_from_product(product)
