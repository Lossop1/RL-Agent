"""机器人知识源描述符（按 robot_id 索引）。

去 taili 化的关键:所有"机器人专属"的东西——奖励代码在哪个文件、奖励函数/配置类/门控清单
叫什么、资产 cfg 在哪——集中到这里一条一条描述,按 **激活机器人的 robot_id** 解析。
推导器逻辑本身通用、不含任何 taili 字样;加一个新机器人 = 在 _ROBOTS 里加一条,不动推导器。

激活机器人从现有的 robot_profile 机制取(get_robot_profile().id),不是写死 taili。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


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


_ROBOTS: Dict[str, RobotSources] = {
    "robot.taili": RobotSources(
        robot_id="robot.taili",
        reward_file="autotuner/taili_core/taili_reward.py",
        reward_func="compute_reward_components",
        reward_cfg_class="RewardConfig",
        reward_gate_const="REWARD_GROUP_GATES",
        reward_group_const="_REWARD_GROUP_OF",
        env_reward_file="autotuner/blind_locomotion/blind_tp_env.py",
        asset_file="autotuner/blind_locomotion/assets/taili.py",
        curriculum_file="autotuner/blind_locomotion/taili_amp_env_cfg.py",
        asset_cfg_call="ArticulationCfg",
        asset_constant_files=(("geometry", "autotuner/taili_core/taili_geometry.py"),),
    ),
}


def active_robot_id() -> str:
    """当前激活机器人的 id。取自现有 robot_profile 机制(不写死 taili);
    将来多机器人时,这里按 framework/config-set 解析。"""
    try:
        from ..robot_profile import get_robot_profile
        return get_robot_profile().id
    except Exception:  # noqa: BLE001
        return "robot.taili"


def get_robot_sources(robot_id: Optional[str] = None) -> RobotSources:
    rid = robot_id or active_robot_id()
    src = _ROBOTS.get(rid)
    if src is None:
        # 未登记的机器人:回退到默认,并留待补描述符(不假装能推导未知机器人)。
        return _ROBOTS["robot.taili"]
    return src
