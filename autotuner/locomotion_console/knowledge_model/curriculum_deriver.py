"""从 taili_amp_env_cfg.py 推导阶段门控阈值等课程常量（纯 AST,不 import → 免 isaaclab）。

替掉 curriculum_facts.py 里手抄的 GATE_DEFAULTS:阈值从代码推导(带 SourceRef 对齐),
当前 run 的真实门槛由 store 从 effective_config 现读、配对呈现(durable 默认 + live 现值)。
"""
from __future__ import annotations

import ast
import re
from typing import Any, Dict, List

from .reward_deriver import _ref
from .robot_deriver import _func_name, _lit
from .schema import GateThreshold

# 门控/课程常量名:阶段门控、惩罚渐入、阶段间隔、回退保护。
_GATE_NAME_RE = re.compile(r"^(phase_gate_\w+|penalty_ramp_intervals|phase_intervals|regress_\w+)$")

# 轻量人读标签(仅展示,非事实;取不到就用常量名)。
_LABELS = {
    "phase_gate_prog_0": "phi0→phi1 四方向进展最小值门槛",
    "phase_gate_prog_1": "phi1→phi2 进展门槛",
    "phase_gate_prog_2": "phi2→phi3 进展门槛",
    "phase_gate_terrain_2": "phi2→phi3 平均地形等级门槛",
    "phase_gate_fall_2": "phi2→phi3 摔倒率门槛",
    "penalty_ramp_intervals": "质量惩罚 0→1 渐入的间隔数",
    "phase_intervals": "推进前需连续满足门控的间隔数",
    "terrain_levels": "地形难度等级数(num_rows)",
}


def derive_gate_thresholds(source_text: str, rel_path: str) -> Dict[str, Any]:
    """返回 {gates: [GateThreshold], errors: [...]}。纯函数,测试可喂改过的文本。"""
    result: Dict[str, Any] = {"gates": [], "errors": []}
    try:
        tree = ast.parse(source_text)
    except SyntaxError as exc:
        result["errors"].append(f"parse: {exc}")
        return result

    gates: List[GateThreshold] = []
    seen: set = set()

    # 类体/模块级 `名 = 常量` 的门控阈值。
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in seen or not _GATE_NAME_RE.match(name):
                continue
            val = _lit(node.value)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                seen.add(name)
                gates.append(GateThreshold(
                    name=name, label=_LABELS.get(name, name), code_default=float(val),
                    source=_ref(source_text, node, rel_path, symbol=name)))

    # 地形等级数:TerrainGeneratorCfg(num_rows=...)。
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _func_name(node) == "TerrainGeneratorCfg":
            for kw in node.keywords:
                if kw.arg == "num_rows" and "terrain_levels" not in seen:
                    v = _lit(kw.value)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        seen.add("terrain_levels")
                        gates.append(GateThreshold(
                            name="terrain_levels", label=_LABELS["terrain_levels"], code_default=float(v),
                            source=_ref(source_text, kw.value, rel_path, symbol="num_rows")))

    result["gates"] = gates
    return result
