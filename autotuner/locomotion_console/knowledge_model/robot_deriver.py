"""从机器人资产 taili.py 源码 AST 推导本体事实（不 import → 免 isaaclab 依赖）。

抽 taili.py 里干净可得的:基座标称高度、默认关节角、执行器关节类型、执行器模型
(力矩/速度限、刚度/阻尼)。这些都是**稳定结构**(随机器人定义变才变),无 live 值。

诚实边界:精确自由度数由 URDF 决定,这里的关节名是正则模式,不是字面数字——
所以 DOF 标为"从执行器关节类型推断(N 类/腿 × 4 腿)",并注明 URDF 是精确来源,不硬编。
"""
from __future__ import annotations

import ast
import operator
import re
from typing import Any, Dict, List, Mapping, Optional

from .reward_deriver import _ref
from .schema import RobotFact


def _call_kwargs(node: ast.AST) -> Dict[str, ast.AST]:
    if not isinstance(node, ast.Call):
        return {}
    return {kw.arg: kw.value for kw in node.keywords if kw.arg}


def _func_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        f = node.func
        return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
    return ""


def _find_assign_call(tree: ast.AST, func_name: str) -> Optional[ast.Call]:
    """找模块级 `X = FuncName(...)` 的那个 Call。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and _func_name(node.value) == func_name:
            return node.value
    return None


_MISSING = object()
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _qualified_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _constant_expr(node: ast.AST, names: Mapping[str, Any]) -> Any:
    """只计算常量容器和算术，不执行调用或任意 Python 表达式。"""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values = [_constant_expr(item, names) for item in node.elts]
        if any(value is _MISSING for value in values):
            return _MISSING
        if isinstance(node, ast.Tuple):
            return tuple(values)
        if isinstance(node, ast.Set):
            return set(values)
        return values
    if isinstance(node, ast.Dict):
        keys = [_constant_expr(item, names) for item in node.keys]
        values = [_constant_expr(item, names) for item in node.values]
        if any(item is _MISSING for item in (*keys, *values)):
            return _MISSING
        return dict(zip(keys, values))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        operand = _constant_expr(node.operand, names)
        if operand is not _MISSING:
            try:
                return _UNARY_OPS[type(node.op)](operand)
            except (TypeError, ValueError, OverflowError):
                return _MISSING
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _constant_expr(node.left, names)
        right = _constant_expr(node.right, names)
        if left is not _MISSING and right is not _MISSING:
            try:
                return _BIN_OPS[type(node.op)](left, right)
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                return _MISSING
    name = _qualified_name(node)
    return names.get(name, _MISSING) if name else _MISSING


def _source_constants(source_text: str, namespace: str) -> Dict[str, Any]:
    """按源码顺序解析模块级纯常量，并以资产使用的命名空间导出。"""
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return {}
    local: Dict[str, Any] = {}
    exported: Dict[str, Any] = {}
    for node in tree.body:
        target: Optional[ast.AST] = None
        value_node: Optional[ast.AST] = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value_node = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value_node = node.target, node.value
        if not isinstance(target, ast.Name) or value_node is None:
            continue
        value = _constant_expr(value_node, local)
        if value is _MISSING:
            continue
        local[target.id] = value
        exported[f"{namespace}.{target.id}"] = value
    return exported


def _lit(node: Optional[ast.AST], default: Any = None, names: Optional[Mapping[str, Any]] = None) -> Any:
    if node is None:
        return default
    value = _constant_expr(node, names or {})
    if value is not _MISSING:
        return value
    try:
        return ast.literal_eval(node)
    except Exception:  # noqa: BLE001
        return default


def _joint_type(expr: str) -> str:
    """把关节名/正则(如 '.*_hip_joint')归一到类型名 'hip'。"""
    m = re.search(r"(hip|thigh|calf|knee|shoulder|elbow|ankle|wheel)", expr.lower())
    if m:
        return m.group(1)
    return expr.strip(".*_").replace("_joint", "").lstrip("_") or expr


def derive_robot_model(
    source_text: str,
    rel_path: str,
    constant_sources: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """返回 {facts: [RobotFact], errors: [...]}。纯函数,测试可喂改过的文本。"""
    result: Dict[str, Any] = {"facts": [], "errors": []}
    try:
        tree = ast.parse(source_text)
    except SyntaxError as exc:
        result["errors"].append(f"parse: {exc}")
        return result

    cfg = _find_assign_call(tree, "ArticulationCfg")
    if cfg is None:
        result["errors"].append("ArticulationCfg 赋值未找到")
        return result
    kw = _call_kwargs(cfg)
    facts: List[RobotFact] = []
    constants: Dict[str, Any] = {}
    for namespace, constant_source in (constant_sources or {}).items():
        constants.update(_source_constants(constant_source, namespace))

    # 基座标称高度 + 默认关节角:来自 init_state=InitialStateCfg(pos=..., joint_pos={...})
    init = kw.get("init_state")
    if isinstance(init, ast.Call):
        ik = _call_kwargs(init)
        pos = _lit(ik.get("pos"), names=constants)
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            facts.append(RobotFact(
                key="base_nominal_height", label="基座标称高度(米)", value=float(pos[2]),
                source=_ref(source_text, ik["pos"], rel_path, symbol="init_state.pos"),
                note="机器人初始/标称机身高度(init_state.pos 的 z)",
            ))
        jp = _lit(ik.get("joint_pos"))
        if isinstance(jp, dict):
            facts.append(RobotFact(
                key="default_joint_pos", label="默认关节角(按名字正则分组)", value=jp,
                source=_ref(source_text, ik["joint_pos"], rel_path, symbol="init_state.joint_pos"),
                note="键为关节名正则(如 F.*_thigh_joint),值为默认弧度",
            ))

    # 执行器:类型 + 关节类型 + 力矩/速度限 + 刚度/阻尼。
    act = kw.get("actuators")
    joint_types: List[str] = []
    if isinstance(act, ast.Dict):
        for vnode in act.values:
            if not isinstance(vnode, ast.Call):
                continue
            ak = _call_kwargs(vnode)
            model = _func_name(vnode).replace("Cfg", "")
            names = _lit(ak.get("joint_names_expr")) or []
            types = []
            for n in names:
                jt = _joint_type(str(n))
                if jt not in types:
                    types.append(jt)
            for jt in types:
                if jt not in joint_types:
                    joint_types.append(jt)
            actuator = {
                "model": model,
                "joint_types": types,
                "effort_limit": _lit(ak.get("effort_limit")),
                "velocity_limit": _lit(ak.get("velocity_limit")),
                "stiffness": _lit(ak.get("stiffness")),
                "damping": _lit(ak.get("damping")),
                "saturation_effort": _lit(ak.get("saturation_effort")),
            }
            facts.append(RobotFact(
                key="actuator_model", label="执行器模型", value=actuator,
                source=_ref(source_text, vnode, rel_path, symbol="actuators"),
                note=f"{model}:各关节类型的力矩/速度限、刚度、阻尼",
            ))

    # 自由度:诚实推断(关节类型数/腿 × 4 腿),URDF 为精确来源,不硬编。
    if joint_types:
        facts.append(RobotFact(
            key="dof_inferred", label="自由度(推断)",
            value=f"{len(joint_types) * 4}（{len(joint_types)} 类关节/腿 × 4 腿 = {len(joint_types) * 4}）",
            source=next((f.source for f in facts if f.key == "actuator_model"), None),
            note="从执行器关节类型推断,假设四足；精确自由度以 URDF(robot.urdf)为准,本文件只有关节名正则。",
        ))

    result["facts"] = facts
    return result
