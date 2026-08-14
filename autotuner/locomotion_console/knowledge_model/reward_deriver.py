"""从 taili_reward.py 源码 AST 推导奖励模型（不手抄、不 import → 免 torch 依赖）。

核心决策:AST 只推**关系骨架**(项名、权重参数、门控、符号、代码位置);核**数学**
(exp(-(err/σ)²) 之类)不转写,只记核类型 + 代码位置,答题时现读源码片段。这样:
- 从文件推的,不可能和文件矛盾(对齐的根)。
- 每段挂 sha256,源码一改就能判过时。
- 分类/覆盖用代码自己的清单(REWARD_GROUP_GATES / _REWARD_GROUP_OF),零手抄。

权重可能是:直接 cfg.w_*、getattr(cfg,"w_..")、或经局部量的混合(如
`slip_w = w_stance_slip + quality_gate*w_stance_slip_late`);还可能分散在多条对同一
comp[key] 的赋值里(自增)。因此收集该项**全部赋值子树 + 相关局部量**里的所有 w_*。

纯函数 derive_reward_terms(source_text, ...):测试可喂改过的文本,不碰文件。遇到不认的
RHS 一律降级 cited_only,绝不抛异常拖垮调用。
"""
from __future__ import annotations

import ast
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from .schema import Edge, Entity, SourceRef


# ── 通用 AST 小工具 ──────────────────────────────────────────────────────────
def _norm(text: str) -> str:
    """规约空白后再 hash:内容变才算变,重排/缩进不算。"""
    return " ".join((text or "").split())


def _sha(text: str) -> str:
    return hashlib.sha256(_norm(text).encode("utf-8")).hexdigest()


def _seg(source_text: str, node: ast.AST) -> str:
    try:
        seg = ast.get_source_segment(source_text, node)
    except Exception:  # noqa: BLE001
        seg = ""
    return seg or ""


def _ref(source_text: str, node: ast.AST, rel_path: str, symbol: str = "") -> SourceRef:
    return SourceRef(
        file=rel_path,
        line_start=getattr(node, "lineno", 0) or 0,
        line_end=getattr(node, "end_lineno", 0) or 0,
        sha256=_sha(_seg(source_text, node)),
        symbol=symbol,
    )


def _find_func(tree: ast.AST, name: str) -> Optional[ast.AST]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _module_const(tree: ast.AST, name: str) -> Any:
    """取模块级常量，支持字面量、常量引用和 tuple/dict 的加法组合。"""
    assignments: Dict[str, ast.AST] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                assignments[tgt.id] = node.value

    def _resolve(node: ast.AST, seen: set[str]) -> Any:
        try:
            return ast.literal_eval(node)
        except Exception:  # noqa: BLE001
            pass
        if isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
            return _resolve(assignments[node.id], seen | {node.id})
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = _resolve(node.left, seen)
            right = _resolve(node.right, seen)
            if left is not None and right is not None:
                try:
                    return left + right
                except TypeError:
                    return None
        return None

    value = assignments.get(name)
    return _resolve(value, {name}) if value is not None else None


def _reward_config_defaults(tree: ast.AST, cfg_class: str = "RewardConfig") -> Dict[str, float]:
    """从权重配置 dataclass 的字段默认里取 {字段名: 默认值}(权威结构默认,非现值)。"""
    out: Dict[str, float] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cfg_class:
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.value is not None:
                    try:
                        out[item.target.id] = ast.literal_eval(item.value)
                    except Exception:  # noqa: BLE001
                        continue
            break
    return out


# ── 权重 / 核 / 门控识别 ─────────────────────────────────────────────────────
def _unwrap(node: ast.AST) -> ast.AST:
    """剥掉 float(...)/int(...)/max(x, ...) 这类包裹,取到里面的权重表达式。"""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in ("float", "int") and node.args:
            return _unwrap(node.args[0])
        if node.func.id == "max" and node.args:
            return _unwrap(node.args[0])
    return node


def _weight_from_expr(node: ast.AST) -> Optional[Tuple[str, Optional[float]]]:
    """单个表达式是否是一个权重:cfg.w_* / getattr(cfg,"w_*",默认)。返回 (参数名, 内联默认) 或 None。"""
    node = _unwrap(node)
    if isinstance(node, ast.Attribute) and isinstance(node.attr, str) and node.attr.startswith(("w_", "rew_")):
        return node.attr, None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
        args = node.args
        if len(args) >= 2 and isinstance(args[1], ast.Constant) and isinstance(args[1].value, str):
            name = args[1].value
            if name.startswith(("w_", "rew_")):
                inline = None
                if len(args) >= 3:
                    try:
                        inline = ast.literal_eval(args[2])
                    except Exception:  # noqa: BLE001
                        inline = None
                return name, inline
    return None


def _collect_weights_direct(node: ast.AST) -> List[Tuple[str, Optional[float]]]:
    """遍历表达式,收集其中**所有**直接权重(cfg.w_* 与 getattr(cfg,"w_.."))。去重保序。"""
    out: List[Tuple[str, Optional[float]]] = []
    for sub in ast.walk(node):
        w = _weight_from_expr(sub)
        if w is not None and w not in out:
            out.append(w)
    return out


def _kernel_from_expr(node: ast.AST) -> Optional[str]:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in ("exp_kernel", "far_kernel"):
            return sub.func.id
    return None


def _is_gate_name(name: str) -> bool:
    return name == "gate" or name.endswith("_gate")


def _gate_norm(name: str) -> str:
    # 裸 `gate` 是基础稳定门控 stable_motion_gate 的局部别名。
    return "stable_motion_gate" if name == "gate" else name


def _is_zeros(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("zeros_like", "zeros"))


def _refs_comp_key(node: ast.AST, key: str) -> bool:
    """RHS 是否引用了 comp[key](= 自增/augment)。"""
    for sub in ast.walk(node):
        if (isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name) and sub.value.id == "comp"
                and isinstance(sub.slice, ast.Constant) and sub.slice.value == key):
            return True
    return False


def _names_in(node: ast.AST) -> List[str]:
    return [sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)]


# ── 主推导 ──────────────────────────────────────────────────────────────────
def derive_reward_terms(
    source_text: str,
    rel_path: str,
    func_name: str = "compute_reward_components",
    cfg_class: str = "RewardConfig",
    gate_const: str = "REWARD_GROUP_GATES",
    group_const: str = "_REWARD_GROUP_OF",
    strict: bool = True,
) -> Dict[str, Any]:
    """推导奖励模型。符号名(函数/配置类/门控/分组常量)由调用方(机器人描述符)给,不写死。
    返回 dict:entities / edges / config_defaults / gate_keys / group_of / errors。"""
    result: Dict[str, Any] = {
        "entities": [], "edges": [], "config_defaults": {},
        "gate_keys": [], "group_of": {}, "errors": [],
    }
    try:
        tree = ast.parse(source_text)
    except SyntaxError as exc:
        result["errors"].append(f"parse: {exc}")
        return result

    config_defaults = _reward_config_defaults(tree, cfg_class)
    gate_tuple = _module_const(tree, gate_const) or ()
    group_of = _module_const(tree, group_const) or {}
    gate_keys = set(gate_tuple)
    result["config_defaults"] = config_defaults
    result["gate_keys"] = sorted(gate_keys)
    result["group_of"] = dict(group_of)

    func = _find_func(tree, func_name)
    if func is None:
        result["errors"].append(f"function {func_name} not found")
        return result

    # 预扫:局部量 -> 该量涉及的全部权重 / 核类型(处理经中间变量的混合权重与核)。
    local_weights: Dict[str, List[Tuple[str, Optional[float]]]] = {}
    local_kernel: Dict[str, str] = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            nm = node.targets[0].id
            ws = _collect_weights_direct(node.value)
            if ws:
                local_weights[nm] = ws
            k = _kernel_from_expr(node.value)
            if k is not None:
                local_kernel.setdefault(nm, k)

    entities: List[Entity] = []
    edges: List[Edge] = []
    seen_params: Dict[str, Entity] = {}
    seen_gates: Dict[str, Entity] = {}
    assigns_by_key: Dict[str, List[ast.Assign]] = {}

    for node in ast.walk(func):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name) and tgt.value.id == "comp"
                and isinstance(tgt.slice, ast.Constant) and isinstance(tgt.slice.value, str)):
            assigns_by_key.setdefault(tgt.slice.value, []).append(node)

    for key, nodes in assigns_by_key.items():
        if key in gate_keys or key == "total":
            continue
        # 选“定义式”(非 zeros、非自增)作 sign/gates/kernel/source 的来源。
        primary = next((n for n in nodes if not _is_zeros(n.value) and not _refs_comp_key(n.value, key)), None)
        if primary is None:
            primary = nodes[0]

        try:
            # 全部权重:遍历该 key 的所有赋值(含自增),收集直接 w_* + 经局部量引入的 w_*。
            all_weights: List[Tuple[str, Optional[float]]] = []

            def _add(ws: List[Tuple[str, Optional[float]]]) -> None:
                for w in ws:
                    if w not in all_weights:
                        all_weights.append(w)

            for n in nodes:
                _add(_collect_weights_direct(n.value))
                for nm in _names_in(n.value):
                    if nm in local_weights:
                        _add(local_weights[nm])

            # sign / gates / kernel:取自 primary 的顶层因子。
            factors, negated = _flatten_mult(primary.value)
            kernel_kind: Optional[str] = None
            gates: List[str] = []
            extra_factors: List[str] = []
            for fexpr in factors:
                if isinstance(fexpr, ast.Name):
                    nm = fexpr.id
                    if nm in local_kernel:
                        kernel_kind = kernel_kind or local_kernel[nm]
                        continue
                    if _is_gate_name(nm):
                        gates.append(_gate_norm(nm))
                        continue
                    if nm not in local_weights:
                        extra_factors.append(nm)
                    continue
                k = _kernel_from_expr(fexpr)
                if k is not None:
                    kernel_kind = kernel_kind or k
            if kernel_kind is None:
                has_arith = any(isinstance(sub, ast.BinOp) for f in factors for sub in ast.walk(f))
                kernel_kind = "linear" if has_arith else "cited"

            # 主权重:优先 w_<key>;否则去掉 _late/_high/_tail 后缀的基权重;再否则第一个。
            params = [p for p, _ in all_weights]
            primary_param: Optional[str] = next((p for p in params if p == f"w_{key}"), None)
            if primary_param is None:
                bases = [p for p in params if not any(s in p for s in ("_late", "_high", "_tail"))]
                primary_param = (bases or params or [None])[0]
            # 主权重排到最前,便于 store 取 weighted_by[0] 作主权重。
            if primary_param in params:
                params = [primary_param] + [p for p in params if p != primary_param]
            primary_inline = next((inl for p, inl in all_weights if p == primary_param), None)

            sign = "penalty" if negated else "reward"
            src = _ref(source_text, primary, rel_path, symbol=f'comp["{key}"]')
            weight_default = config_defaults.get(primary_param) if primary_param else None
            attrs: Dict[str, Any] = {
                "sign": sign,
                "kernel_kind": kernel_kind,
                "weight_default": weight_default,
                "weight_params": params,
                "additional_factors": sorted(set(extra_factors)),
                "extra_assignments": max(0, len(nodes) - 1),
            }
            if primary_param and primary_inline is not None and weight_default is not None and primary_inline != weight_default:
                attrs["inline_default"] = primary_inline
                attrs["inline_default_note"] = "getattr 内联默认与 RewardConfig 字段默认不一致,以字段默认为准"
            if len(params) > 1:
                attrs["blended_weight_note"] = "权重为多项混合(基权重 + 随 quality_gate 渐入的后期/尾部权重),见 weight_params 与代码片段"

            entities.append(Entity(id=f"reward_term:{key}", type="reward_term", name=key, attrs=attrs, source=src))
            entities.append(Entity(id=f"telemetry_field:reward.{key}", type="telemetry_field", name=f"reward.{key}"))
            edges.append(Edge(src=f"reward_term:{key}", rel="measured_as", dst=f"telemetry_field:reward.{key}"))
            edges.append(Edge(src=f"reward_term:{key}", rel="computed_in", dst=f"code:{rel_path}", source=src))
            for p in params:
                pid = f"param:{p}"
                if pid not in seen_params:
                    seen_params[pid] = Entity(id=pid, type="param", name=p, attrs={"weight_default": config_defaults.get(p)})
                edges.append(Edge(src=f"reward_term:{key}", rel="weighted_by", dst=pid, source=src))
            for g in gates:
                gid = f"gate:{g}"
                if gid not in seen_gates:
                    seen_gates[gid] = Entity(id=gid, type="gate", name=g)
                edges.append(Edge(src=f"reward_term:{key}", rel="gated_by", dst=gid))

        except Exception as exc:  # noqa: BLE001 — 绝不因一个奇怪 RHS 拖垮整体
            src = _ref(source_text, primary, rel_path, symbol=f'comp["{key}"]')
            entities.append(Entity(
                id=f"reward_term:{key}", type="reward_term", name=key,
                attrs={"sign": "unknown", "kernel_kind": "cited", "error": f"{type(exc).__name__}: {exc}"},
                source=src, freshness="cited_only",
            ))
            result["errors"].append(f'{key}: {type(exc).__name__}: {exc}')

    entities.extend(seen_params.values())
    entities.extend(seen_gates.values())
    result["entities"] = entities
    result["edges"] = edges
    return result


def _flatten_mult(node: ast.AST) -> Tuple[List[ast.AST], bool]:
    """把乘法链拍平成因子列表;顺带把前导一元负号剥掉并返回 negated 标记。"""
    negated = False
    factors: List[ast.AST] = []

    def walk(n: ast.AST) -> None:
        nonlocal negated
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
            walk(n.left)
            walk(n.right)
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            negated = True
            walk(n.operand)
        else:
            factors.append(n)

    walk(node)
    return factors, negated
