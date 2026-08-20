"""可复用的强化学习机制目录。

``adapt_kind`` 描述更换机器人后的处理方式：保持结构、从资产派生、
重生参考或按几何缩放。组件通过 ``product://`` 角色引用产品实现，
具体组合则由产品合同声明；因此本模块不依赖任何产品包。
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

AdaptKind = Literal["invariant", "derive", "regenerate", "scale"]


@dataclass(frozen=True)
class FrameworkComponent:
    id: str
    label: str
    role: str                                  # one-line: what mechanism it provides
    adapt_kind: AdaptKind                       # the PRIMARY per-robot adaptation action
    applies_to: Tuple[str, ...]                 # robot morphology capabilities it requires
    module: str                                 # provenance path or product:// role
    version: str = "1"
    requires: Tuple[str, ...] = ()              # other component ids it depends on
    note: str = ""


@dataclass(frozen=True)
class FrameworkComposition:
    id: str
    label: str
    component_ids: Tuple[str, ...]
    status: str = "draft"                        # draft | validated | reference | legacy
    note: str = ""


# ── catalog: §7 proven M1-M9 + blind-TP redesign (B-series) ──────────────────
_QUAD = "quadruped_12dof"

CATALOG: Dict[str, FrameworkComponent] = {
    "M1": FrameworkComponent(
        "M1", "异步盲 Actor-Critic", "blind actor hard-slice + privileged critic, one joint RL pass",
        "invariant", (_QUAD, "blind_actor"),
        "product://sources/task_env",
        note="盲机器人通用;A-C 结构=框架本体,不随机器人变。"),
    "M2": FrameworkComponent(
        "M2", "AMP + 解析参考", "command-conditioned trot reference + discriminator",
        "regenerate", (_QUAD, "amp_reference"),
        "product://reference_generator", requires=("M9",),
        note="参考随形态;判别器结构不变,参考 clip 重生成。"),
    "M3": FrameworkComponent(
        "M3", "gait-phase trot clock", "phase clock + contact-matching + diagonal offset",
        "invariant", (_QUAD,),
        "product://sources/task_env",
        note="对角模式=框架本体(§6 不变)。"),
    "M4": FrameworkComponent(
        "M4", "执行器模型", "per-joint effort/vel + Kp/Kd actuator model",
        "derive", (_QUAD,),
        "autotuner/adapter/derive.py",
        note="effort/vel 读 URDF;Kp∝effort、Kd=2√(Kp·I) 派生(§8.1)。"),
    "M5": FrameworkComponent(
        "M5", "统一相位课程", "φ0→φ1→φ2 phase curriculum, gated on mastered subset",
        "invariant", (),
        "product://sources/task_config",
        note="课程结构 + 归一化门值=不变(§8.3 门数值可标定)。"),
    "M6": FrameworkComponent(
        "M6", "设备正确 DR", "mass/friction/CoM/IMU/Kp-Kd/push DR, device-correct pipeline",
        "scale", (),
        "product://sources/task_config",
        note="DR 结构不变,范围 ∝ 质量/腿长缩放(§6)。"),
    "M7": FrameworkComponent(
        "M7", "reward 模式族", "track/gait/stand/clearance/torque/stand-still reward family",
        "scale", (),
        "autotuner/adapter/reward_scale.py",
        note="reward 结构通用,几何/接触阈值数值缩放(§6/§8.0)。"),
    "M8": FrameworkComponent(
        "M8", "镜像增强", "L↔R symmetry augmentation (gated φ≥1)",
        "derive", (_QUAD,),
        "product://symmetry",
        note="置换随形态(关节命名/数)派生(§6)。"),
    "M9": FrameworkComponent(
        "M9", "参考生成器", "URDF→IK→directional clip reference generator",
        "regenerate", (_QUAD, "amp_reference"),
        "autotuner/adapter/regenerate_reference.py",
        note="IK 随形态;部署前重生成全套参考(§6)。"),
    # ── blind-TerrainPerceiver redesign (strategy book) ──
    "B1": FrameworkComponent(
        "B1", "TerrainPerceiver", "causal-TCN proprio-history → z_terrain[32] + geom/risk aux",
        "invariant", (_QUAD, "blind_actor"),
        "product://model",
        note="盲部署地形编码器;结构不变(blind 栈替代 M1 的特权 critic 依赖)。"),
    "B2": FrameworkComponent(
        "B2", "强对称 Equivariant Actor", "structural mean = 0.5[net(x)+M·net(Mx)], equivariant any weights",
        "derive", (_QUAD, "blind_actor"),
        "product://model",
        note="结构等变；镜像置换原语由产品形态派生，替代 M8 的训练时增强。"),
    "B3": FrameworkComponent(
        "B3", "两帧 AMP 判别器", "frame51/102 motion+command+mode discriminator, style-only",
        "regenerate", (_QUAD, "amp_reference"),
        "product://amp_reference", requires=("M9",),
        note="参考随形态重生成(对齐 q_default,terrain-agnostic)。"),
    "B4": FrameworkComponent(
        "B4", "两段门控 reward", "exp-kernel tracking + two-regime stable_motion_gate + collapse penalties",
        "scale", (),
        "product://sources/reward",
        note="门控/penalty 结构不变,阈值(高度/tilt/contact)随几何缩放。"),
}


def _parse_compositions(framework: Mapping[str, Any]) -> Dict[str, FrameworkComposition]:
    """将产品合同中的组合声明解析为强类型对象。

    合同是确定性边界：格式错误必须显式失败，不能在界面或执行阶段
    静默丢弃某个组合。
    """
    raw = framework.get("compositions")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("framework.compositions must be a mapping")

    result: Dict[str, FrameworkComposition] = {}
    for raw_identifier, value in raw.items():
        identifier = str(raw_identifier).strip()
        if not identifier:
            raise ValueError("framework composition id must not be empty")
        if not isinstance(value, Mapping):
            raise ValueError(f"framework.compositions.{identifier} must be a mapping")
        component_ids = value.get("component_ids")
        if not isinstance(component_ids, (list, tuple)) or not component_ids:
            raise ValueError(
                f"framework.compositions.{identifier}.component_ids must be a non-empty list"
            )
        normalized_ids = tuple(str(item).strip() for item in component_ids)
        if any(not item for item in normalized_ids):
            raise ValueError(
                f"framework.compositions.{identifier}.component_ids contains an empty id"
            )
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError(
                f"framework.compositions.{identifier}.component_ids contains duplicates"
            )
        result[identifier] = FrameworkComposition(
            id=identifier,
            label=str(value.get("label") or identifier),
            component_ids=normalized_ids,
            status=str(value.get("status") or "draft"),
            note=str(value.get("note") or ""),
        )
    return result


def get_compositions(product_id: str | None = None) -> Dict[str, FrameworkComposition]:
    """从指定产品合同读取组合；多产品时必须显式传入产品标识。"""
    from autotuner.product import resolve_product_contract

    product = resolve_product_contract(product_id)
    return _parse_compositions(product.framework)


class _DefaultProductCompositions(Mapping[str, FrameworkComposition]):
    """旧 ``COMPOSITIONS`` API 的只读兼容视图。

    视图在访问时解析当前唯一产品，不缓存、不复制产品配置。新代码应调用
    ``get_compositions(product_id)``，以显式保留产品身份。
    """

    @staticmethod
    def _snapshot() -> Dict[str, FrameworkComposition]:
        return get_compositions()

    def __getitem__(self, key: str) -> FrameworkComposition:
        return self._snapshot()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._snapshot())

    def __len__(self) -> int:
        return len(self._snapshot())


COMPOSITIONS: Mapping[str, FrameworkComposition] = _DefaultProductCompositions()


def get_component(cid: str) -> FrameworkComponent:
    try:
        return CATALOG[cid]
    except KeyError as exc:
        raise ValueError(f"unknown component {cid!r}; known: {', '.join(sorted(CATALOG))}") from exc


def get_composition(comp_id: str, product_id: str | None = None) -> FrameworkComposition:
    compositions = get_compositions(product_id)
    try:
        return compositions[comp_id]
    except KeyError as exc:
        raise ValueError(f"unknown composition {comp_id!r}; known: {', '.join(sorted(compositions))}") from exc


def validate_composition(comp: FrameworkComposition,
                         robot_capabilities: Tuple[str, ...]) -> List[str]:
    """Return issues (empty = OK): unknown ids, unmet morphology capabilities, unmet `requires`."""
    issues: List[str] = []
    caps = set(robot_capabilities)
    ids = set(comp.component_ids)
    for cid in comp.component_ids:
        if cid not in CATALOG:
            issues.append(f"{cid}: not in catalog")
            continue
        c = CATALOG[cid]
        missing_caps = [a for a in c.applies_to if a not in caps]
        if missing_caps:
            issues.append(f"{cid} ({c.label}): robot lacks capability {missing_caps}")
        missing_req = [r for r in c.requires if r not in ids]
        if missing_req:
            issues.append(f"{cid} ({c.label}): requires {missing_req} not in composition")
    return issues


def adapt_plan(comp: FrameworkComposition) -> Dict[str, List[str]]:
    """Group the composition's components by adapt_kind → tells ③ Adapter what to do per component.
    invariant=leave framework as-is; derive/regenerate/scale=robot-specific work."""
    plan: Dict[str, List[str]] = {"invariant": [], "derive": [], "regenerate": [], "scale": []}
    for cid in comp.component_ids:
        c = CATALOG.get(cid)
        if c:
            plan[c.adapt_kind].append(cid)
    return plan


def render_catalog(product_id: str | None = None) -> str:
    rows = [f"{'id':4} {'kind':11} {'applies_to':28} label"]
    for c in CATALOG.values():
        rows.append(f"{c.id:4} {c.adapt_kind:11} {','.join(c.applies_to)[:28]:28} {c.label}")
    rows.append("")
    for comp in get_compositions(product_id).values():
        rows.append(f"[{comp.id}] ({comp.status}) = {' '.join(comp.component_ids)}")
    return "\n".join(rows)


if __name__ == "__main__":
    print(render_catalog())
    print()
    for cid, comp in get_compositions().items():
        caps = (_QUAD, "blind_actor", "amp_reference")
        issues = validate_composition(comp, caps)
        plan = adapt_plan(comp)
        print(f"{cid}: {'VALID' if not issues else 'ISSUES ' + str(issues)}")
        print(f"  adapt: invariant={plan['invariant']} derive={plan['derive']} "
              f"regenerate={plan['regenerate']} scale={plan['scale']}")
