"""Framework component catalog (② value core; §7 catalog + §6 adapt-kind table).

adapt_kind semantics (§6 "换 URDF 变哪些" + §7 不变/派生 line):
  invariant   — framework body; NEVER changes per robot (reward STRUCTURE, curriculum structure,
                async A-C, diagonal gait pattern, normalized gate values).
  derive      — computed from URDF/geometry (actuator effort/Kp/Kd, mirror permutation, dims).
  regenerate  — rebuilt from the new robot's IK/geometry (AMP reference clips).
  scale       — numeric ranges scaled ∝ leg-length / mass (DR ranges, reward thresholds).

Each component points to the module that IMPLEMENTS it (provenance, §12). Compositions select a
component set = one framework instance (current Taili blind runtime and strategy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

AdaptKind = Literal["invariant", "derive", "regenerate", "scale"]


@dataclass(frozen=True)
class FrameworkComponent:
    id: str
    label: str
    role: str                                  # one-line: what mechanism it provides
    adapt_kind: AdaptKind                       # the PRIMARY per-robot adaptation action
    applies_to: Tuple[str, ...]                 # robot morphology capabilities it requires
    module: str                                 # provenance: code that implements it
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
        "autotuner/blind_locomotion/env_edit/taili_amp_env.py",
        note="盲机器人通用;A-C 结构=框架本体,不随机器人变。"),
    "M2": FrameworkComponent(
        "M2", "AMP + 解析参考", "command-conditioned trot reference + discriminator",
        "regenerate", (_QUAD, "amp_reference"),
        "autotuner/blind_locomotion/parametric_ref.py", requires=("M9",),
        note="参考随形态;判别器结构不变,参考 clip 重生成。"),
    "M3": FrameworkComponent(
        "M3", "gait-phase trot clock", "phase clock + contact-matching + diagonal offset",
        "invariant", (_QUAD,),
        "autotuner/blind_locomotion/env_edit/taili_amp_env.py",
        note="对角模式=框架本体(§6 不变)。"),
    "M4": FrameworkComponent(
        "M4", "执行器模型", "per-joint effort/vel + Kp/Kd actuator model",
        "derive", (_QUAD,),
        "autotuner/adapter/derive.py",
        note="effort/vel 读 URDF;Kp∝effort、Kd=2√(Kp·I) 派生(§8.1)。"),
    "M5": FrameworkComponent(
        "M5", "统一相位课程", "φ0→φ1→φ2 phase curriculum, gated on mastered subset",
        "invariant", (),
        "autotuner/blind_locomotion/env_edit/taili_amp_env_cfg.py",
        note="课程结构 + 归一化门值=不变(§8.3 门数值可标定)。"),
    "M6": FrameworkComponent(
        "M6", "设备正确 DR", "mass/friction/CoM/IMU/Kp-Kd/push DR, device-correct pipeline",
        "scale", (),
        "autotuner/blind_locomotion/env_edit/taili_amp_env_cfg.py",
        note="DR 结构不变,范围 ∝ 质量/腿长缩放(§6)。"),
    "M7": FrameworkComponent(
        "M7", "reward 模式族", "track/gait/stand/clearance/torque/stand-still reward family",
        "scale", (),
        "autotuner/adapter/reward_scale.py",
        note="reward 结构通用,几何/接触阈值数值缩放(§6/§8.0)。"),
    "M8": FrameworkComponent(
        "M8", "镜像增强", "L↔R symmetry augmentation (gated φ≥1)",
        "derive", (_QUAD,),
        "autotuner/taili_core/taili_symmetry.py",
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
        "autotuner/taili_core/taili_models.py",
        note="盲部署地形编码器;结构不变(blind 栈替代 M1 的特权 critic 依赖)。"),
    "B2": FrameworkComponent(
        "B2", "强对称 Equivariant Actor", "structural mean = 0.5[net(x)+M·net(Mx)], equivariant any weights",
        "derive", (_QUAD, "blind_actor"),
        "autotuner/taili_core/taili_models.py",
        note="结构等变;镜像置换原语自带于 taili_symmetry。**替代** M8 镜像增强(策略书:augmentation 不必要)→ 故不 require M8。"),
    "B3": FrameworkComponent(
        "B3", "两帧 AMP 判别器", "frame51/102 motion+command+mode discriminator, style-only",
        "regenerate", (_QUAD, "amp_reference"),
        "autotuner/taili_core/taili_amp_reference.py", requires=("M9",),
        note="参考随形态重生成(对齐 q_default,terrain-agnostic)。"),
    "B4": FrameworkComponent(
        "B4", "两段门控 reward", "exp-kernel tracking + two-regime stable_motion_gate + collapse penalties",
        "scale", (),
        "autotuner/taili_core/taili_reward.py",
        note="门控/penalty 结构不变,阈值(高度/tilt/contact)随几何缩放。"),
}


COMPOSITIONS: Dict[str, FrameworkComposition] = {
    "taili_amp_proven": FrameworkComposition(
        "taili_amp_proven", "Taili AMP proven baseline (agent_65000)",
        ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"),
        status="validated",
        note="§7 当前基线:sim2sim 合格、各向走、trot 0.89-0.91。"),
    "taili_blind_tp": FrameworkComposition(
        "taili_blind_tp", "Taili blind TerrainPerceiver redesign (strategy book)",
        ("B1", "B2", "B3", "B4", "M4", "M5", "M6", "M9"),
        status="draft",
        note="策略书重设计:盲 TP 编码器 + 结构等变 actor + style-only AMP + 两段门控 reward。"),
}


def get_component(cid: str) -> FrameworkComponent:
    try:
        return CATALOG[cid]
    except KeyError as exc:
        raise ValueError(f"unknown component {cid!r}; known: {', '.join(sorted(CATALOG))}") from exc


def get_composition(comp_id: str) -> FrameworkComposition:
    try:
        return COMPOSITIONS[comp_id]
    except KeyError as exc:
        raise ValueError(f"unknown composition {comp_id!r}; known: {', '.join(sorted(COMPOSITIONS))}") from exc


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


def render_catalog() -> str:
    rows = [f"{'id':4} {'kind':11} {'applies_to':28} label"]
    for c in CATALOG.values():
        rows.append(f"{c.id:4} {c.adapt_kind:11} {','.join(c.applies_to)[:28]:28} {c.label}")
    rows.append("")
    for comp in COMPOSITIONS.values():
        rows.append(f"[{comp.id}] ({comp.status}) = {' '.join(comp.component_ids)}")
    return "\n".join(rows)


if __name__ == "__main__":
    print(render_catalog())
    print()
    for cid, caps in (("taili_amp_proven", (_QUAD, "blind_actor", "amp_reference")),
                      ("taili_blind_tp", (_QUAD, "blind_actor", "amp_reference"))):
        comp = get_composition(cid)
        issues = validate_composition(comp, caps)
        plan = adapt_plan(comp)
        print(f"{cid}: {'VALID' if not issues else 'ISSUES ' + str(issues)}")
        print(f"  adapt: invariant={plan['invariant']} derive={plan['derive']} "
              f"regenerate={plan['regenerate']} scale={plan['scale']}")
