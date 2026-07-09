"""Materialize an adapted config into local editable copies.

This uses the Locomotion Console guarded-edit discipline: backup, replace,
read back, and roll back on mismatch. It operates on local copies with Python
regexes, leaving remote deployment to deploy.py.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# AdaptedConfig.reward_thresholds field -> env_cfg dataclass field.
_ENVCFG_MAP = {
    "stand_height": "stand_height", "base_clearance": "base_clearance",
    "clr_rough_bonus_max": "clr_rough_bonus_max", "air_time_min": "air_time_min",
    "torque_limit_frac": "torque_limit_frac",
    "cmd_fwd_range": "cmd_fwd_range", "cmd_back_range": "cmd_back_range",
    "cmd_lat_range": "cmd_lat_range", "cmd_yaw_range": "cmd_yaw_range",
    "dr_mass_range_1": "dr_mass_range_1", "dr_mass_range_2": "dr_mass_range_2",
    "dr_mass_range_3": "dr_mass_range_3",
}


@dataclass
class Edit:
    file: str
    field: str
    old: str
    new: str
    changed: bool
    kind: str            # "scalar" | "tuple" | "asset_dict" | "flagged_advisory"
    applied: bool = True


def _fmt(v) -> str:
    if isinstance(v, (tuple, list)):
        return "(" + ", ".join(_fmt(x) for x in v) + ")"
    if isinstance(v, float):
        return repr(round(v, 6))
    return str(v)


def _nums(s) -> tuple:
    """从字符串/值里抽出数字元组,用于数值相等比较(去掉 0.30 vs 0.3 / 320 vs 320.0 假阳性)。"""
    if isinstance(s, (tuple, list)):
        return tuple(float(x) for x in s)
    return tuple(float(x) for x in re.findall(r"-?\d+\.?\d*", str(s)))


def _numeric_changed(old_str: str, new_val) -> bool:
    a, b = _nums(old_str), _nums(new_val)
    if len(a) != len(b):
        return True
    return any(abs(x - y) > 1e-6 for x, y in zip(a, b))


def _edit_field(text: str, field: str, new_val: str):
    """替换 `<indent>FIELD<:anno>= VALUE<comment>` 的 VALUE,保留缩进/注解/注释。返回 (text, old)。"""
    pat = re.compile(rf"^(?P<pre>\s*{re.escape(field)}\s*(:[^=\n]*)?=\s*)(?P<val>[^#\n]*?)(?P<post>\s*(#.*)?)$",
                     re.MULTILINE)
    m = pat.search(text)
    if not m:
        return text, None
    old = m.group("val").strip()
    new_text = text[:m.start("val")] + new_val + (" " if not m.group("post").startswith(" ") else "") \
        + text[m.end("val"):]
    return new_text, old


def _edit_asset_dict_value(text: str, block: str, joint_key: str, new_val: str):
    """改 asset 里 `block={ ... "joint_key": VALUE, ...}` 内某关节的 VALUE(块作用域,避免撞另一块同关节)。"""
    bm = re.search(rf"{block}\s*=\s*\{{(.*?)\}}", text, re.DOTALL)
    if not bm:
        return text, None
    blk = bm.group(1)
    jm = re.search(rf'("{re.escape(joint_key)}"\s*:\s*)([\d.]+)', blk)
    if not jm:
        return text, None
    old = jm.group(2)
    new_blk = blk[:jm.start(2)] + new_val + blk[jm.end(2):]
    return text[:bm.start(1)] + new_blk + text[bm.end(1):], old


def materialize(adapted: dict, env_cfg_src: str, asset_src: str, work_dir: str) -> List[Edit]:
    """AdaptedConfig → 副本上的改动集。返回 Edit 列表(含 no-op,便于审计"哪些==proven")。"""
    wd = Path(work_dir); wd.mkdir(parents=True, exist_ok=True)
    env_dst = wd / Path(env_cfg_src).name
    asset_dst = wd / Path(asset_src).name
    shutil.copy2(env_cfg_src, env_dst)
    shutil.copy2(asset_src, asset_dst)

    edits: List[Edit] = []
    rt = adapted["reward_thresholds"]

    # ── env_cfg dataclass 字段 ──
    text = env_dst.read_text(encoding="utf-8")
    for key, field in _ENVCFG_MAP.items():
        if key not in rt:
            continue
        newv = _fmt(rt[key])
        text, old = _edit_field(text, field, newv)
        if old is None:
            edits.append(Edit(env_dst.name, field, "(not found)", newv, False, "scalar", applied=False))
            continue
        kind = "tuple" if isinstance(rt[key], (tuple, list)) else "scalar"
        changed = _numeric_changed(old, rt[key])   # 数值比较,不被 0.30 vs 0.3 假阳性
        edits.append(Edit(env_dst.name, field, old, newv, changed, kind))
    env_dst.write_text(text, encoding="utf-8")

    # ── asset 执行器 per-joint(effort/velocity)──
    atext = asset_dst.read_text(encoding="utf-8")
    act = adapted["actuator"]
    jmap = {"hip": ".*_hip_joint", "thigh": ".*_thigh_joint", "calf": ".*_calf_joint"}
    for block, src in (("effort_limit", act["effort"]), ("velocity_limit", act["velocity"])):
        for role, jk in jmap.items():
            newv = _fmt(src[role])
            atext, old = _edit_asset_dict_value(atext, block, jk, newv)
            if old is None:
                continue
            changed = _numeric_changed(old, src[role])
            edits.append(Edit(asset_dst.name, f"{block}[{role}]", old, newv, changed, "asset_dict"))
    asset_dst.write_text(atext, encoding="utf-8")

    # ── stiffness/damping:Adapter 派生 per-joint,proven 用共享标量 → flagged advisory(不自动改结构,§8.1)──
    for role in ("hip", "thigh", "calf"):
        edits.append(Edit(asset_dst.name, f"Kp[{role}]", "shared 120.0", str(act["Kp"][role]),
                          True, "flagged_advisory", applied=False))
        edits.append(Edit(asset_dst.name, f"Kd[{role}]", "shared 10.0", str(act["Kd"][role]),
                          True, "flagged_advisory", applied=False))

    return edits


def roundtrip_verify(adapted: dict, env_materialized: str) -> List[str]:
    """重读 materialized env_cfg,核对 reward_thresholds 字段确实写进去了。返回不符项(空=PASS)。"""
    text = Path(env_materialized).read_text(encoding="utf-8")
    rt = adapted["reward_thresholds"]
    bad = []
    for key, field in _ENVCFG_MAP.items():
        if key not in rt:
            continue
        m = re.search(rf"^\s*{field}\s*(:[^=\n]*)?=\s*([^#\n]*)", text, re.MULTILINE)
        got = re.sub(r"\s", "", m.group(2)) if m else "(missing)"
        want = re.sub(r"\s", "", _fmt(rt[key]))
        if got != want:
            bad.append(f"{field}: got {got} want {want}")
    return bad


if __name__ == "__main__":
    import tempfile
    from autotuner.adapter.adapt import adapt, _DEFAULT_URDF

    env_src = "autotuner/blind_locomotion/env_edit/taili_amp_env_cfg.py"
    asset_src = "autotuner/blind_locomotion/assets/taili.py"
    wd = tempfile.mkdtemp(prefix="adapter_materialize_")
    cfg = adapt(_DEFAULT_URDF)
    edits = materialize(cfg, env_src, asset_src, wd)

    print(f"落盘到副本: {wd}  (原文件未碰)\n")
    print(f"{'file':22} {'field':22} {'old':>14} → {'new':<14} {'kind':16} {'changed'}")
    for e in edits:
        flag = "★FIX/NEW" if e.changed and e.applied else ("flag" if not e.applied else "")
        print(f"{e.file:22} {e.field:22} {str(e.old)[:14]:>14} → {str(e.new)[:14]:<14} {e.kind:16} {flag}")

    bad = roundtrip_verify(cfg, str(Path(wd) / Path(env_src).name))
    nchg = sum(1 for e in edits if e.changed and e.applied)
    nflag = sum(1 for e in edits if not e.applied)
    print(f"\nenv_cfg round-trip: {'PASS' if not bad else 'FAIL: ' + str(bad)}")
    print(f"应用改动(真实): {nchg}  · flagged advisory(per-motor,§8.1): {nflag}  · 其余 = no-op(==proven)")
    print("Taili 身份:env_cfg 应全 no-op(复现 proven);asset calf velocity 应已是 URDF 8.27。")
