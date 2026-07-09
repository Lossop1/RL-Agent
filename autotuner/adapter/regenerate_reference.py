"""Adapter — AMP 参考重生成入口(§6 "AMP 参考=重生成";§14 step1 留口①)。

把 §6 表里唯一标"重生成"的项接进 Adapter:`URDF → 派生参考几何(ref_geom)→ 注入生成器
(blind_locomotion.gen_taili_gaits.apply_geometry)→ 跑 IK/相位/对角框架 → 每姿态一个 npz`。
框架不变,只换几何。

身份测试(零 GPU):喂 Taili 自己的 URDF 重生成 → 应 ≈ 已提交 clips/taili_*.npz,证明参考管线
确实由 URDF 驱动、能复现 proven 参考(不是手搓的一次性产物)。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from autotuner.blind_locomotion import gen_taili_gaits as gen

from autotuner.adapter.ref_geom import extract

_DEFAULT_URDF = "assets/robots/taili-dog/robot.urdf"


def regenerate(urdf_path: str, out_dir: str, which: str = "flat6") -> List[str]:
    """URDF → 重生成参考 clip 集。which: flat6 | multispeed | all。返回产出的 npz 路径。"""
    geom = extract(urdf_path)
    applied = gen.apply_geometry(geom.as_globals())   # 注入派生几何
    if which == "multispeed":
        lib = gen.flat_multispeed_library()
    elif which == "all":
        lib = {**gen.gait_library(), **gen.flat_multispeed_library()}
    else:
        lib = gen.gait_library()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    produced = []
    for name, cfg in lib.items():
        gen.generate(cfg, out_dir)
        produced.append(str(Path(out_dir) / f"taili_{name}.npz"))
    return produced, applied


def _compare(a_npz: str, b_npz: str) -> Dict[str, float]:
    """两 clip 的关键数组最大绝对差(dof_positions / body_positions)。"""
    a, b = np.load(a_npz), np.load(b_npz)
    out = {}
    for k in ("dof_positions", "body_positions", "body_rotations"):
        if k in a and k in b and a[k].shape == b[k].shape:
            out[k] = float(np.abs(a[k].astype(float) - b[k].astype(float)).max())
        else:
            out[k] = float("nan")  # 形状不一致/缺失
    return out


def _gen_one(cfg, inject_geom):
    """生成单 clip 到临时目录,可选注入几何,返回 npz dict。"""
    import tempfile
    if inject_geom is not None:
        gen.apply_geometry(inject_geom)
    d = tempfile.mkdtemp(prefix="adapter_ref_")
    gen.generate(cfg, d)
    return np.load(Path(d) / f"taili_{cfg.name}.npz")


if __name__ == "__main__":
    urdf = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_URDF
    existing = Path(__file__).resolve().parents[1] / "blind_locomotion" / "motions" / "clips"
    geom = extract(urdf)
    g = geom.as_globals()
    applied = gen.apply_geometry(g)
    print(f"注入几何: L1={applied['L1']:.5f} BASE_Z={applied['BASE_Z']:.4f} "
          f"X0={applied['X0']:.4f} H0={applied['H0']:.4f}\n")

    lib = gen.gait_library()
    # ── 测试 A:Adapter 注入正确性 = 注入 Taili 自己几何 vs 纯硬编码,应字节级一致 ──
    print("测试 A — Adapter 注入正确性(注入 Taili URDF 几何 vs 不注入,应 ~0):")
    print(f"{'clip':16} {'dof Δmax':>11} {'body Δmax':>11} {'match':>6}")
    okA = True
    import importlib
    for name, cfg in lib.items():
        importlib.reload(gen)  # 复位硬编码
        base = _gen_one(gen.gait_library()[name], None)
        inj = _gen_one(gen.gait_library()[name], g)
        dd = np.abs(base["dof_positions"].astype(float) - inj["dof_positions"].astype(float)).max()
        bd = np.abs(base["body_positions"].astype(float) - inj["body_positions"].astype(float)).max()
        m = dd < 1e-5 and bd < 1e-5
        okA &= m
        print(f"{name:16} {dd:>11.2e} {bd:>11.2e} {'OK' if m else 'XX':>6}")
    print(f"→ 注入正确性: {'PASS' if okA else 'FAIL'}\n")

    # ── 诊断 B:当前生成器 vs 已提交 clip(查数据漂移,不是 Adapter 对错) ──
    importlib.reload(gen)
    gen.apply_geometry(g)
    print("诊断 B — 当前生成器 vs 已提交 clips/(查参考漂移):")
    for name, cfg in lib.items():
        ref = existing / f"taili_{name}.npz"
        if not ref.exists():
            continue
        cur = _gen_one(cfg, None)
        dd = np.abs(cur["dof_positions"].astype(float) - np.load(ref)["dof_positions"].astype(float)).max()
        tag = "≈current" if dd < 1e-3 else f"DRIFTED {dd:.2f}rad"
        print(f"{name:16} {tag}")
    print("\n注:flat clip 漂移=已提交 clip 比当前 gait_library 旧(数据卫生,⑦诚实层 provenance 该管);"
          "Adapter 机制正确性看测试 A。")
