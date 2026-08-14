"""验证指定 Taili runtime 的质量跟踪主路径，不依赖 Isaac Lab。"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


def _load_reward_module(runtime_root: Path):
    reward_path = runtime_root / "taili_core" / "taili_reward.py"
    if not reward_path.is_file():
        raise FileNotFoundError(f"找不到奖励实现: {reward_path}")
    module_name = "_taili_runtime_reward_under_test"
    spec = importlib.util.spec_from_file_location(module_name, reward_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载奖励实现: {reward_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_input(n: int = 2, **overrides):
    """构造已经命中 0.5 m/s 前进命令的对角小跑样本。"""
    zeros = torch.zeros(n)
    ones = torch.ones(n)
    values = {
        "cmd": torch.tensor([[0.5, 0.0, 0.0]]).repeat(n, 1),
        "base_lin_vel": torch.tensor([[0.5, 0.0, 0.0]]).repeat(n, 1),
        "base_ang_vel": torch.zeros(n, 3),
        "stable_motion_gate": ones.clone(),
        "stand_gate": zeros.clone(),
        "moving_gate": ones.clone(),
        "quality_gate": ones.clone(),
        "action": torch.zeros(n, 12),
        "last_action": torch.zeros(n, 12),
        "default_pose_error": zeros.clone(),
        "foot_contact": torch.tensor([[1.0, 0.0, 0.0, 1.0]]).repeat(n, 1),
        "local_obstacle_h": zeros.clone(),
        "terrain_response": zeros.clone(),
        "foot_clearance": torch.tensor([[0.0, 0.08, 0.08, 0.0]]).repeat(n, 1),
        "foot_vel_xy": torch.zeros(n, 4),
        "desired_foot_contact": torch.tensor([[1.0, 0.0, 0.0, 1.0]]).repeat(n, 1),
        "touchdown_vz": torch.zeros(n, 4),
        "tilt_rel": zeros.clone(),
        "torque": torch.zeros(n, 12),
        "torque_limit": torch.full((12,), 100.0),
        "torque_clamped": torch.zeros(n, 12),
        "terminal_reason": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def verify(runtime_root: Path) -> None:
    reward = _load_reward_module(runtime_root)

    incoming = reward.touchdown_incoming_planar_speed(
        torch.tensor([[0.42, 0.05]]),
        torch.tensor([[0.00, 0.18]]),
    )
    torch.testing.assert_close(incoming, torch.tensor([[0.42, 0.18]]))

    cfg = reward.RewardConfig(validated_tracking_floor=0.55)
    good_quality = {
        "diagonal_pair_window": torch.ones(2),
        "duty_quality_window": torch.ones(2),
        "stance_slip_high_fraction": torch.zeros(2),
    }
    bad_quality = {
        "diagonal_pair_window": torch.zeros(2),
        "duty_quality_window": torch.zeros(2),
        "stance_slip_high_fraction": torch.ones(2),
    }

    early_good = reward.compute_reward_components(
        _make_input(quality_gate=torch.zeros(2), **good_quality), cfg
    )
    early_bad = reward.compute_reward_components(
        _make_input(quality_gate=torch.zeros(2), **bad_quality), cfg
    )
    torch.testing.assert_close(early_good["tracking_lin"], early_bad["tracking_lin"])
    torch.testing.assert_close(early_good["tracking_lin_far"], early_bad["tracking_lin_far"])

    mature_good = reward.compute_reward_components(
        _make_input(quality_gate=torch.ones(2), **good_quality), cfg
    )
    mature_bad = reward.compute_reward_components(
        _make_input(quality_gate=torch.ones(2), **bad_quality), cfg
    )
    assert mature_good["tracking_lin"].mean() > mature_bad["tracking_lin"].mean()
    assert mature_good["tracking_lin_far"].mean() > mature_bad["tracking_lin_far"].mean()
    assert mature_bad["validated_tracking_gate"].min() >= cfg.validated_tracking_floor

    expected_floor = early_bad["tracking_lin"] * cfg.validated_tracking_floor
    assert torch.all(mature_bad["tracking_lin"] >= expected_floor - 1e-6)

    blind_source = (runtime_root / "blind_tp_env.py").read_text(encoding="utf-8")
    amp_source = (runtime_root / "taili_amp_env.py").read_text(encoding="utf-8")
    config = yaml.safe_load((runtime_root / "taili_blind_config.yaml").read_text(encoding="utf-8"))

    # phase gate 的平地质量与平地 progress 必须使用同一个样本域。
    assert "flat_moving = moving_any & gm_e" in blind_source
    assert blind_source.count('gm_e & transition_clear & (cmd[:,') >= 4
    assert 'quality["duty_by_leg"][flat_moving]' in blind_source
    assert 'quality["slip_speed"][flat_moving]' in blind_source
    assert 'quality["slip_high_fraction"][terrain_moving]' in blind_source
    assert '"_terrain_slip_high_fraction"' in amp_source
    assert "(flat_quality_ok and terrain_quality_ok) if terrain_phase" in amp_source

    # 楼梯方向任务与地形奖励同时开始，但课程等级仍由 terrain_start_phase 控制。
    curriculum = config["env"]["curriculum"]
    assert '"terrain_reward_start_phase"' in blind_source
    assert curriculum["terrain_reward_start_phase"] < curriculum["terrain_start_phase"]
    assert curriculum["stairs_forward_command_prob"] == 1.0

    # 碰障信用覆盖一次摆动，但不能持续到 Actor 历史之外形成隐藏奖励状态。
    trace_tau = float(config["env"]["blind_overrides"]["terrain_collision_trace_s"])
    trace_threshold = float(config["reward"]["terrain_clearance_event_threshold"])
    trace_active_s = trace_tau * math.log(1.0 / trace_threshold)
    control = config["env"]["control"]
    history_s = (
        float(config["model"]["history_encoder"]["history_len"])
        * float(control["dt"])
        * float(control["decimation"])
    )
    gait = config["env"]["gait"]
    swing_s = float(gait["period"]) * (1.0 - float(gait["duty"]))
    assert swing_s <= trace_active_s <= history_s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime_root", type=Path)
    args = parser.parse_args()
    verify(args.runtime_root.resolve())
    print("质量跟踪主路径验证通过")


if __name__ == "__main__":
    main()
