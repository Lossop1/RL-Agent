"""Reward-invariant regression tests for compute_reward_components (taili_reward.py).

These encode the design intent documented in docs/taili_strategy_decisions.md as executable
checks, so a future edit that breaks a principle (per-robot MEAN reduction, no stand-still trap,
wrong-direction pressure, terminal!=timeout, gate zeroing) fails loudly. Pure torch on CPU — no
sim. The reward consumes a duck-typed `inp`; the factory builds a physically-plausible "good
trot" and tests perturb one field at a time.
"""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from autotuner.taili_core.taili_reward import RewardConfig, compute_reward_components


def make_inp(n=2, **over):
    """A plausible forward-walking sample: tracking a 0.5 m/s forward command, upright, trotting."""
    z = torch.zeros(n)
    o = torch.ones(n)
    d = dict(
        cmd=torch.tensor([[0.5, 0.0, 0.0]]).repeat(n, 1),
        base_lin_vel=torch.tensor([[0.5, 0.0, 0.0]]).repeat(n, 1),
        base_ang_vel=torch.zeros(n, 3),
        stable_motion_gate=o.clone(),
        stand_gate=z.clone(),          # moving, not standing
        moving_gate=o.clone(),
        quality_gate=o.clone(),
        action=torch.zeros(n, 12),
        last_action=torch.zeros(n, 12),
        default_pose_error=z.clone(),
        foot_contact=torch.tensor([[1.0, 0.0, 0.0, 1.0]]).repeat(n, 1),  # diagonal FL+RR down
        local_obstacle_h=z.clone(),
        foot_clearance=torch.tensor([[0.0, 0.08, 0.08, 0.0]]).repeat(n, 1),
        foot_vel_xy=torch.zeros(n, 4),   # slip fallback path when no settled-window supplied
        desired_foot_contact=torch.tensor([[1.0, 0.0, 0.0, 1.0]]).repeat(n, 1),
        touchdown_vz=torch.zeros(n, 4),
        tilt_rel=z.clone(),
        torque=torch.zeros(n, 12),
        torque_limit=torch.full((12,), 100.0),
        torque_clamped=torch.zeros(n, 12),
        terminal_reason=None,
    )
    d.update(over)
    return SimpleNamespace(**d)


def _total(inp, cfg=None):
    cfg = cfg or RewardConfig()
    return compute_reward_components(inp, cfg)["total"]


# ── #1 tracking: hitting the command is rewarded, and better than missing it ────────────
def test_perfect_tracking_beats_miss():
    cfg = RewardConfig()
    good = compute_reward_components(make_inp(), cfg)
    miss = compute_reward_components(make_inp(base_lin_vel=torch.tensor([[0.0, 0.0, 0.0]]).repeat(2, 1)), cfg)
    assert good["tracking_lin"].mean() > miss["tracking_lin"].mean()


def test_no_stand_still_trap_under_forward_command():
    # Standing still under a forward command must NOT out-score actually walking forward.
    cfg = RewardConfig()
    walking = _total(make_inp())
    standing = _total(make_inp(
        base_lin_vel=torch.zeros(2, 3),
        foot_contact=torch.ones(2, 4),
        foot_clearance=torch.zeros(2, 4),
        desired_foot_contact=torch.ones(2, 4),
    ))
    assert walking.mean() > standing.mean()


# ── #8 wrong-direction pressure: moving against the command is actively penalized ───────
def test_wrong_direction_is_penalized():
    cfg = RewardConfig()
    backward_under_fwd_cmd = make_inp(base_lin_vel=torch.tensor([[-0.5, 0.0, 0.0]]).repeat(2, 1))
    comp = compute_reward_components(backward_under_fwd_cmd, cfg)
    assert comp["wrong_dir"].mean() < 0.0


def test_off_axis_penalizes_drift_under_stand_command():
    # off_axis is NOT moving-gated, so it must bite lateral drift while standing (A3 anti-drift).
    cfg = RewardConfig()
    inp = make_inp(
        cmd=torch.zeros(2, 3),
        base_lin_vel=torch.tensor([[0.0, 0.3, 0.0]]).repeat(2, 1),  # drifting sideways
        stand_gate=torch.ones(2),
        moving_gate=torch.zeros(2),
    )
    comp = compute_reward_components(inp, cfg)
    assert comp["off_axis"].mean() < 0.0


# ── 3b reduction policy: per-robot MEAN, not raw sum (scale-invariant to joint count) ────
def test_torque_margin_is_mean_not_sum():
    cfg = RewardConfig()
    # one joint over the 0.85 margin vs. all twelve over: a raw-sum would scale ~12x.
    tq_one = torch.zeros(1, 12); tq_one[0, 0] = 95.0    # util 0.95 > 0.85
    tq_all = torch.full((1, 12), 95.0)
    lim = torch.full((12,), 100.0)
    c_one = compute_reward_components(make_inp(1, torque=tq_one, torque_limit=lim), cfg)
    c_all = compute_reward_components(make_inp(1, torque=tq_all, torque_limit=lim), cfg)
    # mean reduction: all-twelve penalty is ~12x the one-joint penalty, never unbounded.
    ratio = float(c_all["torque_margin"].mean() / c_one["torque_margin"].mean())
    assert 11.0 < ratio < 13.0


# ── #4 terminal != timeout ──────────────────────────────────────────────────────────────
def test_timeout_not_penalized_but_terminal_is():
    cfg = RewardConfig()
    timeout = compute_reward_components(make_inp(terminal_reason="timeout"), cfg)
    fell = compute_reward_components(make_inp(terminal_reason="base_contact"), cfg)
    assert float(timeout["terminal_penalty"].mean()) == 0.0
    assert float(fell["terminal_penalty"].mean()) == -cfg.w_terminal


# ── 3a gate: collapse zeroes the shaping terms ──────────────────────────────────────────
def test_collapse_gate_zeroes_tracking():
    cfg = RewardConfig()
    collapsed = compute_reward_components(make_inp(stable_motion_gate=torch.zeros(2)), cfg)
    assert float(collapsed["tracking_lin"].mean()) == 0.0
    assert float(collapsed["tracking_yaw"].mean()) == 0.0
    assert float(collapsed["stand"].mean()) == 0.0


def test_collapse_gate_zeroes_posture_penalties():
    # base_vz/base_wxy/hip_deviation must not pile onto a collapse or tax recovery (gated like orient).
    cfg = RewardConfig()
    collapsed = compute_reward_components(make_inp(
        stable_motion_gate=torch.zeros(2),
        base_lin_vel=torch.tensor([[0.0, 0.0, -2.0]]).repeat(2, 1),   # large downward vz (falling)
        base_ang_vel=torch.tensor([[3.0, 3.0, 0.0]]).repeat(2, 1),    # large roll/pitch rates
        hip_deviation=torch.full((2,), 1.0),
    ), cfg)
    assert float(collapsed["base_vz"].mean()) == 0.0
    assert float(collapsed["base_wxy"].mean()) == 0.0
    assert float(collapsed["hip_deviation"].mean()) == 0.0


# ── A2 yaw is gated on a real yaw command (no standing bonus) ────────────────────────────
def test_yaw_tracking_requires_yaw_command():
    cfg = RewardConfig()
    # zero yaw command -> yaw_cmd_gate off -> no tracking_yaw reward even if wz matches
    no_yaw_cmd = compute_reward_components(make_inp(), cfg)
    assert float(no_yaw_cmd["tracking_yaw"].mean()) == 0.0
    yaw = make_inp(
        cmd=torch.tensor([[0.0, 0.0, 0.8]]).repeat(2, 1),
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.8]]).repeat(2, 1),
    )
    assert float(compute_reward_components(yaw, cfg)["tracking_yaw"].mean()) > 0.0


# ── B2 slip graded & bounded (no zero-gradient skating regime) ───────────────────────────
def test_slip_penalty_graded_and_bounded():
    cfg = RewardConfig()
    slow = compute_reward_components(make_inp(stance_slip_speed_window=torch.full((2,), 0.3)), cfg)
    fast = compute_reward_components(make_inp(stance_slip_speed_window=torch.full((2,), 0.8)), cfg)
    faster = compute_reward_components(make_inp(stance_slip_speed_window=torch.full((2,), 5.0)), cfg)
    # more slip -> more penalty, but saturates (bounded) at high speed
    assert slow["stance_slip"].mean() > fast["stance_slip"].mean()   # 0.3 less negative than 0.8
    assert float(faster["stance_slip"].mean()) >= -(cfg.w_stance_slip + cfg.w_stance_slip_late) - 1e-6


def test_clearance_terrain_aware_semantics():
    # #10 fix relies on this: on FLAT (local_obstacle_h=0) a high swing lift is penalized (target 0.08),
    # but on TERRAIN (local_obstacle_h>0) the target rises so the SAME lift is NOT penalized. The env
    # feeds local_obstacle_h from the height scanner; here we verify the reward's target logic.
    cfg = RewardConfig()
    cfg.w_clearance_over = 1.0  # yaml-authoritative weight, so the sign is unambiguous
    # swinging feet are FR,RL (indices 1,2); give them a high 0.20 m lift
    hi = torch.tensor([[0.0, 0.20, 0.20, 0.0]]).repeat(2, 1)
    flat = compute_reward_components(make_inp(local_obstacle_h=torch.zeros(2), foot_clearance=hi), cfg)
    assert flat["clearance_over"].mean() < -0.01, "flat over-lift must be penalized"
    # terrain: obstacle 0.25 m -> target 0.25+margin; a 0.28 m lift is within band -> not penalized
    terr = compute_reward_components(make_inp(
        local_obstacle_h=torch.full((2,), 0.25),
        foot_clearance=torch.tensor([[0.0, 0.28, 0.28, 0.0]]).repeat(2, 1)), cfg)
    assert float(terr["clearance_over"].mean()) == 0.0, "terrain lift within the raised band must NOT be penalized"
    # flat at target (0.08) is also unpenalized
    ok = compute_reward_components(make_inp(local_obstacle_h=torch.zeros(2)), cfg)
    assert float(ok["clearance_over"].mean()) == 0.0


def test_tracking_credit_is_validated_by_contact_quality():
    cfg = RewardConfig()
    cfg.validated_tracking_floor = 0.40
    cfg.healthy_progress_slip_target = 0.10
    cfg.healthy_progress_slip_width = 0.10

    good = compute_reward_components(make_inp(
        diagonal_pair_window=torch.ones(2),
        duty_quality_window=torch.ones(2),
        stance_slip_high_fraction=torch.zeros(2),
    ), cfg)
    bad = compute_reward_components(make_inp(
        diagonal_pair_window=torch.zeros(2),
        duty_quality_window=torch.zeros(2),
        stance_slip_high_fraction=torch.ones(2),
    ), cfg)

    assert good["validated_tracking_gate"].mean() > bad["validated_tracking_gate"].mean()
    assert good["tracking_lin"].mean() > bad["tracking_lin"].mean()
    assert bad["tracking_lin"].mean() > 0.0  # floor preserves bootstrap gradient


def test_terrain_progress_is_driven_by_clearance_on_obstacles():
    cfg = RewardConfig()
    cfg.terrain_clearance_drive = 0.60
    low = compute_reward_components(make_inp(
        local_obstacle_h=torch.full((2,), 0.25),
        foot_clearance=torch.tensor([[0.0, 0.10, 0.10, 0.0]]).repeat(2, 1),
        diagonal_pair_window=torch.ones(2),
        duty_quality_window=torch.ones(2),
        stance_slip_high_fraction=torch.zeros(2),
    ), cfg)
    high = compute_reward_components(make_inp(
        local_obstacle_h=torch.full((2,), 0.25),
        foot_clearance=torch.tensor([[0.0, 0.30, 0.30, 0.0]]).repeat(2, 1),
        diagonal_pair_window=torch.ones(2),
        duty_quality_window=torch.ones(2),
        stance_slip_high_fraction=torch.zeros(2),
    ), cfg)

    assert high["validated_tracking_gate"].mean() > low["validated_tracking_gate"].mean()
    assert high["terrain_progress"].mean() > low["terrain_progress"].mean()


def test_total_is_finite_everywhere():
    cfg = RewardConfig()
    for over in [{}, {"terminal_reason": "x"}, {"stable_motion_gate": torch.zeros(2)},
                 {"base_lin_vel": torch.full((2, 3), 5.0)}]:
        t = _total(make_inp(**over), cfg)
        assert torch.isfinite(t).all()
