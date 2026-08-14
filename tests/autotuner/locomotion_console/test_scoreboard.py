"""目标记分牌装配器的单测。

重点锁定三件容易出错、且用户最在意的事：
  1. 底线出处必须如实：规范来的标 spec/derived，本次 run 配置来的标 config，
     出处不明的写死显示常量必须标 unknown（绝不冒充成真实门槛）。
  2. 拿不到本次 run 配置时，课程类底线要显示为“未定底线”，而不是拿兜底默认糊上去。
  3. 状态只讲事实（达标/未达/未定/无数据），不含好坏判断。
"""
from autotuner.locomotion_console.schemas import TrainingTelemetry, TrainingTelemetryPoint
from autotuner.locomotion_console.scoreboard import (
    FAMILY_ORDER,
    TENSIONS,
    build_scoreboard,
)


# 本次 run 的最小有效配置：只带课程门控，用来验证 config 出处解析。
_CFG = """
env:
  curriculum:
    phase_gate_prog_2: 0.62
    phase_gate_diag_1: 0.50
    phase_gate_duty_1: 0.58
    phase_gate_fall_2: 0.05
    terrain_start_phase: 2
    phase_gate_terrain_2: 3.5
"""


def _point(**overrides) -> TrainingTelemetryPoint:
    """造一个有代表性的遥测点：处于 phi2、混合命令、地形已开。"""
    pt = TrainingTelemetryPoint(
        step=42500,
        reward={"total": 1.1, "tracking_lin": 0.19, "stance_slip": -0.4,
                "terrain_progress": 0.3, "torque_saturation": -0.05,
                "clearance_under": -0.2, "clearance_over": -0.03,
                "action_rate": -0.01, "landing_impact": -0.1},
        curriculum={"phase": "phi2", "command_mode": "mixed", "active_dirs": "fwd,yaw",
                    "progress_gate": 0.55, "terrain_mean": 3.2, "terrain_max": 5.0,
                    "penalty_gate": 0.6, "dr_level": 1, "diagonal_gate": 0.7},
        health={"fall_rate": 0.03, "terminal_rate": 0.04, "base_h": 0.46,
                "base_h_min": 0.41, "upright": 0.97, "tilt_deg": 6.0,
                "tilt_deg_max": 12.0, "torque_util": 0.72,
                "height_low_risk_window": 0.02, "tilt_high_risk_window": 0.0},
        command={"lin_err": 0.28, "yaw_err": 0.11, "v_along": 0.2,
                 "diagonal_contact": 0.48, "duty_balance": 0.6, "stance_slip": 0.15,
                 "duty_spread_window": 0.25, "stance_slip_high_fraction": 0.08},
    )
    for section, values in overrides.items():
        getattr(pt, section).update(values)
    return pt


def _telemetry(point=None, **kw) -> TrainingTelemetry:
    return TrainingTelemetry(source="real", run_id="t", latest=point or _point(), **kw)


def _find(sb, key):
    for fam in sb.families:
        for m in fam.metrics:
            if m.key == key:
                return m
    raise AssertionError(f"记分牌里找不到指标 {key}")


def test_families_are_more_than_the_named_five():
    # 用户命名的五类不是全集：这里按代码+验收规范补出“命令服从与停站”“触地与冲击”。
    sb = build_scoreboard(_telemetry())
    assert [f.id for f in sb.families] == FAMILY_ORDER
    assert [f.id for f in sb.families] == [
        "velocity", "compliance", "gait", "contact", "efficiency", "terrain", "robustness"
    ]
    # 停站/命令服从这一整簇必须真的在板上，不是只有名字。
    keys = {m.key for f in sb.families for m in f.metrics}
    assert {"stand", "wrong_dir", "off_axis", "heading_hold"} <= keys


def test_coverage_flags_unmapped_reward_terms():
    # 覆盖自检：遥测里被优化、却没上板的奖励项要被列出来，让板子自曝盲区。
    pt = _point()
    pt.reward["hip_deviation"] = -0.3            # 一个还没上板的奖励项
    sb = build_scoreboard(_telemetry(pt))
    assert "hip_deviation" in sb.coverage.unmapped_reward_terms
    assert "tracking_lin" in sb.coverage.mapped_reward_terms
    assert sb.coverage.note


def test_spec_floor_provenance_is_clean():
    sb = build_scoreboard(_telemetry())
    lin = _find(sb, "lin_err")
    assert lin.floor == 0.10 and lin.floor_op == "<=" and lin.floor_confidence == "spec"
    assert "taili_spec" in lin.floor_source
    base = _find(sb, "base_h")
    assert base.floor == 0.47 and base.floor_op == ">=" and base.floor_confidence == "derived"
    torque = _find(sb, "torque_util")
    assert torque.floor == 0.85 and torque.floor_confidence == "spec"


def test_console_unmarked_floor_is_flagged_unknown():
    # duty_spread 的 0.20 是 telemetry.py 里写死、没标出处的显示常量——必须如实标 unknown。
    sb = build_scoreboard(_telemetry())
    m = _find(sb, "duty_spread_window")
    assert m.floor == 0.20 and m.floor_confidence == "unknown"
    assert "telemetry.py" in m.floor_source


def test_curriculum_floor_undefined_without_config():
    # 拿不到本次 run 配置时，课程门控不能编造，应显示未定底线，并在 notes 里说明。
    sb = build_scoreboard(_telemetry(), effective_config_text="")
    prog = _find(sb, "progress_gate")
    assert prog.floor is None and prog.status == "no_floor"
    assert any("effective_config" in n for n in sb.notes)


def test_curriculum_floor_resolves_from_config():
    sb = build_scoreboard(_telemetry(), effective_config_text=_CFG)
    prog = _find(sb, "progress_gate")
    assert prog.floor == 0.62 and prog.floor_op == ">=" and prog.floor_confidence == "config"
    assert prog.status == "below"                       # 0.55 < 0.62
    diag = _find(sb, "diagonal_contact")
    assert diag.floor == 0.50 and diag.floor_confidence == "config"
    terrain = _find(sb, "terrain_mean")
    assert terrain.floor == 3.5 and terrain.floor_confidence == "config"
    fall = _find(sb, "fall_rate")
    assert fall.floor == 0.05 and fall.floor_op == "<=" and fall.status == "meets"  # 0.03 <= 0.05


def test_status_is_facts_only():
    sb = build_scoreboard(_telemetry(), effective_config_text=_CFG)
    allowed = {"meets", "below", "no_floor", "no_data"}
    for fam in sb.families:
        for m in fam.metrics:
            assert m.status in allowed
    # 具体几个：身高 0.46<0.47 未达；力矩利用 0.72<=0.85 达标。
    assert _find(sb, "base_h").status == "below"
    assert _find(sb, "torque_util").status == "meets"


def test_penalty_values_are_absolute():
    # 奖励里的惩罚项是负数；记分牌显示其绝对值，避免“越负越好”的误读。
    sb = build_scoreboard(_telemetry())
    slip = _find(sb, "stance_slip")   # command 优先，取 0.15（正）
    assert slip.numeric_value == 0.15
    # 只有 reward 里有 stance_slip（负）时，应取绝对值。
    pt = _point()
    pt.command.pop("stance_slip")
    sb2 = build_scoreboard(_telemetry(pt))
    assert _find(sb2, "stance_slip").numeric_value == 0.4


def test_no_data_when_field_missing():
    pt = _point()
    pt.command.pop("yaw_err")
    sb = build_scoreboard(_telemetry(pt))
    m = _find(sb, "yaw_err")
    assert m.numeric_value is None and m.status == "no_data"


def test_every_floored_metric_has_provenance():
    # 守卫：凡是给了底线的指标，必须带出处和可信度——不允许出现“有底线但不知哪来的”。
    sb = build_scoreboard(_telemetry(), effective_config_text=_CFG)
    for fam in sb.families:
        for m in fam.metrics:
            if m.floor is not None:
                assert m.floor_source, f"{m.key} 有底线却没出处"
                assert m.floor_confidence in {"spec", "config", "derived", "default", "unknown"}


def test_tensions_are_symmetric():
    sb = build_scoreboard(_telemetry())
    assert len(sb.tensions) == len(TENSIONS)
    # progress_gate 与 stance_slip 应互相在对方的拆台清单里。
    assert "stance_slip" in _find(sb, "progress_gate").tensions
    assert "progress_gate" in _find(sb, "stance_slip").tensions


def test_context_strip_carries_curriculum_state():
    sb = build_scoreboard(_telemetry())
    assert sb.context.phase == "phi2"
    assert sb.context.penalty_gate == 0.6
    assert sb.context.terrain_mean == 3.2
    assert sb.context.note                      # 必须解释为什么跨时间比较要带这条


def test_unavailable_without_latest():
    sb = build_scoreboard(TrainingTelemetry(source="real", run_id="t", latest=None))
    assert sb.available is False and sb.status == "missing"
