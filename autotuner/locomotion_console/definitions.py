"""Definition registry for values shown in the locomotion console.

The registry is intentionally deterministic. LLM answers should use these
definitions as grounding before adding natural-language interpretation.
"""
from __future__ import annotations

from typing import Any, Iterable

from .schemas import DefinitionInfo, DefinitionQueryResult, TrainingTelemetry


_DEFINITIONS: list[DefinitionInfo] = [
    DefinitionInfo(
        key="run_snapshot",
        label="当前运行快照",
        category="system",
        formula="/run/current/telemetry 返回 snapshot：run、路径、最新 step、gate、分组指标、来源证据。",
        meaning="监控和 LLM 的统一事实入口。LLM 不应自己随意扫描远程路径，只能基于快照和允许的只读工具解释。",
        source="TrainingTelemetry.snapshot",
        related=["telemetry_jsonl", "train_log", "tensorboard", "checkpoint"],
        actions=["/status", "/paths"],
    ),
    DefinitionInfo(
        key="telemetry_jsonl",
        label="实时主事实源",
        category="system",
        formula="tail -n train.telemetry.jsonl；每行是一个 train_tick JSON。",
        meaning="当前状态以 JSONL 最新行优先。终端日志和 TensorBoard 只作为补充，不覆盖 JSONL 的最新事实。",
        source="run contract paths.telemetry_jsonl",
        related=["run_snapshot", "train_log", "tensorboard"],
        actions=["/paths", "/status"],
    ),
    DefinitionInfo(
        key="reward",
        label="总奖励",
        category="reward",
        formula="reward.total；终端 fallback 为 [TPREW] total/rew。",
        meaning="训练优化的综合目标。单独看总奖励没有诊断价值，必须同时看推进、步态质量、健康和课程 gate。",
        source="train.telemetry.jsonl reward.total / [TPREW] total / TensorBoard reward tags",
        related=["reward_total", "tracking_lin", "progress_gate", "gait_match", "fall_rate"],
        actions=["/explain progress_gate", "/explain gait_match"],
    ),
    DefinitionInfo(
        key="reward_total",
        label="总奖励",
        category="reward",
        formula="reward.total",
        meaning="同 reward。前端用 reward_total 避免和整组 reward 概念混淆。",
        source="train.telemetry.jsonl reward.total",
        related=["reward", "tracking_lin", "terrain_progress"],
        actions=["/explain reward", "/status"],
    ),
    DefinitionInfo(
        key="tracking_lin",
        label="近场线速度跟踪奖励",
        category="reward",
        formula="reward.tracking_lin；旧日志 fallback 可能叫 track。",
        meaning="越高通常表示实际线速度更接近命令速度，但它不是“会走”的充分条件。必须和 actual_vx、v_along、progress_gate、stance_slip 一起看。",
        source="train.telemetry.jsonl reward.tracking_lin / [TPREW] tracking_lin",
        related=["lin_err", "actual_vx", "progress_gate", "stance_slip"],
        actions=["/explain lin_err", "/explain progress_gate"],
    ),
    DefinitionInfo(
        key="tracking_lin_far",
        label="远场线速度跟踪奖励",
        category="reward",
        formula="reward.tracking_lin_far",
        meaning="误差较大时仍提供学习梯度的远场跟踪项。远场项高、近场项低，通常说明策略在学习方向但还没有精确跟踪。",
        source="train.telemetry.jsonl reward.tracking_lin_far / [TPREW] tracking_lin_far",
        related=["tracking_lin", "lin_err", "progress_gate"],
        actions=["/explain tracking_lin", "/status"],
    ),
    DefinitionInfo(
        key="terrain_progress",
        label="地形推进奖励",
        category="reward",
        formula="reward.terrain_progress",
        meaning="命令方向上的推进/地形相关奖励分量。它能说明策略在尝试推进，但不能替代实际速度和打滑检查。",
        source="train.telemetry.jsonl reward.terrain_progress / [TPREW] terrain_progress",
        related=["progress_gate", "v_along", "terrain"],
        actions=["/explain progress_gate", "/explain terrain"],
    ),
    DefinitionInfo(
        key="lin_err",
        label="线速度误差",
        category="diagnostic",
        formula="||commanded_xy_velocity - measured_base_xy_velocity||",
        meaning="越低越好。lin_err 高而 gait_match 高，说明节律可能有了但没有有效推进；lin_err 低但 slip 高，说明可能在滑。",
        source="train.telemetry.jsonl command.lin_err；fallback reward.lin_err / [TPCMD] lin_err",
        related=["tracking_lin", "cmd_vx", "actual_vx", "progress_gate"],
        actions=["/explain tracking_lin", "/explain stance_slip"],
    ),
    DefinitionInfo(
        key="cmd_vx",
        label="命令前向速度",
        category="diagnostic",
        formula="command.cmd_vx",
        meaning="训练当前要求策略达到的前向速度。只看实际速度不看命令，会误判是否跟踪成功。",
        source="train.telemetry.jsonl command.cmd_vx / [TPCMD] cmd_vx",
        related=["actual_vx", "lin_err", "progress_gate"],
        actions=["/explain actual_vx"],
    ),
    DefinitionInfo(
        key="actual_vx",
        label="实际前向速度",
        category="diagnostic",
        formula="command.actual_vx",
        meaning="仿真 base 的实际 x 方向速度。前进阶段应接近 cmd_vx；mixed/yaw 阶段还要看 v_along。",
        source="train.telemetry.jsonl command.actual_vx / [TPCMD] act_vx",
        related=["cmd_vx", "v_along", "lin_err"],
        actions=["/explain v_along", "/status"],
    ),
    DefinitionInfo(
        key="v_along",
        label="命令方向速度",
        category="diagnostic",
        formula="project(measured_base_xy_velocity, command_direction)",
        meaning="实际速度投影到命令方向。比 actual_vx 更适合 lateral/yaw/mixed 阶段。",
        source="train.telemetry.jsonl command.v_along / [TPCMD] v_along",
        related=["cmd_vx", "actual_vx", "progress_gate"],
        actions=["/explain progress_gate"],
    ),
    DefinitionInfo(
        key="progress_gate",
        label="阶段推进 gate",
        category="curriculum",
        formula="recent-window command progress over active_dirs; compared with phase_gate_prog_N",
        meaning="课程是否允许进入下一阶段的主门槛。如果 blocked_by=progress，训练不是卡死，而是当前策略推进能力没达标。",
        source="train.telemetry.jsonl curriculum.progress_gate / effective_config.yaml phase_gate_prog_*",
        related=["phase", "active_dirs", "lin_err", "tracking_lin"],
        actions=["/explain active_dirs", "/status"],
    ),
    DefinitionInfo(
        key="gait_match",
        label="步态匹配",
        category="diagnostic",
        formula="command.gait_match；fallback curriculum.gait_gate",
        meaning="接触节律与期望步态的匹配度。高值不等于会走，必须和 progress_gate、diagonal_contact、duty_balance、stance_slip 一起判断。",
        source="train.telemetry.jsonl command.gait_match / curriculum.gait_gate",
        related=["diagonal_contact", "duty_balance", "stance_slip", "progress_gate"],
        actions=["/explain diagonal_contact", "/explain stance_slip"],
    ),
    DefinitionInfo(
        key="gait_gate",
        label="课程步态 gate",
        category="curriculum",
        formula="curriculum.gait_gate",
        meaning="课程状态机使用的步态质量代理。它用于过阶段，但不是对前端渲染中“步态好不好”的完整解释。",
        source="train.telemetry.jsonl curriculum.gait_gate / [TPCURR] gait_gate",
        related=["gait_match", "diagonal_contact", "duty_balance"],
        actions=["/explain gait_match"],
    ),
    DefinitionInfo(
        key="diagonal_contact",
        label="对角支撑一致性",
        category="diagnostic",
        formula="command.diagonal_contact",
        meaning="对角腿接触组合是否接近 trot。低值说明可能出现 crawl、拖行或乱步，即使总奖励暂时不低也不能算满足策略目标。",
        source="train.telemetry.jsonl command.diagonal_contact",
        related=["gait_match", "duty_balance", "stance_slip"],
        actions=["/explain duty_balance", "/explain gait_match"],
    ),
    DefinitionInfo(
        key="duty_balance",
        label="占空比平衡",
        category="diagnostic",
        formula="command.duty_balance",
        meaning="前后腿/左右腿站立占空比平衡评分。低值常对应后腿长时间撑地、前腿拖行、步态偏置。",
        source="train.telemetry.jsonl command.duty_balance",
        related=["diagonal_contact", "gait_match", "stance_slip"],
        actions=["/explain diagonal_contact"],
    ),
    DefinitionInfo(
        key="stance_slip",
        label="支撑期打滑",
        category="diagnostic",
        formula="command.stance_slip；fallback abs(reward.stance_slip)",
        meaning="支撑脚在地面上的水平滑动代理，越低越好。高 slip 说明不是稳态推进，而是在拖/滑。",
        source="train.telemetry.jsonl command.stance_slip / reward.stance_slip",
        related=["gait_match", "terrain", "fall_rate"],
        actions=["/explain gait_match", "/explain terrain"],
    ),
    DefinitionInfo(
        key="diagonal_pair_instant",
        label="瞬时严格对角",
        category="diagnostic",
        formula="1 only when contact is exactly FL+RR or FR+RL; otherwise 0",
        meaning="当前帧是否是真正对角支撑。四脚拖行、三脚支撑、单脚支撑都不给对角分。",
        source="train.telemetry.jsonl command.diagonal_pair_instant / [TPCMD] diag_inst",
        related=["diagonal_contact", "gait_match", "duty_balance"],
        actions=["/explain diagonal_contact"],
    ),
    DefinitionInfo(
        key="duty_spread_window",
        label="占空差窗口",
        category="diagnostic",
        formula="max(per_leg_duty_ema) - min(per_leg_duty_ema)",
        meaning="窗口内各腿站立占空比的差。高值说明某些腿长期撑地或长期离地，是 crawl/拖行/偏置步态的直接证据。",
        source="train.telemetry.jsonl command.duty_spread_window / [TPCMD] duty_spread",
        related=["duty_balance", "diagonal_contact", "stance_slip"],
        actions=["/explain duty_balance"],
    ),
    DefinitionInfo(
        key="stance_slip_high_fraction",
        label="高滑移比例",
        category="diagnostic",
        formula="EMA(fraction of settled stance feet with slip speed > slip_high_threshold)",
        meaning="支撑脚高滑移事件比例。它补足平均 stance_slip 被稀释的问题，能暴露局部滑移 bout。",
        source="train.telemetry.jsonl command.stance_slip_high_fraction / [TPCMD] slip_high",
        related=["stance_slip", "diagonal_contact", "fall_rate"],
        actions=["/explain stance_slip"],
    ),
    DefinitionInfo(
        key="height_low_risk_window",
        label="低高度风险窗口",
        category="diagnostic",
        formula="EMA(clamp((h_ok - base_h) / (h_ok - h_gate_close), 0, 1))",
        meaning="训练 stable_motion_gate 使用的近期低身高风险。非零说明近期出现过蹲塌/低高度片段，即使平均 base_h 看起来正常。",
        source="train.telemetry.jsonl health.height_low_risk_window / [TPGATE] height_low_risk",
        related=["base_h", "base_h_min", "stable_motion_gate"],
        actions=["/explain base_h"],
    ),
    DefinitionInfo(
        key="tilt_high_risk_window",
        label="大倾角风险窗口",
        category="diagnostic",
        formula="EMA(clamp((tilt - tilt_ok) / (tilt_gate_close - tilt_ok), 0, 1))",
        meaning="训练 stable_motion_gate 使用的近期大倾角风险。非零说明姿态风险已在折扣正奖励。",
        source="train.telemetry.jsonl health.tilt_high_risk_window / [TPGATE] tilt_high_risk",
        related=["tilt_deg", "tilt_deg_max", "stable_motion_gate"],
        actions=["/explain tilt_deg"],
    ),
    DefinitionInfo(
        key="landing_impact",
        label="落地冲击惩罚",
        category="reward",
        formula="abs(reward.landing_impact)",
        meaning="触地瞬间冲击相关惩罚。高值常表示摆腿轨迹、触地时机或机身姿态不稳。",
        source="train.telemetry.jsonl reward.landing_impact / [TPPEN] landing_impact",
        related=["stance_slip", "tilt_deg", "clearance_under"],
        actions=["/explain stance_slip"],
    ),
    DefinitionInfo(
        key="clearance_under",
        label="清高不足惩罚",
        category="reward",
        formula="abs(reward.clearance_under)",
        meaning="摆动脚低于目标清高带的惩罚，越低越好。高值意味着拖脚/绊脚风险。",
        source="train.telemetry.jsonl reward.clearance_under / [TPPEN] clearance_under",
        related=["clearance_over", "stance_slip", "terrain"],
        actions=["/explain terrain"],
    ),
    DefinitionInfo(
        key="clearance_over",
        label="清高过高惩罚",
        category="reward",
        formula="abs(reward.clearance_over)",
        meaning="摆动脚高于目标清高带的惩罚，越低越好。高值意味着动作浪费、冲击风险增加。",
        source="train.telemetry.jsonl reward.clearance_over / [TPPEN] clearance_over",
        related=["clearance_under", "landing_impact"],
        actions=["/explain landing_impact"],
    ),
    DefinitionInfo(
        key="phase",
        label="训练阶段",
        category="curriculum",
        formula="curriculum.phase",
        meaning="课程状态机当前阶段。它决定命令分布、active_dirs、质量惩罚、地形和 DR 是否开放。",
        source="train.telemetry.jsonl curriculum.phase / [TPCURR] phase",
        related=["command_mode", "active_dirs", "progress_gate"],
        actions=["/explain progress_gate"],
    ),
    DefinitionInfo(
        key="command_mode",
        label="命令采样模式",
        category="curriculum",
        formula="training_recipe.phases[current_phase].command_mode",
        meaning="fixed_forward 只练前进；single_axis 分方向；mixed 允许 vx/vy/wz 同时非零。不同模式下要看不同跟踪指标。",
        source="train.telemetry.jsonl curriculum.command_mode / effective_config.yaml",
        related=["active_dirs", "phase", "cmd_vx"],
        actions=["/explain active_dirs"],
    ),
    DefinitionInfo(
        key="active_dirs",
        label="当前验收方向集合",
        category="curriculum",
        formula="training_recipe.phases[current_phase].active_dirs",
        meaning="当前 progress_gate 统计的方向集合。未进入 back/lat/yaw 阶段时，不应该用那些方向的失败来判定训练卡住。",
        source="train.telemetry.jsonl curriculum.active_dirs / effective_config.yaml",
        related=["command_mode", "progress_gate", "phase"],
        actions=["/explain progress_gate"],
    ),
    DefinitionInfo(
        key="penalty_gate",
        label="质量惩罚渐入",
        category="curriculum",
        formula="phase/curriculum interval controlled ramp from 0 to 1",
        meaning="控制打滑、冲击、清高等质量项逐步变强。早期为 0 时，步态质量差不会被强力压制。",
        source="train.telemetry.jsonl curriculum.penalty_gate / effective_config.yaml",
        related=["stance_slip", "landing_impact", "clearance_under"],
        actions=["/explain stance_slip"],
    ),
    DefinitionInfo(
        key="clearance_gate",
        label="清高/地形渐入",
        category="curriculum",
        formula="curriculum.clearance_gate",
        meaning="控制地形清高相关要求的渐入程度。过早全开会破坏基础步态，过晚会卡地形适应。",
        source="train.telemetry.jsonl curriculum.clearance_gate / effective_config.yaml",
        related=["clearance_under", "terrain", "phase"],
        actions=["/explain clearance_under"],
    ),
    DefinitionInfo(
        key="terrain",
        label="地形课程",
        category="terrain",
        formula="curriculum.terrain_mean / terrain_max",
        meaning="当前地形难度。reward 波动或摔倒上升时，要先看是否刚升级地形或 DR。",
        source="train.telemetry.jsonl curriculum.terrain_mean/terrain_max",
        related=["terrain_mean", "terrain_max", "dr_level", "fall_rate"],
        actions=["/explain dr_level"],
    ),
    DefinitionInfo(
        key="terrain_mean",
        label="平均地形等级",
        category="terrain",
        formula="curriculum.terrain_mean",
        meaning="训练环境当前平均地形等级。",
        source="train.telemetry.jsonl curriculum.terrain_mean",
        related=["terrain", "dr_level", "fall_rate"],
        actions=["/explain terrain"],
    ),
    DefinitionInfo(
        key="dr_level",
        label="Domain Randomization 等级",
        category="curriculum",
        formula="curriculum.dr_level",
        meaning="DR 越高，随机化越强，训练更鲁棒但更难。解释 reward 突变时必须看 DR 是否变化。",
        source="train.telemetry.jsonl curriculum.dr_level / effective_config.yaml",
        related=["terrain", "fall_rate", "progress_gate"],
        actions=["/explain terrain"],
    ),
    DefinitionInfo(
        key="fall_rate",
        label="摔倒率",
        category="diagnostic",
        formula="falls / recent episodes or rollout window proxy",
        meaning="稳定性主指标。fall_rate 高时优先修稳定性；fall_rate 低但 progress 低，说明更可能是推进/跟踪不足。",
        source="train.telemetry.jsonl health.fall_rate / [TPGATE] fall_rate / diagnostics metrics",
        related=["terminal_rate", "tilt_deg", "base_h"],
        actions=["/explain terminal_rate", "/status"],
    ),
    DefinitionInfo(
        key="terminal_rate",
        label="终止率",
        category="diagnostic",
        formula="terminated episodes / recent episodes or rollout window proxy",
        meaning="判断 episode 是否频繁提前结束。终止高会污染 reward 和跟踪曲线。",
        source="train.telemetry.jsonl health.terminal_rate / [TPGATE] terminal_rate",
        related=["fall_rate", "base_h", "tilt_deg"],
        actions=["/explain fall_rate"],
    ),
    DefinitionInfo(
        key="base_h",
        label="机身高度",
        category="diagnostic",
        formula="health.base_h",
        meaning="base link 高度。过低通常是塌陷/蹲伏，过高可能是跳跃或不稳定动作。",
        source="train.telemetry.jsonl health.base_h / [TPGATE] base_h",
        related=["upright", "tilt_deg", "fall_rate"],
        actions=["/explain tilt_deg"],
    ),
    DefinitionInfo(
        key="height_low",
        label="低身高惩罚",
        category="reward",
        formula="reward.height_low = -w_height_low * max((height_min_ok - base_h) / (nominal_base_h - height_min_ok), 0)^2",
        meaning="当 base_h 低于 spec 下界时出现的显式惩罚。当前 Taili nominal_base_h=0.52，height_min_ok=0.47；0.45m 会被明确扣分。",
        source="train.telemetry.jsonl reward.height_low / [TPPEN] height_low",
        related=["base_h", "reward", "stable_motion_gate"],
        actions=["/explain base_h", "/status"],
    ),
    DefinitionInfo(
        key="upright",
        label="直立度",
        category="diagnostic",
        formula="dot(base_up, world_z)",
        meaning="越高越直立。低值说明机身倾斜，会影响接触节律和推进。",
        source="train.telemetry.jsonl health.upright / [TPGATE] upright",
        related=["tilt_deg", "fall_rate"],
        actions=["/explain tilt_deg"],
    ),
    DefinitionInfo(
        key="tilt_deg",
        label="机身倾角",
        category="diagnostic",
        formula="angle(base_up, world_z) in degrees",
        meaning="越低越好。持续偏大说明姿态不稳或步态偏置。",
        source="train.telemetry.jsonl health.tilt_deg / [TPGATE] tilt_deg",
        related=["upright", "base_h", "fall_rate"],
        actions=["/explain upright"],
    ),
    DefinitionInfo(
        key="torque_util",
        label="力矩利用率",
        category="diagnostic",
        formula="normalized applied torque utilization",
        meaning="持续高值说明策略可能靠硬顶力矩运动，部署风险高。",
        source="train.telemetry.jsonl health.torque_util / [TPGATE] torque_util",
        related=["torque_saturation", "fall_rate"],
        actions=["/explain torque_saturation"],
    ),
    DefinitionInfo(
        key="checkpoint",
        label="检查点",
        category="system",
        formula="checkpoints/agent_<step>.pt or catalog-selected checkpoint",
        meaning="诊断和恢复使用的策略文件。控制台必须以远程 catalog/snapshot 为准，不能硬编码本地路径。",
        source="telemetry checkpoint / diagnostics catalog / RunSnapshot provenance",
        related=["checkpoint_dir", "run_snapshot"],
        actions=["/paths", "/probe checkpoints"],
    ),
]


def list_definitions(query: str = "") -> DefinitionQueryResult:
    q = (query or "").strip().lower()
    if not q:
        return DefinitionQueryResult(query=query, items=_DEFINITIONS)

    def score(item: DefinitionInfo) -> int:
        haystack = " ".join(
            [
                item.key,
                item.label,
                item.meaning,
                item.formula,
                item.source,
                " ".join(item.related),
            ]
        ).lower()
        if item.key.lower() == q:
            return 0
        if q in item.key.lower() or q in item.label.lower():
            return 1
        if q in haystack:
            return 2
        return 99

    items = sorted((item for item in _DEFINITIONS if score(item) < 99), key=score)
    return DefinitionQueryResult(query=query, items=items)


def definitions_for_telemetry(telemetry: TrainingTelemetry) -> list[DefinitionInfo]:
    if not telemetry.latest:
        return list_definitions("").items[:8]
    keys = {
        "run_snapshot",
        "telemetry_jsonl",
        "reward",
        "terrain",
        "phase",
        "progress_gate",
        "dr_level",
        "fall_rate",
        "checkpoint",
        "command_mode",
        "active_dirs",
        "penalty_gate",
        "clearance_gate",
    }
    latest = telemetry.latest
    keys.update(latest.reward.keys())
    keys.update(latest.curriculum.keys())
    keys.update(latest.health.keys())
    keys.update(latest.command.keys())
    keys.update(latest.counters.keys())
    keys.update(latest.paths.keys())
    if telemetry.snapshot:
        keys.update(item.key for group in telemetry.snapshot.metric_groups for item in group.items)
        keys.update(item.key for item in telemetry.snapshot.provenance)

    found: list[DefinitionInfo] = []
    seen: set[str] = set()
    for item in _DEFINITIONS:
        if item.key in keys or any(related in keys for related in item.related):
            found.append(_with_current_reading(item, telemetry))
            seen.add(item.key)
    for fallback in ["run_snapshot", "progress_gate", "gait_match", "stance_slip", "fall_rate"]:
        if fallback not in seen:
            matches = list_definitions(fallback).items
            if matches:
                found.append(_with_current_reading(matches[0], telemetry))
                seen.add(fallback)
    return found


def explain_terms(terms: Iterable[str], telemetry: TrainingTelemetry | None = None) -> str:
    lines: list[str] = []
    for term in terms:
        matches = list_definitions(term).items
        if not matches:
            lines.append(f"{term}: 暂无定义。可用 /help 查看可解释项。")
            continue
        item = matches[0]
        if telemetry:
            item = _with_current_reading(item, telemetry)
        reading = f"\n现在：{item.current_reading}" if item.current_reading else ""
        lines.append(
            f"{item.label}（{item.key}）\n"
            f"含义：{item.meaning}\n"
            f"计算：{item.formula}\n"
            f"来源：{item.source}"
            f"{reading}"
        )
    return "\n\n".join(lines)


def _value_for_definition(item: DefinitionInfo, telemetry: TrainingTelemetry) -> Any:
    latest = telemetry.latest
    if latest is None:
        return None
    if item.key == "reward":
        return latest.reward.get("total") or latest.reward.get("rew")
    if item.key == "reward_total":
        return latest.reward.get("total") or latest.reward.get("rew")
    if item.key == "terrain":
        return latest.curriculum.get("terrain_mean")
    if item.key == "checkpoint":
        return latest.checkpoint or None
    if item.key == "run_snapshot":
        return telemetry.snapshot.conclusion if telemetry.snapshot else None
    if item.key == "telemetry_jsonl":
        return telemetry.telemetry_path or latest.paths.get("telemetry_jsonl")
    for mapping in [latest.reward, latest.curriculum, latest.health, latest.command, latest.counters, latest.paths]:
        if item.key in mapping:
            return mapping[item.key]
    if telemetry.snapshot:
        for group in telemetry.snapshot.metric_groups:
            for metric in group.items:
                if metric.key == item.key:
                    return metric.value
    return None


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _with_current_reading(item: DefinitionInfo, telemetry: TrainingTelemetry) -> DefinitionInfo:
    value = _value_for_definition(item, telemetry)
    if value is None:
        return item.model_copy(update={"current_reading": "当前 telemetry 未提供该字段。"})
    latest = telemetry.latest
    context = ""
    if latest:
        if item.key in {"tracking_lin", "lin_err", "actual_vx", "cmd_vx", "v_along"}:
            cmd_vx = latest.command.get("cmd_vx")
            actual_vx = latest.command.get("actual_vx")
            lin_err = latest.command.get("lin_err") or latest.reward.get("lin_err")
            progress = latest.curriculum.get("progress_gate") or latest.command.get("progress_ratio")
            context = _kv_context(
                [
                    ("cmd_vx", cmd_vx),
                    ("actual_vx", actual_vx),
                    ("lin_err", lin_err),
                    ("progress_gate", progress),
                ]
            )
        elif item.key in {"progress_gate", "phase", "active_dirs", "command_mode"}:
            context = _kv_context(
                [
                    ("phase", latest.curriculum.get("phase")),
                    ("active_dirs", latest.curriculum.get("active_dirs")),
                    ("blocked_by", latest.curriculum.get("blocked_by")),
                    ("next_gate", latest.curriculum.get("next_gate") or latest.curriculum.get("next")),
                ]
            )
        elif item.key in {"gait_match", "gait_gate", "diagonal_contact", "duty_balance", "stance_slip"}:
            context = _kv_context(
                [
                    ("gait_match", latest.command.get("gait_match") or latest.curriculum.get("gait_gate")),
                    ("diagonal_contact", latest.command.get("diagonal_contact")),
                    ("duty_balance", latest.command.get("duty_balance")),
                    ("stance_slip", latest.command.get("stance_slip") or latest.reward.get("stance_slip")),
                ]
            )
        elif item.key in {"fall_rate", "terminal_rate", "base_h", "upright", "tilt_deg"}:
            context = _kv_context(
                [
                    ("fall_rate", latest.health.get("fall_rate")),
                    ("terminal_rate", latest.health.get("terminal_rate")),
                    ("base_h", latest.health.get("base_h")),
                    ("tilt_deg", latest.health.get("tilt_deg")),
                ]
            )
    suffix = f"；{context}" if context else ""
    return item.model_copy(update={"current_reading": f"{item.key}={_format_value(value)}{suffix}"})


def _kv_context(items: list[tuple[str, Any]]) -> str:
    parts: list[str] = []
    for key, value in items:
        if value is None or value == "":
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.3f}")
        else:
            parts.append(f"{key}={value}")
    return "，".join(parts)
