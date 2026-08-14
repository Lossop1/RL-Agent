"""目标记分牌装配器（Objective Scoreboard）。

它把训练遥测重排成用户关心的多类目标，每个指标给出三件事：
现在是多少 / 底线是多少 / 这条底线哪来的。

设计原则（和控制台既定规矩一致）：
  1. 只讲事实。状态只有 达标(meets)/未达底线(below)/未定底线(no_floor)/无数据(no_data)，
     绝不替用户下“好/坏”判断。
  2. 出处显式。每条底线都标明来自：规范文档(spec)、本次 run 有效配置(config)、
     由规范量推导(derived)、配置缺失的兜底默认(default)、还是代码里写死但没标出处(unknown)。
  3. 目标族不是全集。用户命名的“速度/步态/效率/地形/鲁棒”五类会漏掉一批被优化的目标
     （最明显的是“命令服从与停站”整簇）。这里按代码和验收规范(它自己把“什么算好”拆成
     A–F 六大族)补全，并用 coverage 自检把仍未上板的奖励项显式列出——绝不假装数全了。

已知需要如实标出来的坑（来自代码勘察）：
  - duty_spread / slip 事件比例 / 低身高风险 / 大倾角风险 这几条底线是 telemetry.py:951-954
    写死的显示常量，没有出处 → 一律标 confidence=unknown。
  - 抗扰恢复(推挤/低摩擦 E1–E3)与域随机迁移(E4/E5)在 physeval 侧测量，活遥测无直读，
    这里以家族附注点明其存在，不留白装作没有这些目标。
"""
from __future__ import annotations

import time

from .schemas import (
    Scoreboard,
    ScoreboardContext,
    ScoreboardCoverage,
    ScoreboardFamily,
    ScoreboardMetric,
    ScoreboardTension,
    TrainingTelemetry,
    TrainingTelemetryPoint,
)
from .telemetry import _actual_phase_gate_from_config, _as_float, _num


# ── 目标族（可扩展，非全集）─────────────────────────────────────────────────
FAMILY_ORDER = ["velocity", "compliance", "gait", "contact", "efficiency", "terrain", "robustness"]
FAMILY_META = {
    "velocity": ("速度跟踪", "机器人是否按命令的速度和方向在走（规范 A1/A2）。"),
    "compliance": ("命令服从与停站",
                   "叫停能不能干净站住、有没有走反方向、纯轴命令下会不会侧向漂移（规范 A3/C）。"
                   "这一类既不属于速度也不属于鲁棒，五类划分里最容易漏掉。"),
    "gait": ("步态质量", "是干净的对角小跑还是拖行/爬行/乱步（规范 B）。跟踪不差也可能步态很烂。"),
    "contact": ("触地与冲击", "落地软不软、触地滑不滑（规范 B1/B2）。和步态节律是两种不同的失败。"),
    "efficiency": ("效率与可部署性", "是否靠硬顶力矩、大动作硬走——影响能耗和能否部署到真机（规范 F）。"),
    "terrain": ("地形推进", "在粗糙地面/台阶上的通过与推进能力（规范 D）。"),
    "robustness": ("稳定与鲁棒", "会不会塌/摔、姿态稳不稳（规范 E 的静态部分）。"
                   "抗扰恢复(推挤/低摩擦)属规范 E1–E3，在 physeval 侧测，活遥测无直读，见家族附注。"),
}


# ── 底线来源（三类）─────────────────────────────────────────────────────────
# A) 规范底线：出处 docs/taili_spec.md，由 acceptance_score.py 执行。
#    元组含义：(方向, 数值, 出处, 可信度, 附注)。方向 "<=" 为上界、">=" 为下界。
SPEC_FLOORS = {
    "lin_err": ("<=", 0.10,
                "acceptance_score.score_A1 / taili_spec §3.1：|v−cmd|≤max(0.10, 0.15|cmd|)",
                "spec", "验收按速度桶测 median 与 p90；此处是训练活窗口的近似读数，口径略宽。"),
    "yaw_err": ("<=", 0.15,
                "acceptance_score.score_A2 / taili_spec §3.1：|ωz−cmd|≤0.15 rad/s",
                "spec", ""),
    "stance_slip": ("<=", 0.05,
                    "acceptance_score.score_B2 平地 / taili_spec §B2（地形放宽到 0.10）",
                    "spec", "训练早期活配置的滑移门控通常更宽松；这里显示的是规范验收底线。"),
    "base_h": (">=", 0.47,
               "nominal_base_h(0.52)−0.05；diagnostics.DiagnosticCriteriaSpec.min_height_m=0.47",
               "derived", ""),
    "tilt_deg": ("<=", 15.0,
                 "tilt_ok_rad=0.2618≈15°（taili_reward.py:210 stable_motion_gate 软上限）",
                 "derived", ""),
    # 说明：base_h_min / tilt_deg_max 是“批次内最差单环境”极值，故意不设达标线——
    # 几百个并行环境里只要有一个在摔/重置就会爆表，拿它比 0.47/15° 会误导。
    # 达标线放在对应的均值行（base_h / tilt_deg），极值行只作短时事件的发现器。
    "upright": (">=", 0.99,
                "acceptance_score.score_A3（立正）：upright≥0.99",
                "spec", "0.99 是立正判据；正常行走时可低于它，别据此判行走失败。"),
    "torque_util": ("<=", 0.85,
                    "acceptance_score.score_F2 / taili_spec §F2：99.5 分位 |τ|/limit≤0.85",
                    "spec", "活值是均值代理；验收测的是 99.5 分位，两者口径不同。"),
    "torque_saturation": ("<=", 0.0,
                          "acceptance_score.score_F2：峰值 |τ|/limit≤1.0、clamp_rate=0",
                          "spec", "这是惩罚代理，>0 即出现接近/超限力矩。"),
}

# B) 出处不明的控制台显示常量：如实标 unknown，绝不冒充成真实底线。
CONSOLE_UNMARKED_FLOORS = {
    "duty_spread_window": ("<=", 0.20,
                           "控制台显示常量（telemetry.py:951 duty_spread_target，未标出处）",
                           "unknown", ""),
    "stance_slip_high_fraction": ("<=", 0.05,
                                  "控制台显示常量（telemetry.py:952 slip_event_target，未标出处）",
                                  "unknown", "事件阈值 slip_high_threshold=0.20 有代码依据，但这个 0.05 的比例底线没有。"),
    "height_low_risk_window": ("<=", 0.05,
                               "控制台显示常量（telemetry.py:953 height_risk_target，未标出处）",
                               "unknown", ""),
    "tilt_high_risk_window": ("<=", 0.05,
                              "控制台显示常量（telemetry.py:954 tilt_risk_target，未标出处）",
                              "unknown", ""),
}

# C) 课程类门控：运行时从本次 run 的 effective_config 读，带 config/default 出处。
#    值含义：(遥测门控条件 key, 方向)。
CONFIG_GATE = {
    "progress_gate": ("progress_min", ">="),
    "diagonal_contact": ("diagonal_min", ">="),
    "duty_balance": ("duty_min", ">="),
    "fall_rate": ("fall_max", "<="),
    "terrain_mean": ("terrain_min", ">="),
}


# ── 互相拆台关系（改一个更好，另一个通常更差）────────────────────────────────
TENSIONS = [
    ("progress_gate", "stance_slip", "推得越快越容易滑"),
    ("v_along", "stance_slip", "速度越高支撑滑移越大"),
    ("progress_gate", "torque_util", "硬推速度会顶高力矩"),
    ("v_along", "torque_util", "同上：速度换力矩"),
    ("clearance_under", "clearance_over", "抬脚高度的两个反向边界，此消彼长"),
    ("terrain_mean", "torque_util", "爬台阶/越障要更大抬腿和力矩"),
    ("terrain_mean", "clearance_over", "越障要抬高脚，与省抬腿相抵触"),
    ("progress_gate", "duty_balance", "抄近路快走会牺牲步态对称"),
    ("yaw_err", "off_axis", "转向时容易在非命令轴上漂移"),
    ("stand", "progress_gate", "站得住与推得动是两种模式，早期常此消彼长"),
    ("v_along", "landing_impact", "走得快落地更硬"),
]


def _tensions_for(key: str) -> list[str]:
    """返回与该指标直接拆台的其它指标 key。"""
    out: list[str] = []
    for a, b, _ in TENSIONS:
        if a == key and b not in out:
            out.append(b)
        elif b == key and a not in out:
            out.append(a)
    return out


# ── 指标表 ───────────────────────────────────────────────────────────────────
# 每个指标：族 / key / 标签 / 取值候选（"段:字段"，按顺序取第一个有数值的）/ 单位 /
# 小数位 / abs=是否取绝对值（惩罚项取正）/ meaning=一句话含义 / note=固定附注 /
# value_confidence=当前值可信度（地形扫描派生量标 low）。
# 底线不写在这里——由 _resolve_floor 按 A/B/C 三类来源统一解析，保证出处可控。
METRICS = [
    # —— 速度跟踪 ——
    dict(family="velocity", key="lin_err", label="线速度误差",
         cands=["command:lin_err", "reward:lin_err"], unit="m/s", precision=3,
         meaning="命令线速度和实际线速度的差，越低越准。"),
    dict(family="velocity", key="yaw_err", label="转向速度误差",
         cands=["command:yaw_err"], unit="rad/s", precision=3,
         meaning="命令转向角速度和实际的差。"),
    dict(family="velocity", key="v_along", label="命令方向速度",
         cands=["command:v_along"], unit="m/s", precision=3,
         meaning="实际速度投影到命令方向，比单看 vx 更适合转向/侧移阶段。"),
    dict(family="velocity", key="progress_gate", label="阶段推进 gate",
         cands=["curriculum:progress_gate", "command:progress_ratio"], unit="", precision=3,
         meaning="课程认为命令方向推进够不够，是过阶段的主门槛。"),
    dict(family="velocity", key="tracking_lin", label="近场跟踪奖励",
         cands=["reward:tracking_lin", "reward:track"], unit="", precision=3,
         meaning="线速度跟踪的奖励项（是奖励不是验收量，本身无底线）。"),

    # —— 命令服从与停站 ——（五类划分里最容易漏掉的一整簇）
    dict(family="compliance", key="stand", label="静止综合奖励",
         cands=["reward:stand"], unit="", precision=3,
         meaning="零命令时低速/低姿态误差/低动作的综合奖励，越高越稳站。"),
    dict(family="compliance", key="stand_contact", label="四脚支撑奖励",
         cands=["reward:stand_contact"], unit="", precision=3,
         meaning="零命令时四脚落地的奖励，压制“身子不动、脚在乱挪”。"),
    dict(family="compliance", key="settle_brake", label="停止减速奖励",
         cands=["reward:settle_brake"], unit="", precision=3,
         meaning="收到停止命令后主动减速的奖励。"),
    dict(family="compliance", key="wrong_dir", label="反向运动惩罚",
         cands=["reward:wrong_dir"], unit="", precision=3, abs=True,
         meaning="朝命令反方向运动的惩罚代理；>0 即在走反方向。"),
    dict(family="compliance", key="off_axis", label="离轴漂移惩罚",
         cands=["reward:off_axis"], unit="", precision=3, abs=True,
         meaning="在近零命令的轴上产生速度（漂移）的惩罚代理。"),
    dict(family="compliance", key="planar_purity", label="跨轴纯度惩罚",
         cands=["reward:planar_purity"], unit="", precision=3, abs=True,
         meaning="纯前进/侧移/转向命令下出现跨轴速度的惩罚代理。"),
    dict(family="compliance", key="heading_hold", label="朝向保持奖励",
         cands=["reward:heading_hold"], unit="", precision=3,
         meaning="无转向命令、只有线速度命令时保持朝向的奖励。"),

    # —— 步态质量 ——
    dict(family="gait", key="diagonal_contact", label="对角支撑一致性",
         cands=["command:diagonal_contact", "curriculum:diagonal_gate"], unit="", precision=3,
         meaning="对角腿接触与 trot 的一致性；低=拖行/爬行/三脚/乱步。"),
    dict(family="gait", key="duty_balance", label="占空比平衡",
         cands=["command:duty_balance", "curriculum:duty_balance_gate"], unit="", precision=3,
         meaning="各腿站立占空比是否均衡；低=偏腿、后腿长撑、前腿拖行。"),
    dict(family="gait", key="duty_spread_window", label="占空差窗口",
         cands=["command:duty_spread_window"], unit="", precision=3,
         meaning="各腿长期站立占空差；高=crawl/拖行/前后偏置。"),
    dict(family="gait", key="gait_match", label="步态匹配",
         cands=["command:gait_match", "curriculum:gait_gate", "reward:gait"], unit="", precision=3,
         note="训练活配置里步态没有单独门控（gait 未进 phase gate），所以这里通常无 config 底线。",
         meaning="接触节律与期望步态的匹配度；高步态分不等于真前进好。"),
    dict(family="gait", key="clearance_under", label="清高不足惩罚",
         cands=["reward:clearance_under"], unit="", precision=3, abs=True,
         meaning="摆动脚低于目标清高带的惩罚代理；高=易拖脚/绊脚。"),
    dict(family="gait", key="clearance_over", label="清高过高惩罚",
         cands=["reward:clearance_over"], unit="", precision=3, abs=True,
         meaning="摆动脚抬太高的惩罚代理；高=动作浪费、冲击风险上升。"),

    # —— 触地与冲击 ——
    dict(family="contact", key="stance_slip", label="支撑期打滑",
         cands=["command:stance_slip", "reward:stance_slip"], unit="m/s", precision=3, abs=True,
         meaning="支撑脚在地上的水平滑动；低=真在走，不是在滑/拖。"),
    dict(family="contact", key="stance_slip_high_fraction", label="高滑移比例",
         cands=["command:stance_slip_high_fraction"], unit="", precision=3,
         meaning="局部高滑移事件比例，补足平均 slip 被稀释的问题。"),
    dict(family="contact", key="landing_impact", label="落地冲击惩罚",
         cands=["reward:landing_impact"], unit="", precision=3, abs=True,
         meaning="触地冲击惩罚代理；规范 B1 实测的是触地垂速 p95，这里只是奖励代理。"),
    dict(family="contact", key="touchdown_slip", label="触地滑移惩罚",
         cands=["reward:touchdown_slip"], unit="", precision=3, abs=True,
         meaning="触地瞬间水平滑移的惩罚代理；高=落脚在蹭。"),

    # —— 效率与可部署性 ——
    dict(family="efficiency", key="torque_util", label="力矩利用率",
         cands=["health:torque_util"], unit="", precision=3,
         meaning="实际力矩/力矩上限；高=靠硬顶力矩走，真机部署风险大。"),
    dict(family="efficiency", key="torque_saturation", label="力矩饱和惩罚",
         cands=["reward:torque_saturation", "reward:torque_sat"], unit="", precision=3, abs=True,
         meaning="接近/超过力矩边界的惩罚代理；>0 即出现力矩饱和。"),
    dict(family="efficiency", key="action_rate", label="动作变化率惩罚",
         cands=["reward:action_rate"], unit="", precision=3, abs=True,
         meaning="相邻动作差的平滑度惩罚（奖励量，无验收底线）；抖动大不利部署。"),

    # —— 地形推进 ——
    dict(family="terrain", key="terrain_mean", label="平均地形等级",
         cands=["curriculum:terrain_mean"], unit="", precision=2,
         meaning="当前平均地形难度（课程位置，不是验收指标）。"),
    dict(family="terrain", key="terrain_max", label="最高地形等级",
         cands=["curriculum:terrain_max"], unit="", precision=2,
         meaning="当前已到的最高地形等级。"),
    dict(family="terrain", key="terrain_progress", label="地形推进奖励",
         cands=["reward:terrain_progress"], unit="", precision=3,
         meaning="命令方向上的地形推进奖励项（奖励量，无底线）。"),

    # —— 稳定与鲁棒 ——
    dict(family="robustness", key="fall_rate", label="摔倒率",
         cands=["health:fall_rate", "curriculum:fall_gate"], unit="", precision=3,
         meaning="稳定性主指标；高于阈值优先修稳定，别急着加速度命令。"),
    dict(family="robustness", key="terminal_rate", label="终止率",
         cands=["health:terminal_rate"], unit="", precision=3,
         meaning="episode 提前终止的比例；高会污染 reward/跟踪曲线。"),
    dict(family="robustness", key="base_h", label="机身高度",
         cands=["health:base_h", "reward:base_h"], unit="m", precision=3,
         meaning="base link 高度；过低=塌陷/蹲伏。"),
    dict(family="robustness", key="base_h_min", label="最低机身高度",
         cands=["health:base_h_min"], unit="m", precision=3,
         note="批次内最差单环境值：几百个并行环境里只要有一个在摔/重置就会很低。"
              "达标线看“机身高度”那行（均值）；此行用来发现短时塌陷事件，不作为达标线。",
         meaning="批次内最低机身高度；均值会隐藏短时塌陷。"),
    dict(family="robustness", key="tilt_deg", label="机身倾角",
         cands=["health:tilt_deg"], unit="deg", precision=1,
         meaning="机身相对竖直的倾角；持续偏大说明姿态不稳。"),
    dict(family="robustness", key="tilt_deg_max", label="最大倾角",
         cands=["health:tilt_deg_max"], unit="deg", precision=1,
         note="批次内最差单环境值：并行环境里有一个在摔就会很大（可达~180°）。"
              "达标线看“机身倾角”那行（均值）；此行用来发现短时大摆动，不作为达标线。",
         meaning="批次内最大倾角；比均值更接近姿态风险。"),
    dict(family="robustness", key="upright", label="直立度",
         cands=["health:upright", "reward:upright"], unit="", precision=3,
         meaning="机身 up 方向与世界竖直的对齐程度，越高越直立。"),
    dict(family="robustness", key="height_low_risk_window", label="低高度风险",
         cands=["health:height_low_risk_window"], unit="", precision=3,
         meaning="训练 gate 使用的近期低身高风险；非零=近期有蹲塌片段。"),
    dict(family="robustness", key="tilt_high_risk_window", label="大倾角风险",
         cands=["health:tilt_high_risk_window"], unit="", precision=3,
         meaning="训练 gate 使用的近期大倾角风险；非零=姿态风险在折扣正奖励。"),
]


# 覆盖自检时忽略的 reward 字段：这些是聚合量/别名，不是被优化的奖励项本身。
_REWARD_AGGREGATE_KEYS = {"total", "rew", "speed", "lin_err", "gait", "base_h", "upright"}


def _section(latest: TrainingTelemetryPoint, name: str) -> dict:
    return {
        "reward": latest.reward,
        "curriculum": latest.curriculum,
        "health": latest.health,
        "command": latest.command,
        "counters": latest.counters,
    }.get(name, {})


def _pick(latest: TrainingTelemetryPoint, cands: list[str], absolute: bool = False):
    """按候选顺序取第一个有数值的字段，返回 (原始值, 数值)。都取不到返回 (None, None)。"""
    for cand in cands:
        section_name, key = cand.split(":", 1)
        mapping = _section(latest, section_name)
        if key in mapping:
            num = _num(mapping.get(key))
            if num is not None:
                if absolute:
                    num = abs(num)
                return mapping.get(key), num
    return None, None


def _resolve_floor(key: str, conditions: dict, conditions_source: dict):
    """解析某指标的底线。返回 (方向, 数值, 出处, 可信度, 附注) 或 None（未定底线）。

    优先级：课程门控(本次 run 配置) > 规范底线 > 出处不明的显示常量。
    """
    # A) 课程门控：只有当本次 run 配置里确实带了该门控（当前阶段可见）才作为底线。
    if key in CONFIG_GATE:
        cond_key, op = CONFIG_GATE[key]
        val = _as_float(conditions.get(cond_key)) if cond_key in conditions else None
        if val is not None:
            src = conditions_source.get(cond_key, "default")
            # config/constant 视为高可信；default 表示配置缺失、用了兜底默认。
            conf = "config" if src in ("config", "constant") else "default"
            origin = "本次 run 有效配置 env.curriculum" if conf == "config" else "配置缺失，用了兜底默认"
            note = "" if conf == "config" else "本次 run 配置里没有这条门控，显示的是代码兜底默认，不是本次真实门槛。"
            return op, float(val), f"{origin}（{cond_key}，来源:{src}）", conf, note
        # 该阶段还没开放这条门控（如地形门控在 terrain_start_phase 之前）→ 暂无底线。
        return None
    # B) 规范底线。
    if key in SPEC_FLOORS:
        return SPEC_FLOORS[key]
    # C) 出处不明的显示常量：如实标 unknown。
    if key in CONSOLE_UNMARKED_FLOORS:
        return CONSOLE_UNMARKED_FLOORS[key]
    return None


def _floor_label(op: str, value: float, unit: str, precision: int) -> str:
    sign = "≤" if op == "<=" else "≥"
    body = f"{value:.{precision}f}"
    return f"{sign}{body}{(' ' + unit) if unit else ''}"


def _status(op: str | None, floor, numeric) -> str:
    if numeric is None:
        return "no_data"
    if floor is None or not op:
        return "no_floor"
    if op == "<=":
        return "meets" if numeric <= floor else "below"
    return "meets" if numeric >= floor else "below"


def _reading(numeric, unit: str, precision: int) -> str:
    if numeric is None:
        return "—"
    return f"{numeric:.{precision}f}{(' ' + unit) if unit else ''}"


def _join_notes(*parts: str) -> str:
    return "；".join(p for p in parts if p)


def _coverage(latest: TrainingTelemetryPoint) -> ScoreboardCoverage:
    """覆盖自检：把遥测里被优化、却还没上板的 reward 项列出来，让板子自曝盲区。"""
    # 一个奖励概念只要在板上有指标就算已覆盖——不管该指标是从 reward 段还是
    # command 段取值（如 diagonal_contact/duty_balance 从 command 取，但 reward 段
    # 也有同名分量，不能因此误报“未上板”）。
    reward_fields_read = {
        cand.split(":", 1)[1]
        for spec in METRICS
        for cand in spec["cands"]
        if cand.startswith("reward:")
    }
    covered = reward_fields_read | {spec["key"] for spec in METRICS}
    present = {k for k, v in latest.reward.items() if isinstance(v, (int, float))}
    unmapped = sorted(present - covered - _REWARD_AGGREGATE_KEYS)
    if unmapped:
        note = (f"遥测里还有 {len(unmapped)} 个被优化的奖励项没纳入记分牌（下方列出）。"
                "这不是遗忘，而是刻意把盲区显示出来——可据此继续补 family/指标。")
    else:
        note = "当前遥测里的奖励项都已在板上有对应指标。"
    return ScoreboardCoverage(
        mapped_reward_terms=sorted(reward_fields_read),
        unmapped_reward_terms=unmapped,
        note=note,
    )


def build_scoreboard(telemetry: TrainingTelemetry, effective_config_text: str = "") -> Scoreboard:
    """从遥测 + 本次 run 有效配置装配目标记分牌。纯读，不改任何训练状态。"""
    latest = telemetry.latest
    notes: list[str] = []

    if latest is None:
        return Scoreboard(
            available=False,
            run_id=telemetry.run_id,
            generated_at=time.time(),
            status="error" if telemetry.error else "missing",
            notes=[telemetry.error or "没有可解析的训练遥测；无法装配记分牌。"],
        )

    curr = latest.curriculum
    phase_raw = curr.get("phase")
    phase = "" if phase_raw is None else str(phase_raw)

    # 复用既有的“带出处”门控解析（config/default/constant）。
    gate = _actual_phase_gate_from_config(effective_config_text, phase_raw)
    conditions = gate.get("conditions") or {}
    conditions_source = gate.get("conditions_source") or {}
    if not effective_config_text.strip():
        notes.append(
            "未拿到本次 run 的 effective_config.yaml：课程类底线（推进/对角/占空/摔倒/地形）"
            "暂时无法确认为本次 run 的真实门槛，这些行会显示为未定底线。"
        )

    families = {fid: ScoreboardFamily(id=fid, title=FAMILY_META[fid][0], summary=FAMILY_META[fid][1])
                for fid in FAMILY_ORDER}

    below_count = 0
    for spec in METRICS:
        raw, numeric = _pick(latest, spec["cands"], spec.get("abs", False))
        floor_info = _resolve_floor(spec["key"], conditions, conditions_source)
        if floor_info is not None:
            op, floor_val, floor_src, floor_conf, floor_note = floor_info
            floor_label = _floor_label(op, floor_val, spec["unit"], spec["precision"])
        else:
            op, floor_val, floor_src, floor_conf, floor_note = "", None, "", "unknown", ""
            floor_label = ""
        status = _status(op or None, floor_val, numeric)
        if status == "below":
            below_count += 1
        metric = ScoreboardMetric(
            key=spec["key"],
            label=spec["label"],
            family=spec["family"],
            value=raw,
            numeric_value=numeric,
            unit=spec["unit"],
            precision=spec["precision"],
            reading=_reading(numeric, spec["unit"], spec["precision"]),
            floor=floor_val,
            floor_op=op,  # type: ignore[arg-type]
            floor_label=floor_label,
            floor_source=floor_src,
            floor_confidence=floor_conf,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            value_confidence=spec.get("value_confidence", "high"),  # type: ignore[arg-type]
            tensions=_tensions_for(spec["key"]),
            meaning=spec["meaning"],
            note=_join_notes(spec.get("note", ""), floor_note),
        )
        families[spec["family"]].metrics.append(metric)

    context = ScoreboardContext(
        phase=phase,
        command_mode="" if curr.get("command_mode") is None else str(curr.get("command_mode")),
        active_dirs="" if curr.get("active_dirs") is None else str(curr.get("active_dirs")),
        terrain_mean=_num(curr.get("terrain_mean")),
        terrain_max=_num(curr.get("terrain_max")),
        penalty_gate=_num(curr.get("penalty_gate")),
        dr_level=_num(curr.get("dr_level")),
        note="penalty_gate 会重标所有晚期惩罚的强度，地形/阶段决定当前在考哪一关。"
             "跨时间比较这些数字时务必带上这条状态带，否则今天和昨天不可比。",
    )

    notes.append(
        "抗扰恢复(推挤/低摩擦 E1–E3)与域随机迁移(E4/E5)属验收 physeval 侧测量，"
        "活遥测里没有直读值，本板暂不以数值呈现，但它们是真实存在的目标，别当作已覆盖。"
    )

    status = "watch" if below_count else "ok"
    if below_count:
        notes.append(f"当前有 {below_count} 项低于其底线（仅陈述事实，未做好坏判断）。")

    return Scoreboard(
        available=True,
        run_id=telemetry.run_id,
        generated_at=time.time(),
        step=latest.step,
        stale=telemetry.stale,
        status=status,  # type: ignore[arg-type]
        context=context,
        families=[families[fid] for fid in FAMILY_ORDER],
        tensions=[ScoreboardTension(a=a, b=b, reason=r) for a, b, r in TENSIONS],
        coverage=_coverage(latest),
        notes=notes,
    )
