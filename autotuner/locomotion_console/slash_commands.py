"""Deterministic slash command layer for the LLM Copilot.

Slash commands are intentionally routed before the open-ended LLM loop. They
use allowlisted backend tools and return auditable evidence.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import shlex
from typing import Any

from .config import LocomotionConsoleSettings
from .datasource import RunDataSource
from .definitions import explain_terms, list_definitions


HELP = """/status：读取实时 telemetry，总结 step/ETA/reward/课程状态
/log tail [N]：读取远程训练日志尾部（只读）
/explain <term>：解释 reward/track/lin_err/terrain/phase/progress_gate 等术语
/why stalled：基于 telemetry 判断当前可能卡在哪个 gate
/why blocked：直接读取 RunSnapshot.blockers，说明当前最可能卡住的门控
/compare <metric> last [N]：比较某个 telemetry 指标最近 N 个点的变化
/diagnose current：基于 telemetry 给出当前只读诊断判断
/diag explain [job_id]：读取诊断历史/报告并解释这次机器人表现、失败点和下一步
/run diag <preset>：生成启动诊断的确认动作，不直接执行
/probe checkpoints：通过白名单 diagnostics.catalog 探测 checkpoint
/probe telemetry：只读检查远程 telemetry JSONL/log/emitter 是否真的部署
/context [query]：读取 Taili spec/策略/YAML 白名单知识包摘要
/ask <问题> 或 /llm <问题>：跳过快速命令层，强制走 LLM Agent
/ 开头不等于一定调用 LLM；/status、/log、/probe 等默认是快速只读命令，/diag explain 会调用 LLM 解读
/help：显示命令列表"""


def _telemetry_reply(telemetry) -> str:
    if not telemetry.available or not telemetry.latest:
        detail = telemetry.error or "没有 telemetry 数据。"
        limits = "\n".join(f"- {item}" for item in telemetry.limitations)
        return f"没有可用训练 telemetry：{detail}\n{limits}".strip()
    latest = telemetry.latest
    lines = [
        f"训练状态：{'运行中' if telemetry.running else '未检测到训练进程'}",
        f"进度：step {latest.step}{f' / {latest.total_steps}' if latest.total_steps else ''}",
    ]
    if latest.eta or latest.fps is not None:
        lines.append(f"速度：ETA {latest.eta or '未知'}，{latest.fps:.2f} it/s" if latest.fps is not None else f"ETA {latest.eta}")
    if latest.reward:
        reward = ", ".join(
            f"{key}={value:.3f}"
            for key, value in latest.reward.items()
            if isinstance(value, (int, float))
        )
        lines.append(f"奖励分解：{reward}")
    if latest.curriculum:
        curr = ", ".join(f"{key}={value}" for key, value in latest.curriculum.items())
        lines.append(f"课程/门控：{curr}")
    if latest.health:
        health = ", ".join(f"{key}={value:.3f}" for key, value in latest.health.items())
        lines.append(f"健康：{health}")
    if telemetry.limitations:
        lines.append("限制：" + "；".join(telemetry.limitations))
    return "\n".join(lines)


def _remote_status_reply(remote_status) -> str:
    if not remote_status.available:
        return f"远程机器状态：不可用。{remote_status.error or 'remote unavailable'}"
    lines = [
        f"远程机器：{remote_status.host or 'unknown'}",
        f"系统：uptime={remote_status.uptime or 'unknown'}，load={remote_status.load_avg or 'unknown'}，cpu={remote_status.cpu_count or 'unknown'}",
    ]
    if remote_status.memory_total_mb:
        used = remote_status.memory_used_mb or 0.0
        total = remote_status.memory_total_mb or 0.0
        avail = remote_status.memory_available_mb
        pct = used / total * 100 if total else 0.0
        lines.append(
            f"内存：{used:.0f}/{total:.0f} MB ({pct:.1f}%)"
            + (f"，available={avail:.0f} MB" if avail is not None else "")
        )
    if remote_status.gpus:
        for gpu in remote_status.gpus:
            used = gpu.memory_used_mb
            total = gpu.memory_total_mb
            mem = (
                f"{used:.0f}/{total:.0f} MB ({used / total * 100:.1f}%)"
                if used is not None and total
                else "unknown"
            )
            util = f"{gpu.utilization_gpu_pct:.0f}%" if gpu.utilization_gpu_pct is not None else "unknown"
            temp = f"{gpu.temperature_c:.0f}°C" if gpu.temperature_c is not None else "unknown"
            power = f"{gpu.power_w:.0f}W" if gpu.power_w is not None else "unknown"
            proc = ", ".join(
                f"{p.get('pid')}:{p.get('name')}:{p.get('used_memory_mb')}MB"
                for p in gpu.processes[:4]
            )
            lines.append(f"GPU{gpu.index} {gpu.name}：显存={mem}，util={util}，temp={temp}，power={power}")
            if proc:
                lines.append(f"  GPU进程：{proc}")
    else:
        lines.append("GPU：未读到 nvidia-smi 输出。")
    if remote_status.disks:
        disk = "；".join(f"{d.mount} {d.used}/{d.size} ({d.use_pct}, avail {d.avail})" for d in remote_status.disks)
        lines.append(f"磁盘：{disk}")
    if remote_status.tmux_sessions:
        tmux = "；".join(
            f"{s.name}({s.windows}w{' attached' if s.attached else ''})"
            for s in remote_status.tmux_sessions[:8]
        )
        lines.append(f"tmux：{tmux}")
    else:
        lines.append("tmux：未发现会话。")
    if remote_status.training_processes:
        lines.append("训练/Isaac 相关进程：")
        for proc in remote_status.training_processes[:5]:
            lines.append(
                f"- pid={proc.get('pid')} cpu={proc.get('cpu_pct')}% mem={proc.get('mem_pct')}% "
                f"etime={proc.get('etime')} cmd={str(proc.get('command') or '')[:120]}"
            )
    else:
        lines.append("训练/Isaac 相关进程：未发现。")
    return "\n".join(lines)


def _combined_status_reply(telemetry, remote_status) -> str:
    return _telemetry_reply(telemetry) + "\n\n" + _remote_status_reply(remote_status)


def _context_pack_reply(pack: dict[str, Any]) -> str:
    yaml = pack.get("yaml") if isinstance(pack.get("yaml"), dict) else {}
    recipe = yaml.get("recipe") if isinstance(yaml.get("recipe"), dict) else {}
    model = yaml.get("model") if isinstance(yaml.get("model"), dict) else {}
    actor = model.get("actor") if isinstance(model.get("actor"), dict) else {}
    phases = yaml.get("phases") if isinstance(yaml.get("phases"), dict) else {}
    reward_core = yaml.get("reward_core") if isinstance(yaml.get("reward_core"), dict) else {}
    docs = pack.get("docs") if isinstance(pack.get("docs"), list) else []
    hits = (pack.get("search_hits") or {}).get("matches") if isinstance(pack.get("search_hits"), dict) else []
    lines = [
        "LLM 受控知识包已读取：",
        f"- 权限边界：{pack.get('permission_boundary')}",
        f"- YAML：profile={yaml.get('profile')}，recipe={recipe.get('id')}，task={yaml.get('task_default_id')}",
        f"- 架构：actor input={actor.get('input_dim')} hidden={actor.get('hidden')}；terrain_perceiver={model.get('terrain_perceiver')}",
        f"- phase：{len(phases)} 个；策略应保持累积 active_dirs，避免新方向训练时旧方向退化不可见。",
        "- 核心奖励/门控：" + ", ".join(f"{k}={v}" for k, v in reward_core.items()),
        "- 文档白名单：" + ", ".join(str(doc.get("name")) for doc in docs if doc.get("available")),
    ]
    if hits:
        lines.append("相关文档片段：")
        for item in hits[:3]:
            preview = str(item.get("text") or "").strip().replace("\n", " ")[:180]
            lines.append(f"- [{item.get('source')}] {item.get('title')}: {preview}")
    lines.append("下一步可以问：/ask 按 spec 看当前训练最大问题是什么？ 或 /ask YAML 里哪个奖励项最可能需要调？")
    return "\n".join(lines)


def _why_stalled(telemetry) -> str:
    if not telemetry.available or not telemetry.latest:
        return "无法判断是否卡住：没有可用 telemetry。先让远程训练输出 JSONL，或至少保留 [TPREW]/[TPSTAT] 日志。"
    latest = telemetry.latest
    reward = latest.reward
    curr = latest.curriculum
    blocked = curr.get("blocked_by") or curr.get("blocked")
    if blocked:
        return f"当前 telemetry 显示 blocked_by={blocked}。这通常表示训练还在跑，但对应 gate 未满足。\n\n{_telemetry_reply(telemetry)}"
    clues = []
    if reward.get("gait", 0) >= 0.9 and reward.get("track", 1) < 0.25:
        clues.append("gait 高但 track 低：步态节奏可能有了，但有效推进/速度跟踪不足。")
    if reward.get("lin_err", 0) > 0.25:
        clues.append("lin_err 偏高：速度误差仍明显。")
    if reward.get("speed", 1) < 0.25:
        clues.append("speed 偏低：可能存在推进不足或原地踏步。")
    if not curr:
        clues.append("缺少 terrain/phase/DR telemetry，不能判断是否卡课程 gate。")
    return "\n".join(clues or ["没有明显 stalled 证据；需要更多 telemetry 或诊断回放确认。"])


def _why_blocked(telemetry) -> str:
    snapshot = telemetry.snapshot
    if not telemetry.available or not telemetry.latest or snapshot is None:
        return "无法判断当前卡点：没有可用 telemetry snapshot。先检查 /probe telemetry。"
    lines = [
        f"当前结论：{snapshot.conclusion}",
        f"run={snapshot.run_id or telemetry.run_id} step={snapshot.step} phase={snapshot.phase or 'unknown'} blocked_by={snapshot.blocked_by or '—'}",
    ]
    if not snapshot.blockers:
        lines.append("snapshot.blockers 为空：当前没有结构化卡点；如果 UI 仍显示卡住，优先检查 telemetry 是否是旧 payload 或 stale。")
        return "\n".join(lines)
    lines.append("结构化 blocker：")
    for blocker in snapshot.blockers[:6]:
        value = "—" if blocker.value is None else f"{blocker.value:.3f}"
        target = "" if blocker.target is None else f" target {blocker.direction} {blocker.target:.3f}"
        gap = "" if blocker.gap is None else f" gap={blocker.gap:.3f}"
        lines.append(f"- [{blocker.severity}] {blocker.label or blocker.key}: {value}{target}{gap}")
        if blocker.detail:
            lines.append(f"  {blocker.detail}")
        if blocker.source:
            lines.append(f"  source: {blocker.source}")
    lines.append("这是快速只读判断，没有调用 LLM；要做策略原因分析再用 /ask 为什么这个 blocker 会出现？")
    return "\n".join(lines)


def _metric_value(point, metric: str):
    if metric in point.reward:
        return point.reward[metric]
    if metric == "reward":
        return point.reward.get("total") or point.reward.get("rew")
    value = point.curriculum.get(metric)
    if isinstance(value, (int, float)):
        return float(value)
    if metric in point.health:
        return point.health[metric]
    return None


def _compare_metric(telemetry, metric: str, window: int) -> str:
    if not telemetry.available or not telemetry.history:
        return "无法比较：没有可用 telemetry。"
    samples = [(p.step, _metric_value(p, metric)) for p in telemetry.history[-window:]]
    samples = [(step, value) for step, value in samples if isinstance(value, (int, float))]
    if len(samples) < 2:
        return f"无法比较 {metric}：最近 {window} 个点里有效样本不足。"
    first_step, first = samples[0]
    last_step, last = samples[-1]
    values = [float(v) for _, v in samples]
    delta = float(last) - float(first)
    mean = sum(values) / len(values)
    direction = "上升" if delta > 0 else "下降" if delta < 0 else "基本不变"
    return (
        f"{metric} 最近 {len(samples)} 个有效点：{direction}。\n"
        f"起点 step {first_step}: {first:.4f}\n"
        f"终点 step {last_step}: {last:.4f}\n"
        f"delta={delta:+.4f}, mean={mean:.4f}, min={min(values):.4f}, max={max(values):.4f}"
    )


def _diagnose_current(telemetry) -> str:
    if not telemetry.available or not telemetry.latest:
        return "无法诊断当前训练：没有 telemetry。先确认远程训练输出 JSONL 或 [TPREW] 日志。"
    latest = telemetry.latest
    lines = [_telemetry_reply(telemetry), "", "只读判断："]
    reward = latest.reward
    curr = latest.curriculum
    health = latest.health
    if reward.get("gait", 0) >= 0.9 and reward.get("track", 1) < 0.25:
        lines.append("- gait 高但 track 低：更像“节奏有了、推进/速度跟踪不足”，不是单纯没学到步态。")
    if reward.get("lin_err", 0) > 0.25:
        lines.append("- lin_err 偏高：速度误差仍明显，建议优先跑方向诊断或比较 track/lin_err 趋势。")
    if reward.get("speed", 1) < 0.25:
        lines.append("- speed 偏低：需要警惕原地踏步/小步慢走。")
    if health.get("fall_rate", 0) > 0.05 or health.get("reset_rate", 0) > 0.05:
        lines.append("- fall/reset 偏高：先看稳定性和地形/DR，而不是直接加跟踪奖励。")
    if curr.get("blocked_by"):
        lines.append(f"- 当前 blocked_by={curr.get('blocked_by')}：课程推进被该 gate 限制。")
    if not curr:
        lines.append("- 缺少 phase/terrain/DR：当前只能做奖励项级别判断，不能判断课程 gate。")
    lines.append("\n建议命令：/compare track last 20；/explain progress_gate；/run diag directions")
    return "\n".join(lines)


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_number(value: Any, *, unit: str = "", digits: int = 3) -> str:
    number = _float_or_none(value)
    if number is None:
        return "—"
    return f"{number:.{digits}f}{unit}"


def _fmt_percent(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return "—"
    return f"{number * 100:.1f}%"


def _report_value_explanation(report, key: str) -> str:
    for item in report.value_explanations:
        if item.key != key:
            continue
        formula = f"；计算：{item.formula}" if item.formula else ""
        source = f"；来源：{item.source}" if item.source else ""
        return f"{item.label}：{item.interpretation}{formula}{source}"
    return ""


def _event_count(report, event_type: str) -> int:
    for event in report.events:
        if str(event.get("type") or "") == event_type:
            try:
                return int(event.get("count") or 0)
            except (TypeError, ValueError):
                return 0
    return 0


def _parse_duties(report) -> dict[str, float]:
    for note in report.notes:
        message = str(note.get("message") or "")
        if "duties={" not in message:
            continue
        match = re.search(r"duties=(\{.*\})", message)
        if not match:
            continue
        try:
            parsed = json.loads(match.group(1).replace("'", '"'))
        except json.JSONDecodeError:
            continue
        result = {}
        for key, value in parsed.items():
            number = _float_or_none(value)
            if number is not None and math.isfinite(number):
                result[str(key)] = number
        if result:
            return result
    return {}


def _test_intent(report) -> dict[str, Any]:
    preset = str(report.preset or "").lower()
    label = str(report.preset_label or report.preset or "diagnostic")
    modes = [str(command.get("mode") or "unknown") for command in report.commands]
    target_modes = [mode for mode in modes if mode != "stand"] or modes
    descriptions = {
        "quick": "快速冒烟测试：确认 checkpoint、任务注册和 record/metrics 管线能跑通，不用于完整评价步态。",
        "forward": "前进测试：先站立基线，然后给正向速度命令；目标是向前稳定移动，同时保持机身高度和干净接触。",
        "backward": "后退测试：先站立基线，然后给负向速度命令；目标是按命令向后走，而不是继续向前跑或原地拖行。",
        "lateral": "侧移测试：先站立基线，然后给横向速度命令；目标是侧向移动，不能靠旋转或滑移冒充。",
        "yaw": "转向测试：先站立基线，然后给 yaw 角速度命令；目标是主要原地转身，平移应很小，四足接触节律不能塌成拖行。",
        "directions": "全方向测试：分别检查前进、后退、侧移和 yaw，目标是确认策略真的按命令条件化，而不是只会一种默认运动。",
        "terrain": "地形覆盖测试：检查不同地形上的姿态、接触和推进是否仍稳定。",
        "dr": "域随机化测试：检查不同随机化等级下行为是否一致，不只看单一环境。",
        "push": "外力扰动测试：施加已知推力后观察恢复，而不是看正常无扰动行走。",
    }
    return {
        "preset": preset,
        "label": label,
        "target_modes": target_modes,
        "description": descriptions.get(preset, f"{label} 测试：根据命令段评估策略行为。"),
    }


def _command_behavior(command: dict[str, Any], preset: str) -> dict[str, Any]:
    mode = str(command.get("mode") or "unknown")
    stable = _float_or_none(command.get("stable_fraction"))
    controlled = _float_or_none(command.get("controlled_progress_m"))
    raw = _float_or_none(command.get("raw_progress_m"))
    tracking = _float_or_none(command.get("tracking_error"))
    major = _float_or_none(command.get("major_event_fraction")) or 0.0
    reset = _float_or_none(command.get("reset_fraction")) or 0.0
    attempts = int(command.get("attempts") or 0)
    judgement: list[str] = []
    if mode == "stand":
        if tracking is not None and tracking > 0.10:
            judgement.append("站立基线不够安静，存在速度/角速度漂移")
        else:
            judgement.append("站立段没有明显位移指标，但仍需结合接触事件判断是否在抖动")
    elif mode == "yaw":
        if tracking is None:
            judgement.append("缺少 yaw 跟踪误差，无法确认是否跟住转向命令")
        elif tracking > 0.55:
            judgement.append("yaw 跟踪误差偏大，说明没有可靠地按角速度命令转身")
        else:
            judgement.append("yaw 跟踪误差不算灾难，但仍要看是否靠滑移/拖行完成")
    elif mode == "backward":
        if controlled is not None and controlled < 0.05:
            judgement.append("后退方向没有形成有效受控位移")
        if raw is not None and controlled is not None and raw - controlled > 0.10:
            judgement.append("raw 位移明显大于受控位移，可能有滑行或异常段贡献")
    elif mode in {"forward", "lateral"}:
        if controlled is not None and controlled < 0.10:
            judgement.append(f"{mode} 命令下推进不足")
    if major > 0:
        judgement.append("出现主要异常事件")
    if reset > 0:
        judgement.append("发生 reset")
    if stable is not None and stable < 0.75:
        judgement.append("姿态稳定占比偏低")
    if not judgement:
        judgement.append("聚合指标没有显示硬失败，但需要回放确认步态质量")
    return {
        "mode": mode,
        "attempts": attempts,
        "stable_fraction": stable,
        "major_event_fraction": major,
        "reset_fraction": reset,
        "controlled_progress_m": controlled,
        "raw_progress_m": raw,
        "tracking_error": tracking,
        "behavior_judgement": judgement,
    }


def _diagnostic_behavior_context(report) -> dict[str, Any]:
    intent = _test_intent(report)
    duties = _parse_duties(report)
    duty_spread = max(duties.values()) - min(duties.values()) if duties else None
    events = {str(event.get("type") or "unknown"): int(event.get("count") or 0) for event in report.events}
    posture = {
        "stable_fraction": _float_or_none(report.posture.get("stable_fraction")),
        "min_height_m": _float_or_none(report.posture.get("min_height_m")),
        "max_roll_deg": _float_or_none(report.posture.get("max_roll_deg")),
        "max_pitch_deg": _float_or_none(report.posture.get("max_pitch_deg")),
    }
    contact_reading: list[str] = []
    if duties:
        rear = [duties.get("RL"), duties.get("RR")]
        front = [duties.get("FL"), duties.get("FR")]
        rear_values = [v for v in rear if v is not None]
        front_values = [v for v in front if v is not None]
        if rear_values and sum(rear_values) / len(rear_values) > 0.85:
            contact_reading.append("后腿占空比很高，像后腿长时间压地支撑")
        if front_values and sum(front_values) / len(front_values) < 0.55:
            contact_reading.append("前腿占空比较低/不均，前后支撑不平衡")
        if duty_spread is not None and duty_spread > 0.35:
            contact_reading.append("四腿 duty spread 很大，步态节律不均衡")
    if events.get("high_slip_bout", 0) > 0:
        contact_reading.append("检测到 stance slip，高概率有站立脚滑移")
    if events.get("hard_impact_bout", 0) > 0:
        contact_reading.append("检测到 hard impact，落脚或姿态恢复不干净")
    if not contact_reading:
        contact_reading.append("聚合事件里没有明显接触硬失败；仍需要看回放确认足端轨迹")
    height = posture["min_height_m"]
    posture_reading: list[str] = []
    if height is not None and height < 0.45:
        posture_reading.append("机身偏低，接近蹲走/低姿态运行")
    if posture["stable_fraction"] is not None and posture["stable_fraction"] >= 0.95:
        posture_reading.append("姿态判定稳定，并不等于步态好；它只说明没有明显翻滚/俯仰失稳")
    if posture["max_roll_deg"] is not None and posture["max_roll_deg"] > 20:
        posture_reading.append("横滚峰值偏大")
    if posture["max_pitch_deg"] is not None and posture["max_pitch_deg"] > 20:
        posture_reading.append("俯仰峰值接近风险边界")
    if not posture_reading:
        posture_reading.append("姿态聚合值没有明显硬失败")
    return {
        "test": intent,
        "checkpoint": report.checkpoint.rstrip("/").rsplit("/", 1)[-1] if report.checkpoint else "",
        "job_id": report.job_id,
        "output_dir": report.output_dir,
        "coverage": report.coverage,
        "commands": [_command_behavior(command, intent["preset"]) for command in report.commands],
        "posture": posture,
        "events": events,
        "duties": duties,
        "duty_spread": duty_spread,
        "contact_reading": contact_reading,
        "posture_reading": posture_reading,
        "notes": [str(note.get("message") or "") for note in report.notes if note.get("message")][:8],
        "value_definitions": [
            {
                "key": item.key,
                "label": item.label,
                "source": item.source,
                "formula": item.formula,
                "interpretation": item.interpretation,
            }
            for item in report.value_explanations[:16]
        ],
    }


def _llm_behavior_explain(context: dict[str, Any], fallback: str) -> tuple[str, dict[str, Any]]:
    from autotuner.llm_gateway.client import call_llm_with_schema

    system = (
        "你是四足机器人强化学习诊断助手。你只能使用用户消息中给出的诊断上下文，"
        "不能声称自己看了未提供的视频、不能扫描远程路径、不能执行动作。"
        "你的任务不是复述字段，而是像工程师看完诊断一样描述机器人实际表现。"
        "必须说明：这是什么测试、命令要求机器人做什么、实际看起来像什么、哪些证据支持、"
        "哪些只是推断、下一步应该看什么。"
        "不要输出大表格，不要逐字段罗列。"
        "返回严格 JSON：{\"reply\":\"...\"}。用中文。"
    )
    user = json.dumps(
        {
            "instruction": (
                "基于这些受控证据写一段行为层面的诊断说明。"
                "如果是 yaw，要判断是否像原地转身；如果是 forward/backward，要判断是否按方向移动；"
                "如果 evidence 只能间接说明，要明确写“从聚合指标推断”。"
            ),
            "diagnostic_context": context,
            "output_shape": [
                "先一句话结论",
                "测试目的",
                "机器人表现",
                "关键证据",
                "下一步",
            ],
        },
        ensure_ascii=False,
    )
    resp = call_llm_with_schema(system, user, schema_name="diagnostic_behavior_explain", max_retries=1)
    transcript = {
        "schema": "diagnostic_behavior_explain",
        "model": resp.model,
        "elapsed_s": resp.elapsed_s,
        "error": resp.error,
    }
    if resp.parsed and isinstance(resp.parsed.get("reply"), str) and resp.parsed["reply"].strip():
        return resp.parsed["reply"].strip(), transcript
    return fallback + f"\n\n（LLM 行为解读不可用，已使用受控 fallback：{resp.error or 'unknown'}）", transcript


def _command_judgement(command: dict[str, Any]) -> str:
    mode = str(command.get("mode") or "unknown")
    major = _float_or_none(command.get("major_event_fraction")) or 0.0
    reset = _float_or_none(command.get("reset_fraction")) or 0.0
    done = _float_or_none(command.get("done_fraction")) or 0.0
    stable = _float_or_none(command.get("stable_fraction"))
    controlled = _float_or_none(command.get("controlled_progress_m"))
    tracking = _float_or_none(command.get("tracking_error"))
    clues: list[str] = []
    if major > 0:
        clues.append("出现主要事件，优先看回放中第一次异常前后的接触/姿态")
    if reset > 0:
        clues.append("有 reset，说明该方向存在明确失败段")
    if done > 0 and reset <= 0:
        clues.append("有 done/终止事件，需要区分是正常结束还是异常结束")
    if stable is not None and stable < 0.75:
        clues.append("稳定姿态占比偏低")
    if controlled is not None:
        if controlled < -0.02:
            clues.append("受控位移为负，疑似朝命令反方向或被滑移拖走")
        elif controlled < 0.05:
            clues.append("受控位移几乎没有，不能证明会走")
    if tracking is not None:
        threshold = 0.7 if mode == "yaw" else 0.35
        if tracking > threshold:
            clues.append("跟踪误差偏大")
    if not clues:
        return "未看到明显硬失败；仍要结合回放判断步态是否干净。"
    return "；".join(clues) + "。"


def _diagnostic_report_reply(report) -> str:
    behavior = _diagnostic_behavior_context(report)
    commands = behavior["commands"]
    target_modes = behavior["test"]["target_modes"]
    stable = report.posture.get("stable_fraction")
    min_height = report.posture.get("min_height_m")
    max_roll = report.posture.get("max_roll_deg")
    max_pitch = report.posture.get("max_pitch_deg")
    weak_modes = [
        str(command.get("mode") or "unknown")
        for command in commands
        if (
            (_float_or_none(command.get("major_event_fraction")) or 0.0) > 0.0
            or ((_float_or_none(command.get("stable_fraction")) is not None)
                and (_float_or_none(command.get("stable_fraction")) or 0.0) < 0.75)
            or ((_float_or_none(command.get("controlled_progress_m")) is not None)
                and (_float_or_none(command.get("controlled_progress_m")) or 0.0) < 0.05)
            or ((_float_or_none(command.get("tracking_error")) is not None)
                and str(command.get("mode")) == "yaw"
                and (_float_or_none(command.get("tracking_error")) or 0.0) > 0.55)
        )
    ]
    rows = report.coverage.get("rows") or 0
    attempts = report.coverage.get("attempts") or 0
    checkpoint_name = report.checkpoint.rstrip("/").rsplit("/", 1)[-1] if report.checkpoint else "—"

    lines = [
        f"诊断行为解读：{report.preset_label} / job_id={report.job_id}",
        "权限边界：这次只读取 diagnostics.history 和 diagnostics.report，不扫描远程任意路径，也不执行动作。",
        "",
        f"一句话：这是 {report.preset_label} 测试，目标命令是 {', '.join(target_modes)}；"
        + (
            f"从聚合指标看，主要问题在 {', '.join(dict.fromkeys(weak_modes))}。"
            if weak_modes else
            "聚合指标没有显示硬失败，但仍需要回放确认步态质量。"
        ),
        "",
        "它做的是什么测试：",
        f"- {behavior['test']['description']}",
        f"- checkpoint={checkpoint_name}；数据量={rows} 行 / {attempts} 个命令段。",
        "",
        "机器人现在看起来像什么：",
    ]
    for command in commands:
        mode = command["mode"]
        if mode == "stand":
            lines.append(f"- stand：作为基线段，聚合位移为 0；{'; '.join(command['behavior_judgement'])}。")
        else:
            lines.append(
                f"- {mode}：{'; '.join(command['behavior_judgement'])}。"
                f" controlled={_fmt_number(command.get('controlled_progress_m'), unit=' m')}，"
                f"tracking_error={_fmt_number(command.get('tracking_error'))}。"
            )
    lines.extend([
        f"- 姿态：稳定占比 {_fmt_percent(stable)}，最低机身高度 {_fmt_number(min_height, unit=' m')}，最大横滚 {_fmt_number(max_roll, unit='°', digits=1)}，最大俯仰 {_fmt_number(max_pitch, unit='°', digits=1)}。",
        f"- 接触/步态：{'；'.join(behavior['contact_reading'])}。",
        f"- 姿态解释：{'；'.join(behavior['posture_reading'])}。",
    ])

    lines.append("")
    lines.append("判断和限制：")
    if rows == 0:
        lines.append("- record/metrics 数据为空，不能评价步态；应先修复诊断采集。")
    elif weak_modes:
        lines.append(f"- 优先怀疑这些方向：{', '.join(dict.fromkeys(weak_modes))}。这是基于聚合指标的推断，最终应以回放确认足端接触和机身运动。")
    else:
        lines.append("- 这份聚合报告没有暴露明显硬失败；但它不是视频理解，下一步仍应看回放里的足端接触、滑移和机身高度。")
    if _float_or_none(min_height) is not None and (_float_or_none(min_height) or 0.0) < 0.45:
        lines.append("- 最低机身高度偏低；如果回放也显示低姿态，就是蹲走/拖行风险。")
    if _float_or_none(stable) is not None and (_float_or_none(stable) or 0.0) < 0.75:
        lines.append("- 稳定占比低于诊断阈值，调参前先确认失败是否来自姿态/接触，而不是只加 tracking 权重。")

    lines.extend([
        "",
        "建议下一步：",
        "- 打开当前 job 的回放，优先看目标命令段：机身有没有按测试目标移动，足端是否在 stance 期滑动。",
        "- 对 yaw：重点看它是不是原地转身，而不是后腿撑住、前腿拖/滑；对前后方向：重点看实际方向是否和命令一致。",
        "- 如果要让系统执行新诊断，用 /run diag <preset>，仍会走确认门控。",
    ])
    return "\n".join(lines)


def _diagnostic_explain(settings: LocomotionConsoleSettings, source: RunDataSource, job_id: str = "") -> dict[str, Any]:
    from .diagnostics import DiagnosticsController

    controller = DiagnosticsController(settings, source)
    history = controller.history.list(limit=10)
    selected_job_id = job_id.strip()
    if not selected_job_id:
        completed = next((item for item in history.items if item.state == "complete"), None)
        selected_job_id = completed.job_id if completed else (history.items[0].job_id if history.items else "")
    transcript: list[dict[str, Any]] = [
        {
            "tool": "diagnostics.history",
            "args": {"limit": 10},
            "result": history.model_dump(),
        }
    ]
    if not selected_job_id:
        return {
            "reply": "还没有可解读的诊断记录。先运行 /run diag directions 或在诊断页启动一个 preset；执行仍需确认。",
            "transcript": transcript,
            "steps": 1,
        }
    try:
        report = asyncio.run(controller.report_for_job(job_id=selected_job_id))
    except RuntimeError as exc:
        rows = [
            f"- {item.job_id} · {item.preset_label or item.preset} · {item.state} · {item.message}"
            for item in history.items[:8]
        ]
        reply = (
            f"诊断 job_id={selected_job_id} 还不能解读：{exc}\n"
            "可选历史：\n"
            + ("\n".join(rows) if rows else "- 无")
        )
        return {
            "reply": reply,
            "transcript": transcript,
            "steps": 1,
        }
    transcript.append({
        "tool": "diagnostics.report",
        "args": {"job_id": selected_job_id},
        "result": report.model_dump(),
    })
    context = _diagnostic_behavior_context(report)
    fallback = _diagnostic_report_reply(report)
    reply, llm_trace = _llm_behavior_explain(context, fallback)
    reply = (
        f"诊断证据：{report.preset_label} / job_id={selected_job_id}；"
        "只读 diagnostics.history + diagnostics.report。\n\n"
        f"{reply}"
    )
    transcript.append({
        "tool": "llm.diagnostic_behavior_explain",
        "args": {
            "job_id": selected_job_id,
            "evidence": "diagnostics.report + derived behavior context",
            "permission": "read-only; no remote path scan; no execution",
        },
        "result": llm_trace,
    })
    return {
        "reply": reply,
        "transcript": transcript,
        "steps": 3,
    }


def _probe_telemetry(settings: LocomotionConsoleSettings, source: RunDataSource) -> dict[str, Any]:
    telemetry = asyncio.run(source.training_telemetry())
    result: dict[str, Any] = {
        "telemetry": telemetry.model_dump(),
        "checks": [],
    }
    checks: list[str] = []
    if telemetry.available:
        checks.append(f"telemetry readable via {telemetry.mode}: {telemetry.summary}")
    else:
        checks.append(f"telemetry unavailable: {telemetry.error or '; '.join(telemetry.limitations) or 'unknown'}")

    from .datasource import RealDataSource

    if isinstance(source, RealDataSource):
        import shlex

        remote = source._get_remote()
        run = source._newest_run(remote)
        paths = source._resolve_telemetry_paths(remote, run)
        log_path = paths.log_path
        jsonl_path = paths.telemetry_path
        console_log_path = paths.console_log_path
        inspect_cmd = (
            f"echo run={shlex.quote(run or '(none)')}; "
            f"test -s {shlex.quote(jsonl_path)} && echo jsonl_present || echo jsonl_missing; "
            f"test -s {shlex.quote(log_path)} && echo train_log_present || echo train_log_missing; "
            f"test -s {shlex.quote(console_log_path)} && echo console_log_present || echo console_log_missing; "
            f"test -d {shlex.quote(paths.checkpoint_dir)} && echo checkpoint_dir_present || echo checkpoint_dir_missing; "
            f"wc -l {shlex.quote(jsonl_path)} 2>/dev/null | awk '{{print \"jsonl_lines=\" $1}}'; "
            f"wc -l {shlex.quote(log_path)} 2>/dev/null | awk '{{print \"train_log_lines=\" $1}}'; "
            f"wc -l {shlex.quote(console_log_path)} 2>/dev/null | awk '{{print \"console_log_lines=\" $1}}'; "
            f"grep -q '\\[TPPATH\\]' {shlex.quote(log_path)} 2>/dev/null && echo tppath_present || echo tppath_missing; "
            f"grep -q '\\[TPSTAT\\]' {shlex.quote(log_path)} 2>/dev/null && echo tpstat_present || echo tpstat_missing; "
            f"grep -q '\\[TPCURR\\]' {shlex.quote(log_path)} 2>/dev/null && echo tpcurr_present || echo tpcurr_missing"
        )
        raw = remote.exec_out(inspect_cmd) or ""
        checks.extend(line.strip() for line in raw.splitlines() if line.strip())
        emitter_probe = (
            "python - <<'PY'\n"
            "mods=['taili_blind_runtime.telemetry_emit','telemetry_emit','autotuner.blind_locomotion.telemetry_emit']\n"
            "ok=[]\n"
            "for m in mods:\n"
            "    try:\n"
            "        __import__(m); ok.append(m)\n"
            "    except Exception as e:\n"
            "        pass\n"
            "print('emitter_import=' + (ok[0] if ok else 'missing'))\n"
            "PY"
        )
        try:
            checks.append((remote.exec_out(emitter_probe, timeout=15) or "").strip())
        except Exception as exc:  # noqa: BLE001
            checks.append(f"emitter_import_probe_failed={type(exc).__name__}: {exc}")
        if paths.evidence:
            checks.append("path_discovery=" + "; ".join(paths.evidence))
        result["remote_run"] = run
        result["remote_log_path"] = log_path
        result["remote_jsonl_path"] = jsonl_path
        result["remote_console_log_path"] = console_log_path
        result["remote_checkpoint_dir"] = paths.checkpoint_dir
        result["path_evidence"] = list(paths.evidence)
    result["checks"] = checks
    reply = "telemetry 部署/读取检查：\n" + "\n".join(f"- {line}" for line in checks if line)
    if telemetry.limitations:
        reply += "\n限制：\n" + "\n".join(f"- {item}" for item in telemetry.limitations)
    return {
        "reply": reply,
        "transcript": [{"tool": "telemetry_probe", "args": {}, "result": result}],
        "steps": 1,
    }


def handle_slash_command(message: str, settings: LocomotionConsoleSettings, source: RunDataSource) -> dict[str, Any] | None:
    text = (message or "").strip()
    if not text.startswith("/"):
        return None
    try:
        parts = shlex.split(text)
    except ValueError:
        parts = text.split()
    command = parts[0].lower()
    args = parts[1:]

    if command in {"/help", "/"}:
        return {"reply": HELP, "transcript": [{"tool": "slash_help", "args": {}, "result": {"commands": HELP}}], "steps": 1}

    if command == "/status":
        telemetry = asyncio.run(source.training_telemetry())
        remote_status = asyncio.run(source.remote_machine_status())
        return {
            "reply": _combined_status_reply(telemetry, remote_status),
            "transcript": [
                {"tool": "training_telemetry", "args": {}, "result": telemetry.model_dump()},
                {"tool": "remote_machine_status", "args": {}, "result": remote_status.model_dump()},
            ],
            "steps": 2,
        }

    if command in {"/remote", "/gpu"} or (command == "/probe" and args[:1] in (["remote"], ["gpu"])):
        remote_status = asyncio.run(source.remote_machine_status())
        return {
            "reply": _remote_status_reply(remote_status),
            "transcript": [{"tool": "remote_machine_status", "args": {}, "result": remote_status.model_dump()}],
            "steps": 1,
        }

    if command in {"/context", "/strategy"}:
        from . import knowledge

        query = " ".join(args)
        pack = knowledge.build_taili_context_pack(query=query, include_docs=True)
        return {
            "reply": _context_pack_reply(pack),
            "transcript": [{"tool": "taili_context_pack", "args": {"query": query}, "result": pack}],
            "steps": 1,
        }

    if command == "/why" and args[:1] == ["stalled"]:
        telemetry = asyncio.run(source.training_telemetry())
        return {
            "reply": _why_stalled(telemetry),
            "transcript": [{"tool": "training_telemetry", "args": {}, "result": telemetry.model_dump()}],
            "steps": 1,
        }

    if command == "/why" and args[:1] in (["blocked"], ["blocker"], ["卡住"]):
        telemetry = asyncio.run(source.training_telemetry())
        return {
            "reply": _why_blocked(telemetry),
            "transcript": [{"tool": "run_snapshot.blockers", "args": {}, "result": telemetry.model_dump()}],
            "steps": 1,
            "mode": "slash_fast",
        }

    if command == "/compare":
        metric = args[0] if args else "track"
        window = 20
        if len(args) >= 3 and args[1] == "last":
            try:
                window = max(2, min(int(args[2]), 240))
            except ValueError:
                window = 20
        telemetry = asyncio.run(source.training_telemetry())
        return {
            "reply": _compare_metric(telemetry, metric, window),
            "transcript": [{"tool": "training_telemetry", "args": {"metric": metric, "window": window}, "result": telemetry.model_dump()}],
            "steps": 1,
        }

    if command == "/diagnose" and (not args or args[0] == "current"):
        telemetry = asyncio.run(source.training_telemetry())
        return {
            "reply": _diagnose_current(telemetry),
            "transcript": [{"tool": "training_telemetry", "args": {}, "result": telemetry.model_dump()}],
            "steps": 1,
        }

    if command in {"/diag", "/diagnostic", "/diagnostics"}:
        subcommand = args[0].lower() if args else "explain"
        if subcommand in {"explain", "report", "brief"}:
            job_id = args[1] if len(args) >= 2 else ""
            return _diagnostic_explain(settings, source, job_id=job_id)
        if subcommand in {"history", "list"}:
            from .diagnostics import DiagnosticsController

            history = DiagnosticsController(settings, source).history.list(limit=10)
            if not history.items:
                reply = "没有诊断历史。可以用 /run diag directions 生成一次诊断确认动作。"
            else:
                rows = [
                    f"- {item.job_id} · {item.preset_label or item.preset} · {item.state} · {item.checkpoint_name or 'checkpoint'}"
                    for item in history.items
                ]
                reply = "最近诊断历史：\n" + "\n".join(rows) + "\n\n用 /diag explain <job_id> 解读指定报告。"
            return {
                "reply": reply,
                "transcript": [{"tool": "diagnostics.history", "args": {"limit": 10}, "result": history.model_dump()}],
                "steps": 1,
            }

    if command == "/explain":
        terms = args or ["reward"]
        telemetry = asyncio.run(source.training_telemetry())
        return {
            "reply": explain_terms(terms, telemetry),
            "transcript": [
                {"tool": "definitions", "args": {"terms": terms}, "result": list_definitions(" ".join(terms)).model_dump()},
                {"tool": "training_telemetry", "args": {}, "result": telemetry.model_dump()},
            ],
            "steps": 2,
        }

    if command == "/log":
        lines = 80
        if len(args) >= 2 and args[0] == "tail":
            try:
                lines = max(10, min(int(args[1]), 500))
            except ValueError:
                lines = 80
        from .datasource import RealDataSource

        if not isinstance(source, RealDataSource):
            telemetry = asyncio.run(source.training_telemetry())
            raw = "\n".join(point.raw for point in telemetry.history[-lines:] if point.raw)
            return {
                "reply": raw or telemetry.summary or "fake source has no raw remote log tail",
                "transcript": [{"tool": "training_telemetry", "args": {"lines": lines}, "result": telemetry.model_dump()}],
                "steps": 1,
            }
        from .agent import _tool_get_train_log  # existing read-only allowlisted tool

        result = _tool_get_train_log(source, lines=lines)
        return {
            "reply": str(result.get("tail") or result.get("log") or result)[:4000],
            "transcript": [{"tool": "get_train_log", "args": {"lines": lines}, "result": result}],
            "steps": 1,
        }

    if command == "/run" and len(args) >= 2 and args[0] == "diag":
        preset = args[1]
        from .diagnostics import DiagnosticsController

        catalog = asyncio.run(DiagnosticsController(settings, source).catalog())
        if catalog.training_running:
            # Do NOT blanket-block: run the diagnostic CONCURRENTLY when the box has RAM+VRAM headroom.
            from .diagnostics import probe_concurrent_headroom
            try:
                _remote = source._get_remote()
                _ok, _detail = asyncio.run(probe_concurrent_headroom(_remote, 4))
            except Exception:
                _ok, _detail = (False, "无法读取余量")
            if not _ok:
                return {
                    "reply": f"训练运行中且余量不足({_detail})——诊断暂停以避免 OOM/显存争用。降低 num_envs 或先停训练；也可查看历史诊断或 /diag explain。",
                    "transcript": [{"tool": "diagnostics.catalog", "args": {"preset": preset}, "result": catalog.model_dump()}],
                    "steps": 1,
                    "mode": "slash_fast",
                }
            # headroom OK → fall through to propose the diagnostic (it will run alongside training)
        return {
            "reply": f"准备运行诊断 preset={preset}。这会占用远程 GPU，可能影响正在训练的进程；确认后才会执行。",
            "proposed_action": {
                "name": "run_diagnostic",
                "kind": "diagnostic",
                "risk": "medium",
                "args": {"preset": preset},
                "evidence": ["slash_command", "diagnostic_allowlist"],
                "requires": ["操作者确认", "远程可达", "存在可用 checkpoint", "若训练正在运行则接受 GPU/IsaacLab 并发风险"],
                "expected_result": "后端通过诊断白名单启动 preset，并持续暴露 status/log/report/playback。",
                "rollback": "如果诊断仍在运行，可以使用诊断取消动作停止 tmux 诊断会话。",
            },
            "transcript": [{"tool": "slash_action_gate", "args": {"preset": preset}, "result": {"requires_confirmation": True}}],
            "steps": 1,
        }

    if command == "/probe" and args[:1] == ["checkpoints"]:
        from .diagnostics import DiagnosticsController

        catalog = asyncio.run(DiagnosticsController(settings, source).catalog())
        if not catalog.available:
            reply = f"checkpoint catalog 不可用：{catalog.message or 'remote unavailable'}"
        elif not catalog.checkpoints:
            reply = "checkpoint catalog 可访问，但没有发现可用 checkpoint。请检查框架 checkpoint_roots/远程同步状态。"
        else:
            rows = [
                f"- {item.name} · iter={item.iteration or 'unknown'} · {item.path}"
                for item in catalog.checkpoints[:10]
            ]
            reply = "通过白名单 diagnostics.catalog 发现 checkpoint：\n" + "\n".join(rows)
        return {
            "reply": reply,
            "transcript": [{"tool": "diagnostics.catalog", "args": {"target": "checkpoints"}, "result": catalog.model_dump()}],
            "steps": 1,
        }

    if command == "/probe" and args[:1] == ["telemetry"]:
        return _probe_telemetry(settings, source)

    return {
        "reply": f"未知命令：{command}\n\n{HELP}",
        "transcript": [{"tool": "slash_unknown", "args": {"command": command}, "result": {"help": HELP}}],
        "steps": 1,
    }
