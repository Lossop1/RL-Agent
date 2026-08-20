from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


W, H = 1920, 1080
BG = "#F5F5F7"
PAPER = "#FFFFFF"
INK = "#1D1D1F"
MUTED = "#6E6E73"
FAINT = "#A1A1A6"
LINE = "#DEDEE3"
LINE_SOFT = "#ECECF0"
BLUE = "#0066CC"
BLUE_SOFT = "#EAF2FF"
GREEN = "#1A8A4E"
GREEN_SOFT = "#E7F6EC"
RED = "#D23A2E"
RED_SOFT = "#FDECEA"
AMBER = "#8A6100"
AMBER_SOFT = "#FDF2D4"


def font(size: int, bold: bool = False):
    paths = [
        Path(r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


F10, F11, F12, F13, F14, F15, F16, F18 = [font(n) for n in (10, 11, 12, 13, 14, 15, 16, 18)]
F20B, F24B, F30B, F38B, F48B = [font(n, True) for n in (20, 24, 30, 38, 48)]


def fetch(path: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"http://127.0.0.1:8000{path}", timeout=25) as response:
        return json.load(response)


def get(record: Any, *keys: str, fallback=None):
    value = record
    for key in keys:
        if not isinstance(value, dict):
            return fallback
        value = value.get(key)
    return fallback if value is None else value


def num(value: Any, digits: int = 3, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}{suffix}"


def txt(draw: ImageDraw.ImageDraw, xy, value: Any, *, fill=INK, f=F14, anchor=None):
    draw.text(xy, str(value), fill=fill, font=f, anchor=anchor)


def line(draw: ImageDraw.ImageDraw, xy, *, fill=LINE, width=1):
    draw.line(xy, fill=fill, width=width)


def wrap(draw: ImageDraw.ImageDraw, value: str, width: int, f=F14, max_rows: int | None = None):
    rows: list[str] = []
    current = ""
    for char in value:
        candidate = current + char
        if current and draw.textlength(candidate, font=f) > width:
            rows.append(current)
            current = char
            if max_rows and len(rows) == max_rows:
                break
        else:
            current = candidate
    if current and (not max_rows or len(rows) < max_rows):
        rows.append(current)
    return rows


def paragraph(draw: ImageDraw.ImageDraw, xy, value: str, width: int, *, fill=MUTED, f=F14, gap=7, max_rows=None):
    x, y = xy
    rows = wrap(draw, value, width, f, max_rows)
    for row in rows:
        txt(draw, (x, y), row, fill=fill, f=f)
        y += f.size + gap
    return y


def tag(draw: ImageDraw.ImageDraw, box, label: str, *, fill, ink, border=None, f=F12):
    draw.rounded_rectangle(box, radius=6, fill=fill, outline=border or fill)
    txt(draw, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), label, fill=ink, f=f, anchor="mm")


def metric_from_scoreboard(board: dict[str, Any], key: str) -> dict[str, Any]:
    for family in board.get("families") or []:
        for metric in family.get("metrics") or []:
            if metric.get("key") == key:
                return metric
    return {}


def plot(draw: ImageDraw.ImageDraw, box, values, *, color, lo=0.0, hi=1.0, width=3):
    clean = [float(value) if isinstance(value, (int, float)) else None for value in values]
    valid = [value for value in clean if value is not None]
    if len(valid) < 2:
        return
    x1, y1, x2, y2 = box
    points = []
    for index, value in enumerate(clean):
        if value is None:
            continue
        x = x1 + (x2 - x1) * index / max(1, len(clean) - 1)
        y = y2 - (y2 - y1) * max(0.0, min(1.0, (value - lo) / (hi - lo)))
        points.append((x, y))
    draw.line(points, fill=color, width=width, joint="curve")
    if points:
        x, y = points[-1]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)


def main() -> None:
    workbench = fetch("/agent/workbench")
    telemetry = fetch("/run/current/telemetry")
    latest = telemetry.get("latest") or {}
    curriculum = latest.get("curriculum") or {}
    command = latest.get("command") or {}
    history = (telemetry.get("history") or [])[-48:]
    attempt = workbench.get("attempt") or {}
    judgement = workbench.get("judgement") or {}

    progress = curriculum.get("progress_gate")
    progress_floor = get(curriculum, "phase_gate", "conditions", "progress_min")
    slip = command.get("stance_slip_high_fraction")
    diagonal = command.get("diagonal_contact")
    diagonal_floor = get(curriculum, "phase_gate", "conditions", "diagonal_min")
    duty = command.get("duty_balance")
    duty_floor = get(curriculum, "phase_gate", "conditions", "duty_min")

    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)

    # Existing product shell: same brand, same four workspaces, same light foundation.
    draw.rectangle((0, 0, W, 64), fill="#FBFBFD")
    line(draw, (0, 63, W, 63), fill=LINE_SOFT)
    draw.rounded_rectangle((28, 15, 62, 49), radius=8, fill=INK)
    txt(draw, (45, 32), "策", fill=PAPER, f=F15, anchor="mm")
    txt(draw, (76, 18), "运动策略智能体", f=F14)
    txt(draw, (76, 38), "训练迭代工作台", fill=MUTED, f=F11)

    nav = [("智能体", True), ("目标记分牌", False), ("诊断工具", False), ("配置工具", False)]
    nx = 684
    draw.rounded_rectangle((nx - 8, 13, nx + 494, 51), radius=9, fill="#ECECEF")
    for label, active in nav:
        width = 118
        if active:
            draw.rounded_rectangle((nx, 17, nx + width, 47), radius=7, fill=PAPER, outline=LINE)
        txt(draw, (nx + width / 2, 32), label, fill=INK if active else MUTED, f=F13, anchor="mm")
        nx += width + 2
    tag(draw, (1735, 17, 1812, 47), "实时", fill=GREEN_SOFT, ink=GREEN, border="#B9DFC8")
    txt(draw, (1888, 32), "刷新", fill=BLUE, f=F13, anchor="ra")

    left, split, right = 52, 1280, 1868

    # Objective is context, not a hero card.
    txt(draw, (left, 90), "当前目标", fill=MUTED, f=F11)
    objective = str(workbench.get("objective") or "")
    paragraph(draw, (left, 110), objective, 1420, fill=INK, f=F16, gap=4, max_rows=2)
    txt(draw, (right, 111), str(attempt.get("run_id") or ""), fill=MUTED, f=F12, anchor="ra")

    # One continuous run ribbon establishes comparability for every number below.
    ribbon_y1, ribbon_y2 = 162, 230
    draw.rectangle((left, ribbon_y1, right, ribbon_y2), fill=INK)
    draw.ellipse((left + 20, 183, left + 32, 195), fill="#63D68C")
    txt(draw, (left + 44, 178), "训练运行中", fill=PAPER, f=F14)
    txt(draw, (left + 44, 201), str(attempt.get("runtime_state") or "unknown"), fill="#A8E6BE", f=F11)
    ribbon = [
        ("STEP", f"{int(attempt.get('step') or 0):,} / {int(attempt.get('total_steps') or 0):,}"),
        ("PHASE", attempt.get("phase") or "—"),
        ("COMMAND", attempt.get("command_mode") or "—"),
        ("TELEMETRY", f"fresh {num(attempt.get('telemetry_age_s'), 1, 's')}"),
        ("HISTORY", f"{len(telemetry.get('history') or [])} points"),
    ]
    x = 280
    for label, value in ribbon:
        txt(draw, (x, 177), label, fill="#8F8F93", f=F10 if 'F10' in globals() else F11)
        txt(draw, (x, 199), value, fill=PAPER, f=F14)
        x += 230
    condition_complete = bool(get(curriculum, "phase_gate", "condition_set_complete", fallback=False))
    tag(draw, (right - 230, 179, right - 20, 214), "门控条件集完整" if condition_complete else "展示阈值不是完整门控", fill="#4A2E2A", ink="#FFB5AA", f=F12)

    # Service loop remains visible as product structure, with evidence/judgement active now.
    service_loop = workbench.get("service_loop") or []
    y = 254
    x = left
    for index, label in enumerate(service_loop):
        active = label in {"证据", "判断"}
        draw.ellipse((x, y + 5, x + 8, y + 13), fill=BLUE if active else FAINT)
        txt(draw, (x + 16, y), label, fill=INK if active else MUTED, f=F12)
        x += 105
        if index < len(service_loop) - 1:
            line(draw, (x - 25, y + 9, x - 10, y + 9), fill=LINE)

    # Main stage: the current contradiction determines the composition.
    main_top = 302
    line(draw, (left, main_top, right, main_top), fill=LINE)
    txt(draw, (left, main_top + 26), "当前训练矛盾", fill=RED, f=F12)
    txt(draw, (left, main_top + 54), "推进已经达标，", f=F38B)
    txt(draw, (left, main_top + 101), "但接触质量仍在拖后腿。", f=F38B)
    summary = str(get(telemetry, "snapshot", "conclusion", fallback=attempt.get("summary") or ""))
    paragraph(draw, (left, main_top + 160), summary, 790, fill=MUTED, f=F15, gap=6, max_rows=2)

    # Relationship field: these are not independent cards; lines explain the tension.
    field_y = 510
    nodes = [
        (left, "阶段推进", progress, progress_floor, "≥", GREEN, "本次配置"),
        (left + 300, "高滑移比例", slip, 0.05, "≤", RED, "显示底线·出处待核"),
        (left + 600, "对角支撑", diagonal, diagonal_floor, "≥", GREEN, "本次配置"),
        (left + 900, "占空平衡", duty, duty_floor, "≥", GREEN, "本次配置"),
    ]
    line(draw, (left, field_y + 52, split - 42, field_y + 52), fill=LINE, width=2)
    for index, (x, label, value, floor, direction, color, source) in enumerate(nodes):
        draw.ellipse((x, field_y + 43, x + 18, field_y + 61), fill=PAPER, outline=color, width=4)
        txt(draw, (x, field_y), label, fill=MUTED, f=F12)
        txt(draw, (x, field_y + 74), num(value), fill=color, f=F30B)
        txt(draw, (x, field_y + 113), f"底线 {direction}{num(floor)}", fill=INK, f=F12)
        txt(draw, (x, field_y + 136), source, fill=MUTED, f=F11)
        if index == 0:
            txt(draw, (x + 204, field_y + 38), "速度提升会牵制接触质量", fill=RED, f=F11, anchor="ma")

    # Real recent trend, shared timeline, raw 0..1 scale.
    chart = (left, 704, split - 42, 828)
    draw.rectangle(chart, fill="#FAFAFC")
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = chart[3] - (chart[3] - chart[1]) * tick
        line(draw, (chart[0], yy, chart[2], yy), fill=LINE_SOFT)
        txt(draw, (chart[0] - 8, yy), f"{tick:.2f}", fill=FAINT, f=F10 if 'F10' in globals() else F11, anchor="rm")
    progress_values = [get(row, "curriculum", "progress_gate") for row in history]
    slip_values = [get(row, "command", "stance_slip_high_fraction") for row in history]
    plot(draw, chart, progress_values, color=GREEN, lo=0, hi=1, width=3)
    plot(draw, chart, slip_values, color=RED, lo=0, hi=1, width=3)
    floor_y = chart[3] - (chart[3] - chart[1]) * 0.05
    line(draw, (chart[0], floor_y, chart[2], floor_y), fill="#E7AAA3", width=2)
    txt(draw, (chart[0], chart[1] - 24), "最近 48 个遥测点 · 原始值", fill=MUTED, f=F11)
    tag(draw, (chart[2] - 254, chart[1] - 31, chart[2] - 132, chart[1] - 7), "推进 gate", fill=GREEN_SOFT, ink=GREEN, f=F11)
    tag(draw, (chart[2] - 124, chart[1] - 31, chart[2], chart[1] - 7), "高滑移比例", fill=RED_SOFT, ink=RED, f=F11)

    # Agent channel is attached to this evidence, not a detached chatbot.
    agent_x = split + 8
    draw.rectangle((agent_x, main_top, right, 884), fill=PAPER)
    line(draw, (agent_x, main_top, agent_x, 884), fill=LINE)
    txt(draw, (agent_x + 28, main_top + 24), "智能体判断", fill=BLUE, f=F12)
    confidence = str(judgement.get("confidence") or "unknown")
    tag(draw, (right - 125, main_top + 18, right - 24, main_top + 46), f"可信度 {confidence}", fill=BLUE_SOFT, ink=BLUE, f=F11)
    paragraph(draw, (agent_x + 28, main_top + 62), str(judgement.get("title") or ""), 480, fill=INK, f=F24B, gap=8, max_rows=2)
    paragraph(draw, (agent_x + 28, main_top + 138), str(judgement.get("summary") or ""), 480, fill=MUTED, f=F13, gap=6, max_rows=4)

    line(draw, (agent_x + 28, main_top + 248, right - 24, main_top + 248), fill=LINE_SOFT)
    txt(draw, (agent_x + 28, main_top + 270), "这次判断使用的证据", fill=MUTED, f=F11)
    evidence = workbench.get("evidence") or []
    ey = main_top + 300
    for item in evidence:
        status = str(item.get("status") or "unknown")
        color = GREEN if status == "ok" else AMBER if status == "warning" else RED
        draw.ellipse((agent_x + 28, ey + 5, agent_x + 38, ey + 15), fill=color)
        txt(draw, (agent_x + 50, ey), item.get("label") or "", f=F13)
        txt(draw, (right - 24, ey), status, fill=color, f=F11, anchor="ra")
        ey += 34

    line(draw, (agent_x + 28, main_top + 486, right - 24, main_top + 486), fill=LINE_SOFT)
    txt(draw, (agent_x + 28, main_top + 508), "下一步", fill=MUTED, f=F11)
    diagnostic = next((item for item in workbench.get("actions") or [] if item.get("id") == "run-diagnostic"), {})
    txt(draw, (agent_x + 28, main_top + 536), diagnostic.get("label") or "运行物理诊断", f=F16)
    txt(draw, (agent_x + 28, main_top + 563), "检查点回放 · 需要确认 · 不改变训练权重", fill=MUTED, f=F11)
    tag(draw, (right - 150, main_top + 526, right - 24, main_top + 562), "打开诊断", fill=BLUE, ink=PAPER, f=F12)

    # Composer remains available without displacing the evidence.
    draw.rounded_rectangle((agent_x + 28, 766, right - 24, 830), radius=8, fill="#F6F7F9", outline=LINE)
    txt(draw, (agent_x + 48, 787), "追问当前判断，或要求核对代码机制…", fill=MUTED, f=F13)
    draw.rounded_rectangle((right - 72, 778, right - 36, 818), radius=7, fill=BLUE)
    txt(draw, (right - 54, 798), "↑", fill=PAPER, f=F18, anchor="mm")

    # Evidence ledger and action boundary are one bottom operational band.
    lower_y = 904
    line(draw, (left, lower_y, right, lower_y), fill=LINE)
    txt(draw, (left, lower_y + 20), "证据账本", fill=MUTED, f=F11)
    ex = left
    for item in evidence:
        status = str(item.get("status") or "unknown")
        color = GREEN if status == "ok" else AMBER if status == "warning" else RED
        txt(draw, (ex, lower_y + 48), item.get("label") or "", f=F13)
        txt(draw, (ex, lower_y + 72), status, fill=color, f=F11)
        source = str(item.get("source") or "")
        txt(draw, (ex, lower_y + 94), source, fill=MUTED, f=F10 if 'F10' in globals() else F11)
        ex += 198
    line(draw, (1050, lower_y + 18, 1050, H - 36), fill=LINE)
    txt(draw, (1080, lower_y + 20), "动作边界", fill=MUTED, f=F11)
    actions = workbench.get("actions") or []
    action_x = 1080
    for action_id in ("run-diagnostic", "deploy-payload", "start-training", "resume-training", "kill-training"):
        action = next((item for item in actions if item.get("id") == action_id), {})
        enabled = bool(action.get("enabled"))
        risk = str(action.get("risk") or "")
        color = RED if risk == "high" and enabled else BLUE if enabled else FAINT
        label = str(action.get("label") or action_id)
        txt(draw, (action_x, lower_y + 49), label, fill=color if enabled else MUTED, f=F12)
        txt(draw, (action_x, lower_y + 72), "可用 · 需确认" if enabled else "当前锁定", fill=color, f=F10 if 'F10' in globals() else F11)
        action_x += 158

    txt(draw, (left, H - 24), "CONCEPT 02 · 基于实时 AgentWorkbench / Telemetry / Scoreboard 数据", fill=FAINT, f=F10 if 'F10' in globals() else F11)
    txt(draw, (right, H - 24), "不改变现有前端 · 设计研究稿", fill=FAINT, f=F11, anchor="ra")

    output = Path("output/concepts/agent-system-ui-concept-v2.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG", optimize=True)
    print(output.resolve())


if __name__ == "__main__":
    main()
