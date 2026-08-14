from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


W, H = 1920, 1080
BG = "#161815"
SURFACE = "#20221E"
SURFACE_2 = "#292B26"
INK = "#F2F1E8"
MUTED = "#999C91"
FAINT = "#575A52"
LINE = "#383B35"
LIME = "#D2FF00"
CYAN = "#45D6D0"
CORAL = "#FF6854"
AMBER = "#FFB84A"


def font(size: int, bold: bool = False):
    candidates = [
        Path(r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


F12 = font(12)
F14 = font(14)
F16 = font(16)
F18 = font(18)
F20 = font(20)
F22 = font(22, True)
F28 = font(28, True)
F36 = font(36, True)
F46 = font(46, True)
F64 = font(64, True)


def text(draw: ImageDraw.ImageDraw, xy, value: str, *, fill=INK, f=F16, anchor=None):
    draw.text(xy, value, fill=fill, font=f, anchor=anchor)


def rule(draw: ImageDraw.ImageDraw, x1, y1, x2, y2, *, fill=LINE, width=1):
    draw.line((x1, y1, x2, y2), fill=fill, width=width)


def pill(draw: ImageDraw.ImageDraw, box, label: str, *, fill=SURFACE_2, ink=INK, border=None, f=F14):
    draw.rounded_rectangle(box, radius=5, fill=fill, outline=border, width=1)
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2
    text(draw, (cx, cy), label, fill=ink, f=f, anchor="mm")


def wrap(draw: ImageDraw.ImageDraw, value: str, width: int, f=F16):
    rows: list[str] = []
    current = ""
    for char in value:
        candidate = current + char
        if draw.textlength(candidate, font=f) > width and current:
            rows.append(current)
            current = char
        else:
            current = candidate
    if current:
        rows.append(current)
    return rows


def sparkline(draw: ImageDraw.ImageDraw, box, values, color):
    x1, y1, x2, y2 = box
    lo, hi = min(values), max(values)
    points = []
    for i, value in enumerate(values):
        x = x1 + (x2 - x1) * i / (len(values) - 1)
        t = 0.5 if hi == lo else (value - lo) / (hi - lo)
        y = y2 - (y2 - y1) * t
        points.append((x, y))
    draw.line(points, fill=color, width=3, joint="curve")
    draw.ellipse((points[-1][0] - 4, points[-1][1] - 4, points[-1][0] + 4, points[-1][1] + 4), fill=color)


def main() -> None:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)

    # Top command rail
    draw.rectangle((0, 0, W, 72), fill="#111310")
    text(draw, (28, 22), "TAILI", fill=INK, f=F22)
    text(draw, (102, 25), "/ AGENT OPERATIONS", fill=MUTED, f=F14)
    draw.rectangle((315, 0, 319, 72), fill=LIME)
    text(draw, (344, 26), "训练调查台", fill=INK, f=F18)
    pill(draw, (1540, 17, 1640, 53), "REAL", fill="#253126", ink=LIME, border="#466044")
    draw.ellipse((1662, 29, 1674, 41), fill=LIME)
    text(draw, (1684, 26), "LIVE", fill=INK, f=F16)
    text(draw, (1780, 26), "03:27:18", fill=MUTED, f=F16)

    # Compact navigation rail
    draw.rectangle((0, 72, 76, H), fill="#121411")
    nav = [("A", 126, True), ("T", 198, False), ("C", 270, False), ("D", 342, False), ("H", 414, False)]
    for label, cy, active in nav:
        if active:
            draw.rectangle((0, cy - 28, 4, cy + 28), fill=LIME)
            draw.rounded_rectangle((15, cy - 25, 61, cy + 25), radius=5, fill=SURFACE_2)
        text(draw, (38, cy), label, fill=INK if active else MUTED, f=F18, anchor="mm")
    text(draw, (38, 1018), "?", fill=MUTED, f=F18, anchor="mm")

    # Main stage header
    left, right = 108, 1436
    text(draw, (left, 104), "当前调查", fill=MUTED, f=F14)
    text(draw, (left, 130), "为什么 phase_count 没有增加？", fill=INK, f=F36)
    pill(draw, (108, 184, 216, 218), "因果调查", fill="#302D22", ink=AMBER, border="#5C5134")
    pill(draw, (228, 184, 346, 218), "需要代码证据", fill="#20302E", ink=CYAN, border="#365956")
    text(draw, (right, 190), "REQUEST 0142", fill=FAINT, f=F14, anchor="ra")

    # Continuous investigation timeline
    timeline_y = 270
    stages = [
        (145, "01", "路由", LIME, True),
        (405, "02", "运行态", CYAN, True),
        (665, "03", "代码核对", AMBER, True),
        (925, "04", "证据门控", CORAL, True),
        (1185, "05", "结论", FAINT, False),
    ]
    rule(draw, 145, timeline_y, 1280, timeline_y, fill=LINE, width=2)
    rule(draw, 145, timeline_y, 925, timeline_y, fill=LIME, width=3)
    for x, num, label, color, done in stages:
        draw.ellipse((x - 11, timeline_y - 11, x + 11, timeline_y + 11), fill=BG, outline=color, width=3)
        if done:
            draw.ellipse((x - 4, timeline_y - 4, x + 4, timeline_y + 4), fill=color)
        text(draw, (x, timeline_y - 42), num, fill=color, f=F12, anchor="mm")
        text(draw, (x, timeline_y + 30), label, fill=INK if done else MUTED, f=F16, anchor="mm")
    text(draw, (1340, timeline_y - 7), "2.8s", fill=MUTED, f=F14, anchor="ra")

    # Investigation body, built as unframed bands rather than nested cards
    body_top = 336
    rule(draw, left, body_top, right, body_top)
    text(draw, (left, body_top + 26), "正在核对运行时代码", fill=AMBER, f=F20)
    text(draw, (left, body_top + 58), "Agent 正在定位计数器的写入条件，并与界面显示的阈值集合进行交叉验证。", fill=MUTED, f=F16)

    # Source evidence window
    code_x1, code_y1, code_x2, code_y2 = left, 435, 905, 694
    draw.rounded_rectangle((code_x1, code_y1, code_x2, code_y2), radius=6, fill="#111310", outline=LINE)
    draw.rectangle((code_x1, code_y1, code_x2, code_y1 + 44), fill="#1B1D19")
    draw.ellipse((code_x1 + 18, code_y1 + 17, code_x1 + 28, code_y1 + 27), fill=CORAL)
    text(draw, (code_x1 + 42, code_y1 + 13), "taili_amp_env.py", fill=INK, f=F14)
    text(draw, (code_x2 - 18, code_y1 + 13), "1528–1586", fill=MUTED, f=F14, anchor="ra")
    code_rows = [
        ("1528", "phase_ready = execution_ok and transition_safe", CYAN),
        ("1534", "phase_ready &= duty_valid and period_valid", INK),
        ("1541", "phase_ready &= yaw_gait_ok and event_count_ok", INK),
        ("1557", "if phase_ready:", AMBER),
        ("1558", "    self._phase_count += 1", LIME),
        ("1560", "else: self._phase_count = 0", CORAL),
    ]
    y = code_y1 + 62
    for line_no, row, color in code_rows:
        text(draw, (code_x1 + 22, y), line_no, fill=FAINT, f=F14)
        text(draw, (code_x1 + 82, y), row, fill=color, f=F16)
        y += 30

    # Evidence interpretation
    insight_x1, insight_y1, insight_x2, insight_y2 = 935, 435, right, 694
    draw.rounded_rectangle((insight_x1, insight_y1, insight_x2, insight_y2), radius=6, fill=SURFACE, outline=LINE)
    text(draw, (insight_x1 + 24, insight_y1 + 22), "证据解释", fill=MUTED, f=F14)
    text(draw, (insight_x1 + 24, insight_y1 + 58), "显示阈值通过", fill=INK, f=F28)
    text(draw, (insight_x1 + 24, insight_y1 + 98), "≠", fill=CORAL, f=F46)
    text(draw, (insight_x1 + 24, insight_y1 + 154), "完整运行时门控通过", fill=INK, f=F28)
    rule(draw, insight_x1 + 24, insight_y2 - 50, insight_x2 - 24, insight_y2 - 50)
    draw.ellipse((insight_x1 + 24, insight_y2 - 30, insight_x1 + 34, insight_y2 - 20), fill=AMBER)
    text(draw, (insight_x1 + 46, insight_y2 - 36), "仍需核对 6 个未展示条件", fill=AMBER, f=F14)

    # Claim gate band
    gate_y = 730
    rule(draw, left, gate_y, right, gate_y)
    text(draw, (left, gate_y + 24), "CLAIM GATE", fill=CORAL, f=F14)
    text(draw, (left, gate_y + 54), "阻止无依据的“所有条件都满足”结论", fill=INK, f=F22)
    text(draw, (left, gate_y + 91), "回答将被限制在已有证据范围内；缺失条件会作为明确调查项保留。", fill=MUTED, f=F16)
    pill(draw, (1180, gate_y + 42, 1436, gate_y + 86), "正在生成受约束结论  ···", fill="#2D2522", ink=AMBER, border="#59443B", f=F14)

    # Bottom composer
    composer_y = 928
    draw.rounded_rectangle((left, composer_y, right, 1037), radius=6, fill=SURFACE, outline="#4A4D45", width=2)
    text(draw, (left + 24, composer_y + 20), "继续追问或指定下一步调查…", fill=MUTED, f=F16)
    pill(draw, (left + 24, composer_y + 63, left + 146, composer_y + 95), "读取代码", fill=SURFACE_2, ink=CYAN, f=F12)
    pill(draw, (left + 156, composer_y + 63, left + 278, composer_y + 95), "对比运行态", fill=SURFACE_2, ink=LIME, f=F12)
    pill(draw, (right - 65, composer_y + 53, right - 20, composer_y + 98), "↑", fill=LIME, ink=BG, f=F22)

    # Right live telemetry rail
    rail_x1, rail_x2 = 1472, 1892
    draw.rectangle((rail_x1, 72, W, H), fill="#1B1D19")
    text(draw, (rail_x1 + 24, 104), "LIVE TRAINING", fill=LIME, f=F14)
    text(draw, (rail_x1 + 24, 136), "TAILI AMP / BLIND", fill=INK, f=F22)
    text(draw, (rail_x2, 140), "RUN 027", fill=MUTED, f=F14, anchor="ra")
    rule(draw, rail_x1 + 24, 184, rail_x2, 184)

    text(draw, (rail_x1 + 24, 210), "ITERATION", fill=MUTED, f=F12)
    text(draw, (rail_x1 + 24, 232), "18,420", fill=INK, f=F46)
    text(draw, (rail_x2, 256), "/ 50,000", fill=MUTED, f=F16, anchor="ra")
    sparkline(draw, (rail_x1 + 24, 296, rail_x2, 368), [8, 12, 11, 18, 22, 21, 28, 33, 31, 38, 42, 47, 51, 54, 60], LIME)
    text(draw, (rail_x1 + 24, 382), "reward  18.42", fill=LIME, f=F14)
    text(draw, (rail_x2, 382), "+6.8% / 1k", fill=MUTED, f=F14, anchor="ra")

    rule(draw, rail_x1 + 24, 424, rail_x2, 424)
    text(draw, (rail_x1 + 24, 450), "PHASE GATE", fill=MUTED, f=F12)
    text(draw, (rail_x1 + 24, 478), "Phase 0 → 1", fill=INK, f=F28)
    gates = [
        ("progress", "0.72 / 0.68", True),
        ("height", "0.31 / 0.28", True),
        ("tilt", "8.4° / 12°", True),
        ("runtime extras", "6 pending", False),
    ]
    y = 532
    for label, value, ok in gates:
        draw.ellipse((rail_x1 + 24, y + 4, rail_x1 + 36, y + 16), fill=LIME if ok else AMBER)
        text(draw, (rail_x1 + 50, y), label, fill=INK, f=F16)
        text(draw, (rail_x2, y), value, fill=MUTED if ok else AMBER, f=F14, anchor="ra")
        y += 46

    rule(draw, rail_x1 + 24, 726, rail_x2, 726)
    text(draw, (rail_x1 + 24, 752), "GAIT CONTACT", fill=MUTED, f=F12)
    # Abstract four-leg contact rhythm, domain signal instead of decoration
    feet = [(rail_x1 + 62, 814, "FL", True), (rail_x1 + 162, 786, "FR", False),
            (rail_x1 + 262, 786, "RL", False), (rail_x1 + 362, 814, "RR", True)]
    rule(draw, rail_x1 + 62, 814, rail_x1 + 362, 814, fill=FAINT, width=2)
    for x, y, label, contact in feet:
        draw.ellipse((x - 18, y - 18, x + 18, y + 18), fill=LIME if contact else SURFACE_2,
                     outline=LIME if contact else FAINT, width=2)
        text(draw, (x, y), label, fill=BG if contact else MUTED, f=F12, anchor="mm")
    text(draw, (rail_x1 + 24, 862), "diagonal pair", fill=INK, f=F16)
    text(draw, (rail_x2, 862), "stable", fill=LIME, f=F14, anchor="ra")

    rule(draw, rail_x1 + 24, 908, rail_x2, 908)
    text(draw, (rail_x1 + 24, 934), "SYSTEM", fill=MUTED, f=F12)
    system_rows = [("telemetry", "fresh 0.8s", LIME), ("code index", "ready", CYAN), ("actions", "locked", CORAL)]
    y = 966
    for label, value, color in system_rows:
        text(draw, (rail_x1 + 24, y), label, fill=INK, f=F14)
        text(draw, (rail_x2, y), value, fill=color, f=F14, anchor="ra")
        y += 27

    # Fine framing marks carry the engineered editorial character.
    for x in (104, 1436, 1472, 1892):
        rule(draw, x, 72, x, 84, fill=FAINT)
    text(draw, (82, 1058), "CONCEPT 01  /  EVIDENCE-FIRST AGENT UI", fill=FAINT, f=F12)

    output = Path("output/concepts/agent-system-ui-concept.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)
    print(output.resolve())


if __name__ == "__main__":
    main()
