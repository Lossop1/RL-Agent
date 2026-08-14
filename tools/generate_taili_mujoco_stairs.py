"""Generate a parameterized Taili staircase MJCF and optionally launch SAR."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shlex
import xml.etree.ElementTree as ET


SCENE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def scalar(value: float) -> str:
    return f"{value:.8g}"


def vector(*values: float) -> str:
    return " ".join(scalar(value) for value in values)


def add_geom(
    worldbody: ET.Element,
    *,
    name: str,
    geom_type: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    rgba: str,
    friction: float,
) -> None:
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": name,
            "type": geom_type,
            "pos": vector(*pos),
            "size": vector(*size),
            "rgba": rgba,
            "friction": vector(friction, 0.005, 0.0001),
            "contype": "2",
            "conaffinity": "1",
        },
    )


def build_scene(args: argparse.Namespace) -> ET.ElementTree:
    total_height = args.height * args.steps
    course_length = args.approach + args.depth * args.steps + args.landing
    center_x = 0.5 * (args.approach + args.depth * args.steps)
    center_z = 0.5 * total_height if args.direction == "up" else -0.5 * total_height
    extent = max(2.5, 0.65 * course_length, 1.2 * args.width)
    half_width = 0.5 * args.width

    root = ET.Element(
        "mujoco",
        {"model": f"taili stairs {args.direction} {scalar(args.height)}m"},
    )
    root.append(
        ET.Comment(
            " generated parameters: "
            f"direction={args.direction} height={scalar(args.height)} "
            f"depth={scalar(args.depth)} steps={args.steps} "
            f"width={scalar(args.width)} approach={scalar(args.approach)} "
            f"landing={scalar(args.landing)} friction={scalar(args.friction)} "
        )
    )
    ET.SubElement(root, "include", {"file": "taili.xml"})
    ET.SubElement(
        root,
        "statistic",
        {
            "center": vector(center_x, 0.0, center_z + 0.25),
            "extent": scalar(extent),
        },
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        {
            "diffuse": "0.6 0.6 0.6",
            "ambient": "0.3 0.3 0.3",
            "specular": "0 0 0",
        },
    )
    ET.SubElement(visual, "global", {"azimuth": "-130", "elevation": "-20"})
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {"pos": vector(center_x, 0.0, max(2.5, total_height + 1.5)), "dir": "0 0 -1", "directional": "true"},
    )

    if args.direction == "up":
        add_geom(
            worldbody,
            name="floor",
            geom_type="plane",
            pos=(0.0, 0.0, 0.0),
            size=(0.0, 0.0, 0.05),
            rgba="0.22 0.28 0.34 1",
            friction=args.friction,
        )
        for index in range(1, args.steps + 1):
            top = index * args.height
            center = args.approach + (index - 0.5) * args.depth
            add_geom(
                worldbody,
                name=f"stair_up_{index:02d}",
                geom_type="box",
                pos=(center, 0.0, 0.5 * top),
                size=(0.5 * args.depth, half_width, 0.5 * top),
                rgba="0.52 0.59 0.68 1",
                friction=args.friction,
            )
        landing_start = args.approach + args.steps * args.depth
        add_geom(
            worldbody,
            name="upper_landing",
            geom_type="box",
            pos=(landing_start + 0.5 * args.landing, 0.0, 0.5 * total_height),
            size=(0.5 * args.landing, half_width, 0.5 * total_height),
            rgba="0.42 0.50 0.60 1",
            friction=args.friction,
        )
    else:
        lower_z = -total_height
        add_geom(
            worldbody,
            name="lower_ground",
            geom_type="plane",
            pos=(0.0, 0.0, lower_z),
            size=(0.0, 0.0, 0.05),
            rgba="0.22 0.28 0.34 1",
            friction=args.friction,
        )
        add_geom(
            worldbody,
            name="top_platform",
            geom_type="box",
            pos=(0.0, 0.0, -0.05),
            size=(args.approach, half_width, 0.05),
            rgba="0.42 0.50 0.60 1",
            friction=args.friction,
        )
        for index in range(1, args.steps):
            top = -index * args.height
            center = args.approach + (index - 0.5) * args.depth
            half_height = 0.5 * (top - lower_z)
            add_geom(
                worldbody,
                name=f"stair_down_{index:02d}",
                geom_type="box",
                pos=(center, 0.0, lower_z + half_height),
                size=(0.5 * args.depth, half_width, half_height),
                rgba="0.52 0.59 0.68 1",
                friction=args.friction,
            )
        landing_start = args.approach + args.steps * args.depth
        add_geom(
            worldbody,
            name="lower_landing",
            geom_type="box",
            pos=(landing_start + 0.5 * args.landing, 0.0, lower_z - 0.025),
            size=(0.5 * args.landing, half_width, 0.025),
            rgba="0.42 0.50 0.60 1",
            friction=args.friction,
        )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    return tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Taili staircase scene for the unified SAR MuJoCo runner."
    )
    parser.add_argument("--direction", choices=("up", "down"), required=True)
    parser.add_argument(
        "--height",
        type=positive_float,
        required=True,
        help="step height in meters",
    )
    parser.add_argument("--depth", type=positive_float, default=0.40, help="step tread depth in meters")
    parser.add_argument("--steps", type=positive_int, default=5)
    parser.add_argument("--width", type=positive_float, default=2.40, help="full staircase width in meters")
    parser.add_argument("--approach", type=positive_float, default=1.20, help="distance from spawn to first riser")
    parser.add_argument("--landing", type=positive_float, default=2.00, help="landing length after the staircase")
    parser.add_argument("--friction", type=positive_float, default=1.0)
    parser.add_argument("--scene-name", default="stairs_runtime")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--launch", action="store_true", help="replace this process with the unified SAR runner")
    args = parser.parse_args()
    if not SCENE_NAME_RE.fullmatch(args.scene_name):
        parser.error("--scene-name may contain only letters, digits, underscore, and hyphen")
    return args


def main() -> None:
    args = parse_args()
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else Path(__file__).resolve().parent.parent
    )
    mjcf_dir = project_root / "src" / "rl_sar_zoo" / "taili_description" / "mjcf"
    if not (mjcf_dir / "taili.xml").is_file():
        raise FileNotFoundError(f"Taili MJCF not found under {mjcf_dir}")
    output = mjcf_dir / f"{args.scene_name}.xml"
    temporary = mjcf_dir / f".{args.scene_name}.xml.tmp"
    build_scene(args).write(temporary, encoding="utf-8", xml_declaration=False)
    with temporary.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    os.replace(temporary, output)

    command = [
        str(project_root / "cmake_build" / "bin" / "rl_sim_mujoco"),
        "taili",
        args.scene_name,
    ]
    print(f"scene={output}")
    print("launch=" + " ".join(shlex.quote(part) for part in command))
    if args.launch:
        if not Path(command[0]).is_file():
            raise FileNotFoundError(command[0])
        os.chdir(project_root)
        os.execv(command[0], command)


if __name__ == "__main__":
    main()
