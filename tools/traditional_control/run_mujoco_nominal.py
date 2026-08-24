"""Run one serial nominal MuJoCo episode for the Taili controller."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from products.taili.traditional_control.evaluation import run_mujoco_episode, write_episode_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="episode seconds; defaults to the scenario-specific evaluation duration",
    )
    parser.add_argument("--result", type=Path, default=None)
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument(
        "--trace-dynamics",
        action="store_true",
        help="include full synchronized dynamics in every trace row",
    )
    args = parser.parse_args()
    result = run_mujoco_episode(
        args.scenario,
        duration_s=args.duration,
        trace_path=args.trace,
        trace_dynamics=args.trace_dynamics,
    )
    print(result.as_dict())
    if args.result is not None:
        print(write_episode_result(args.result, result))


if __name__ == "__main__":
    main()
