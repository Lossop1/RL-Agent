"""Standalone Taili blind training entry point inside the runtime payload.

This is the payload-owned replacement for calling a train.py from a remote
RobotLab checkout. It relies only on IsaacLab/isaaclab_rl/skrl/PyTorch being
available in the Python environment and on this payload being on PYTHONPATH.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random

from isaaclab.app import AppLauncher


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Taili blind runtime with skrl AMP.")
    parser.add_argument("--task", type=str, default="RobotLab-Isaac-Taili-AMP-Blind-Direct-v0")
    parser.add_argument("--agent-yaml", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional skrl checkpoint to resume from.")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser


SKRL_VERSION = "1.4.3"


def _load_experiment_cfg(args) -> dict:
    import yaml
    from isaaclab_tasks.utils import load_cfg_from_registry

    from .taili_blind_config import build_skrl_config, load_taili_blind_config

    if args.agent_yaml:
        path = Path(args.agent_yaml)
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    else:
        try:
            loaded = load_taili_blind_config()
        except Exception:
            loaded = load_cfg_from_registry(args.task, "skrl_amp_cfg_entry_point")
    if isinstance(loaded, dict) and "skrl" in loaded:
        return build_skrl_config(loaded)
    return loaded


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint}")
        args.checkpoint = str(checkpoint)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import skrl
    import torch  # noqa: F401  # imported before Runner to match IsaacLab skrl examples
    from packaging import version
    from skrl.utils.runner.torch import Runner

    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    import taili_blind_runtime  # noqa: F401  # registers task + skrl policy components

    if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
        raise RuntimeError(f"Unsupported skrl version: {skrl.__version__}; need >= {SKRL_VERSION}")

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    experiment_cfg = _load_experiment_cfg(args)

    if args.seed == -1:
        args.seed = random.randint(0, 10000)
    if args.seed is not None:
        experiment_cfg["seed"] = args.seed
    env_cfg.seed = experiment_cfg.get("seed", getattr(env_cfg, "seed", None))

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    experiment_cfg.setdefault("trainer", {})["close_environment_at_exit"] = False
    runner = Runner(env, experiment_cfg)
    if args.checkpoint:
        runner.agent.load(args.checkpoint)
        if hasattr(runner.agent, "set_running_mode"):
            runner.agent.set_running_mode("train")
        print(f"[TAILI_TRAIN] loaded checkpoint={args.checkpoint}", flush=True)
    runner.run()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
