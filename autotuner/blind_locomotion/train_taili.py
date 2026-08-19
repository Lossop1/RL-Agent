"""Standalone Taili blind training entry point inside the runtime payload.

This is the payload-owned replacement for calling a train.py from a remote
RobotLab checkout. It relies only on IsaacLab/isaaclab_rl/skrl/PyTorch being
available in the Python environment and on this payload being on PYTHONPATH.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import sys
import traceback

from isaaclab.app import AppLauncher

if __name__ == "__main__" and __package__:
    # ``python -m taili_blind_runtime.train_taili`` executes as ``__main__``;
    # alias it so runtime proof does not import the train module a second time.
    sys.modules.setdefault(f"{__package__}.train_taili", sys.modules[__name__])

try:
    from .runtime_manifest import (
        REQUIRED_OPTIMIZATION_FIELDS,
        capture_optimization_state,
        capture_runtime_execution,
        checkpoint_inventory,
        initial_manifest,
        update_manifest,
        write_manifest,
    )
except ImportError:  # payload package is also executed as a top-level module.
    from runtime_manifest import (  # type: ignore
        REQUIRED_OPTIMIZATION_FIELDS,
        capture_optimization_state,
        capture_runtime_execution,
        checkpoint_inventory,
        initial_manifest,
        update_manifest,
        write_manifest,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Taili blind runtime with skrl AMP.")
    parser.add_argument("--task", type=str, default="RobotLab-Isaac-Taili-AMP-Blind-Direct-v0")
    parser.add_argument("--agent-yaml", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional skrl checkpoint to resume from.")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--allow-unverified-runtime",
        action="store_true",
        help="Explicitly bypass runtime evidence preflight; never use for a research run.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


SKRL_VERSION = "1.4.3"


def _manifest_path() -> Path:
    run_dir = os.environ.get("TAILI_RUN_DIR", "").strip()
    return Path(os.environ.get("TAILI_RUNTIME_MANIFEST", "runtime_manifest.json")) if not run_dir else Path(
        os.environ.get("TAILI_RUNTIME_MANIFEST", str(Path(run_dir) / "runtime_manifest.json"))
    )


def _state_ready(snapshot: dict) -> bool:
    fields = snapshot.get("fields") if isinstance(snapshot, dict) else None
    if not isinstance(fields, dict):
        return False
    accepted = {"captured", "fresh_initialization", "not_configured"}
    return all(isinstance(fields.get(name), dict) and fields[name].get("status") in accepted for name in REQUIRED_OPTIMIZATION_FIELDS)


def _resume_parity(
    checkpoint: str,
    inventory: dict,
    snapshot: dict,
) -> tuple[bool, dict]:
    """Require a parent runtime manifest before calling a resume complete."""
    if not checkpoint:
        ready = _state_ready(snapshot)
        return ready, {"mode": "fresh", "status": "proven" if ready else "blocked", "restored": {}}

    parent_run = Path(checkpoint).resolve().parent.parent
    parent_manifest_path = parent_run / "runtime_manifest.json"
    parent: dict = {}
    try:
        import json

        parent = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        parent = {}
    checkpoint_keys = set(str(item) for item in inventory.get("keys", []))
    # skrl checkpoint naming is version-dependent; accept the known aliases but
    # never infer optimizer/curriculum/RNG restoration from actor weights alone.
    key_aliases = {
        "policy": {"policy", "actor"},
        "value": {"value", "critic"},
        "optimizer": {"optimizer", "optimizers"},
        "normalizer": {"state_preprocessor", "value_preprocessor", "normalizer"},
        "amp": {"amp", "discriminator"},
    }
    restored = {
        name: bool(any(alias in checkpoint_keys for alias in aliases))
        for name, aliases in key_aliases.items()
    }
    parent_state = parent.get("optimization_state") if isinstance(parent, dict) else None
    parent_fields = parent_state.get("fields") if isinstance(parent_state, dict) else None
    parent_complete = isinstance(parent_fields, dict) and all(
        isinstance(parent_fields.get(name), dict) and parent_fields[name].get("status") in {
            "captured", "fresh_initialization", "not_configured"
        }
        for name in REQUIRED_OPTIMIZATION_FIELDS
    )
    current_complete = _state_ready(snapshot)
    # Scheduler, curriculum and RNG cannot be proven from a legacy .pt file;
    # they require the parent runtime manifest and an explicit restoration path.
    restored.update({
        "scheduler": bool(parent_complete and parent_fields.get("scheduler")),
        "log_std": bool(restored.get("policy") and parent_complete),
        "curriculum": bool(parent_complete and parent_fields.get("curriculum")),
        "rng": bool(parent_complete and parent_fields.get("rng")),
    })
    parity = bool(inventory.get("status") == "captured" and parent_complete and current_complete and all(restored.values()))
    return parity, {
        "mode": "resume",
        "status": "proven" if parity else "blocked",
        "parent_run": str(parent_run),
        "parent_manifest": str(parent_manifest_path),
        "checkpoint_inventory": inventory,
        "restored": restored,
    }


def _write_preflight(manifest_path: Path, payload: dict) -> None:
    update_manifest(manifest_path, runtime_preflight=payload)
    preflight_path = manifest_path.with_name("runtime_preflight.json")
    write_manifest(preflight_path, payload)


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
        # 环境在创建时据此恢复检查点所属 run 的地形课程 sidecar。
        os.environ["TAILI_RESUME_CHECKPOINT"] = args.checkpoint

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

    runtime_path = _manifest_path()
    if not runtime_path.is_file():
        run_dir = Path(os.environ.get("TAILI_RUN_DIR", "."))
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime = initial_manifest(
            run_id=os.environ.get("TAILI_RUN_ID", run_dir.name),
            run_dir=run_dir,
            task=args.task,
            payload_root=Path(__file__).resolve().parent.parent,
            source_config=os.environ.get("TAILI_CONFIG_PATH", str(Path(__file__).resolve().with_name("taili_blind_config.yaml"))),
            effective_config=os.environ.get("TAILI_EFFECTIVE_CONFIG", str(run_dir / "effective_config.yaml")),
            agent_config=args.agent_yaml or str(run_dir / "agent.skrl.yaml"),
            resume_checkpoint=args.checkpoint,
            seed=args.seed,
        )
        write_manifest(runtime_path, runtime)

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    experiment_cfg.setdefault("trainer", {})["close_environment_at_exit"] = False
    runner = Runner(env, experiment_cfg)
    # Capture modules after task construction and Runner setup. Importing an
    # expected module just to make a preflight pass would not prove that this
    # runtime actually used it.
    expected_modules = (
        "taili_blind_runtime",
        "taili_blind_runtime.train_taili",
        "taili_blind_runtime.blind_tp_env",
        "taili_blind_runtime.taili_amp_env",
        "taili_blind_runtime.taili_blind_config",
        "taili_blind_runtime.telemetry_emit",
        "taili_blind_runtime.telemetry_payloads",
        "taili_blind_runtime.taili_core.taili_reward",
        "taili_blind_runtime.taili_core.terrain_curriculum",
    )
    runtime_execution = capture_runtime_execution(
        Path(__file__).resolve().parent.parent,
        expected_modules,
        source_paths=(
            os.environ.get("TAILI_CONFIG_PATH", ""),
            os.environ.get("TAILI_EFFECTIVE_CONFIG", ""),
            args.agent_yaml,
        ),
    )
    update_manifest(
        runtime_path,
        runtime_execution=runtime_execution,
        run={"seed": experiment_cfg.get("seed"), "skrl_version": skrl.__version__},
    )
    run_dir = Path(os.environ.get("TAILI_RUN_DIR", runtime_path.parent))
    failure: BaseException | None = None
    try:
        inventory = checkpoint_inventory(args.checkpoint, torch_module=torch) if args.checkpoint else {
            "status": "not_applicable",
            "path": "",
            "keys": [],
        }
        if args.checkpoint:
            runner.agent.load(args.checkpoint)
            if hasattr(runner.agent, "set_running_mode"):
                runner.agent.set_running_mode("train")
            print(f"[TAILI_TRAIN] loaded checkpoint={args.checkpoint}", flush=True)

        snapshot = capture_optimization_state(
            runner.agent,
            env,
            run_dir=run_dir,
            torch_module=torch,
            amp_expected="AMP" in args.task.upper() or "amp" in str(experiment_cfg).lower(),
            parity_check=False,
            resume=bool(args.checkpoint),
        )
        parity, resume_edge = _resume_parity(args.checkpoint, inventory, snapshot)
        snapshot["parity_check"] = parity
        snapshot["status"] = "proven" if _state_ready(snapshot) and parity else snapshot.get("status", "missing")
        update_manifest(
            runtime_path,
            optimization_state=snapshot,
            resume_edge=resume_edge,
        )
        write_manifest(run_dir / "optimization_state.json", snapshot)
        preflight = {
            "schema_version": "rl-agent.runtime-preflight/v1",
            "status": "pass" if runtime_execution.get("status") == "proven" and snapshot.get("status") == "proven" else "blocked",
            "runtime_execution_status": runtime_execution.get("status"),
            "optimization_state_status": snapshot.get("status"),
            "resume_status": resume_edge.get("status"),
            "checkpoint": args.checkpoint,
            "reason": "" if runtime_execution.get("status") == "proven" and snapshot.get("status") == "proven" else "runtime source or optimization/resume evidence is incomplete",
        }
        _write_preflight(runtime_path, preflight)
        if preflight["status"] != "pass" and not args.allow_unverified_runtime:
            raise RuntimeError(
                "Taili runtime preflight blocked training; inspect "
                f"{run_dir / 'runtime_preflight.json'} or pass --allow-unverified-runtime explicitly"
            )
        if preflight["status"] != "pass":
            print("[TAILI_TRAIN] WARNING: bypassing unverified runtime preflight", flush=True)
        runner.run()
    except BaseException as exc:  # preserve failure evidence before re-raising
        failure = exc
        update_manifest(
            runtime_path,
            status="failed",
            errors=[{"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}],
        )
        raise
    finally:
        checkpoints = []
        checkpoint_dir = Path(os.environ.get("TAILI_CHECKPOINT_DIR", str(run_dir / "checkpoints")))
        if checkpoint_dir.is_dir():
            checkpoints = [
                {"checkpoint": item.name, **checkpoint_inventory(item, torch_module=None)}
                for item in sorted(checkpoint_dir.glob("agent_*.pt"), key=lambda item: item.stat().st_mtime_ns)[-20:]
            ]
        update_manifest(
            runtime_path,
            status="failed" if failure is not None else "complete",
            checkpoint_capabilities=checkpoints,
            evidence={"telemetry": [os.environ.get("TAILI_TELEMETRY_JSONL", "")]},
        )
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
