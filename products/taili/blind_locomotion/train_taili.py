"""Standalone Taili blind training entry point inside the runtime payload.

This is the payload-owned replacement for calling a train.py from a remote
RobotLab checkout. It relies only on IsaacLab/isaaclab_rl/skrl/PyTorch being
available in the Python environment and on this payload being on PYTHONPATH.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import traceback
from typing import Any, Mapping

from isaaclab.app import AppLauncher

try:
    # Prefer the compatibility checker shipped inside the payload.  Falling
    # back to a remote/source-tree package would make resume identity depend
    # on an unrelated checkout.
    from .resume_compatibility import check_resume_compatibility  # type: ignore
except ImportError:  # source-tree compatibility: the helper lives in execution/
    try:
        from autotuner.execution.compatibility import check_resume_compatibility
    except ImportError:
        check_resume_compatibility = None  # type: ignore

if __name__ == "__main__" and __package__:
    # ``python -m taili_blind_runtime.train_taili`` executes as ``__main__``;
    # alias it so runtime proof does not import the train module a second time.
    sys.modules.setdefault(f"{__package__}.train_taili", sys.modules[__name__])

try:
    from .runtime_manifest import (
        REQUIRED_OPTIMIZATION_FIELDS,
        capture_optimization_state,
        capture_runtime_execution,
        capture_runtime_identity,
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
        capture_runtime_identity,
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


def _load_parent_manifest(checkpoint: str) -> tuple[Path, dict]:
    """Load the immutable parent run evidence without touching the agent."""
    parent_run = Path(checkpoint).resolve().parent.parent
    manifest_path = parent_run / "runtime_manifest.json"
    try:
        parent = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        parent = {}
    return manifest_path, parent if isinstance(parent, dict) else {}


def _resume_identity_preflight(
    checkpoint: str,
    inventory: dict,
    current_manifest: dict,
) -> dict:
    """Prove checkpoint identity compatibility before ``agent.load``.

    Loading a mismatched checkpoint can fail with a low-level tensor-shape
    error or, worse, partially mutate an agent before the error is raised.
    Identity comparison is therefore a separate preflight stage. Complete
    optimizer/curriculum/RNG restoration is checked later, after loading.
    """
    if not checkpoint:
        return {
            "status": "proven",
            "decision": "fresh",
            "compatible": True,
            "compatibility": {"decision": "fresh", "compatible": True},
        }
    manifest_path, parent = _load_parent_manifest(checkpoint)
    if check_resume_compatibility is None:
        return {
            "status": "blocked",
            "decision": "blocked",
            "compatible": False,
            "parent_manifest": str(manifest_path),
            "reasons": ["resume compatibility implementation is unavailable"],
            "compatibility": {"decision": "blocked", "compatible": False},
        }
    decision = check_resume_compatibility(
        parent,
        current_manifest,
        checkpoint=inventory,
        requested=True,
        strict=True,
    )
    return {
        "status": "proven" if decision.compatible else "blocked",
        "decision": decision.decision,
        "compatible": decision.compatible,
        "parent_manifest": str(manifest_path),
        "reasons": list(decision.reasons),
        "compatibility": decision.to_dict(),
    }


def _resume_parity(
    checkpoint: str,
    inventory: dict,
    snapshot: dict,
    *,
    current_manifest: dict | None = None,
    identity_preflight: Mapping[str, Any] | None = None,
) -> tuple[bool, dict]:
    """Require a parent runtime manifest before calling a resume complete."""
    if not checkpoint:
        ready = _state_ready(snapshot)
        return ready, {"mode": "fresh", "status": "proven" if ready else "blocked", "restored": {}}

    parent_manifest_path, parent = _load_parent_manifest(checkpoint)
    parent_run = parent_manifest_path.parent
    raw_keys = inventory.get("keys", []) if isinstance(inventory, dict) else []
    raw_key_paths = inventory.get("key_paths", []) if isinstance(inventory, dict) else []
    raw_keys = raw_keys if isinstance(raw_keys, (list, tuple, set)) else []
    raw_key_paths = raw_key_paths if isinstance(raw_key_paths, (list, tuple, set)) else []
    checkpoint_keys = {
        str(item).lower().replace("\\", ".")
        for item in (*raw_keys, *raw_key_paths)
    }
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
        name: bool(any(alias == key or alias in key for alias in aliases for key in checkpoint_keys))
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
    compatibility: dict[str, Any] = {"status": "not_evaluated"}
    if identity_preflight is not None:
        compatibility = dict(identity_preflight.get("compatibility") or {})
    elif current_manifest is not None and check_resume_compatibility is not None:
        decision = check_resume_compatibility(
            parent,
            current_manifest,
            checkpoint=inventory,
            requested=True,
            strict=True,
        )
        compatibility = decision.to_dict()
        parity = parity and decision.compatible
    return parity, {
        "mode": "resume",
        "status": "proven" if parity else "blocked",
        "parent_run": str(parent_run),
        "parent_manifest": str(parent_manifest_path),
        "checkpoint_inventory": inventory,
        "restored": restored,
        "compatibility": compatibility,
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

    import skrl
    import torch  # noqa: F401  # imported before Runner to match IsaacLab skrl examples
    from packaging import version
    from skrl.utils.runner.torch import Runner

    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    try:
        # 载荷内自带（taili_blind_runtime/taili_sim）：远端只有载荷在 PYTHONPATH 上。
        from .taili_sim.backend_factory import create_backend  # type: ignore
    except ImportError:  # 源码树兼容：工厂住在 autotuner/simulation/
        from autotuner.simulation.backend_factory import create_backend

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

    backend = create_backend(args.task, env_cfg, backend_type="isaaclab")
    # skrl 的包装器要的是 IsaacLab 原生环境：它按 env.unwrapped 认类型，并把 step/reset
    # 直接发给传进来的那个对象。SimulatorBackend 适配器的 reset 只返回观测、step 返回
    # 四元组，契约不同，所以把底层环境取出来交给它。
    env = SkrlVecEnvWrapper(backend.unwrapped, ml_framework="torch")

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
    current_before_execution = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.is_file() else {}
    runtime_evidence = capture_runtime_identity(current_before_execution.get("runtime", {}))
    update_manifest(
        runtime_path,
        runtime_execution=runtime_execution,
        runtime_evidence=runtime_evidence,
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
        current_manifest = json.loads(runtime_path.read_text(encoding="utf-8"))
        identity_preflight = _resume_identity_preflight(args.checkpoint, inventory, current_manifest)
        update_manifest(runtime_path, resume_preflight=identity_preflight)
        if args.checkpoint and identity_preflight.get("status") != "proven":
            _write_preflight(
                runtime_path,
                {
                    "schema_version": "rl-agent.runtime-preflight/v1",
                    "status": "blocked",
                    "runtime_execution_status": runtime_execution.get("status"),
                    "resume_identity_status": identity_preflight.get("status"),
                    "checkpoint": args.checkpoint,
                    "reason": "checkpoint identity is incompatible; agent.load was not attempted",
                    "reasons": identity_preflight.get("reasons", []),
                },
            )
            raise RuntimeError(
                "Taili resume blocked before checkpoint load; inspect "
                f"{run_dir / 'runtime_preflight.json'}"
            )
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
        parity, resume_edge = _resume_parity(
            args.checkpoint,
            inventory,
            snapshot,
            current_manifest=current_manifest,
            identity_preflight=identity_preflight,
        )
        snapshot["parity_check"] = parity
        snapshot["status"] = "proven" if _state_ready(snapshot) and parity else snapshot.get("status", "missing")
        update_manifest(
            runtime_path,
            optimization_state=snapshot,
            resume_edge=resume_edge,
        )
        write_manifest(run_dir / "optimization_state.json", snapshot)
        runtime_evidence_status = str(runtime_evidence.get("status") or "unknown")
        runtime_evidence_ok = runtime_evidence_status not in {"mismatch", "invalid"}
        preflight = {
            "schema_version": "rl-agent.runtime-preflight/v1",
            # ``observed`` is intentionally not reported as ``matched``:
            # package observations cannot prove an immutable remote image.
            "status": "pass" if runtime_execution.get("status") == "proven" and snapshot.get("status") == "proven" and runtime_evidence_ok else "blocked",
            "runtime_execution_status": runtime_execution.get("status"),
            "runtime_evidence_status": runtime_evidence_status,
            "runtime_identity_proven": runtime_evidence_status == "matched",
            "optimization_state_status": snapshot.get("status"),
            "resume_status": resume_edge.get("status"),
            "checkpoint": args.checkpoint,
            "reason": "" if runtime_execution.get("status") == "proven" and snapshot.get("status") == "proven" and runtime_evidence_ok else "runtime source, runtime identity, or optimization/resume evidence is incomplete",
        }
        _write_preflight(runtime_path, preflight)
        if preflight["status"] != "pass" and not args.allow_unverified_runtime:
            raise RuntimeError(
                "Taili runtime preflight blocked training; inspect "
                f"{run_dir / 'runtime_preflight.json'} or pass --allow-unverified-runtime explicitly"
            )
        if preflight["status"] != "pass":
            print("[TAILI_TRAIN] WARNING: bypassing unverified runtime preflight", flush=True)

        # P4.2集成：安装检查点管理钩子
        try:
            from .checkpoint_hook import install_checkpoint_hook
            checkpoint_dir_path = str(Path(os.environ.get("TAILI_CHECKPOINT_DIR", str(run_dir / "checkpoints"))))
            install_checkpoint_hook(runner.agent, env.unwrapped, checkpoint_dir_path)
        except Exception as e:
            print(f"[TAILI_TRAIN] checkpoint hook install failed: {e}", flush=True)

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
        # P4.2集成：训练结束时执行最终清理和能力提升
        try:
            from .checkpoint_hook import finalize_checkpoint_management
            finalize_checkpoint_management(env.unwrapped, config=experiment_cfg)
        except Exception as e:
            print(f"[TAILI_TRAIN] checkpoint finalize failed: {e}", flush=True)

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
