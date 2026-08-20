"""Bounded frozen-policy rollout for curriculum-gate calibration.

This entry point deliberately does not call ``Runner.run`` and never enables
training.  The environment keeps its real command sampler and gate telemetry;
the loaded checkpoint supplies deterministic mean actions for a fixed number
of steps.  The resulting artifacts are evidence inputs, not a capability
verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import traceback

from isaaclab.app import AppLauncher

if __name__ == "__main__" and __package__:
    sys.modules.setdefault(f"{__package__}.calibrate_taili_gates", sys.modules[__name__])

from .runtime_manifest import (  # noqa: E402
    capture_runtime_execution,
    checkpoint_inventory,
    file_digest,
    initial_manifest,
    update_manifest,
    write_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a frozen Taili gate-calibration rollout.")
    parser.add_argument("--task", default="RobotLab-Isaac-Taili-AMP-Blind-Direct-v0")
    parser.add_argument("--agent-yaml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", type=int, default=-1, help="Freeze one phase; -1 preserves configured progression.")
    parser.add_argument("--dr-level", type=int, default=-1, help="Override initial DR level; -1 preserves config.")
    parser.add_argument(
        "--restore-curriculum",
        action="store_true",
        help="Explicitly restore the checkpoint run's exact curriculum_state.json before scenario overrides.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_agent_config(path: Path) -> dict:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"agent YAML root must be a mapping: {path}")
    return value


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _curriculum_state_path(checkpoint: Path) -> Path:
    return checkpoint.parent.parent / "curriculum_state.json"


def _verify_restored_curriculum(base, state: dict) -> dict:
    """Prove that the environment loaded the exact per-environment state."""
    expected_levels = state.get("terrain_levels")
    expected_types = state.get("terrain_types")
    terrain = getattr(base, "_terrain", None)
    actual_levels = getattr(terrain, "terrain_levels", None)
    actual_types = getattr(terrain, "terrain_types", None)
    if not isinstance(expected_levels, list) or not isinstance(expected_types, list):
        raise ValueError("curriculum state lacks terrain_levels or terrain_types")
    if actual_levels is None or actual_types is None:
        raise RuntimeError("runtime environment does not expose terrain curriculum tensors")
    actual_level_values = actual_levels.detach().cpu().tolist()
    actual_type_values = actual_types.detach().cpu().tolist()
    checks = {
        "environment_count": len(actual_level_values) == len(expected_levels),
        "terrain_levels_exact": actual_level_values == expected_levels,
        "terrain_types_exact": actual_type_values == expected_types,
        "phase_restored": int(getattr(base, "_phase", -1)) >= int(state.get("phase", 0)),
        "dr_level_restored": int(getattr(base, "_dr_level", -1)) >= int(state.get("dr_level", 0)),
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"exact curriculum restore verification failed: {failed}")
    return {
        "status": "proven",
        "checks": checks,
        "restored_phase": int(getattr(base, "_phase", 0)),
        "restored_dr_level": int(getattr(base, "_dr_level", 0)),
        "environment_count": len(actual_level_values),
    }


def _apply_scenario_overrides(base, *, phase: int, dr_level: int) -> dict:
    """Apply fixed evaluation coordinates after any curriculum restoration."""
    if phase >= 0:
        base._phase = int(phase)
        base._max_training_phase = int(phase)
        base._phase_count = 0
    if dr_level >= 0:
        base._dr_level = int(dr_level)
        base._dr_gate_count = 0
        push_s = float(getattr(base.cfg, f"dr_push_interval_s_{dr_level}", 0.0))
        step_dt = float(base.cfg.dt * base.cfg.decimation)
        base._push_steps = max(1, int(push_s / step_dt)) if push_s > 0 else int(1e9)
    return {
        "phase": int(getattr(base, "_phase", 0)),
        "max_training_phase": int(getattr(base, "_max_training_phase", 0)),
        "dr_level": int(getattr(base, "_dr_level", 0)),
    }


def _gate_snapshot(base) -> dict:
    """Read the exact post-gate-evaluation state without changing it."""
    import torch

    phase = int(getattr(base, "_phase", 0))
    progress = {
        "fwd": _float(getattr(base, "_fwd_prog", 0.0)),
        "back": _float(getattr(base, "_back_prog", 0.0)),
        "lat": _float(getattr(base, "_lat_prog", 0.0)),
        "yaw": _float(getattr(base, "_yaw_prog", 0.0)),
    }
    try:
        from .taili_blind_config import active_direction_progress

        progress_gate, active_dirs = active_direction_progress(progress, base.cfg, phase)
    except Exception:
        active_dirs = tuple(progress)
        progress_gate = min(progress.values())
    terrain_stats = {}
    try:
        terrain_stats = base._terrain_level_stats()
    except Exception:
        terrain_stats = {}
    air = 0.0
    try:
        air = float(base._contact_sensor.data.last_air_time[:, base._feet_contact_ids].mean())
    except Exception:
        pass
    fall_rate = 0.0
    try:
        base._ensure_gate_mask()
        support_height = base._support_height_under_base()
        height = (base.robot.data.root_pos_w[:, 2] - support_height)[base._gate_mask]
        below = height < base.cfg.termination_height
        tilted = base.robot.data.projected_gravity_b[base._gate_mask, 2] >= -0.7
        fall_rate = float((below & tilted).float().mean())
    except Exception:
        try:
            below = base.robot.data.root_pos_w[:, 2] < base.cfg.termination_height
            tilted = base.robot.data.projected_gravity_b[:, 2] >= -0.7
            fall_rate = float((below & tilted).float().mean())
        except Exception:
            pass

    values = {
        "progress": progress_gate,
        "slip": _float(getattr(base, "_slip_ema", 0.0)),
        "diagonal": _float(getattr(base, "_diag_contact", 0.0)),
        "duty_target": _float(getattr(base, "_duty_target_score", 0.0)),
        "duty_symmetry": _float(getattr(base, "_duty_symmetry_score", 0.0)),
        "duty_valid": _float(getattr(base, "_duty_cycle_valid_frac", 0.0)),
        "period": _float(getattr(base, "_gait_period_score", 0.0)),
        "yaw_gait": _float(getattr(base, "_yaw_gait_gate", 0.0)),
        "air": air,
        "execution": _float(getattr(base, "_execution_gate", 0.0)),
        "terminal_rate": _float(getattr(base, "_direction_terminal_rate", 0.0)),
        "terrain_level": _float(terrain_stats.get("terrain_real_mean", terrain_stats.get("terrain_mean", 0.0))),
        "terrain_discrete": _float(terrain_stats.get("terrain_discrete_mean", 0.0)),
        "terrain_boxes": _float(terrain_stats.get("terrain_boxes_mean", 0.0)),
        "terrain_stairs_down": _float(terrain_stats.get("terrain_stairs_mean", 0.0)),
        "terrain_stairs_up": _float(terrain_stats.get("terrain_stairs_up_mean", 0.0)),
        "terrain_boxes_success": _float(getattr(base, "_terrain_boxes_success_ema", 0.0)),
        "terrain_stairs_down_success": _float(getattr(base, "_terrain_stairs_down_success_ema", 0.0)),
        "terrain_stairs_up_success": _float(getattr(base, "_terrain_stairs_up_success_ema", 0.0)),
        "terrain_boxes_collapse": _float(getattr(base, "_terrain_boxes_collapse_ema", 1.0)),
        "terrain_stairs_down_collapse": _float(getattr(base, "_terrain_stairs_down_collapse_ema", 1.0)),
        "terrain_stairs_up_collapse": _float(getattr(base, "_terrain_stairs_up_collapse_ema", 1.0)),
    }
    tilt_deg = 0.0
    try:
        base._ensure_gate_mask()
        gravity = base.robot.data.projected_gravity_b[base._gate_mask]
        if gravity.shape[0] > 0:
            tilt_deg = float(torch.rad2deg(torch.acos((-gravity[:, 2]).clamp(-1.0, 1.0))).mean())
    except Exception:
        pass
    runtime_values = getattr(base, "_phase_gate_values", {})
    if isinstance(runtime_values, dict):
        values.update({str(key): _float(value) for key, value in runtime_values.items()})
    flat = getattr(base, "_flat_quality_metrics", {})
    if isinstance(flat, dict):
        values.update({
            "flat_tilt_p95": _float(flat.get("tilt_p95", 0.0)),
            "flat_wxy": _float(flat.get("wxy_mean", 0.0)),
            "flat_height_error_p95": _float(flat.get("height_error_p95", 0.0)),
            "flat_touchdown_vz_p95": _float(flat.get("touchdown_vz_p95", 0.0)),
            "flat_slip_high": _float(flat.get("slip_high", 0.0)),
            "flat_trajectory_p95": _float(flat.get("trajectory_worst_p95", 0.0)),
            "flat_false_terrain_response": _float(flat.get("false_terrain_response", 0.0)),
        })
    curriculum = {
        "phase": f"phi{phase}",
        "dr_level": int(getattr(base, "_dr_level", 0)),
        "active_dirs": ",".join(str(item) for item in active_dirs),
        "phase_gate_eval_step": int(getattr(base, "_log_step", 0)),
        "progress_gate": progress_gate,
        "fall_gate": fall_rate,
        "execution_gate": _float(getattr(base, "_execution_gate", 0.0)),
        "dr_gate_terrain_level": values["terrain_level"],
        "terrain_health_slip_high_value": _float(
            getattr(base, "_terrain_slip_high_fraction", getattr(base, "_slip_high_fraction", 0.0))
        ),
        "dr_gate_gait_value": _float(
            getattr(base, "_gait_match_best_lag", getattr(base, "_best_lag_gait_match", 0.0))
        ),
        "dr_gate_duty_value": _float(getattr(base, "_duty_balance", 0.0)),
        "dr_gate_slip_value": _float(getattr(base, "_slip_ema", 0.0)),
        "dr_gate_tilt_deg_value": tilt_deg,
    }
    for direction, value in progress.items():
        curriculum[f"progress_{direction}"] = value
    for name, value in values.items():
        curriculum[f"phase_gate_{name}_value"] = value
    return {
        "step": int(getattr(base, "_log_step", 0)),
        "curriculum": curriculum,
    }


def _install_gate_trace(base, path: Path, *, phase: int = -1, dr_level: int = -1):
    """Wrap the existing diagnostic tick and persist only post-evaluation facts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", buffering=1)
    original = base._log_training_diag

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        # 诊断回合结束时运行时可能推进课程；在标记样本和下一步回放前，
        # 把请求的阶段和随机化等级恢复到显式值。
        if phase >= 0 or dr_level >= 0:
            _apply_scenario_overrides(base, phase=phase, dr_level=dr_level)
        handle.write(json.dumps(_gate_snapshot(base), ensure_ascii=False, sort_keys=True) + "\n")
        return result

    base._log_training_diag = wrapped
    return handle


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    agent_yaml = Path(args.agent_yaml).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if not agent_yaml.is_file():
        raise FileNotFoundError(f"agent YAML does not exist: {agent_yaml}")
    if args.steps <= 0 or args.num_envs <= 0:
        raise ValueError("--steps and --num_envs must be positive")

    curriculum_source = _curriculum_state_path(checkpoint)
    if args.restore_curriculum:
        if not curriculum_source.is_file():
            raise FileNotFoundError(f"exact curriculum state does not exist: {curriculum_source}")
        os.environ["TAILI_RESUME_CHECKPOINT"] = str(checkpoint)
    else:
        # 回放不能因为策略来自 resume 就隐式继承 sidecar；是否恢复必须显式指定。
        os.environ.pop("TAILI_RESUME_CHECKPOINT", None)
    if args.phase >= 0:
        os.environ["TAILI_INIT_PHASE"] = str(args.phase)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    run_dir = Path(os.environ.get("TAILI_RUN_DIR", ".")).resolve()
    runtime_path = Path(os.environ.get("TAILI_RUNTIME_MANIFEST", run_dir / "runtime_manifest.json"))
    effective_config = Path(os.environ.get("TAILI_EFFECTIVE_CONFIG", run_dir / "effective_config.yaml")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    curriculum_restore: dict = {"requested": bool(args.restore_curriculum), "status": "not_requested"}
    curriculum_state: dict = {}
    if args.restore_curriculum:
        curriculum_state = json.loads(curriculum_source.read_text(encoding="utf-8"))
        if not isinstance(curriculum_state, dict):
            raise ValueError(f"curriculum state root must be a mapping: {curriculum_source}")
        curriculum_copy = run_dir / "restored_curriculum_state.json"
        shutil.copy2(curriculum_source, curriculum_copy)
        source_ref = file_digest(curriculum_source)
        artifact_ref = file_digest(curriculum_copy)
        if source_ref.get("sha256") != artifact_ref.get("sha256"):
            raise RuntimeError("curriculum evidence copy digest differs from restore source")
        curriculum_restore = {
            "requested": True,
            "status": "captured",
            "source": source_ref,
            "artifact_ref": curriculum_copy.name,
            "artifact": artifact_ref,
        }
    if not runtime_path.is_file():
        write_manifest(runtime_path, initial_manifest(
            run_id=os.environ.get("TAILI_RUN_ID", run_dir.name),
            run_dir=run_dir,
            task=args.task,
            payload_root=Path(__file__).resolve().parent.parent,
            source_config=os.environ.get("TAILI_CONFIG_PATH", run_dir / "taili_blind_config.yaml"),
            effective_config=effective_config,
            agent_config=agent_yaml,
            resume_checkpoint=str(checkpoint),
            seed=args.seed,
        ))

    import gymnasium as gym
    import torch
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from skrl.utils.runner.torch import Runner

    import taili_blind_runtime  # noqa: F401

    env = None
    failure: BaseException | None = None
    runtime_execution: dict = {}
    inventory: dict = {}
    gate_trace_handle = None
    try:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        env_cfg.seed = args.seed
        if args.phase >= 0:
            env_cfg.init_phase = args.phase
            env_cfg.max_training_phase = args.phase
        if args.dr_level >= 0:
            env_cfg.dr_start_level = args.dr_level

        experiment_cfg = _read_agent_config(agent_yaml)
        experiment_cfg["seed"] = args.seed
        experiment_cfg.setdefault("trainer", {})["close_environment_at_exit"] = False
        agent_cfg = experiment_cfg.setdefault("agent", {})
        if isinstance(agent_cfg, dict):
            exp = agent_cfg.setdefault("experiment", {})
            if isinstance(exp, dict):
                exp["write_interval"] = 0
                exp["checkpoint_interval"] = 0

        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        base = env.unwrapped
        if args.restore_curriculum:
            curriculum_restore.update(_verify_restored_curriculum(base, curriculum_state))
        scenario_coordinates = _apply_scenario_overrides(
            base,
            phase=args.phase,
            dr_level=args.dr_level,
        )
        env = SkrlVecEnvWrapper(env, ml_framework="torch")
        runner = Runner(env, experiment_cfg)
        inventory = checkpoint_inventory(checkpoint, torch_module=torch)
        runner.agent.load(str(checkpoint))
        runner.agent.set_running_mode("eval")
        gate_trace_path = run_dir / "gate_runtime.telemetry.jsonl"
        gate_trace_handle = _install_gate_trace(
            env.unwrapped,
            gate_trace_path,
            phase=args.phase,
            dr_level=args.dr_level,
        )

        expected_modules = (
            "taili_blind_runtime",
            "taili_blind_runtime.calibrate_taili_gates",
            "taili_blind_runtime.blind_tp_env",
            "taili_blind_runtime.taili_amp_env",
            "taili_blind_runtime.telemetry_emit",
            "taili_blind_runtime.telemetry_payloads",
            "taili_blind_runtime.taili_core.taili_reward",
        )
        runtime_execution = capture_runtime_execution(
            Path(__file__).resolve().parent.parent,
            expected_modules,
            source_paths=(
                os.environ.get("TAILI_CONFIG_PATH", ""),
                effective_config,
                agent_yaml,
            ),
        )
        update_manifest(runtime_path, runtime_execution=runtime_execution)
        if runtime_execution.get("status") != "proven":
            raise RuntimeError("gate-calibration runtime source proof failed")

        observation, _ = env.reset()
        with torch.inference_mode():
            for step in range(args.steps):
                outputs = runner.agent.act(observation, timestep=step, timesteps=args.steps)
                extra = outputs[-1] if isinstance(outputs, (tuple, list)) and isinstance(outputs[-1], dict) else {}
                actions = extra.get("mean_actions", outputs[0])
                observation, _, _, _, _ = env.step(actions)

        checkpoint_hash = str(inventory.get("sha256") or "")
        config_hash = _sha256(effective_config)
        scenario = {
            "schema_version": "rl-agent.gate-calibration-scenario/v1",
            "training_enabled": False,
            "policy_mode": "mean_action",
            "command_source": "environment_sampler",
            "task": args.task,
            "num_envs": int(args.num_envs),
            "steps": int(args.steps),
            "seed": int(args.seed),
            "phase_override": int(args.phase),
            "dr_level_override": int(args.dr_level),
            "resolved_curriculum": scenario_coordinates,
            "curriculum_restore": curriculum_restore,
            "checkpoint_sha256": checkpoint_hash,
            "effective_config_sha256": config_hash,
            "telemetry_ref": os.environ.get("TAILI_TELEMETRY_JSONL", str(run_dir / "train.telemetry.jsonl")),
            "gate_trace_ref": str(gate_trace_path),
        }
        reference = {
            "schema_version": "rl-agent.gate-calibration-rollout/v1",
            "status": "complete",
            "training_enabled": False,
            "policy_mode": "mean_action",
            "task": args.task,
            "runtime_execution": runtime_execution,
            "checkpoint": inventory,
            "effective_config": {
                "path": str(effective_config),
                "sha256": config_hash,
            },
            "curriculum_restore": curriculum_restore,
            "scenario_contract_ref": str(run_dir / "gate_scenario_contract.json"),
        }
        write_manifest(run_dir / "gate_scenario_contract.json", scenario)
        write_manifest(run_dir / "reference_policy.json", reference)
        update_manifest(
            runtime_path,
            status="complete",
            evaluation={
                "kind": "gate_calibration",
                "training_enabled": False,
                "policy_mode": "mean_action",
                "checkpoint": inventory,
                "scenario_contract": str(run_dir / "gate_scenario_contract.json"),
            },
        )
        return 0
    except BaseException as exc:
        failure = exc
        update_manifest(runtime_path, status="failed", errors=[{
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }])
        raise
    finally:
        if env is not None:
            env.close()
        if gate_trace_handle is not None:
            gate_trace_handle.close()
        simulation_app.close()
        if failure is not None:
            print(f"[TAILI_GATE_CAL] failed: {type(failure).__name__}: {failure}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
