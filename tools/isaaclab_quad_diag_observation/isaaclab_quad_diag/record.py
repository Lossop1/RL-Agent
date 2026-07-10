from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from types import MethodType
from typing import Any

import numpy as np

from .util import derive_mode_from_cmd, load_yaml


TERRAIN_ALIASES = {
    "plane": "flat",
    "flat": "flat",
    "slope_up": "slope",
    "slope": "slope",
    "slope_down": "slope_inv",
    "slope_inv": "slope_inv",
    "rough": "rough",
    "boxes": "boxes",
    "stairs": "stairs",
    "stairs_up": "stairs",
    "stairs_down": "stairs",
}


def normalize_terrain_cases(value: Any) -> list[dict[str, Any]]:
    if not value:
        return [{"type": "flat", "level": 0}]
    if isinstance(value, dict):
        value = [value]
    out = []
    for item in value:
        out.append({"type": item} if isinstance(item, str) else dict(item))
    return out


def normalize_dr_cases(value: Any) -> list[dict[str, Any]]:
    if not value:
        return [{"level": 0}]
    if isinstance(value, dict):
        if value.get("enabled") in [False, "false", "disabled"]:
            return [{"level": 0}]
        if "cases" in value:
            value = value["cases"]
        elif "level" not in value:
            return [{"level": 0}]
        else:
            value = [value]
    return [{"level": int(v)} if isinstance(v, (int, float)) else dict(v) for v in value]


def normalize_pushes(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    if isinstance(value, dict):
        if value.get("enabled") in [False, "false", "disabled"]:
            return []
        value = value.get("events", value.get("cases", [value]))
    return [dict(v) for v in value]


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - requires Isaac Lab runtime
    p = argparse.ArgumentParser(description="Record Isaac Lab quadruped rollout data in observation-only ILQD schema.")
    p.add_argument("--task", required=True)
    p.add_argument("--robot-spec", required=True)
    p.add_argument("--policy-backend", default="skrl", choices=["skrl", "none"])
    p.add_argument("--checkpoint", default="")
    p.add_argument("--suite", required=True)
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--record-capture", default="post_step_terminal_safe", choices=["post_step_terminal_safe"])
    p.add_argument("--case-index", type=int, default=None, help=argparse.SUPPRESS)
    args, unknown = p.parse_known_args(argv)

    suite = load_yaml(args.suite) or {}
    case_count = len(normalize_terrain_cases(suite.get("terrains"))) * len(
        normalize_dr_cases(suite.get("dr_cases", suite.get("dr")))
    )
    if args.case_index is None and case_count > 1:
        run_isolated_cases(args, unknown, suite, case_count)
        return

    try:
        from isaaclab.app import AppLauncher
    except Exception as exc:
        raise RuntimeError("ilqd-record must be run inside an Isaac Lab Python environment") from exc

    launch_args = argparse.Namespace(**vars(args))
    launch_args.headless = bool(args.headless)
    launch_args.enable_cameras = False
    app = AppLauncher(launch_args).app
    try:
        run_isaaclab_record(args)
    finally:
        app.close()


def run_isolated_cases(args, unknown: list[str], suite: dict[str, Any], case_count: int) -> None:
    """Run each terrain/DR case in a fresh Isaac process and merge the observations."""
    out_dir = Path(args.out)
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    child_dirs = [cases_dir / f"case_{case_index:03d}" for case_index in range(case_count)]

    for case_index, child_out in enumerate(child_dirs):
        cmd = [
            sys.executable,
            "-c",
            "from isaaclab_quad_diag.record import main; main()",
            "--task",
            args.task,
            "--robot-spec",
            args.robot_spec,
            "--policy-backend",
            args.policy_backend,
            "--suite",
            args.suite,
            "--out",
            str(child_out),
            "--device",
            args.device,
            "--record-capture",
            args.record_capture,
            "--case-index",
            str(case_index),
        ]
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])
        if args.num_envs is not None:
            cmd.extend(["--num-envs", str(args.num_envs)])
        if args.headless:
            cmd.append("--headless")
        cmd.extend(unknown)
        print(f"[ILQD] isolated case {case_index + 1}/{case_count} starting", flush=True)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            write_merged_case_outputs(out_dir, child_dirs, args, suite, status="failed")
            raise RuntimeError(f"isolated diagnostic case {case_index} exited with code {result.returncode}")
        print(f"[ILQD] isolated case {case_index + 1}/{case_count} complete", flush=True)
        write_merged_case_outputs(out_dir, child_dirs, args, suite, status="running")

    write_merged_case_outputs(out_dir, child_dirs, args, suite, status="complete")
    print(f"record written to: {out_dir / 'record.csv'}")


def write_merged_case_outputs(
    out_dir: Path,
    child_dirs: list[Path],
    args,
    suite: dict[str, Any],
    status: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    executed_cases: list[dict[str, Any]] = []
    notes: set[str] = set()
    rows_written = 0
    completed_cases = 0
    output_csv = out_dir / "record.csv"

    with output_csv.open("w", newline="", encoding="utf-8") as dst:
        writer = None
        for child_dir in child_dirs:
            csv_path = child_dir / "record.csv"
            meta_path = child_dir / "record_meta.json"
            if not csv_path.exists() or not meta_path.exists():
                continue
            child_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if child_meta.get("status") != "complete":
                continue
            completed_cases += 1
            executed_cases.extend(child_meta.get("executed_runtime_config", {}).get("cases", []))
            notes.update(child_meta.get("recording_notes", []))
            with csv_path.open("r", newline="", encoding="utf-8") as src:
                reader = csv.DictReader(src)
                if writer is None:
                    writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
                    rows_written += 1

    meta = {
        "schema_version": "ilqd_observation_record_v0.5.1",
        "status": status,
        "task": args.task,
        "checkpoint": args.checkpoint,
        "suite_path": args.suite,
        "requested_suite_config": suite,
        "executed_runtime_config": {
            "cases": executed_cases,
            "record_capture": args.record_capture,
            "case_isolation": "one_isaac_process_per_terrain_dr_case",
            "completed_cases": completed_cases,
            "requested_cases": len(child_dirs),
        },
        "num_envs_requested": args.num_envs or int(suite.get("num_envs", 1)),
        "rows_written": rows_written,
        "recording_notes": sorted(notes),
        "semantics": "Observation-only. No success/failure or pass/fail labels are emitted by the recorder.",
    }
    (out_dir / "record_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "record_progress.json").write_text(
        json.dumps({
            "completed_cases": completed_cases,
            "requested_cases": len(child_dirs),
            "rows_written": rows_written,
            "status": status,
        }, indent=2),
        encoding="utf-8",
    )


def run_isaaclab_record(args) -> None:  # pragma: no cover - requires Isaac Lab runtime
    import torch
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from skrl.utils.runner.torch import Runner
    import robot_lab.tasks  # noqa: F401

    from .spec import load_robot_spec

    suite = load_yaml(args.suite) or {}
    spec = load_robot_spec(args.robot_spec)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_envs = args.num_envs or int(suite.get("num_envs", 1))
    reset_policy = str(suite.get("reset_policy", "per_segment"))
    reset_initialization = str(suite.get("reset_initialization", "command_start"))

    # This recorder is intentionally conservative: terrain/DR/push are requested
    # through suite metadata and best-effort env-cfg mutation. Every request and
    # every observed runtime value are written to metadata; missing support is not
    # hidden behind fake values.
    terrains = normalize_terrain_cases(suite.get("terrains"))
    commands = suite.get("commands", []) or [{"mode": "stand", "vx": 0, "vy": 0, "wz": 0, "duration": 2.0}]
    pushes = normalize_pushes(suite.get("pushes"))
    dr_cases = normalize_dr_cases(suite.get("dr_cases", suite.get("dr")))

    recording_notes: list[str] = []
    executed_cases: list[dict[str, Any]] = []
    cols = build_columns(12, ["FL", "FR", "RL", "RR"])
    out_csv = out_dir / "record.csv"
    csv_handle = out_csv.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_handle, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    rows_written = 0
    global_step = 0

    def write_meta(status: str) -> None:
        meta = {
            "schema_version": "ilqd_observation_record_v0.5.1",
            "status": status,
            "task": args.task,
            "checkpoint": args.checkpoint,
            "suite_path": args.suite,
            "requested_suite_config": suite,
            "executed_runtime_config": {
                "cases": executed_cases,
                "record_capture": args.record_capture,
                "reset_initialization": reset_initialization,
            },
            "num_envs_requested": num_envs,
            "rows_written": rows_written,
            "recording_notes": sorted(set(recording_notes)),
            "semantics": "Observation-only. No success/failure or pass/fail labels are emitted by the recorder.",
        }
        (out_dir / "record_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    write_meta("running")

    case_id = 0
    for terrain_case in terrains:
        for dr_case in dr_cases:
            active_case_id = case_id
            case_id += 1
            if args.case_index is not None and active_case_id != args.case_index:
                continue
            case_id = active_case_id
            env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=num_envs)
            if reset_initialization == "command_start":
                env_cfg.reset_strategy = "start"
            terrain_apply = apply_terrain_case_best_effort(env_cfg, terrain_case, recording_notes)
            dr_apply = apply_dr_case_best_effort(env_cfg, dr_case, recording_notes)
            if not terrain_apply["applied"] or not dr_apply["applied"]:
                reason = terrain_apply.get("reason") or dr_apply.get("reason")
                executed_cases.append({
                    "case_id": case_id,
                    "terrain_case_requested": terrain_case,
                    "dr_case_requested": dr_case,
                    "status": "skipped",
                    "reason": reason,
                })
                case_id = active_case_id + 1
                continue
            env = gym.make(args.task, cfg=env_cfg, render_mode=None)
            env = SkrlVecEnvWrapper(env, ml_framework="torch")
            base = env.unwrapped
            if hasattr(base, "use_external_commands"):
                base.use_external_commands = True
            command_reset = install_command_conditioned_reset(base) if reset_initialization == "command_start" else None

            runner = None
            if args.policy_backend == "skrl":
                print("[ILQD] loading SKRL experiment config", flush=True)
                experiment_cfg = load_cfg_from_registry(args.task, "skrl_amp_cfg_entry_point")
                experiment_cfg["trainer"]["close_environment_at_exit"] = False
                experiment_cfg["agent"]["experiment"]["write_interval"] = 0
                experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
                print("[ILQD] constructing SKRL Runner", flush=True)
                runner = Runner(env, experiment_cfg)
                print("[ILQD] SKRL Runner ready", flush=True)
                if args.checkpoint:
                    print(f"[ILQD] loading checkpoint: {args.checkpoint}", flush=True)
                    load_evaluation_checkpoint(runner.agent, args.checkpoint)
                    print("[ILQD] checkpoint loaded", flush=True)
                runner.agent.set_running_mode("eval")
                print("[ILQD] policy in eval mode", flush=True)

            def act_mean(obs):
                if runner is None:
                    return torch.zeros((base.num_envs, base.action_space.shape[-1]), device=base.device)
                out = runner.agent.act(obs, timestep=0, timesteps=0)
                return out[-1].get("mean_actions", out[0])

            robot = base.robot
            body_names = list(robot.data.body_names)
            joint_names = list(robot.data.joint_names)
            joint_order = spec.joint_order or joint_names[:12]
            foot_names = [spec.foot_body_names.get(leg, f"{leg}_foot") for leg in spec.leg_order]
            joint_idx = [joint_names.index(j) for j in joint_order]
            foot_body_idx = [body_names.index(f) for f in foot_names]
            contact_sensor = getattr(base, "_contact_sensor", None)
            foot_contact_idx = None
            if contact_sensor is not None:
                try:
                    foot_contact_idx, _ = contact_sensor.find_bodies(foot_names)
                except Exception:
                    foot_contact_idx = None
                    recording_notes.append("contact_sensor present but foot bodies were not found")
            else:
                recording_notes.append("contact_sensor unavailable; contact/touchdown/slip metrics may be unavailable")

            dt = float(getattr(base, "step_dt", getattr(base, "physics_dt", 0.005) * getattr(base.cfg, "decimation", 1)))
            first_cmd = commands[0]
            first_target = torch.tensor(
                [float(first_cmd.get("vx", 0.0)), float(first_cmd.get("vy", 0.0)), float(first_cmd.get("wz", 0.0))],
                device=base.device,
                dtype=torch.float32,
            )
            if command_reset is not None:
                command_reset["set_target"](first_target)
            print(f"[ILQD] resetting case {case_id} pass=1", flush=True)
            obs, _ = env.reset()
            print(f"[ILQD] resetting case {case_id} pass=1 complete", flush=True)
            force_terrain_level_best_effort(base, terrain_case, recording_notes)
            print(f"[ILQD] resetting case {case_id} pass=2", flush=True)
            obs, _ = env.reset()
            print(f"[ILQD] resetting case {case_id} pass=2 complete", flush=True)
            print(f"[ILQD] rollout case {case_id} started", flush=True)
            refresh_dr_snapshot_best_effort(base, recording_notes)
            episode_id = np.zeros(base.num_envs, dtype=int)
            prev_contact = np.zeros((base.num_envs, len(spec.leg_order)), dtype=bool)
            contact_history_valid = np.zeros(base.num_envs, dtype=bool)
            t = 0.0
            seg_id = 0
            terminal_capture = install_terminal_capture_hook(base)
            applied_pushes: set[tuple[int, int]] = set()
            last_rows_by_env: dict[int, dict[str, Any]] = {}

            executed_cases.append({
                "case_id": case_id,
                "terrain_case_requested": terrain_case,
                "terrain_case_applied": terrain_apply,
                "dr_case_requested": dr_case,
                "dr_case_applied": dr_apply,
                "num_envs": int(num_envs),
                "reset_policy": reset_policy,
                "status": "executed",
            })
            for cmd in commands:
                print(f"[ILQD] segment {seg_id} mode={cmd.get('mode', 'unknown')} starting", flush=True)
                target = torch.tensor(
                    [float(cmd.get("vx", 0.0)), float(cmd.get("vy", 0.0)), float(cmd.get("wz", 0.0))],
                    device=base.device,
                    dtype=torch.float32,
                )
                if command_reset is not None:
                    command_reset["set_target"](target)
                if reset_policy == "per_segment" and seg_id > 0:
                    print(f"[ILQD] segment {seg_id} reset pass=1", flush=True)
                    obs, _ = env.reset()
                    print(f"[ILQD] segment {seg_id} reset pass=1 complete", flush=True)
                    force_terrain_level_best_effort(base, terrain_case, recording_notes)
                    print(f"[ILQD] segment {seg_id} reset pass=2", flush=True)
                    obs, _ = env.reset()
                    print(f"[ILQD] segment {seg_id} reset pass=2 complete", flush=True)
                    refresh_dr_snapshot_best_effort(base, recording_notes)
                    episode_id += 1
                    prev_contact[:] = False
                    contact_history_valid[:] = False
                    last_rows_by_env.clear()
                mode = str(cmd.get("mode", "unknown"))
                duration = float(cmd.get("duration", 1.0))
                steps = max(1, int(round(duration / dt)))
                for k in range(steps):
                    set_external_command(base, target)
                    with torch.inference_mode():
                        action_mean = act_mean(obs)
                    action_applied = read_action_applied_best_effort(base, action_mean)
                    push_info = maybe_apply_push_best_effort(
                        base, pushes, k * dt, seg_id, recording_notes, applied_pushes
                    )
                    common = dict(
                        base=base, robot_data=robot.data, contact_sensor=contact_sensor, action_mean=action_mean,
                        action_applied=action_applied, episode_id=episode_id, step=global_step,
                        t=t + dt, dt=dt, seg_id=seg_id, mode=mode, target=target,
                        joint_idx=joint_idx, foot_body_idx=foot_body_idx, foot_contact_idx=foot_contact_idx,
                        legs=spec.leg_order, foot_names=foot_names, spec=spec,
                        terrain_case=terrain_case, dr_case=dr_case, prev_contact=prev_contact,
                        contact_history_valid=contact_history_valid, case_id=case_id,
                        time_since_command_switch=(k + 1) * dt, push_info=push_info,
                        recording_notes=recording_notes,
                    )
                    arm_terminal_capture(
                        terminal_capture,
                        lambda env_ids, common=common: make_rows_from_current_state(
                            **common,
                            env_ids=env_ids,
                            command_applied=read_env_command_best_effort(base, target),
                            capture_stage="terminal_pre_reset",
                        ),
                    )
                    obs, _, terminated, truncated, info = env.step(action_mean)
                    terminated_np = tensor_bool_np(terminated)
                    truncated_np = tensor_bool_np(truncated)
                    done = terminated_np | truncated_np
                    terminal_rows = {int(r["env_id"]): r for r in collect_terminal_rows(terminal_capture)}
                    live_ids = [env_id for env_id in range(base.num_envs) if not done[env_id]]
                    live_rows = make_rows_from_current_state(
                        **common,
                        env_ids=live_ids,
                        command_applied=read_env_command_best_effort(base, target),
                        capture_stage="post_step",
                    )
                    selected_rows = {int(r["env_id"]): r for r in live_rows}
                    missing_terminal_ids = [
                        env_id for env_id in range(base.num_envs)
                        if done[env_id] and env_id not in terminal_rows and env_id not in last_rows_by_env
                    ]
                    reset_rows = {
                        int(r["env_id"]): r
                        for r in make_rows_from_current_state(
                            **common,
                            env_ids=missing_terminal_ids,
                            command_applied=read_env_command_best_effort(base, target),
                            capture_stage="post_reset_fallback",
                        )
                    } if missing_terminal_ids else {}
                    for env_id in range(base.num_envs):
                        if done[env_id]:
                            row = terminal_rows.get(env_id)
                            if row is None:
                                fallback = last_rows_by_env.get(env_id)
                                if fallback is None:
                                    fallback = reset_rows[env_id]
                                row = dict(fallback)
                                row["time"] = float(t + dt)
                                row["step"] = int(global_step)
                                row["capture_stage"] = "previous_state_fallback"
                            row["terminal_state_available"] = int(env_id in terminal_rows)
                            row["post_step_state_may_be_after_reset"] = int(env_id not in terminal_rows)
                        else:
                            row = selected_rows[env_id]
                            row["terminal_state_available"] = 1
                            row["post_step_state_may_be_after_reset"] = 0
                        row["transition_done_after_action"] = int(done[env_id])
                        row["terminated"] = int(terminated_np[env_id])
                        row["truncated"] = int(truncated_np[env_id])
                        row["done"] = int(done[env_id])
                        row["reset_observed"] = int(done[env_id])
                        writer.writerow(row)
                        rows_written += 1
                        last_rows_by_env[env_id] = dict(row)
                        if done[env_id]:
                            episode_id[env_id] += 1
                            prev_contact[env_id, :] = False
                            contact_history_valid[env_id] = False
                            last_rows_by_env.pop(env_id, None)
                        else:
                            prev_contact[env_id, :] = [
                                bool(row[f"foot_{leg}_contact"]) for leg in spec.leg_order
                            ]
                            contact_history_valid[env_id] = True
                    done_ids = np.flatnonzero(done)
                    if len(done_ids):
                        refresh_dr_snapshot_best_effort(base, recording_notes, done_ids)
                    t += dt
                    global_step += 1
                seg_id += 1
                csv_handle.flush()
                (out_dir / "record_progress.json").write_text(
                    json.dumps({
                        "case_id": case_id,
                        "completed_segments": seg_id,
                        "rows_written": rows_written,
                        "last_mode": mode,
                    }, indent=2),
                    encoding="utf-8",
                )
                write_meta("running")
            restore_terminal_capture_hook(base, terminal_capture)
            if command_reset is not None:
                command_reset["restore"]()
            env.close()
            case_id = active_case_id + 1

    csv_handle.close()
    write_meta("complete")
    print(f"record written to: {out_csv}")


def install_command_conditioned_reset(base) -> dict[str, Any]:  # pragma: no cover - requires Isaac Lab runtime
    """Make every diagnostic reset sample a reference pose for the active test command."""
    original = base._resample_commands
    state: dict[str, Any] = {"target": None}

    def diagnostic_resample(self, env_ids, snap=False):
        target = state["target"]
        if target is None:
            return original(env_ids, snap=snap)
        expanded = target.view(1, 3).expand(len(env_ids), -1)
        self._cmd_target[env_ids] = expanded
        if snap:
            self.commands[env_ids] = expanded

    base._resample_commands = MethodType(diagnostic_resample, base)

    def set_target(target) -> None:
        state["target"] = target.detach().clone()

    def restore() -> None:
        base._resample_commands = original

    return {"set_target": set_target, "restore": restore}


def build_columns(n_joints: int, legs: list[str]) -> list[str]:
    cols = [
        "run_id", "case_id", "env_id", "episode_id", "step", "time", "control_dt", "physics_dt", "decimation",
        "task_name", "robot_name", "nominal_stand_height",
        "terrain_type_requested", "terrain_type", "terrain_level", "terrain_height_source",
        "dr_level_requested", "dr_level",
        "capture_stage", "terminal_state_available", "post_step_state_may_be_after_reset", "transition_done_after_action",
        "terminated", "truncated", "done", "reset_observed",
        "cmd_target_vx", "cmd_target_vy", "cmd_target_wz", "cmd_target_mode",
        "cmd_vx", "cmd_vy", "cmd_wz", "cmd_mode", "cmd_segment_id", "time_since_command_switch",
        "push_event", "push_vector_x", "push_vector_y", "push_vector_z", "push_equivalent_delta_v",
        "base_pos_w_x", "base_pos_w_y", "base_pos_w_z", "base_terrain_height", "base_height_local",
        "base_quat_w", "base_quat_x", "base_quat_y", "base_quat_z",
        "base_lin_vel_b_x", "base_lin_vel_b_y", "base_lin_vel_b_z",
        "base_ang_vel_b_x", "base_ang_vel_b_y", "base_ang_vel_b_z",
        "projected_gravity_b_x", "projected_gravity_b_y", "projected_gravity_b_z",
        "dr_mass", "dr_friction", "dr_com_x", "dr_com_y", "dr_com_z",
        "dr_stiffness_scale", "dr_damping_scale", "dr_latency",
    ]
    for i in range(n_joints):
        cols += [f"joint_pos_{i}", f"joint_vel_{i}", f"joint_pos_des_{i}", f"joint_error_{i}", f"torque_applied_{i}", f"torque_limit_{i}", f"torque_utilization_{i}", f"action_mean_{i}", f"action_applied_{i}"]
    for leg in legs:
        cols += [
            f"foot_{leg}_pos_w_x", f"foot_{leg}_pos_w_y", f"foot_{leg}_pos_w_z",
            f"foot_{leg}_vel_w_x", f"foot_{leg}_vel_w_y", f"foot_{leg}_vel_w_z",
            f"foot_{leg}_terrain_height", f"foot_{leg}_clearance_local", f"foot_{leg}_contact",
            f"foot_{leg}_force_w_x", f"foot_{leg}_force_w_y", f"foot_{leg}_force_w_z", f"foot_{leg}_force_norm",
            f"foot_{leg}_normal_force", f"foot_{leg}_tangent_force",
            f"foot_{leg}_air_time", f"foot_{leg}_stance_time", f"foot_{leg}_touchdown", f"foot_{leg}_liftoff", f"foot_{leg}_touchdown_vz", f"foot_{leg}_stance_slip_xy",
        ]
    return cols


def tensor_bool_np(x):
    try:
        return x.detach().cpu().numpy().astype(bool)
    except Exception:
        return np.asarray(x).astype(bool)


def load_evaluation_checkpoint(agent, path: str) -> None:  # pragma: no cover - requires torch/skrl runtime
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"unsupported checkpoint payload: {type(checkpoint).__name__}")
    required = ["policy", "state_preprocessor"]
    for name in required:
        if name not in checkpoint:
            raise KeyError(f"checkpoint is missing evaluation module: {name}")
        module = agent.checkpoint_modules.get(name)
        if module is None or not hasattr(module, "load_state_dict"):
            raise KeyError(f"agent is missing evaluation module: {name}")
        print(f"[ILQD] loading module: {name}", flush=True)
        module.load_state_dict(checkpoint[name])
        if hasattr(module, "eval"):
            module.eval()
        print(f"[ILQD] module loaded: {name}", flush=True)


def set_external_command(base, target):
    if hasattr(base, "_cmd_target"):
        base._cmd_target[:] = target.unsqueeze(0).expand(base.num_envs, -1)
    elif hasattr(base, "commands"):
        base.commands[:] = target.unsqueeze(0).expand(base.num_envs, -1)
    else:
        raise AttributeError("Cannot set command on this env. Provide a custom recorder adapter.")


def read_env_command_best_effort(base, fallback_target):
    try:
        if hasattr(base, "commands"):
            return base.commands.detach().clone()
        if hasattr(base, "_commands"):
            return base._commands.detach().clone()
    except Exception:
        pass
    return fallback_target.unsqueeze(0).expand(base.num_envs, -1)


def read_action_applied_best_effort(base, fallback_action):
    """Read the action buffer that will drive the next control transition."""
    for name in ["_delayed_action", "actions"]:
        value = getattr(base, name, None)
        if value is not None:
            try:
                return value.detach().clone()
            except Exception:
                pass
    try:
        return fallback_action.detach().clone()
    except Exception:
        return np.asarray(fallback_action).copy()


def install_terminal_capture_hook(base):
    """Capture the true post-physics state immediately before DirectRLEnv resets it."""
    original = base._reset_idx
    state = {"original": original, "callback": None, "rows": []}

    def wrapped(env_ids):
        callback = state.get("callback")
        if callback is not None:
            state["rows"].extend(callback(env_ids))
        return original(env_ids)

    base._reset_idx = wrapped
    return state


def arm_terminal_capture(state, callback) -> None:
    state["rows"] = []
    state["callback"] = callback


def collect_terminal_rows(state) -> list[dict[str, Any]]:
    rows = list(state.get("rows", []))
    state["rows"] = []
    state["callback"] = None
    return rows


def restore_terminal_capture_hook(base, state) -> None:
    base._reset_idx = state["original"]


def read_effort_limits_best_effort(base):
    try:
        return base.robot.actuators["legs"].effort_limit
    except Exception:
        return getattr(base.robot.data, "joint_effort_limits", None)


def tensor_value_best_effort(value, env_id: int, index: int) -> float:
    if value is None:
        return np.nan
    try:
        if len(value.shape) == 1:
            return float(value[index])
        return float(value[env_id, index])
    except Exception:
        return np.nan


def contact_transition_flags(contact_prev: bool, contact_now: bool, history_valid: bool) -> tuple[int, int]:
    if not history_valid:
        return 0, 0
    return int((not contact_prev) and contact_now), int(contact_prev and (not contact_now))


def read_contacts_best_effort(base, contact_sensor, foot_contact_idx, num_envs, n_legs):
    out = np.zeros((num_envs, n_legs), dtype=bool)
    if contact_sensor is None or foot_contact_idx is None:
        return out
    try:
        nf = contact_sensor.data.net_forces_w[:, foot_contact_idx, :]
        return (nf.norm(dim=-1).detach().cpu().numpy() > 5.0)
    except Exception:
        return out


def make_rows_from_current_state(**kw) -> list[dict[str, Any]]:  # pragma: no cover
    import torch

    base = kw["base"]; d = kw["robot_data"]; contact_sensor = kw["contact_sensor"]
    env_ids = [int(x) for x in kw["env_ids"]]
    if not env_ids:
        return []
    ids = torch.as_tensor(env_ids, device=base.device, dtype=torch.long)
    joint_idx = list(kw["joint_idx"])
    foot_body_idx = list(kw["foot_body_idx"])
    foot_contact_idx = list(kw["foot_contact_idx"]) if kw["foot_contact_idx"] is not None else None

    def select(value, second_idx=None):
        if value is None:
            return None
        selected = value[ids]
        if second_idx is not None:
            selected = selected[:, second_idx]
        return selected

    tensors = {
        "pos": select(d.root_pos_w),
        "quat": select(d.root_quat_w),
        "lin": select(d.root_lin_vel_b),
        "ang": select(d.root_ang_vel_b),
        "gravity": select(d.projected_gravity_b),
        "joint_pos": select(d.joint_pos, joint_idx),
        "joint_vel": select(getattr(d, "joint_vel", None), joint_idx),
        "joint_des": select(getattr(d, "joint_pos_target", None), joint_idx),
        "torque": select(getattr(d, "applied_torque", getattr(d, "computed_torque", None)), joint_idx),
        "foot_pos": select(d.body_pos_w, foot_body_idx),
        "foot_vel": select(getattr(d, "body_lin_vel_w", None), foot_body_idx),
        "command": select(kw["command_applied"]),
        "action_mean": select(kw["action_mean"]),
        "action_applied": select(kw.get("action_applied")),
    }
    if contact_sensor is not None and foot_contact_idx is not None:
        try:
            tensors["force"] = select(contact_sensor.data.net_forces_w, foot_contact_idx)
        except Exception:
            tensors["force"] = None
        try:
            tensors["air"] = select(contact_sensor.data.current_air_time, foot_contact_idx)
        except Exception:
            tensors["air"] = None
    else:
        tensors["force"] = None
        tensors["air"] = None
    scanner = getattr(base, "_height_scanner", None)
    if scanner is not None:
        try:
            tensors["height_hits"] = select(scanner.data.ray_hits_w)
        except Exception:
            tensors["height_hits"] = None
    else:
        tensors["height_hits"] = None

    packed_parts = []
    layout = {}
    offset = 0
    for name, tensor in tensors.items():
        if tensor is None:
            continue
        shape = tuple(tensor.shape[1:])
        width = int(np.prod(shape))
        packed_parts.append(tensor.reshape(len(env_ids), width))
        layout[name] = (offset, offset + width, shape)
        offset += width
    packed = torch.cat(packed_parts, dim=1).detach().cpu().numpy()
    arrays = {name: None for name in tensors}
    for name, (start, end, shape) in layout.items():
        arrays[name] = packed[:, start:end].reshape((len(env_ids),) + shape)

    limits = read_effort_limits_best_effort(base)
    if limits is None:
        arrays["torque_limit"] = None
    else:
        limits_np = getattr(base, "_ilqd_effort_limits_np", None)
        if limits_np is None:
            limits_np = limits.detach().cpu().numpy() if hasattr(limits, "detach") else np.asarray(limits)
            base._ilqd_effort_limits_np = limits_np
        if limits_np.ndim == 1:
            arrays["torque_limit"] = np.broadcast_to(limits_np[joint_idx], (len(env_ids), len(joint_idx)))
        else:
            arrays["torque_limit"] = limits_np[env_ids][:, joint_idx]

    if arrays["force"] is not None:
        arrays["force_norm"] = np.linalg.norm(arrays["force"], axis=-1)
        arrays["contact"] = arrays["force_norm"] > 5.0
    else:
        arrays["force_norm"] = None
        arrays["contact"] = np.zeros((len(env_ids), len(kw["legs"])), dtype=bool)

    if arrays["height_hits"] is not None:
        height_cache = {
            env_id: arrays["height_hits"][index][np.isfinite(arrays["height_hits"][index]).all(axis=-1)]
            for index, env_id in enumerate(env_ids)
        }
    else:
        height_cache = build_height_query_cache(base, env_ids)
    terrain_identities = read_terrain_identities_best_effort(base, env_ids)
    target = kw["target"].detach().cpu().numpy() if hasattr(kw["target"], "detach") else np.asarray(kw["target"])
    rows = []
    for local_i, env_id in enumerate(env_ids):
        pos = arrays["pos"][local_i]
        quat = arrays["quat"][local_i]
        terrain_h, terrain_source = terrain_height_at_point_best_effort(
            base, env_id, pos[0], pos[1], height_cache
        )
        cmd_app = arrays["command"][local_i]
        actual_terrain_type, actual_terrain_level = terrain_identities[env_id]
        dr = read_dr_snapshot(base, env_id)
        push = kw.get("push_info", {}).get(env_id, {})
        cmd_values = [float(cmd_app[0]), float(cmd_app[1]), float(cmd_app[2])]
        row: dict[str, Any] = {
            "run_id": f"case_{int(kw['case_id'])}",
            "case_id": int(kw["case_id"]),
            "env_id": env_id, "episode_id": int(kw["episode_id"][env_id]), "step": int(kw["step"]), "time": float(kw["t"]),
            "control_dt": float(kw["dt"]), "physics_dt": getattr(base, "physics_dt", np.nan), "decimation": getattr(base.cfg, "decimation", np.nan),
            "task_name": getattr(base.cfg, "task_name", "unknown"), "robot_name": getattr(kw["spec"], "robot_name", "unknown"), "nominal_stand_height": kw["spec"].nominal_stand_height,
            "terrain_type_requested": str(kw["terrain_case"].get("type", "unknown")),
            "terrain_type": actual_terrain_type, "terrain_level": actual_terrain_level, "terrain_height_source": terrain_source,
            "dr_level_requested": str(kw["dr_case"].get("level", "unknown")),
            "dr_level": str(getattr(base, "_dr_level", getattr(base, "dr_level", "unknown"))),
            "capture_stage": kw["capture_stage"], "terminal_state_available": 0, "post_step_state_may_be_after_reset": 0, "transition_done_after_action": 0,
            "terminated": 0, "truncated": 0, "done": 0, "reset_observed": 0,
            "cmd_target_vx": float(target[0]), "cmd_target_vy": float(target[1]), "cmd_target_wz": float(target[2]), "cmd_target_mode": kw["mode"],
            "cmd_vx": cmd_values[0], "cmd_vy": cmd_values[1], "cmd_wz": cmd_values[2],
            "cmd_mode": derive_mode_from_cmd(*cmd_values), "cmd_segment_id": kw["seg_id"],
            "time_since_command_switch": float(kw["time_since_command_switch"]),
            "push_event": int(bool(push)), "push_vector_x": push.get("x", np.nan),
            "push_vector_y": push.get("y", np.nan), "push_vector_z": push.get("z", np.nan),
            "push_equivalent_delta_v": push.get("delta_v", np.nan),
            "base_pos_w_x": float(pos[0]), "base_pos_w_y": float(pos[1]), "base_pos_w_z": float(pos[2]), "base_terrain_height": terrain_h, "base_height_local": float(pos[2]) - terrain_h,
            "base_quat_w": float(quat[0]), "base_quat_x": float(quat[1]), "base_quat_y": float(quat[2]), "base_quat_z": float(quat[3]),
            "base_lin_vel_b_x": float(arrays["lin"][local_i, 0]), "base_lin_vel_b_y": float(arrays["lin"][local_i, 1]), "base_lin_vel_b_z": float(arrays["lin"][local_i, 2]),
            "base_ang_vel_b_x": float(arrays["ang"][local_i, 0]), "base_ang_vel_b_y": float(arrays["ang"][local_i, 1]), "base_ang_vel_b_z": float(arrays["ang"][local_i, 2]),
            "projected_gravity_b_x": float(arrays["gravity"][local_i, 0]), "projected_gravity_b_y": float(arrays["gravity"][local_i, 1]), "projected_gravity_b_z": float(arrays["gravity"][local_i, 2]),
            "dr_mass": dr.get("mass", np.nan), "dr_friction": dr.get("friction", np.nan),
            "dr_com_x": dr.get("com_x", np.nan), "dr_com_y": dr.get("com_y", np.nan),
            "dr_com_z": dr.get("com_z", np.nan),
            "dr_stiffness_scale": dr.get("stiffness_scale", np.nan),
            "dr_damping_scale": dr.get("damping_scale", np.nan),
            "dr_latency": dr.get("latency", np.nan),
        }
        # Joint state.
        q = arrays["joint_pos"][local_i]
        for k in range(len(joint_idx)):
            row[f"joint_pos_{k}"] = float(q[k])
            row[f"joint_vel_{k}"] = float(arrays["joint_vel"][local_i, k]) if arrays["joint_vel"] is not None else np.nan
            desired = float(arrays["joint_des"][local_i, k]) if arrays["joint_des"] is not None else np.nan
            row[f"joint_pos_des_{k}"] = desired
            row[f"joint_error_{k}"] = desired - float(q[k]) if np.isfinite(desired) else np.nan
            row[f"torque_applied_{k}"] = float(arrays["torque"][local_i, k]) if arrays["torque"] is not None else np.nan
            limit = float(arrays["torque_limit"][local_i, k]) if arrays["torque_limit"] is not None else np.nan
            row[f"torque_limit_{k}"] = limit
            row[f"torque_utilization_{k}"] = (
                abs(row[f"torque_applied_{k}"]) / abs(limit)
                if np.isfinite(limit) and abs(limit) > 1e-9
                else np.nan
            )
            row[f"action_mean_{k}"] = float(arrays["action_mean"][local_i, k]) if arrays["action_mean"] is not None else np.nan
            row[f"action_applied_{k}"] = (
                float(arrays["action_applied"][local_i, k]) if arrays["action_applied"] is not None else np.nan
            )

        for li, leg in enumerate(kw["legs"]):
            fx, fy, fz = [float(x) for x in arrays["foot_pos"][local_i, li]]
            th, thsrc = terrain_height_at_point_best_effort(base, env_id, fx, fy, height_cache)
            contact_now = bool(arrays["contact"][local_i, li])
            contact_prev = bool(kw["prev_contact"][env_id, li])
            transition_valid = bool(kw["contact_history_valid"][env_id])
            touchdown, liftoff = contact_transition_flags(contact_prev, contact_now, transition_valid)
            if arrays["foot_vel"] is not None:
                vx, vy, vz = [float(x) for x in arrays["foot_vel"][local_i, li]]
            else:
                vx = vy = vz = np.nan
            if arrays["force"] is not None:
                fwx, fwy, fwz = [float(x) for x in arrays["force"][local_i, li]]
                fnorm = float(arrays["force_norm"][local_i, li])
            else:
                fwx = fwy = fwz = fnorm = np.nan
            row.update({
                f"foot_{leg}_pos_w_x": fx, f"foot_{leg}_pos_w_y": fy, f"foot_{leg}_pos_w_z": fz,
                f"foot_{leg}_vel_w_x": vx, f"foot_{leg}_vel_w_y": vy, f"foot_{leg}_vel_w_z": vz,
                f"foot_{leg}_terrain_height": th, f"foot_{leg}_clearance_local": fz - th,
                f"foot_{leg}_contact": int(contact_now),
                f"foot_{leg}_force_w_x": fwx, f"foot_{leg}_force_w_y": fwy, f"foot_{leg}_force_w_z": fwz, f"foot_{leg}_force_norm": fnorm,
                f"foot_{leg}_normal_force": fwz, f"foot_{leg}_tangent_force": math.sqrt(fwx*fwx + fwy*fwy) if np.isfinite(fwx) and np.isfinite(fwy) else np.nan,
                f"foot_{leg}_air_time": float(arrays["air"][local_i, li]) if arrays["air"] is not None else np.nan,
                f"foot_{leg}_stance_time": np.nan,
                f"foot_{leg}_touchdown": touchdown,
                f"foot_{leg}_liftoff": liftoff,
                f"foot_{leg}_touchdown_vz": abs(vz) if (touchdown and np.isfinite(vz)) else np.nan,
                f"foot_{leg}_stance_slip_xy": math.sqrt(vx*vx + vy*vy) if (contact_now and np.isfinite(vx) and np.isfinite(vy)) else np.nan,
            })
            if thsrc != row["terrain_height_source"] and row["terrain_height_source"] == "env_origin_fallback":
                row["terrain_height_source"] = thsrc
        rows.append(row)
    return rows


def build_height_query_cache(base, env_ids: list[int]) -> dict[int, np.ndarray]:  # pragma: no cover
    if not env_ids:
        return {}
    scanner = getattr(base, "_height_scanner", None)
    if scanner is not None:
        try:
            hits = scanner.data.ray_hits_w[env_ids].detach().cpu().numpy()
            return {
                env_id: hits[index][np.isfinite(hits[index]).all(axis=-1)]
                for index, env_id in enumerate(env_ids)
            }
        except Exception:
            pass
    return {}


def terrain_height_at_point_best_effort(
    base,
    env_id: int,
    x: float,
    y: float,
    height_cache: dict[int, np.ndarray] | None = None,
) -> tuple[float, str]:  # pragma: no cover
    valid = (height_cache or {}).get(env_id)
    if valid is not None and len(valid):
        distance_sq = (valid[:, 0] - x) ** 2 + (valid[:, 1] - y) ** 2
        return float(valid[int(np.argmin(distance_sq)), 2]), "height_scanner_nearest"
    terrain = getattr(base, "_terrain", None)
    for obj in [terrain, getattr(base, "terrain", None)]:
        if obj is None:
            continue
        for name in ["height_at", "get_height", "get_terrain_height"]:
            fn = getattr(obj, name, None)
            if callable(fn):
                try:
                    return float(fn(x, y)), name
                except Exception:
                    pass
    try:
        return float(base._terrain.env_origins[env_id, 2]), "env_origin_fallback"
    except Exception:
        return 0.0, "unavailable"


def read_terrain_identities_best_effort(
    base, env_ids: list[int]
) -> dict[int, tuple[str, float]]:  # pragma: no cover
    terrain_cfg = getattr(getattr(base, "cfg", None), "terrain", None)
    if getattr(terrain_cfg, "terrain_type", None) == "plane":
        return {env_id: ("flat", 0.0) for env_id in env_ids}
    try:
        terrain = base._terrain
        levels = terrain.terrain_levels[env_ids].detach().cpu().numpy()
        columns = terrain.terrain_types[env_ids].detach().cpu().numpy().astype(int)
        generator = base.cfg.terrain.terrain_generator
        names = list(generator.sub_terrains.keys())
        proportions = np.asarray([generator.sub_terrains[n].proportion for n in names], dtype=float)
        proportions = proportions / proportions.sum()
        type_by_column = np.searchsorted(
            np.cumsum(proportions),
            (np.arange(generator.num_cols, dtype=float) + 0.5) / generator.num_cols,
            side="right",
        )
        return {
            env_id: (names[int(type_by_column[columns[index]])], float(levels[index]))
            for index, env_id in enumerate(env_ids)
        }
    except Exception:
        return {env_id: ("unknown", np.nan) for env_id in env_ids}


def read_terrain_identity_best_effort(base, env_id: int) -> tuple[str, float]:  # pragma: no cover
    return read_terrain_identities_best_effort(base, [env_id])[env_id]


def apply_terrain_case_best_effort(env_cfg, terrain_case: dict[str, Any], notes: list[str]) -> dict[str, Any]:  # pragma: no cover
    requested = str(terrain_case.get("type", "flat"))
    canonical = TERRAIN_ALIASES.get(requested)
    if canonical is None:
        reason = f"unsupported terrain type: {requested}"
        notes.append(reason)
        return {"applied": False, "reason": reason}
    try:
        if canonical == "flat":
            env_cfg.terrain.terrain_type = "plane"
            return {"applied": True, "canonical_type": "flat", "level": 0}
        generator = env_cfg.terrain.terrain_generator
        sub_terrains = generator.sub_terrains
        if canonical not in sub_terrains:
            reason = f"terrain generator does not provide {canonical}"
            notes.append(reason)
            return {"applied": False, "reason": reason}
        for name, cfg in sub_terrains.items():
            cfg.proportion = 1.0 if name == canonical else 0.0
        env_cfg.terrain.terrain_type = "generator"
        generator.curriculum = True
        if requested in ["stairs_up", "stairs_down"]:
            notes.append(f"{requested} uses the actual 'stairs' terrain; ascent/descent is determined from observed motion")
        return {"applied": True, "canonical_type": canonical, "level": int(terrain_case.get("level", 0))}
    except Exception as exc:
        reason = f"terrain case could not be applied: {terrain_case} ({type(exc).__name__}: {exc})"
        notes.append(reason)
        return {"applied": False, "reason": reason}


def force_terrain_level_best_effort(base, terrain_case: dict[str, Any], notes: list[str]) -> bool:  # pragma: no cover
    level = int(terrain_case.get("level", 0))
    if getattr(getattr(base.cfg, "terrain", None), "terrain_type", None) == "plane":
        return level == 0
    try:
        terrain = base._terrain
        max_level = int(terrain.terrain_origins.shape[0]) - 1
        level = max(0, min(level, max_level))
        terrain.terrain_levels[:] = level
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        return True
    except Exception as exc:
        notes.append(f"fixed terrain level unavailable: {type(exc).__name__}: {exc}")
        return False


def apply_dr_case_best_effort(env_cfg, dr_case: dict[str, Any], notes: list[str]) -> dict[str, Any]:  # pragma: no cover
    level = int(dr_case.get("level", 0))
    if level not in [0, 1, 2, 3]:
        reason = f"unsupported DR level: {level}"
        notes.append(reason)
        return {"applied": False, "reason": reason}
    try:
        env_cfg.dr_start_level = level
        env_cfg.dr_enable = level > 0
        for name, value in dr_case.items():
            if name in {"level", "enabled", "note"}:
                continue
            attr = f"dr_{name}_range_{level}" if not name.startswith("dr_") else name
            if hasattr(env_cfg, attr):
                setattr(env_cfg, attr, value)
        return {"applied": True, "level": level}
    except Exception as exc:
        reason = f"DR case could not be applied: {dr_case} ({type(exc).__name__}: {exc})"
        notes.append(reason)
        return {"applied": False, "reason": reason}


def refresh_dr_snapshot_best_effort(base, notes: list[str], env_ids=None) -> None:  # pragma: no cover
    count = int(base.num_envs)
    ids = np.arange(count, dtype=int) if env_ids is None else np.asarray(env_ids, dtype=int)
    cache = getattr(base, "_ilqd_dr_snapshot", None)
    if cache is None:
        cache = {
            key: np.full(count, np.nan, dtype=float)
            for key in ["mass", "friction", "com_x", "com_y", "com_z", "stiffness_scale", "damping_scale", "latency"]
        }
        base._ilqd_dr_snapshot = cache
    try:
        masses = base.robot.root_physx_view.get_masses()
        masses_np = masses.detach().cpu().numpy() if hasattr(masses, "detach") else np.asarray(masses)
        cache["mass"][ids] = masses_np[ids, 0]
    except Exception as exc:
        notes.append(f"actual DR mass unavailable: {type(exc).__name__}")
    try:
        mats = base.robot.root_physx_view.get_material_properties()
        mats_np = mats.detach().cpu().numpy() if hasattr(mats, "detach") else np.asarray(mats)
        cache["friction"][ids] = np.nanmean(mats_np[ids, :, 0], axis=1)
    except Exception as exc:
        notes.append(f"actual DR friction unavailable: {type(exc).__name__}")
    try:
        coms = base.robot.root_physx_view.get_coms()
        coms_np = coms.detach().cpu().numpy() if hasattr(coms, "detach") else np.asarray(coms)
        cache["com_x"][ids] = coms_np[ids, 0, 0]
        cache["com_y"][ids] = coms_np[ids, 0, 1]
        cache["com_z"][ids] = coms_np[ids, 0, 2]
    except Exception as exc:
        notes.append(f"actual DR CoM unavailable: {type(exc).__name__}")
    try:
        actuator = base.robot.actuators["legs"]
        stiffness = actuator.stiffness.detach().cpu().numpy()
        damping = actuator.damping.detach().cpu().numpy()
        cache["stiffness_scale"][ids] = np.nanmean(stiffness[ids], axis=1) / 120.0
        cache["damping_scale"][ids] = np.nanmean(damping[ids], axis=1) / 10.0
    except Exception as exc:
        notes.append(f"actual DR actuator gains unavailable: {type(exc).__name__}")
    cache["latency"][ids] = float(getattr(base.cfg, "action_delay_steps", 0)) * float(
        getattr(base, "step_dt", getattr(base.cfg, "dt", 0.0) * getattr(base.cfg, "decimation", 1))
    )


def read_dr_snapshot(base, env_id: int) -> dict[str, float]:
    cache = getattr(base, "_ilqd_dr_snapshot", {})
    return {
        key: float(value[env_id])
        for key, value in cache.items()
        if value is not None and len(value) > env_id
    }


def maybe_apply_push_best_effort(
    base,
    pushes: list[dict[str, Any]],
    t: float,
    seg_id: int,
    notes: list[str],
    applied: set[tuple[int, int]],
) -> dict[int, dict[str, float]]:  # pragma: no cover
    result: dict[int, dict[str, float]] = {}
    for push_index, p in enumerate(pushes):
        key = (seg_id, push_index)
        if key in applied:
            continue
        if int(p.get("segment", -999)) != int(seg_id):
            continue
        pt = float(p.get("time", -1.0))
        if t + 1e-9 < pt:
            continue
        try:
            import torch
            vector = p.get("vector", [p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)])
            vector = torch.as_tensor(vector, device=base.device, dtype=torch.float32)
            env_ids = p.get("env_ids")
            ids = (
                torch.arange(base.num_envs, device=base.device, dtype=torch.long)
                if env_ids is None
                else torch.as_tensor(env_ids, device=base.device, dtype=torch.long)
            )
            velocity = torch.cat(
                [base.robot.data.root_lin_vel_w[ids], base.robot.data.root_ang_vel_w[ids]], dim=-1
            ).clone()
            velocity[:, :3] += vector[None, :3]
            base.robot.write_root_com_velocity_to_sim(velocity, ids)
            magnitude = float(torch.linalg.norm(vector[:3]))
            for env_id in ids.detach().cpu().tolist():
                result[int(env_id)] = {
                    "x": float(vector[0]), "y": float(vector[1]), "z": float(vector[2]), "delta_v": magnitude,
                }
            applied.add(key)
        except Exception as exc:
            notes.append(f"push request could not be applied: {p} ({type(exc).__name__}: {exc})")
    return result


if __name__ == "__main__":
    main()
