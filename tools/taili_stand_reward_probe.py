"""量化 Taili 在运动命令归零后继续踏步时的真实奖励驱动。

该工具只加载既有检查点并执行确定性均值动作，不修改环境奖励、策略或训练状态。
输出同时包含任务奖励、AMP 风格奖励、站立分量和关键物理量，便于判断静止失败
究竟来自相反驱动、奖励不可达，还是样本与优化不足。
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe zero-command stand reward causality")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--task", default="RobotLab-Isaac-Taili-AMP-Blind-Direct-v0")
    parser.add_argument("--agent-yaml", default="")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--sequence", choices=("full", "forward"), default="full")
    return parser


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(max(int(round(q * (len(ordered) - 1))), 0), len(ordered) - 1)
    return float(ordered[index])


def _summarize(rows: list[dict[str, float]]) -> dict[str, Any]:
    keys = sorted({key for row in rows for key in row})
    result: dict[str, Any] = {"steps": len(rows)}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        if values:
            result[key] = {
                "mean": _mean(values),
                "p10": _quantile(values, 0.10),
                "p50": _quantile(values, 0.50),
                "p90": _quantile(values, 0.90),
                "sum": float(sum(values)),
            }
    return result


def _tensor_mean(value: Any) -> float | None:
    import torch

    if not torch.is_tensor(value) or value.numel() == 0:
        return None
    return float(torch.nan_to_num(value.detach(), nan=0.0, posinf=0.0, neginf=0.0).mean())


def _load_modules(agent: Any, checkpoint_path: str) -> list[str]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = ("policy", "state_preprocessor", "discriminator", "amp_state_preprocessor")
    loaded: list[str] = []
    for name in required:
        if name not in checkpoint:
            raise KeyError(f"checkpoint is missing module: {name}")
        module = agent.checkpoint_modules.get(name)
        if module is None or not hasattr(module, "load_state_dict"):
            raise KeyError(f"agent is missing checkpoint module: {name}")
        module.load_state_dict(checkpoint[name])
        if hasattr(module, "eval"):
            module.eval()
        loaded.append(name)
    return loaded


def _style_reward(agent: Any, amp_states: Any) -> Any:
    import torch

    logits, _, _ = agent.discriminator.act(
        {"states": agent._amp_state_preprocessor(amp_states)},
        role="discriminator",
    )
    scale = float(agent._discriminator_reward_scale)
    return -torch.log(torch.clamp(1.0 - torch.sigmoid(logits), min=1e-4)) * scale


def _run(args: argparse.Namespace) -> dict[str, Any]:  # pragma: no cover - 需要 IsaacLab
    import numpy as np
    import torch
    import gymnasium as gym
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from skrl.utils.runner.torch import Runner

    import taili_blind_runtime  # noqa: F401 - 注册任务
    from taili_blind_runtime import diagnose_taili
    from taili_blind_runtime.taili_core import taili_reward

    # 捕获环境每一步实际使用的完整奖励分量；包装器不改变返回值或梯度语义。
    captured: dict[str, Any] = {}
    original_compute = taili_reward.compute_reward_components

    def capture_components(inp: Any, cfg: Any) -> dict[str, Any]:
        components = original_compute(inp, cfg)
        captured["components"] = components
        return components

    taili_reward.compute_reward_components = capture_components

    experiment_cfg = diagnose_taili._load_experiment_cfg(args)
    experiment_cfg.setdefault("trainer", {})["close_environment_at_exit"] = False
    experiment_cfg.setdefault("agent", {}).setdefault("experiment", {})["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    diagnose_taili._configure_terrain(env_cfg, {"type": "flat", "level": 0, "params": {}})
    if hasattr(env_cfg, "early_termination"):
        env_cfg.early_termination = False
    if hasattr(env_cfg, "reset_strategy"):
        env_cfg.reset_strategy = "start"

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    try:
        env = SkrlVecEnvWrapper(env, ml_framework="torch")
        base = env.unwrapped
        # A diagnostic plane bypasses the terrain generator, whose fallback mask
        # labels every environment as non-flat. Preserve the training meaning of
        # the generator's `flat` subterrain for reward-component inspection.
        if getattr(getattr(base.cfg, "terrain", None), "terrain_type", None) == "plane":
            base._flat_terrain_mask = torch.ones(
                base.num_envs, dtype=torch.bool, device=base.device
            )
            base._real_terrain_mask = torch.zeros(
                base.num_envs, dtype=torch.bool, device=base.device
            )
        base.use_external_commands = True
        runner = Runner(env, experiment_cfg)
        loaded_modules = _load_modules(runner.agent, args.checkpoint)
        runner.agent.set_running_mode("eval")

        dt = float(getattr(base, "step_dt", 0.02))
        if args.sequence == "forward":
            segments = [
                ("initial_zero", (0.0, 0.0, 0.0), 0.5),
                ("forward", (0.5, 0.0, 0.0), 2.0),
                ("zero_after_forward", (0.0, 0.0, 0.0), 3.0),
            ]
        else:
            segments = [
                ("initial_zero", (0.0, 0.0, 0.0), 1.0),
                ("forward", (0.5, 0.0, 0.0), 3.0),
                ("zero_after_forward", (0.0, 0.0, 0.0), 4.0),
                ("lateral", (0.0, 0.35, 0.0), 3.0),
                ("zero_after_lateral", (0.0, 0.0, 0.0), 4.0),
                ("yaw", (0.0, 0.0, 0.6), 3.0),
                ("zero_after_yaw", (0.0, 0.0, 0.0), 4.0),
            ]
        planned_steps = sum(max(1, int(round(duration / dt))) for _, _, duration in segments)

        first_target = torch.tensor(segments[0][1], device=base.device, dtype=torch.float32)
        diagnose_taili._set_external_command(base, first_target)
        obs, _ = diagnose_taili._force_env_reset(env)
        diagnose_taili._set_external_command(base, first_target)
        obs = diagnose_taili._sync_external_command_observation(obs, base)
        diagnose_taili._extend_diagnostic_episode_horizon(
            base,
            planned_steps=planned_steps,
            margin_steps=max(10, int(round(2.0 / dt))),
        )

        task_weight = float(runner.agent._task_reward_weight)
        style_weight = float(runner.agent._style_reward_weight)
        previous_action = torch.zeros(
            (base.num_envs, int(getattr(base.cfg, "action_space", 12))),
            device=base.device,
        )
        all_rows: dict[str, list[dict[str, float]]] = {}
        all_components: dict[str, list[dict[str, float]]] = {}
        interpolation_rows: dict[str, list[dict[str, float]]] = {}

        for segment_name, command, duration in segments:
            target = torch.tensor(command, device=base.device, dtype=torch.float32)
            diagnose_taili._set_external_command(base, target)
            obs = diagnose_taili._sync_external_command_observation(obs, base)
            steps = max(1, int(round(duration / dt)))
            rows: list[dict[str, float]] = []
            component_rows: list[dict[str, float]] = []
            interp: list[dict[str, float]] = []

            with torch.no_grad():
                reference_amp = base.collect_reference_motions(
                    base.num_envs,
                    current_times=np.zeros(base.num_envs, dtype=np.float32),
                )
                reference_style = float(_style_reward(runner.agent, reference_amp).mean())

            for step_index in range(steps):
                diagnose_taili._set_external_command(base, target)
                obs = diagnose_taili._sync_external_command_observation(obs, base)
                with torch.inference_mode():
                    action_out = runner.agent.act(obs, timestep=0, timesteps=0)
                    action = action_out[-1].get("mean_actions", action_out[0])
                action_delta_rms = torch.linalg.vector_norm(
                    action - previous_action, dim=-1
                ) / math.sqrt(action.shape[-1])
                # IsaacLab 会复用并原地更新下一步观测，不能在 inference tensor 语义下创建它。
                obs, task_reward, _terminated, _truncated, _info = env.step(action)
                previous_action = action.detach().clone()
                with torch.no_grad():
                    amp_states = base.extras["amp_obs"]
                    style_raw = _style_reward(runner.agent, amp_states).reshape(-1)
                    style_scale = torch.as_tensor(
                        base.extras.get("amp_style_scale", 1.0),
                        dtype=style_raw.dtype,
                        device=style_raw.device,
                    ).reshape(-1)
                    style_effective = style_raw * style_scale
                    task_flat = task_reward.reshape(-1)
                    combined = task_weight * task_flat + style_weight * style_effective

                    root_lin = base.robot.data.root_lin_vel_b
                    root_ang = base.robot.data.root_ang_vel_b
                    joint_rms = torch.linalg.vector_norm(
                        base.robot.data.joint_vel, dim=-1
                    ) / math.sqrt(base.robot.data.joint_vel.shape[-1])
                    foot_vel = base.robot.data.body_lin_vel_w[:, base.foot_indexes]
                    foot_speed = torch.linalg.vector_norm(foot_vel, dim=-1)
                    contact = base._in_contact
                    contact_count = contact.sum(dim=-1)
                    static_mask = (
                        (torch.linalg.vector_norm(root_lin[:, :2], dim=-1) < 0.05)
                        & (root_ang[:, 2].abs() < 0.05)
                        & (torch.linalg.vector_norm(root_ang[:, :2], dim=-1) < 0.10)
                        & (joint_rms < 0.10)
                        & (foot_speed.max(dim=-1).values < 0.10)
                        & (contact_count >= 3.5)
                    )
                    tread_mask = (joint_rms > 0.30) | (contact_count < 3.5)

                    row = {
                        "time": (step_index + 1) * dt,
                        "task_reward": float(task_flat.mean()),
                        "style_raw": float(style_raw.mean()),
                        "style_effective": float(style_effective.mean()),
                        "style_reference": reference_style,
                        "combined_reward": float(combined.mean()),
                        "amp_style_scale": float(style_scale.mean()),
                        "speed_xy": float(torch.linalg.vector_norm(root_lin[:, :2], dim=-1).mean()),
                        "yaw_abs": float(root_ang[:, 2].abs().mean()),
                        "wxy": float(torch.linalg.vector_norm(root_ang[:, :2], dim=-1).mean()),
                        "vz_abs": float(root_lin[:, 2].abs().mean()),
                        "joint_velocity_rms": float(joint_rms.mean()),
                        "foot_speed_mean": float(foot_speed.mean()),
                        "foot_speed_worst": float(foot_speed.max(dim=-1).values.mean()),
                        "contact_count": float(contact_count.mean()),
                        "four_contact_fraction": float((contact_count >= 3.5).float().mean()),
                        "action_delta_rms": float(action_delta_rms.mean()),
                        "strict_static_fraction": float(static_mask.float().mean()),
                        "treading_fraction": float(tread_mask.float().mean()),
                    }
                    if bool(static_mask.any()):
                        row["style_static_samples"] = float(style_raw[static_mask].mean())
                    if bool(tread_mask.any()):
                        row["style_treading_samples"] = float(style_raw[tread_mask].mean())
                    rows.append(row)

                    components = captured.get("components", {})
                    component_row: dict[str, float] = {}
                    for key, value in components.items():
                        mean_value = _tensor_mean(value)
                        if mean_value is not None:
                            component_row[key] = mean_value
                    component_rows.append(component_row)

                    # 零命令尾段每 0.2 秒比较“当前状态 -> 静态解析参考”的 AMP 回报曲线。
                    if command == (0.0, 0.0, 0.0) and step_index % max(1, int(round(0.2 / dt))) == 0:
                        alpha_row = {"time": (step_index + 1) * dt}
                        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
                            mixed = amp_states + alpha * (reference_amp - amp_states)
                            alpha_row[f"alpha_{alpha:.2f}"] = float(
                                _style_reward(runner.agent, mixed).mean()
                            )
                        interp.append(alpha_row)

            all_rows[segment_name] = rows
            all_components[segment_name] = component_rows
            interpolation_rows[segment_name] = interp

        summaries: dict[str, Any] = {}
        for segment_name, rows in all_rows.items():
            duration = rows[-1]["time"] if rows else 0.0
            tail_start = max(0.0, duration - 2.0)
            tail_rows = [row for row in rows if row["time"] >= tail_start]
            component_rows = all_components[segment_name]
            tail_index = max(0, len(component_rows) - len(tail_rows))
            summaries[segment_name] = {
                "full": _summarize(rows),
                "tail_2s": _summarize(tail_rows),
                "components_full": _summarize(component_rows),
                "components_tail_2s": _summarize(component_rows[tail_index:]),
                "amp_static_interpolation": _summarize(interpolation_rows[segment_name]),
            }

        return {
            "checkpoint": args.checkpoint,
            "num_envs": int(base.num_envs),
            "dt": dt,
            "loaded_modules": loaded_modules,
            "task_reward_weight": task_weight,
            "style_reward_weight": style_weight,
            "discriminator_reward_scale": float(runner.agent._discriminator_reward_scale),
            "segments": summaries,
            "raw_series": all_rows,
        }
    finally:
        taili_reward.compute_reward_components = original_compute
        try:
            env.close()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - 需要 IsaacLab
    args = _parser().parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["TAILI_DIAGNOSTIC"] = "1"
    os.environ.setdefault("TAILI_INIT_PHASE", "2")
    os.environ.setdefault("TAILI_RUN_DIR", str(out_path.parent))
    os.environ.setdefault("TAILI_RUN_ID", out_path.parent.name)
    try:
        from isaaclab.app import AppLauncher
    except Exception as exc:
        raise RuntimeError("该探针必须在 IsaacLab Python 环境中运行") from exc

    launch_args = argparse.Namespace(**vars(args))
    launch_args.headless = bool(args.headless)
    launch_args.enable_cameras = False
    app_launcher = AppLauncher(launch_args)
    app = app_launcher.app
    try:
        result = _run(args)
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(f"[STAND_PROBE] complete: {out_path}", flush=True)
    except BaseException:
        error_path = out_path.with_suffix(".error.json")
        error_path.write_text(
            json.dumps({"traceback": traceback.format_exc()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
