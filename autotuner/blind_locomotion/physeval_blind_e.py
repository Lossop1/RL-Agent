"""physeval_blind_e.py — E-robustness (push recovery) battery for the blind stack (taili_spec E1/E2).

The one acceptance MEASUREMENT physeval_blind doesn't cover. Applies a deterministic lateral base
push (the same write_root_com_velocity_to_sim the env uses for DR pushes) and measures, per spec:
  E1 walking push (cmd fwd-0.5): no fall, recover ≤1.0s, resume tracking.
  E2 standing push (cmd 0):      no fall, recover ≤1.0s, return to clean stand.
Scored via acceptance_score.score_E1/E2 (the threshold logic is unit-tested off-sim). Separate file
from physeval_blind.py so the cron's flat acceptance run is never disturbed.

  python physeval_blind_e.py --checkpoint ...best_agent.pt --headless --num_envs 64 --push 1.0
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--push", type=float, default=1.0)          # lateral push m/s (spec E1/E2 ≥1.0)
parser.add_argument("--report", type=str, default="", help="Write E1/E2 verdicts as JSON (for physeval_suite).")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Taili-Blind-Direct-v0")
parser.add_argument("--config", type=str, default="", help="Taili single-source config YAML")
parser.add_argument("--agent-yaml", type=str, default="", help="Optional generated skrl runtime YAML")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False
app = AppLauncher(args).app

import torch
import gymnasium as gym
import yaml
from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper
try:
    import taili_blind_runtime as taili_runtime  # noqa: F401
    from taili_blind_runtime.taili_blind_env_cfg import TailiBlindEnvCfg
    from taili_blind_runtime.taili_blind_config import build_skrl_config, load_taili_blind_config
except Exception:
    import autotuner.blind_locomotion as taili_runtime  # noqa: F401
    from autotuner.blind_locomotion.taili_blind_env_cfg import TailiBlindEnvCfg
    from autotuner.blind_locomotion.taili_blind_config import build_skrl_config, load_taili_blind_config
import acceptance_score as ACC

# Control-step dt = physics dt (yaml env.control.dt = 0.005) * decimation (4) = 0.02 s.
# Each env.step() advances one control step, so recov_step counts control steps and rec_t =
# recov_step * DT is real seconds. (Was 0.02*4 = 0.08 — a 4x error that made the spec E1/E2
# "recover <=1.0s" gate reject any recovery slower than ~0.25s real, false-failing good policies.)
DT = 0.005 * 4


def main():
    cfg = TailiBlindEnvCfg(); cfg.scene.num_envs = args.num_envs
    import isaaclab.terrains as tg
    cfg.terrain.terrain_generator.sub_terrains = {"flat": tg.MeshPlaneTerrainCfg(proportion=1.0)}
    env = gym.make(args.task, cfg=cfg, render_mode=None)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    if args.agent_yaml:
        ac = yaml.safe_load(open(args.agent_yaml, encoding="utf-8"))
    else:
        ac = build_skrl_config(load_taili_blind_config(args.config or None))
    ac.setdefault("trainer", {})["close_environment_at_exit"] = False
    ac["agent"].setdefault("experiment", {}).update(write_interval=0, checkpoint_interval=0)
    runner = Runner(env, ac); runner.agent.load(args.checkpoint); runner.agent.set_running_mode("eval")
    base = env.unwrapped; base.use_external_commands = True; robot = base.robot
    elim = robot.actuators["legs"].effort_limit          # SAME source as env reward / physeval_blind F2

    def step(cmd):
        with torch.inference_mode():
            getattr(base, "_cmd_target", base.commands)[:] = cmd.unsqueeze(0).expand(base.num_envs, -1)
            out = runner.agent.act(obs_ref[0], timestep=0, timesteps=0)
            o, *_ = env.step(out[-1].get("mean_actions", out[0]))
            obs_ref[0] = o

    def upright():
        return (-robot.data.projected_gravity_b[:, 2])           # ~1 upright, <0.5 fallen
    def speed():
        return robot.data.root_lin_vel_b[:, :2].norm(dim=-1)

    def apply_push(mag):
        vel = torch.cat([robot.data.root_lin_vel_w, robot.data.root_ang_vel_w], dim=-1).clone()
        vel[:, 1] += mag                                         # lateral shove (y), deterministic
        robot.write_root_com_velocity_to_sim(vel)

    def run_push_case(cmd, label, target_speed):
        for _ in range(80):                                      # settle into the commanded behavior
            step(cmd)
        apply_push(args.push)
        fell = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
        recov_step = torch.full((base.num_envs,), -1, dtype=torch.long, device=base.device)
        sat_hits = 0; sat_tot = 0                                # torque-saturation event counting
        for t in range(80):                                     # 1.6s recovery window
            step(cmd)
            fell |= (upright() < 0.5)
            # E1 clause: "no torque saturation" — measure the actual applied torque vs the effort
            # limit during the recovery window (same |tau|>=limit event as physeval_blind F2).
            with torch.inference_mode():
                tq = robot.data.applied_torque.abs()
                sat_hits += int((tq >= elim).sum().item()); sat_tot += tq.numel()
            # recovered = upright AND speed near the command's target (tracking resumed / standing)
            ok = (upright() > 0.93) & ((speed() - target_speed).abs() < 0.20)
            newly = ok & (recov_step < 0)
            recov_step[newly] = t
        fall_rate = float(fell.float().mean())
        recovered = recov_step >= 0
        rec_t = float(recov_step[recovered].float().mean()) * DT if bool(recovered.any()) else 99.0
        tracking_resumed = float(recovered.float().mean()) > 0.8
        sat_rate = sat_hits / max(1, sat_tot)
        # allow a tiny transient at the instant of impact; sustained saturation fails E1.
        torque_saturated = sat_rate > 0.005
        print(f"{label}: fall_rate={fall_rate:.3f} recovery_time={rec_t:.2f}s "
              f"recovered_frac={float(recovered.float().mean()):.2f} sat_rate={sat_rate:.4f}", flush=True)
        return fall_rate, rec_t, tracking_resumed, torque_saturated

    obs_ref = [env.reset()[0]]
    print(f"\n{'='*80}\nBLIND E-BATTERY  push={args.push}m/s  num_envs={args.num_envs}\n{'='*80}", flush=True)
    results = {}
    fr1, rt1, tr1, sat1 = run_push_case(torch.tensor([0.5, 0, 0], device=base.device), "E1 walking push", 0.5)
    results.update(ACC.score_E1(fr1, rt1, tr1, torque_saturated=sat1))
    fr2, rt2, ret2, _sat2 = run_push_case(torch.tensor([0.0, 0, 0], device=base.device), "E2 standing push", 0.0)
    results.update(ACC.score_E2(fr2, rt2, ret2))
    print(f"\n{'-'*80}\nE-GATES:", flush=True)
    for k, v in results.items():
        print(f"  {k}  {'PASS' if v['ok'] else 'FAIL'}  {v['detail']}", flush=True)
    if getattr(args, "report", ""):
        import json as _json
        from pathlib import Path as _Path
        payload = {"push": args.push, "checkpoint": args.checkpoint,
                   "items": {k: ("PASS" if v["ok"] else "FAIL") for k, v in results.items()},
                   "details": {k: v["detail"] for k, v in results.items()}}
        _Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        _Path(args.report).write_text(_json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[report] -> {args.report}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
