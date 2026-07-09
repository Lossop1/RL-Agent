"""physeval_blind.py — DEPLOYMENT-口径 acceptance eval for the BLIND TerrainPerceiver stack.

Faithfully replicates the payload training assembly:
    gym.make("RobotLab-Isaac-Taili-Blind-Direct-v0", cfg=TailiBlindEnvCfg) + SkrlVecEnvWrapper
    + Runner(env, skrl config generated from taili_blind_config.yaml) + agent.load(checkpoint)
then runs the mean-action diagnostic battery and SCORES each result pass/fail against the
acceptance ruler docs/taili_spec.md via acceptance_score.py (the threshold logic is unit-tested
separately, off-sim). Mean-action = no exploration noise = deployment口径 (taili_spec §1). The blind
policy ignores the privileged/label obs tail (slices proprioception only), so eval is
deployment-faithful for the parts measurable on a given terrain.

  python physeval_blind.py --checkpoint /root/tp_runs/blind_tp_v1/checkpoints/best_agent.pt \
                           --headless --num_envs 64 --steps 220 --terrain flat

Flat scores hard gates A1 A2 A3 B1 B2 B3 B4 C F2. Terrain (D1-D4) needs --terrain stairs/slope/
rough/boxes (scored D-style: controlled progress, tracking-band relaxed per spec §3.6). Robustness
(E1-E5) needs the push+DR battery — reported NOT-EVALUATED, never silently passed. Soft items
(A5 B5 F1 F3) reported, never gating.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=220)
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Taili-Blind-Direct-v0")
parser.add_argument("--config", type=str, default="", help="Taili single-source config YAML")
parser.add_argument("--agent-yaml", type=str, default="", help="Optional generated skrl runtime YAML")
parser.add_argument("--terrain", type=str, default="flat",
                    choices=["mix", "flat", "rough", "stairs", "stairs_up", "boxes", "slope"])
parser.add_argument("--terrain-level", type=int, default=4)
parser.add_argument("--report", type=str, default="", help="Write the scorecard as JSON to this path (for physeval_suite).")
parser.add_argument("--friction", type=float, default=0.0,
                    help="E3/E5: force foot/ground friction (spec E3 low-friction μ>=0.4). 0 = no override (default clean eval).")
parser.add_argument("--com-offset", type=float, default=0.0,
                    help="E5: force base CoM planar offset (m) on all envs. 0 = none.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import numpy as np
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
import acceptance_score as ACC                          # pure spec-threshold judge (off-sim unit-tested)

LEGS = ["FL", "FR", "RL", "RR"]
# Settled-stance vertical-force threshold for the B2 slip metric: a foot bearing more than ~10% of
# the 38.98 kg Taili's weight (0.10 * 38.98 * 9.81 ≈ 38 N) is weight-bearing/settled; below it the
# foot is in a touchdown or liftoff TRANSITION (still moving), which taili_spec excludes from slip.
_SETTLE_N = 38.0
COMMANDS = [("stand",  (0.0, 0.0, 0.0)),
            ("fwd03",  (0.3, 0.0, 0.0)), ("fwd05",  (0.5, 0.0, 0.0)), ("fwd07", (0.7, 0.0, 0.0)),
            ("back04", (-0.4, 0.0, 0.0)), ("lat03",  (0.0, 0.3, 0.0)),
            ("yaw04",  (0.0, 0.0, 0.4)), ("yaw08",  (0.0, 0.0, 0.8)),
            ("mixed",  (0.4, 0.2, 0.3))]
SYMMETRIC = {"stand", "fwd03", "fwd05", "fwd07", "back04"}
LINEAR = {"fwd03": (0, 0.3), "fwd05": (0, 0.5), "fwd07": (0, 0.7),
          "back04": (0, -0.4), "lat03": (1, 0.3)}
YAW = {"yaw04": 0.4, "yaw08": 0.8}


def pf(ok):
    return "PASS" if ok else "FAIL"


def main():
    cfg = TailiBlindEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.terrain != "mix":
        import isaaclab.terrains as tg
        single = {
            "flat":   tg.MeshPlaneTerrainCfg(proportion=1.0),
            "rough":  tg.HfRandomUniformTerrainCfg(proportion=1.0, noise_range=(0.02, 0.12),
                                                   noise_step=0.02, border_width=0.25),
            "stairs": tg.MeshPyramidStairsTerrainCfg(proportion=1.0, step_height_range=(0.05, 0.18),
                                                     step_width=0.3, platform_width=3.0, border_width=1.0,
                                                     holes=False),
            # ASCENDING stairs (0706): inverted pyramid = stepped pit, robot spawns at the BOTTOM and must
            # CLIMB OUT. This makes 爬楼梯 a scored benchmark gate (D[stairs_up]); step up to 0.25 = the 25cm floor.
            "stairs_up": tg.MeshInvertedPyramidStairsTerrainCfg(proportion=1.0, step_height_range=(0.10, 0.25),
                                                     step_width=0.32, platform_width=1.5, border_width=1.0,
                                                     holes=False),
            "boxes":  tg.MeshRandomGridTerrainCfg(proportion=1.0, grid_width=0.45,
                                                  grid_height_range=(0.05, 0.2), platform_width=2.0),
            "slope":  tg.HfPyramidSlopedTerrainCfg(proportion=1.0, slope_range=(0.0, 0.4),
                                                   platform_width=2.0, border_width=0.25),
        }[args.terrain]
        cfg.terrain.terrain_generator.sub_terrains = {args.terrain: single}
    if cfg.terrain.terrain_generator is not None:
        cfg.terrain.terrain_generator.curriculum = True
    cfg.terrain.max_init_terrain_level = max(0, args.terrain_level)

    env = gym.make(args.task, cfg=cfg, render_mode=None)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    if args.agent_yaml:
        ac = yaml.safe_load(open(args.agent_yaml, encoding="utf-8"))
    else:
        ac = build_skrl_config(load_taili_blind_config(args.config or None))
    ac.setdefault("trainer", {})["close_environment_at_exit"] = False
    exp = ac["agent"].setdefault("experiment", {})
    exp["write_interval"] = 0
    exp["checkpoint_interval"] = 0
    runner = Runner(env, ac)
    runner.agent.load(args.checkpoint)
    runner.agent.set_running_mode("eval")

    base = env.unwrapped
    base.use_external_commands = True
    robot = base.robot
    # ── E3/E5 DR-force override (0708): apply the spec DR envelope DIRECTLY on the eval env, using the
    # SAME physx-view writes the training env uses (taili_amp_env _reset_idx). This measures whether the
    # hard gates hold under the low-friction / CoM-shifted condition — the E3/E5 battery that physeval_suite
    # flagged NOT-BUILT. Absolute set on ALL envs; persists across resets (material props are not reset). ──
    if args.friction > 0.0 or args.com_offset != 0.0:
        _N = base.num_envs
        _idx = torch.arange(_N, device=base.device)
        if args.friction > 0.0:
            try:
                _mats = robot.root_physx_view.get_material_properties().clone()
                _mats[:, :, 0:2] = float(args.friction)           # static + dynamic friction, absolute
                robot.root_physx_view.set_material_properties(_mats, _idx.cpu())
                print(f"[E3/E5] forced friction μ={args.friction} on all {_N} envs", flush=True)
            except Exception as _e:
                print(f"[E3/E5] friction override FAILED: {type(_e).__name__}: {_e}", flush=True)
        if args.com_offset != 0.0:
            try:
                _coms = robot.root_physx_view.get_coms().clone()
                _coms[:, 0, 0:2] += float(args.com_offset)
                robot.root_physx_view.set_coms(_coms, _idx.cpu())
                print(f"[E3/E5] forced CoM +{args.com_offset}m on all {_N} envs", flush=True)
            except Exception as _e:
                print(f"[E3/E5] CoM override FAILED: {type(_e).__name__}: {_e}", flush=True)
    foot_b = [robot.data.body_names.index(f"{lg}_foot") for lg in LEGS]
    cs = base._contact_sensor
    foot_c = base._feet_contact_ids
    hs = base._height_scanner
    elim = robot.actuators["legs"].effort_limit                       # SAME source as env reward
    elim = torch.as_tensor(elim, device=base.device, dtype=torch.float32)
    if elim.ndim == 2:
        elim = elim[0]
    elim = elim.reshape(-1)

    print(f"\n{'='*108}\nBLIND ACCEPTANCE  ckpt={os.path.basename(args.checkpoint)}  "
          f"terrain={args.terrain}@{args.terrain_level}  num_envs={args.num_envs}  steps/cmd={args.steps}\n"
          f"effort_limit(12)={elim.tolist()}\n{'='*108}", flush=True)

    def foot_clear(fp):
        hits = hs.data.ray_hits_w
        dxy = fp[:, :, None, :2] - hits[:, None, :, :2]
        near = torch.argmin((dxy * dxy).sum(-1), dim=-1)
        gz = torch.gather(hits[:, :, 2], 1, near)
        return torch.clamp(fp[:, :, 2] - gz, min=0.0)

    obs, _ = env.reset()
    warm = args.steps // 3

    SC = {}
    tq_abs_max = torch.zeros(12, device=base.device)
    tq_samples = []
    clamp_hits = 0; clamp_tot = 0
    td_vz_all = []
    fall_any = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)   # D fall proxy
    bh_drop_max = 0.0

    print(f"{'cmd':6s}| base_h | vx/vy/wz actual (cmd)        | swing FL/FR/RL/RR cm (peak) | "
          f"duty FL/FR/RL/RR | slip FL/FR/RL/RR cm/s | sym dduty/dclr | up | th/ca", flush=True)
    for label, (vx, vy, wz) in COMMANDS:
        c = torch.tensor([vx, vy, wz], dtype=torch.float32, device=base.device)
        A = {k: [] for k in ("bh", "vx", "vy", "wz", "swz", "swzpk", "duty", "slip", "up", "th", "ca")}
        prev_contact = (cs.data.net_forces_w[:, foot_c, :].norm(dim=-1) > 1.0)
        prev_firm = (cs.data.net_forces_w[:, foot_c, :].norm(dim=-1) > 10.0)   # B1: firm-landing edge, not 1N graze
        td_vz = []
        for t in range(args.steps):
            with torch.inference_mode():
                getattr(base, "_cmd_target", base.commands)[:] = c.unsqueeze(0).expand(base.num_envs, -1)
                out = runner.agent.act(obs, timestep=0, timesteps=0)
                obs, _, _, _, _ = env.step(out[-1].get("mean_actions", out[0]))
                _ff = cs.data.net_forces_w[:, foot_c, :].norm(dim=-1)
                contact = (_ff > 1.0)
                # SETTLED stance = foot bearing real weight (> ~10% body weight, _SETTLE_N). Per
                # taili_spec, slip is measured ONLY on settled frames — touchdown/liftoff TRANSITION
                # frames (low force, foot still moving) are excluded. The 1N `contact` mask (used for
                # duty) wrongly counted those transitions as stance and inflated B2. (0705 metric fix.)
                settled = (_ff > _SETTLE_N)
                # B1 touchdown vz measured at the FIRM-landing edge (first frame > 10N), not the 1N
                # first-graze where the foot is still descending fast (which inflated the p95). (D3m)
                firm = (_ff > 10.0)
                firm_rising = firm & (~prev_firm)
                if firm_rising.any() and t >= warm and label != "stand":
                    fvz = robot.data.body_lin_vel_w[:, foot_b, 2].abs()
                    td_vz.append(fvz[firm_rising].detach())
                prev_contact = contact
                prev_firm = firm
                up_now = -robot.data.projected_gravity_b[:, 2]
                fall_any |= (up_now < 0.5)                                       # >60deg tilt = fall
                if t >= warm:
                    tq = robot.data.applied_torque.abs()
                    tq_abs_max = torch.maximum(tq_abs_max, tq.max(0).values)
                    clamp_hits += int((tq >= elim).sum().item())
                    clamp_tot += tq.numel()
                    if t % 5 == 0:
                        tq_samples.append(tq.reshape(-1, 12).detach())
                if t < warm:
                    continue
                fp = robot.data.body_pos_w[:, foot_b, :]
                clr = foot_clear(fp)
                inc = contact.float()                       # 1N contact — for DUTY (contact timing)
                settled_f = settled.float()                 # weight-bearing — for SLIP (spec: settled only)
                swing = 1.0 - inc
                # CONTACT-POINT velocity (not foot-CoM): a sphere foot (r=0.014m) has a ROLLING CoM
                # velocity even with a non-sliding contact; the spec's slip is the bottom-of-sphere
                # (contact-point) velocity = v_com + omega x r, r=(0,0,-radius). Removing the roll
                # deletes the ~0.05 m/s phantom slip that was the whole B2 budget (D3m).
                _omega = robot.data.body_ang_vel_w[:, foot_b, :]                      # (N,4,3)
                _r = torch.tensor([0.0, 0.0, -0.014], device=robot.data.body_lin_vel_w.device).view(1, 1, 3)
                _vc = robot.data.body_lin_vel_w[:, foot_b, :] + torch.cross(_omega, _r.expand_as(_omega), dim=-1)
                fvel = _vc[:, :, :2].norm(dim=-1)
                A["bh"].append(float((robot.data.root_pos_w[:, 2] - base._terrain.env_origins[:, 2]).mean()))
                A["vx"].append(robot.data.root_lin_vel_b[:, 0].detach())
                A["vy"].append(robot.data.root_lin_vel_b[:, 1].detach())
                A["wz"].append(robot.data.root_ang_vel_b[:, 2].detach())
                A["swz"].append(((clr * swing).sum(0) / swing.sum(0).clamp(min=1.0) * 100).tolist())
                A["swzpk"].append((clr * swing).max(0).values.mul(100).tolist())
                A["duty"].append(inc.mean(0).tolist())
                A["slip"].append(((fvel * settled_f).sum(0) / settled_f.sum(0).clamp(min=1.0) * 100).tolist())
                A["up"].append(float(up_now.mean()))
                jp = robot.data.joint_pos
                A["th"].append(float(jp[:, 4:8].mean())); A["ca"].append(float(jp[:, 8:12].mean()))
        bh = np.mean(A["bh"]); up = np.mean(A["up"])
        vx_a = torch.stack(A["vx"]); vy_a = torch.stack(A["vy"]); wz_a = torch.stack(A["wz"])
        swz = np.mean(A["swz"], 0); swpk = np.max(A["swzpk"], 0)
        duty = np.mean(A["duty"], 0); slip = np.mean(A["slip"], 0)
        th = np.degrees(np.mean(A["th"])); ca = np.degrees(np.mean(A["ca"]))
        dd = abs(duty[0] - duty[1]) + abs(duty[2] - duty[3]); dc = abs(swz[0] - swz[1]) + abs(swz[2] - swz[3])
        bh_drop_max = max(bh_drop_max, float(cfg.stand_height - bh) if hasattr(cfg, "stand_height") else 0.0)
        if td_vz:
            td_vz_all.append(torch.cat(td_vz))
        climb = float(A["bh"][-1] - A["bh"][0]) if len(A["bh"]) > 1 else 0.0   # height GAINED = ascent
        SC[label] = dict(bh=bh, up=up, vx=vx_a.mean().item(), vy=vy_a.mean().item(), wz=wz_a.mean().item(),
                         swz=swz, swpk=swpk, duty=duty, slip=slip, dd=dd, dc=dc, climb=climb,
                         vx_a=vx_a, vy_a=vy_a, wz_a=wz_a)
        print(f"{label:6s}| {bh:.3f}m | {SC[label]['vx']:+.2f}/{SC[label]['vy']:+.2f}/{SC[label]['wz']:+.2f} "
              f"({vx:+.1f}/{vy:+.1f}/{wz:+.1f}) | {swz[0]:.1f}/{swz[1]:.1f}/{swz[2]:.1f}/{swz[3]:.1f} "
              f"({swpk.max():.0f}) | {duty[0]:.2f}/{duty[1]:.2f}/{duty[2]:.2f}/{duty[3]:.2f} | "
              f"{slip[0]:.0f}/{slip[1]:.0f}/{slip[2]:.0f}/{slip[3]:.0f} | {dd:.2f}/{dc:.1f} | "
              f"{up:.3f} | {th:.0f}/{ca:.0f}", flush=True)

    # =========================== STOP-RECOVERY (A3 settle) ===========================
    cw = torch.tensor([0.5, 0.0, 0.0], device=base.device); cz = torch.zeros(3, device=base.device)
    for t in range(120):
        with torch.inference_mode():
            getattr(base, "_cmd_target", base.commands)[:] = cw.unsqueeze(0).expand(base.num_envs, -1)
            out = runner.agent.act(obs, timestep=0, timesteps=0)
            obs, _, _, _, _ = env.step(out[-1].get("mean_actions", out[0]))
    spd_tr, h_tr, duty_tr, up_tr, v_tr, wz_tr = [], [], [], [], [], []
    HALT = 160
    for t in range(HALT):
        with torch.inference_mode():
            getattr(base, "_cmd_target", base.commands)[:] = cz.unsqueeze(0).expand(base.num_envs, -1)
            out = runner.agent.act(obs, timestep=0, timesteps=0)
            obs, _, _, _, _ = env.step(out[-1].get("mean_actions", out[0]))
            spd = float(robot.data.root_lin_vel_b[:, :2].norm(dim=-1).mean())
            wzr = float(robot.data.root_ang_vel_b[:, 2].abs().mean())
            v_tr.append(spd); wz_tr.append(wzr); spd_tr.append(spd + wzr)   # spd_tr kept for resid/peak report
            h_tr.append(float((robot.data.root_pos_w[:, 2] - base._terrain.env_origins[:, 2]).mean()))
            inc = (cs.data.net_forces_w[:, foot_c, :].norm(dim=-1) > 10.0).float()  # DUTY = foot down (10N, matches train), not the 38N settled-slip mask
            duty_tr.append(inc.mean(0).tolist())
            up_tr.append(float((-robot.data.projected_gravity_b[:, 2]).mean()))
    spd_tr, v_tr, wz_tr = np.array(spd_tr), np.array(v_tr), np.array(wz_tr)
    # A3 settle per spec §3.1: first frame after which |v|<=0.05 AND |wz|<=0.05 BOTH hold for the rest
    # (separate bands + dwell), NOT the old conflated (|v|+|wz|)<0.08 forward-max.
    _settled = (v_tr <= 0.05) & (wz_tr <= 0.05)
    settle = next((i for i in range(len(_settled)) if _settled[i:].all()), HALT)
    # A3-settle ARTIFACT CHECK (0706): the strict "holds for ALL remaining frames" makes settle hypersensitive
    # to a single late micro-blip. Report the FIRST settled frame + how many later blips push the strict settle
    # out, so we can tell a genuine slow-settle from a 1-frame artifact.
    _first = next((i for i in range(len(_settled)) if _settled[i]), HALT)
    _dt = float(getattr(base, "step_dt", 0.02))
    _blips = int((~_settled[_first:]).sum()) if _first < HALT else 0
    print(f"[A3_SETTLE_DIAG] first_settled={_first*_dt:.2f}s strict_settle={settle*_dt:.2f}s "
          f"blips_after_first={_blips} maxv_after_first={float(v_tr[_first:].max()) if _first<HALT else 0:.3f} "
          f"maxwz_after_first={float(wz_tr[_first:].max()) if _first<HALT else 0:.3f}", flush=True)
    resid = float(spd_tr[-30:].mean()); peak = float(spd_tr[:20].max())
    fduty = np.mean(duty_tr[-30:], 0); fh = float(np.mean(h_tr[-30:])); fup = float(np.mean(up_tr[-30:]))
    fsym = abs(fduty[0] - fduty[1]) + abs(fduty[2] - fduty[3])
    print(f"\n{'-'*108}\nSTOP-RECOVERY (walk0.5 -> halt {HALT}): settle={settle} ({settle*0.02:.1f}s)  "
          f"resid={resid:.3f}m/s  peak={peak:.2f}  duty={fduty[0]:.2f}/{fduty[1]:.2f}/{fduty[2]:.2f}/{fduty[3]:.2f}  "
          f"sym={fsym:.2f}  h={fh:.3f}m  up={fup:.3f}", flush=True)

    # =========================== SCORECARD via acceptance_score (vs taili_spec) ===========================
    def p(a, q):
        return float(torch.quantile(a.reshape(-1).float(), q))
    print(f"\n{'='*108}\nSCORECARD vs taili_spec ({args.terrain}@{args.terrain_level}, mean-action)\n{'='*108}", flush=True)
    results = {}
    flat = (args.terrain == "flat")

    if flat:
        a1b = {}
        for lab, (axis, cmd) in LINEAR.items():
            if lab not in SC:
                continue
            a = SC[lab]["vx_a"] if axis == 0 else SC[lab]["vy_a"]
            err = (a - cmd).abs()
            a1b[lab] = (p(err, 0.5), p(err, 0.9), cmd)
        results.update(ACC.score_A1(a1b))
        a2b = {}
        for lab, cmd in YAW.items():
            if lab in SC:
                err = (SC[lab]["wz_a"] - cmd).abs()
                a2b[lab] = (p(err, 0.5), p(err, 0.9))
        results.update(ACC.score_A2(a2b))
        if "stand" in SC:
            s = SC["stand"]
            speed = (s["vx"] ** 2 + s["vy"] ** 2) ** 0.5
            results.update(ACC.score_A3(speed, s["wz"], settle * 0.02, float(min(s["duty"])), s["up"]))
        if td_vz_all:
            results.update(ACC.score_B1(p(torch.cat(td_vz_all), 0.95), args.terrain))
        slip_pool = np.concatenate([SC[l]["slip"] for l in SC if l != "stand"]) / 100.0
        results.update(ACC.score_B2(float(np.percentile(slip_pool, 90)), args.terrain))
        # per-swing-peak p90 distribution (not a global max-of-max outlier over ~300k samples)
        _swpk = np.concatenate([np.asarray(SC[l]["swpk"], dtype=float).reshape(-1)
                                for l in SC if l != "stand"])
        peakcm = float(np.percentile(_swpk, 90)) / 100.0
        results.update(ACC.score_B3(peakcm, args.terrain))
        symb = {l: (SC[l]["dd"], SC[l]["dc"] / 100.0) for l in SYMMETRIC if l in SC}
        results.update(ACC.score_B4(symb))
        a3ok = results.get("A3", {"ok": False})["ok"]
        results.update(ACC.score_C(a3ok, SC.get("stand", {}).get("dd", 9.0), fup, float(min(fduty))))
    else:
        if args.terrain == "stairs_up":
            # ASCENDING: judge by HEIGHT CLIMBED (ascent is slow, so a 0.30 m/s horizontal bar would
            # false-fail a robot that IS slowly climbing). Pass = climbed >= 0.15 m AND no fall.
            fwd = SC.get("fwd05", SC.get("fwd03", {}))
            climb_m = fwd.get("climb", 0.0)
            results.update(ACC.score_D_ascend(climb_m, float(fall_any.float().mean())))
        else:
            # D terrain: controlled progress at fwd05, no fall, bounded base-height drop
            fwd = SC.get("fwd05", SC.get("fwd03", {}))
            fwd_speed = fwd.get("vx", 0.0)
            results.update(ACC.score_D(fwd_speed, float(fall_any.float().mean()), bh_drop_max, args.terrain))

    # F2 always (torque is terrain-independent hardware gate)
    tqs = torch.cat(tq_samples) if tq_samples else torch.zeros(1, 12, device=base.device)
    q995 = torch.quantile(tqs.float(), 0.995, dim=0)
    ratio995 = (q995 / elim).max().item()
    peak_ratio = (tq_abs_max / elim).max().item()
    clamp_rate = clamp_hits / max(1, clamp_tot)
    results.update(ACC.score_F2(ratio995, peak_ratio, clamp_rate, int((q995 / elim).argmax().item())))

    for k, v in results.items():
        print(f"{k:10s} {pf(v['ok'])}  {v['detail']}", flush=True)
    s = ACC.summarize(results)
    print(f"\n{'-'*108}\nHARD GATES on {args.terrain}: {s['n_pass']}/{s['n_total']} pass  "
          f"PASS[{', '.join(s['passed'])}]  FAIL[{', '.join(s['failed'])}]", flush=True)
    print("NOT-EVALUATED here: " + ("D1-D4 (run --terrain stairs/slope/rough/boxes); " if flat else "")
          + "E1-E5 (push + full DR battery); B4 pi(M obs)=M pi(obs) [structural; unit-tested in taili_core]. "
            "Soft report-only: A5 envelope, B5 jerk, F1 CoT, F3 jitter.", flush=True)
    if args.report:
        import json as _json
        from pathlib import Path as _Path
        payload = {
            "terrain": args.terrain,
            "terrain_level": args.terrain_level,
            "checkpoint": args.checkpoint,
            "items": {k: ("PASS" if v["ok"] else "FAIL") for k, v in results.items()},
            "details": {k: v["detail"] for k, v in results.items()},
            "summary": s,
        }
        _Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        _Path(args.report).write_text(_json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[report] -> {args.report}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
