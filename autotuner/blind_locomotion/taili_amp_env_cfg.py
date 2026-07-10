# Taili quadruped AMP locomotion env (adapted from g1_amp; task=velocity tracking, style=AMP trot)
from __future__ import annotations
import os
from pathlib import Path
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import PhysxCfg, RigidBodyMaterialCfg, SimulationCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from .assets.taili import TAILI_DOG_CFG

MOTIONS_DIR = os.environ.get("TAILI_MOTIONS_DIR", str(Path(__file__).resolve().parent / "motions"))

TAILI_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=20,
    horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
    use_cache=False, curriculum=True,
    sub_terrains={
        "flat":      terrain_gen.MeshPlaneTerrainCfg(proportion=0.28),
        "slope":     terrain_gen.HfPyramidSlopedTerrainCfg(
                         proportion=0.18, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25),
        "slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                         proportion=0.14, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25),
        "rough":     terrain_gen.HfRandomUniformTerrainCfg(
                         proportion=0.13, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25),
        "boxes":     terrain_gen.MeshRandomGridTerrainCfg(
                         proportion=0.10, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0),
        "stairs":    terrain_gen.MeshPyramidStairsTerrainCfg(   # spawn on center platform -> DESCEND
                         proportion=0.05, step_height_range=(0.05, 0.15), step_width=0.3,
                         platform_width=3.0, border_width=1.0, holes=False),
        # ASCENDING stairs (0706): inverted pyramid = a stepped pit, robot spawns at the bottom and must
        # CLIMB OUT. The benchmark's 爬楼梯 (25cm floor) needs this; the normal pyramid above spawns on top
        # so it only ever practised DESCENT. Tall step band (up to 0.30) so the foot_h4 latent + adaptive
        # clearance + climb reward actually get trained on real risers. Curriculum scales height by level.
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                         proportion=0.12, step_height_range=(0.08, 0.30), step_width=0.32,
                         platform_width=3.0, border_width=1.0, holes=False),
    },
)


@configclass
class TailiAmpEnvCfg(DirectRLEnvCfg):
    # 鈹€鈹€ TRACKING REWARDS 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    # AMP-CORE AUDIT: the discriminator (achieved [vx,vy,wz] now in amp_obs) is the primary speed signal, so
    # track_lin stays a LIGHT command anchor (1.5). track_ang RAISED to 2.0 鈥?yaw is the weakest axis (prog
    # ~0.67) and the discriminator's single wz channel under-drives it.
    # TWO-SIDED tracking (P4): velocity is the SOLE owner of A1-A3 now (AMP no longer touches speed), so make it a
    # PRIMARY term comparable to gait/style 鈥?was 1.5/2.0 and got dominated by imit+gait (4.7 vs 0.9) -> the policy
    # ignored the weak speed signal and overspeed went unpunished. r=rew*exp(-||achieved-cmd||^2/track_sigma).
    rew_track_lin    = 3.5     # 2.5->3.5 (0708 user): 增大速度跟踪
    rew_track_ang    = 3.5     # 2.5->3.5 增大角速度跟踪
    track_sigma_lin  = 0.25      # m/s scale: err 0.25 -> 0.78, err 0.5 -> 0.37 (two-sided; clean-stop precision via P5)
    track_sigma_ang  = 0.25      # rad/s scale
    stand_sigma      = 0.10      # (legacy; used by stand-pose terms)
    # AMP-CORE AUDIT: r_air REMOVED (0.0). The AMP reference already lifts the feet and gait_match enforces the
    # contact timing, so feet_air_time is redundant 鈥?AND its target (0.30s) exceeded the reference swing
    # duration (0.5*period 鈮?0.22-0.26s), pulling toward a LONGER air than the AMP gait. Drop it.
    rew_feet_air_time = 0.0
    air_time_min     = 0.08
    air_time_target  = 0.30
    rew_alive        = 0.0
    rew_action_rate  = -0.025  # -0.01->-0.025: smoother action transitions -> the move->stop "绔嬫" recovery
                               # eases in instead of snapping (user: recovery worked but felt abrupt/unnatural)
    rew_joint_acc    = -2.5e-7
    rew_lin_vel_z    = -1.5
    rew_ang_vel_xy   = -0.4
    # MOTOR PROTECTION: penalize joint torque ONLY in the stress zone (above torque_limit_frac of the effort
    # limit) so normal-gait torque is untouched but near-saturation / violent torque (motor damage risk) is
    # discouraged -> a gentler, motor-safe policy. effort limits: hip 320 / thigh 110 / calf 220 Nm.
    rew_torque       = -2.0e-4
    torque_limit_frac = 0.85
    # STAND STILLNESS: when commanded to stand, penalize joint velocity so the robot SETTLES smoothly into the
    # nominal "绔嬫" stance instead of fidgeting. Gated to phi>=1 (refinement, off during phi0 bootstrap).
    rew_stand_still  = -2.0e-2  # P5: -6e-3->-2e-2. physeval showed a 0.19 m/s creep + feet not planted at cmd=0 鈥?                               # the old weight was too weak to matter. Now ALWAYS-ON (trains stop from phi0) + stronger.
                               # Penalizes joint velocity ONLY when commanded ~0 -> settle to a still 绔嬫, not fidget.

    # 鈹€鈹€ GAIT-PHASE TROT CONTACT 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    rew_gait_phase   = 3.0     # AMP-CORE AUDIT (restored 1.0->3.0): the discriminator is BLIND to foot contact
                               # (amp_obs has no contact force), so gait_match is the ONLY contact-timing signal 鈥?                               # NOT redundant with the reference (which only sees foot POSITION). The clock is
                               # verified consistent with the AMP reference (same trot offsets/duty/period law), so
                               # this reinforces AMP, not fights it. Overspeed is now fixed, so 3.0 can finally
                               # push gait_match past the old 0.74 ceiling (was structurally locked by overspeed).
    # Target trot cadence ~0.5 s (period base). The robot's revealed cadence (~0.35 s fast
    # shuffle) is treated as an instability response, NOT accommodated: the clock stays at a
    # proper trot period and the air-time reward (target = period*(1-duty)) pulls the policy toward
    # it. Overridable from the YAML env.gait section (single source).
    gait_period      = 0.55   # slow cadence for heavy legs (see YAML env.gait; 0.38->0.55, 1.8 Hz)
    gait_period_min  = 0.40
    gait_period_slope = 0.10
    gait_duty        = 0.5
    rew_swing_drag   = 0.0     # framework hardening: penalize a foot loaded during its SWING phase (=drag/skate).
                               # DEFAULT 0 = exact no-op (Taili already lifts 9-19cm). Heavier robots that slide
                               # (B2 swing_z鈮?) set a negative weight to force feet off the ground. Validate on both.
    rew_gait_enforce = 0.0     # HARD gait-schedule penalty (both ways: drag during swing OR lift during stance).
                               # DEFAULT 0 = no-op. Negative weight pushes gait_match past the ~0.93 soft ceiling
                               # toward 0.95. Relaxed on rough terrain (so the policy can break the clock to climb).
    dr_unlock_terrain = 0.0    # DR-defer: don't start stacking DR until mean terrain level >= this (reduce phi2 load).
                               # DEFAULT 0 = DR unlocks at phi2 as before.

    # TERRAIN gait relaxation: scale the trot-clock reward DOWN as terrain roughness rises (terrain_ctx[2]),
    # so the policy can break the rigid flat timing to climb. gait_match dropping on hard terrain is correct.
    gait_rough_scale = 0.08     # roughness at which the gait-phase reward fully relaxes to 0

    # 鈹€鈹€ GAIT QUALITY (new 鈥?directly fixes slip=54-85% observed in physdiag) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    # Stance slip: penalize feet sliding during contact (fixes physdiag slip=54-85%)
    rew_stance_slip  = -0.8     # AMP-CORE AUDIT (restored -0.4->-0.8): the discriminator is BLIND to foot velocity
                                # (amp_obs has foot POSITION, not velocity), so it CANNOT see ground slip. This is
                                # the only slip signal 鈥?restore it strong (reference stance foot is planted=0 slip).
    # Swing direction: swing foot should move along the commanded direction
    rew_swing_dir    = 0.0      # REDESIGN 1.0->0.0 REMOVED: this LINEAR, UNCAPPED term was the overspeed engine
                                # (faster swing = more reward, no brake). The reference foot trajectory already
                                # specifies swing direction/magnitude; with velocity-in-obs it is redundant + harmful.
    # Landing deceleration: brake the foot's horizontal speed in the touchdown phase (clean plant, no drift/slip)
    rew_land_decel   = -1.5     # per m/s of foot horizontal speed during late swing
    # Clearance: swing foot should reach target height above ground.
    rew_clearance    = -1.5     # RESTORED to agent_65000: weakening this let the foot OVER-LIFT (physeval: 7.4cm vs
                                # the 4cm reference, > spec-3 of 3-5cm). This pulls the foot DOWN toward the target so
                                # it matches the reference's low, efficient lift. (r_imitate alone under-enforced it.)
    base_clearance   = 0.11     # FINAL STRATEGY 0708: 0.09->0.11 抬脚太低是打滑/脚步重/地形差的上游根因 — raise swing target so foot swings clean above ground (source of the causal chain)
    # ROUGHNESS-GATED phi2 clearance (calibrated 2026-06-20 from live terrain<->rough: flat rough~0.007, the
    # terrain-4.8 plateau rough~0.016-0.021). The FIRST attempt (uniform -8 ramp) backfired: it forced wasteful 9cm
    # high-stepping on FLAT terrain -> velocity tracking collapsed -> regression guard pulled terrain DOWN. Fix:
    # gate BOTH weight and target by per-env terrain roughness, so flat stays efficient (-1.5, 7cm) and only
    # rough/boxes/stairs tiles demand high lift. gate cg = clr_gate(phi2 time fade) x rough_factor; in phi0/phi1
    # cg=0 -> EXACTLY the 245k base (-1.5, 7cm), so bootstrap is untouched. Calibration ALSO showed the old target
    # was too low (8cm can't clear 13cm L5 boxes) -> raise target up to +6cm on full-rough (-> 13cm, clears boxes).
    rew_clearance_heavy   = -8.0    # weight on full-rough terrain (forces foot high enough to clear obstacles)
    clr_rough_flat        = 0.008   # rough(m) at/below this = treated as flat -> no clearance boost
    clr_rough_span        = 0.012   # rough range over which clearance ramps to full (0.008 -> 0.020)
    clr_rough_bonus_max   = 0.06    # m 鈥?max target boost on full-rough (target 0.07 -> 0.13, clears L5 boxes/stairs)
    clearance_ramp_intervals = 25   # clr_gate phi2 time fade-in (avoids a sudden reward shock at phi2 entry)
    # AMP reference clearance also rises with terrain roughness (so the discriminator teaches high-lift climbing,
    # not just the task reward). 0.30 * roughness(0..0.3) -> up to +9cm lift in the reference on stair terrain.
    ref_clearance_rough_gain = 0.30
    # (A) IMITATION reward 鈥?DELETED (P6, alignment decision 2): r_imitate was the phase-locked DeepMimic tracker, ONE
    # of the "triple reference lock" (imit + r_gait + AMP style) that over-constrained the policy. With AMP now STYLE-
    # ONLY (P1) holding posture/foot-placement and r_gait holding contact timing, this exact joint tracker is redundant
    # and over-constraining. Set to 0 -> the compute skips it (r_imitate=0). The phi1->phi2 gate no longer reads
    # style_err (it came from here) 鈥?gates on min_prog (now meaningful via two-sided tracking). See env phase gate.
    # 0706 D3s (REINSTATED, TURN-GATED): P6 deleted this global joint tracker as over-constraining, on the
    # premise that AMP-style + r_gait cover foot placement. That has a DEFINITIONAL hole: AMP's reference
    # distribution is forward-dense so its style gradient is weak in the turning region, and r_gait constrains
    # contact TIMING not foot PLACEMENT. Result: for turning commands, NOTHING anchors the swing foot to the
    # (correct, command-conditional) reference → the "脚前勾" forward-hook and the A2 yaw p90 tail. Fix: re-enable
    # the tracker but scale its weight by the command's TURN fraction (see rew loop) so it is ~0 for pure forward
    # (keeps P6's freedom where AMP already works) and full for pure yaw/lateral (fills the gap). Not a proportion
    # band-aid — it re-introduces the missing direct kinematic anchor exactly where the reward field was blind.
    rew_imitate      = 1.5
    imitate_sigma    = 0.5
    # RANK-2A (0707): forward imitation FLOOR — pure-forward buckets get a modest (0.3) trajectory anchor for
    # gait SHAPE (B1 touchdown vz / B2 slip / B4 forward duty symmetry, the failing forward gait-quality gates)
    # instead of turn_frac→0 leaving forward unanchored. Speed-covariant ref constrains shape not net speed →
    # A1-forward protected. 0.0 = off (legacy); 0.3 = modest anchor bounding the A4 tradeoff.
    imitate_fwd_floor = 0.5     # FINAL STRATEGY 0708: 0.3→0.5 anchor FORWARD swing too (fwd was the worst slip/landing bucket) — bounds the A4 tradeoff, still speed-covariant so A1-fwd protected
    # LIVE TP joint-imitation (RANK-1, 0707): rew_imitate/imitate_sigma ABOVE are DEAD in the TerrainPerceiver
    # subclass (TailiBlindTPEnv OVERRIDES _get_rewards and never calls super), so they never reach the live
    # policy — the crude skating gait was the reward optimum. These NEW knobs drive a dense POSITIVE joint-
    # imitation attractor added directly in blind_tp_env._get_rewards: reward a MATCH to the clean analytic
    # reference joints each step, which reproduces the reference's soft sin^2 landing (B1), no-slip retraction
    # (B2) and L/R symmetry (B4/C) BY CONSTRUCTION. Positive → enters the positive-reward sum, immune to the
    # penalty-budget homeostat. Kept SEPARATE from the parent's dead rew_imitate/imitate_sigma so tuning one
    # never disturbs the other. w_imitate_live=0.0 disables it (falls back to the pre-fix behavior).
    w_imitate_live     = 1.6    # FINAL STRATEGY 0708: 1.0→1.6 master lever — reproduces no-slip retraction(B2/打滑) + soft landing(B1/脚步重) + L/R symmetry(B4/不对称) BY CONSTRUCTION. Safe on a WALKING base(agent_30000), not fresh.  [prev note: 0707 1.5→1.0 — 1.5 destabilized the FRESH policy, fall_rate pinned 10%
                                # >5% phase gate; a GENTLER attractor lets it learn to walk stably while cleaning)
    imitate_sigma_live = 0.09   # FINAL STRATEGY 0708: 0.12→0.09 tighter kernel — pull the swing CLOSER to the clean reference (sharper no-slip/soft-landing enforcement)
    w_swing_dir = 0.0        # 脚在空中往滑 fix: positive mid-swing command-directional attractor (foot fore-aft vel -> commanded vfx=vx-wz*foot_y0)
    swing_dir_margin = 0.15
    swing_dir_sigma  = 0.08
    rew_climb = 0.0
    climb_slip_gate = 0.18
    climb_slip_soft_span = 0.18
    climb_vz_cap = 0.8
    rew_terrain_up = 0.0
    rew_terrain_down = 0.0
    terrain_transition_eps = 0.05
    terrain_transition_span = 0.08
    terrain_up_vz_cap = 0.6
    terrain_down_vz_cap = 0.5
    terrain_down_vz_target = 0.18
    terrain_curriculum_height_gain = 0.08
    terrain_curriculum_height_loss = 0.08
    w_settle_brake = 2.0    # A3/C settle: reward per-step deceleration when cmd~0 & still-moving (breaks the coast; settle 1.54s->target<1.0s)   # per-step joint-error scale (0707: 0.12→0.18 — a broader kernel gives a gradient
                                # even when far from the reference, a smoother/gentler pull than the tight 0.12)
    # (B) WITHDRAWN: forcing a fixed 30cm lift on stair tiles violates spec-3 (3-5cm) and needs to KNOW the terrain
    # type (perception). Replaced by COLLISION-TRIGGERED lift: keep 3-5cm nominal; when the swing foot hits a riser
    # (foot in contact during its swing phase) the rew_swing_drag penalty + terrain latent z drive a higher,
    # ADAPTIVE lift only where needed. (The terrain latent anticipates; swing_drag reacts.) So this stays off.
    discrete_clearance = False
    # Fore-aft stance widen baked into the AMP reference (front feet +stance_dx, rear -stance_dx) so the
    # discriminator target == the RSI clips (which use it). 0.0 = narrow base stance; blind config overrides to 0.05.
    stance_dx        = 0.0

    # 鈹€鈹€ STAND / POSTURE 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    rew_stand_pose   = 5.0
    stand_height     = 0.52
    rew_base_height  = -9.0     # -12->-9 (0708 first-principles): -12 fought physics(crouch-for-stability on unseen bumps); let high clr do the work
    rew_hip_neutral  = -8.0     # RESTORED to agent_65000 (the proven straight walker): a weak hip penalty let the
                                # policy pigeon-toe. physeval(agent_135000) confirmed the regression. AMP/discriminator
                                # does NOT enforce this tightly enough (aggregate match hides per-leg hip drift).
    hip_neutral_lat_scale = 0.25
    rew_offaxis_vel  = -3.5     # RESTORED to agent_65000: a weak off-axis penalty let the policy DRIFT/veer ("鏂滅潃璧?
                                # on a fwd command). The discriminator's achieved-vel is a 64-env-AVERAGE signal 鈥?it
                                # does NOT correct per-env lateral drift; this explicit penalty does. (Was over-cut.)
    rew_overshoot    = -2.0     # RE-ENABLED (with capped-progress r_lin): the overspeed BRAKE. amp_obs no longer
    rew_wrong_dir    = 0.0      # Default no-op; Taili blind config enables this when diagnostics show opposite-direction motion.
    rew_underspeed   = 0.0      # Default no-op; generic banded underspeed penalty outside A1 tolerance.
    rew_backward_underspeed = 0.0  # Directional aux: only for backward vx commands, avoids damaging forward retention.
    rew_backward_wrong_dir  = 0.0  # Directional aux: only for backward vx commands moving forward.
    rew_lateral_underspeed = 0.0   # Directional aux: only for near-pure lateral commands.
    directional_aux_backward_only = True
    speed_tol_abs    = 0.10     # A1 absolute tolerance used by banded underspeed reward and diagnostics.
    speed_tol_rel    = 0.15     # A1 relative tolerance used by banded underspeed reward and diagnostics.
                                # carries velocity (P1, AMP is style-only), so the discriminator does NOT brake
                                # overspeed 鈥?this quadratic penalty on speed-beyond-command does. Paired with the
                                # capped-progress reward it gives the two-sided effect (move up to cmd, don't exceed)
                                # without the bootstrap-killing standing-reward of a pure exp tracker.

    # 鈹€鈹€ COMMAND DISTRIBUTION (balanced 鈥?all 4 directions equal during early training) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    cmd_lin_x        = (-1.0,  1.0)
    cmd_lin_y        = (-0.6,  0.6)
    cmd_ang_z        = (-1.0,  1.0)
    cmd_resample_s   = 5.0
    # COMMAND BUFFER: self.commands low-passes toward the sampled target (first-order, per control step at 50Hz).
    # alpha=0.93 -> ~0.3s ramp -> a NATURAL decel on stop / smooth accel-turn, no hard brake (spec-5 "natural").
    # Steady-state unaffected (commands==target after the ramp), so velocity-tracking accuracy is preserved.
    cmd_smooth_alpha = 0.93
    stand_prob       = 0.25    # P5: 0.10->0.25. physeval: 绔嬫/鍋滆溅閲嶄激 (0.19 m/s creep, can't settle). Stand was
                               # barely practiced; 0.25 gives far more stand exposure AND more walk->stop transitions
                               # (an env resampling from a moving cmd to 0 = a deceleration drill) -> learns clean stop.
    cmd_prob_fwd     = 0.25    # equal 4-way split during bootstrap phase
    cmd_prob_back    = 0.25
    cmd_prob_lat     = 0.25
    cmd_prob_yaw     = 0.25
    # Per-direction speed ranges (lower bound, curriculum starting upper bound)
    cmd_fwd_range    = (0.30, 0.70)
    cmd_back_range   = (0.20, 0.45)
    cmd_lat_range    = (0.15, 0.35)
    cmd_yaw_range    = (0.25, 0.80)

    # 鈹€鈹€ PER-DIRECTION VELOCITY CURRICULUM 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    # Each direction has its OWN progress tracker and speed ceiling 鈥?no direction carries another.
    cmd_fwd_max      = 2.5     # fwd ceiling target (TCP-96 safely delivers ~3 m/s)
    cmd_back_max     = 0.8     # back ceiling target
    cmd_lat_max      = 0.7     # lateral ceiling target
    cmd_yaw_max      = 1.2     # yaw ceiling target (rad/s)
    vel_cur_up       = 0.85    # raise ceiling when direction progress >= this
    vel_cur_down     = 0.55    # lower ceiling when direction progress <= this
    vel_cur_step     = 0.04    # speed change per log interval
    vel_terrain_decouple = True

    # 鈹€鈹€ DOMAIN RANDOMIZATION (CONDITIONAL 鈥?see _dr_level in env) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    # DR starts at level 0 (no DR except very weak push). Levels gate on demonstrated capability.
    # Level 0 鈫?1: upright>0.98, gait_match>0.85, all 4 direction progress>0.65 for 5+ intervals
    # Level 1 鈫?2: all 4 direction progress>0.75, fall_rate low, terrain not collapsing
    # SYMMETRY: mirror-augment the PPO batch (swap L<->R legs etc.) to force a left-right symmetric gait.
    sym_augment      = True

    # 鈹€鈹€ UNIFIED CURRICULUM PHASE (phi 0->1->2 coordinates penalties / velocity / terrain / DR) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    # phi0 bootstrap locomotion (no quality penalties, flat, slow, no DR) -> phi1 clean gait (penalties fade
    # in) -> phi2 scale speed + terrain + DR ALL IN PARALLEL (each on its own capability gate). DR is DECOUPLED
    # from terrain level (was phi3 = terrain>=6): a blind real robot needs sim2real DR regardless of whether
    # terrain reaches 6. phi3 remains only a vestigial "terrain>=6 reached" marker (adds no new behavior).
    # Each gate is sustained over phase_intervals.
    phase_intervals       = 5      # consecutive log intervals meeting a gate before advancing
    penalty_ramp_intervals = 25    # phi1: quality penalties fade 0->1 over this many intervals (avoid critic shock)
    phase_gate_prog_0     = 0.60   # phi0->1: min over 4 directions of tracking progress (locomotion exists)
    # Phase gates use command progress plus de-clocked gait quality (slip/diagonal/duty/air).
    # gait_match remains a readout only; it is not a curriculum gate.
    phase_gate_prog_1     = 0.70   # phi1->2: lowered 0.75->0.70 鈥?the last run STALLED ~120k steps at min_prog 0.70
                                   # (fwd the laggard, never reached terrain/stairs). Restored offaxis should also
                                   # lift fwd prog (less drift = more forward speed); 0.70 ensures it reaches phi2.
    phase_gate_slip_1     = 0.20   # (legacy 鈥?no longer gates)
    phase_gate_terrain_2  = 6.0    # phi2->3: mean terrain level reached
    phase_gate_fall_2     = 0.05   # phi2->3: fall rate below this
    phase_gate_prog_2     = 0.50   # phi2->3: min-direction progress still healthy on terrain
    regress_fall          = 0.10   # phi2 regression guard: pause terrain/speed if fall rate exceeds this
    regress_prog          = 0.40   # phi2 regression guard: pause if min-direction progress drops below this

    dr_enable        = True
    dr_start_level   = 0    # gate DR up from 0 (set >0 to skip levels). Resume resets to this each run.
    # Level 0 (no DR 鈥?bootstrap clean gait)
    dr_push_interval_s_0 = 0.0
    dr_push_vel_0        = 0.0
    # Level 1 (light)
    dr_push_interval_s_1 = 30.0
    dr_push_vel_1        = 0.2
    dr_mass_range_1      = (-1.0, 2.0)
    dr_stiffness_scale_1 = (0.9,  1.1)
    dr_damping_scale_1   = (0.85, 1.15)
    # Level 2 (moderate sim-to-real)
    dr_push_interval_s_2 = 20.0
    dr_push_vel_2        = 0.5
    dr_mass_range_2      = (-3.0, 6.0)
    dr_stiffness_scale_2 = (0.8,  1.2)
    dr_damping_scale_2   = (0.7,  1.3)
    # Level 3 (COMPREHENSIVE 鈥?deployment-grade sim2real envelope; goal: break past this level)
    dr_push_interval_s_3 = 15.0
    dr_push_vel_3        = 0.8
    dr_push_ang_scale    = 0.8     # yaw angular kick = this x linear push_vel (rad/s) -> rotational push recovery
    dr_mass_range_3      = (-5.0, 20.0)    # up to +10 kg payload (deployment target)
    dr_stiffness_scale_3 = (0.6,  1.4)
    dr_damping_scale_3   = (0.6,  1.4)
    # 鈫?extra channels ACTIVE ONLY at level >= 3 (the full menu)
    dr_full_start_level  = 1               # ROOT-4 fix: DR level at which friction/CoM/IMU turn on (was hard-coded 3=never)
    dr_friction_range_1  = (0.7, 1.3)      # friction ladder: gentle -> full E5 (0.4-1.4) by level 3
    dr_friction_range_2  = (0.5, 1.4)
    dr_com_offset_1      = 0.02            # CoM-offset ladder (m)
    dr_com_offset_2      = 0.035
    dr_imu_gyro_bias_1   = 0.02            # gyro-bias ladder (rad/s)
    dr_imu_gyro_bias_2   = 0.035
    dr_imu_grav_bias_1   = 0.015           # gravity-tilt ladder
    dr_imu_grav_bias_2   = 0.022
    dr_friction_range_3  = (0.4, 1.4)      # foot/ground friction (spec E5 envelope)
    dr_com_offset_3      = 0.05            # 卤 m base CoM shift in x,y (payload / mounting asymmetry)
    dr_imu_gyro_bias_3   = 0.05            # 卤 rad/s constant gyro bias (IMU miscalibration, per episode)
    dr_imu_grav_bias_3   = 0.03            # 卤 constant gravity-vector tilt error (accelerometer bias)
    # DR level gate thresholds (rise with level 鈥?stronger gait required to harden further)
    dr_gate_progress     = 0.65   # 0 -> 1
    dr_gate_progress_l2  = 0.75   # 1 -> 2
    dr_gate_progress_l3  = 0.80   # 2 -> 3 (stronger)
    dr_gate_intervals    = 5      # consecutive intervals above gate before level-up

    # OBSERVATION NOISE (applied to blind policy obs, not the privileged extras)
    obs_noise_jpos    = 0.01
    obs_noise_jvel    = 0.25
    obs_noise_angvel  = 0.05
    obs_noise_gravity = 0.05

    # CONTROL DELAY: 1-step (~20ms) to simulate real robot compute + comm latency
    action_delay_steps = 1

    episode_length_s = 20.0
    decimation       = 4
    dt               = 1 / 200

    # 鈹€鈹€ OBSERVATION SPACES 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    # ACTOR (blind): motor+IMU+history only 鈥?what gets deployed on the real robot.
    #   current_blind(53) = ang_vel(3)+gravity(3)+cmd(3)+jpos(12)+jvel(12)+last_act(12)+gait(8)
    #   history(420)      = 42 脳 10 steps of proprio
    #   BLIND_OBS_DIM     = 53 + 420 = 473   (see blind_policy.py)
    #
    # ENV observation_space = 670 (FULL PRIVILEGED 鈥?what skrl sees for ALL networks):
    #   blind(473) + lin_vel(3) + height_scan(187) + terrain_ctx(3) + foot_contact(4) = 670
    #
    # The BlindGaussianPolicy model explicitly slices [:, :473] 鈥?critic sees all 670 for free.
    obs_history_len  = 10         # H=10 (200ms): detects stair impacts + good velocity estimation
    obs_history_dim  = 42         # per-step: jpos(12)+jvel(12)+angvel(3)+gravity(3)+lastact(12)
    n_height_scan    = 17 * 11    # GridPatternCfg(resolution=0.1, size=(1.6,1.0)) 鈫?187 rays
    observation_space = (53 + obs_history_dim * obs_history_len   # blind part: 473
                         + 3 + n_height_scan + 3 + 4)             # privileged extra: 197 鈫?total 670
    action_space     = 12
    state_space      = 0          # not using skrl state_space mechanism; model slices internally

    # AMP obs = STYLE ONLY: base kinematic style (43) + terrain_ctx (3) = 46.
    # base 43 = jp12+jv12+bh1+tn6+foot_rel12. Velocity and command are EXCLUDED 鈥?the discriminator judges pure
    # kinematic style conditioned on terrain_ctx ("style | terrain"); speed/command tracking is owned by the
    # two-sided tracking reward, NOT AMP. (The old achieved-velocity/command channels made AMP half-do speed 鈥?    # weakly: physeval showed the policy still overspeeds 鈥?and penalized real slip as off-style. Removed.)
    terrain_ctx_dim  = 3          # [fore_slope, lat_slope, roughness] 鈥?critic/AMP only, not actor
    command_ctx_dim  = 3          # (kept for reference; no longer part of amp_observation_space)
    num_amp_observations = 2
    amp_observation_space = 43 + terrain_ctx_dim  # = 46 (pure style + terrain_ctx; velocity & command removed)
    # AMP STRIDE WINDOW (TAILI_AMP_STRIDE=1 only): make the style window span ~one gait period (~400ms)
    # instead of 2 adjacent frames. When the flag is set the env overrides num_amp_observations->amp_stride_frames
    # and subsamples the AMP obs ring by amp_frame_stride (~4 steps ~80ms) on both policy and reference sides;
    # the discriminator input auto-scales to amp_observation_space * num_amp_observations. Ignored when unset.
    amp_frame_stride  = 4          # subsample stride in env steps (~80ms at 50Hz)
    amp_stride_frames = 6          # frames in the strided window: (6-1)*4+1 = 21 steps ~= 420ms

    action_scale     = 0.35
    early_termination = True
    termination_height = 0.35
    contact_force_threshold = 10.0
    actuator_stiffness_by_joint = [120.0] * 12
    actuator_damping_by_joint = [10.0] * 12

    # 鈹€鈹€ REFERENCE MOTIONS 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    motion_files = [os.path.join(MOTIONS_DIR, "clips", f"taili_{g}.npz") for g in
                    ["fwd_030", "fwd_060", "fwd_090", "fwd_120", "fwd_160", "fwd_200",
                     "back_030", "back_060",
                     "left_025", "left_045", "right_025", "right_045",
                     "yawl_040", "yawl_080", "yawr_040", "yawr_080",
                     "uphill_fwd", "downhill_fwd", "uphill_yawl", "uphill_yawr",
                     "cross_fwd", "cross_back"]]
    cond_sigma       = 0.30
    slope_sigma      = 0.10
    log_every        = 500
    reference_body   = "base_link"
    foot_body_names  = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    reset_strategy   = "random-start"

    # 鈹€鈹€ SIMULATION 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    sim: SimulationCfg = SimulationCfg(dt=dt, render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**24,
            gpu_total_aggregate_pairs_capacity=2**25,
            gpu_found_lost_aggregate_pairs_capacity=2**25,
            gpu_max_rigid_contact_count=2**24,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**27,
        ))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    robot: ArticulationCfg = TAILI_DOG_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="generator", terrain_generator=TAILI_TERRAINS_CFG,
        max_init_terrain_level=0, collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0),
        debug_vis=False,
    )
    # Height scanner 鈥?always in scene (used for terrain_ctx + critic privileged obs + clearance reward).
    # The ACTOR policy never sees height_scan; it is privileged info for critic/AMP/reward only.
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=False, mesh_prim_paths=["/World/ground"],
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, track_air_time=True)
    terrain_move_up_dist = TAILI_TERRAINS_CFG.size[0] / 2.0
