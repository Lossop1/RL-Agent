"""Generate Taili reference motions for MULTIPLE gaits/directions + terrain variants.

Generalizes gen_taili_trot.py (kinematic, synthetic — no hardware data). Reuses the proven
2-link sagittal IK + FK assembly; adds:
  * per-leg ground-velocity (vx, vy, wz) -> per-leg stride vector  (covers fwd/back/lateral/yaw)
  * decoupled hip-abduction IK for lateral foot offset (theta_hip = atan2(-fy,-fz); remaining
    in-plane target solved by the existing 2-link ik_foot; validated by full FK roundtrip)
  * base world trajectory with yaw  Tb = H(Rz(wz*t), integral of R(yaw)·[vx,vy,0] + [0,0,base_z])
  * slope clips with pitched/rolled base alignment
  * finite-difference base angular velocity (no longer hardcoded 0)

Output: one npz per gait, AMP format identical to gen_taili_trot.py
  keys: fps, dof_names, body_names, dof_positions(N,12), dof_velocities,
        body_positions(N,17,3), body_rotations(N,17,4 wxyz),
        body_linear_velocities, body_angular_velocities, [terrain_type]
  dof order  [FL,FR,RL,RR] x [hip,thigh,calf];  body order base_link + per-leg[hip,thigh,calf,foot]

Leg geometry from URDF: thigh L1=0.36385 (axis Y), calf->foot (0.0253,0,-0.318) (axis Y), hip axis X.
"""
import argparse
import os
from dataclasses import dataclass, field

import numpy as np

# ---------------- leg geometry (sagittal plane, x=fwd z=up, thigh joint at origin) -------------
L1 = 0.36385                                    # thigh length (URDF FL_calf_joint z)
FOOT_OFF = np.array([0.0252765, 0.0, -0.3179635])   # calf->foot (URDF FL_foot_joint xyz, exact)


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def H(R, p):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.asarray(p, dtype=float); return M


def trans(p):
    return H(np.eye(3), p)


def fk_foot(theta_t, theta_c):
    """foot position relative to thigh(hip) joint, in leg sagittal frame (y≈0)."""
    calf_joint = Ry(theta_t) @ np.array([0, 0, -L1])
    foot = calf_joint + Ry(theta_t) @ Ry(theta_c) @ FOOT_OFF
    return foot


def ik_foot(fx, fz, seed=(0.6, -1.2)):
    """numerical 2-link IK: desired foot (fx,fz) rel hip -> (theta_t, theta_c)."""
    th = np.array(seed, dtype=float)
    for _ in range(60):
        f = fk_foot(th[0], th[1]); err = np.array([fx - f[0], fz - f[2]])
        if np.linalg.norm(err) < 1e-6:
            break
        J = np.zeros((2, 2)); d = 1e-5
        for k in range(2):
            tp = th.copy(); tp[k] += d; fp = fk_foot(tp[0], tp[1])
            J[0, k] = (fp[0] - f[0]) / d; J[1, k] = (fp[2] - f[2]) / d
        th = th + np.linalg.solve(J, err)
    return th


def ik_foot_3dof(fx, fy, fz, seed=(0.0, 0.6, -1.2)):
    """Decoupled 3-DOF IK: hip Rx abduction for the lateral (y) offset, then 2-link sagittal.

    theta_hip rotates the leg about body-X. A foot the 2-link chain places at in-plane
    (fx, 0, -r) maps, after Rx(theta_hip), to (fx, r*sin th, -r*cos th). Match (fy, fz):
        theta_hip = atan2(fy, -fz) ;  r = sqrt(fy**2 + fz**2)  ;  solve 2-link for (fx, -r)
    Returns (theta_hip, theta_thigh, theta_calf). FK roundtrip validated by caller.
    """
    th_h = np.arctan2(fy, -fz)
    r = np.hypot(fy, fz)
    th_t, th_c = ik_foot(fx, -r, seed=seed[1:])
    return np.array([th_h, th_t, th_c])


def fk_foot_3dof(th_h, th_t, th_c):
    """full-leg foot position rel hip joint (incl hip Rx), for IK validation."""
    return Rx(th_h) @ fk_foot(th_t, th_c)


# ---------------- nominal stance ----------------
_nom = fk_foot(0.6, -1.2)
X0 = _nom[0]          # nominal foot x rel hip
H0 = -_nom[2]         # nominal thigh->foot vertical (FK at nominal joints)
BASE_Z = 0.56         # taller nominal standing base height for a more upright flat/slope style
                      # FK feet-on-ground -> stance foot link origin sits ~0.014 (foot sphere radius)

LEGS = ['FL', 'FR', 'RL', 'RR']
# hip offset rel base center: front sx=+1 rear -1; left sy=+1 right -1
SGN = {'FL': (1, 1), 'FR': (1, -1), 'RL': (-1, 1), 'RR': (-1, -1)}
HIPX, HIPY = 0.30414, 0.065
THIGHY, CALFZ = 0.1432, -0.36385
# trot diagonal phase
TROT_PHASE = {'FL': 0.0, 'FR': 0.5, 'RL': 0.5, 'RR': 0.0}
STANCE_DX = 0.0       # fore-aft stance widen: front feet +STANCE_DX, rear -STANCE_DX (wider wheelbase, pitch-stable)


def apply_geometry(geom):
    """Adapter 注入点(§6 参考重生成):用从 URDF 派生的腿几何覆盖模块级常量,再重算 X0/H0。

    框架(IK/相位/对角模式/gait_library 速度结构)不变,只换机器人几何。geom = RefGeom.as_globals()
    或含 {L1,FOOT_OFF,HIPX,HIPY,THIGHY,CALFZ,BASE_Z} 子集的 dict。换 URDF 后调一次再 generate。
    """
    global L1, FOOT_OFF, HIPX, HIPY, THIGHY, CALFZ, BASE_Z, _nom, X0, H0, STANCE_DX
    for k in ("L1", "FOOT_OFF", "HIPX", "HIPY", "THIGHY", "CALFZ", "BASE_Z", "STANCE_DX"):
        if k in geom:
            globals()[k] = geom[k]
    _nom = fk_foot(0.6, -1.2)   # 用新 L1/FOOT_OFF 重算名义站姿
    X0 = _nom[0]
    H0 = -_nom[2]
    return dict(L1=L1, X0=float(X0), H0=float(H0), BASE_Z=BASE_Z, HIPX=HIPX, HIPY=HIPY)


@dataclass
class GaitCfg:
    name: str
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    T: float = 0.6                 # gait period (s)
    cycles: int = 4
    fps: int = 50
    clearance: float = 0.09        # swing lift (m)
    # Layer A continuous-terrain slope: WORLD-frame gradient (rad). height(x,y)=slope_x*x+slope_y*y.
    # body aligns to surface (pitch=fore-aft slope, roll=lateral slope), both rotate with yaw.
    #   uphill_fwd: slope_x>0,vx>0 ; downhill_fwd: slope_x<0,vx>0 ; cross: slope_y!=0
    slope_x: float = 0.0
    slope_y: float = 0.0
    step_rise: float = 0.0         # (unused in Layer A; kept for compat)
    terrain_type: str = "flat"     # flat | slope
    phase: dict = field(default_factory=lambda: dict(TROT_PHASE))


def hip_xy(lg):
    sx, sy = SGN[lg]
    return sx * HIPX, sy * HIPY


def per_leg_stride(cfg, lg):
    """Effective ground velocity at this leg's FOOT contact -> stride vector over a half-period (stance).

    vg = v_cmd + wz x r_foot = (vx - wz*foot_y, vy + wz*foot_x)   (rigid-body velocity at the foot)
    stride = vg * (T/2)                                            (stance covers half the trot period)

    YAW LEVER FIX: r must be the nominal stance-FOOT base-frame position, not the hip. The foot sits ahead of
    the hip by x0 (=X0 ± STANCE_DX, matching foot_traj_rel_hip) and lateral by sy*THIGHY; using the shorter hip
    lever under-rotated commanded yaw ~22%. STANCE_DX is folded into foot_x so the widened stance is consistent.
    """
    sx, sy = SGN[lg]
    hx, hy = hip_xy(lg)
    x0 = X0 + (STANCE_DX if lg in ("FL", "FR") else -STANCE_DX)
    foot_x = hx + x0
    foot_y = hy + sy * THIGHY
    vgx = cfg.vx - cfg.wz * foot_y
    vgy = cfg.vy + cfg.wz * foot_x
    return vgx * cfg.T / 2.0, vgy * cfg.T / 2.0


def foot_traj_rel_hip(cfg, lg, t):
    """foot position rel hip over time t (N,), returns fx,fy,fz arrays. Trot stance/swing."""
    N = len(t)
    sdx, sdy = per_leg_stride(cfg, lg)
    p = ((t / cfg.T) + cfg.phase[lg]) % 1.0
    # STANCE-WIDEN: shift front feet forward + rear feet back by STANCE_DX -> wider fore-aft support polygon
    # (user: "前后足靠得太近" → pitchy). Front=FL/FR get +dx, rear=RL/RR get -dx.
    x0 = X0 + (STANCE_DX if lg in ("FL", "FR") else -STANCE_DX)
    fx = np.zeros(N); fy = np.zeros(N); fz = np.zeros(N)
    for i, pp in enumerate(p):
        if pp < 0.5:                       # stance: planted, moving back rel body
            s = pp / 0.5
            fx[i] = x0 + sdx / 2 - sdx * s
            fy[i] = sdy / 2 - sdy * s
            fz[i] = -H0
        else:                              # swing: lift + return to front foothold (SOFT, NO-SLAM, NO-SCUFF)
            s = (pp - 0.5) / 0.5
            h = -4.0 * s ** 3 + 6.0 * s ** 2 - s    # velocity-matched retraction: h'(0)=h'(1)=-1 -> no touchdown scuff
            fx[i] = x0 - sdx / 2 + sdx * h
            fy[i] = -sdy / 2 + sdy * h
            fz[i] = -H0 + cfg.clearance * np.sin(np.pi * s) ** 2     # sin^2 -> dz/dt=0 at touchdown (no slam)
    return fx, fy, fz


def solve_leg(cfg, lg, t):
    """foot trajectory -> per-leg (hip,thigh,calf) angle arrays, with FK-roundtrip check."""
    fx, fy, fz = foot_traj_rel_hip(cfg, lg, t)
    N = len(t)
    th = np.zeros((N, 3)); seed = (0.0, 0.6, -1.2); maxerr = 0.0
    for i in range(N):
        a = ik_foot_3dof(fx[i], fy[i], fz[i], seed=seed)
        th[i] = a; seed = (a[0], a[1], a[2])
        f = fk_foot_3dof(*a); maxerr = max(maxerr, np.linalg.norm(f - np.array([fx[i], fy[i], fz[i]])))
    return th[:, 0], th[:, 1], th[:, 2], maxerr


def base_world_pose(cfg, t):
    """base world pose over time -> (pos(N,3), yaw(N), s_fore(N), s_lat(N)).

    Integrate body-frame [vx,vy] under yaw. On a planar slope height=slope_x*x+slope_y*y, the base
    rides the surface (z = BASE_Z + slope.xy) and aligns to it: body-frame slope (s_fore,s_lat) =
    rotate world gradient by -yaw, used as pitch (about Y) and roll (about X) in the assembly.
    """
    N = len(t); dt = 1.0 / cfg.fps
    pos = np.zeros((N, 3)); yaw = cfg.wz * t
    s_fore = np.zeros(N); s_lat = np.zeros(N)
    p = np.array([0.0, 0.0])
    for i in range(N):
        if i > 0:
            R = Rz(yaw[i - 1])[:2, :2]
            p = p + R @ np.array([cfg.vx, cfg.vy]) * dt
        pos[i, 0] = p[0]; pos[i, 1] = p[1]
        # world planar slope height + body-frame slope components (rotate gradient by -yaw)
        c, s = np.cos(yaw[i]), np.sin(yaw[i])
        s_fore[i] = cfg.slope_x * c + cfg.slope_y * s
        s_lat[i] = -cfg.slope_x * s + cfg.slope_y * c
        pos[i, 2] = BASE_Z + cfg.slope_x * p[0] + cfg.slope_y * p[1] \
            + cfg.step_rise * (t[i] / cfg.T)
    return pos, yaw, s_fore, s_lat


def rot2quat(R):  # -> wxyz
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2; w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S; y = (R[0, 2] - R[2, 0]) / S; z = (R[1, 0] - R[0, 1]) / S
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            S = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S; x = 0.25 * S; y = (R[0, 1] + R[1, 0]) / S; z = (R[0, 2] + R[2, 0]) / S
        elif i == 1:
            S = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S; x = (R[0, 1] + R[1, 0]) / S; y = 0.25 * S; z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S; x = (R[0, 2] + R[2, 0]) / S; y = (R[1, 2] + R[2, 1]) / S; z = 0.25 * S
    return np.array([w, x, y, z])


def quat_angular_velocity(quats, fps):
    """finite-difference body angular velocity (world frame) from a quaternion sequence (N,4 wxyz)."""
    N = quats.shape[0]; dt = 1.0 / fps; w = np.zeros((N, 3))
    for i in range(1, N):
        q0 = quats[i - 1]; q1 = quats[i]
        # relative rotation q1 * conj(q0)
        w0, x0, y0, z0 = q0; w1, x1, y1, z1 = q1
        cw, cx, cy, cz = w0, -x0, -y0, -z0
        rw = w1 * cw - x1 * cx - y1 * cy - z1 * cz
        rx = w1 * cx + x1 * cw + y1 * cz - z1 * cy
        ry = w1 * cy - x1 * cz + y1 * cw + z1 * cx
        rz = w1 * cz + x1 * cy - y1 * cx + z1 * cw
        ang = 2.0 * np.arccos(np.clip(rw, -1, 1))
        axis = np.array([rx, ry, rz]); n = np.linalg.norm(axis)
        w[i] = (axis / n) * (ang / dt) if n > 1e-8 else 0.0
    if N > 1:
        w[0] = w[1]
    return w


# ---------------- assemble one gait -> npz ----------------
def generate(cfg: GaitCfg, out_dir: str):
    N = int(cfg.T * cfg.fps * cfg.cycles)
    t = np.arange(N) / cfg.fps

    hip_a = {}; thigh_a = {}; calf_a = {}; errs = {}
    for lg in LEGS:
        h, tt, cc, e = solve_leg(cfg, lg, t)
        hip_a[lg], thigh_a[lg], calf_a[lg], errs[lg] = h, tt, cc, e

    # dof arrays  [FL,FR,RL,RR] x [hip,thigh,calf]
    dof_names = [f"{lg}_{j}_joint" for j in ['hip', 'thigh', 'calf'] for lg in LEGS]
    dof_pos = np.zeros((N, 12))
    for k, arr in enumerate([hip_a, thigh_a, calf_a]):
        for li, lg in enumerate(LEGS):
            dof_pos[:, k * 4 + li] = arr[lg]
    dof_vel = np.gradient(dof_pos, 1.0 / cfg.fps, axis=0)

    # body poses
    body_names = ['base_link'] + [f"{lg}_{seg}" for lg in LEGS for seg in ['hip', 'thigh', 'calf', 'foot']]
    B = len(body_names)
    bpos = np.zeros((N, B, 3)); brot = np.zeros((N, B, 4))
    pos, yaw, s_fore, s_lat = base_world_pose(cfg, t)
    for i in range(N):
        # align body to slope surface: Ry(-s_fore) = nose UP when climbing (Ry(+)@fwd points down);
        # Rx(s_lat) raises the uphill (left) side. pitch/roll rotate with yaw.
        Rb = Rz(yaw[i]) @ Ry(-s_fore[i]) @ Rx(s_lat[i])
        Tb = H(Rb, pos[i])
        bpos[i, 0] = Tb[:3, 3]; brot[i, 0] = rot2quat(Tb[:3, :3]); bi = 1
        for lg in LEGS:
            sx, sy = SGN[lg]
            Th = Tb @ trans([sx * HIPX, sy * HIPY, -0.0003]) @ H(Rx(hip_a[lg][i]), [0, 0, 0])
            Tt = Th @ trans([0, sy * THIGHY, 0]) @ H(Ry(thigh_a[lg][i]), [0, 0, 0])
            Tc = Tt @ trans([0, 0, CALFZ]) @ H(Ry(calf_a[lg][i]), [0, 0, 0])
            Tf = Tc @ trans(FOOT_OFF)
            for Tm in [Th, Tt, Tc, Tf]:
                bpos[i, bi] = Tm[:3, 3]; brot[i, bi] = rot2quat(Tm[:3, :3]); bi += 1
    blin = np.gradient(bpos, 1.0 / cfg.fps, axis=0)
    bang = np.zeros((N, B, 3))
    bang[:, 0] = quat_angular_velocity(brot[:, 0], cfg.fps)   # base only (legs approx 0 for AMP)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"taili_{cfg.name}.npz")
    np.savez(out, fps=np.int64(cfg.fps), dof_names=np.array(dof_names), body_names=np.array(body_names),
             dof_positions=dof_pos.astype(np.float32), dof_velocities=dof_vel.astype(np.float32),
             body_positions=bpos.astype(np.float32), body_rotations=brot.astype(np.float32),
             body_linear_velocities=blin.astype(np.float32), body_angular_velocities=bang.astype(np.float32),
             terrain_type=np.array(cfg.terrain_type),
             # condition metadata for Stage-4 condition-matched soft sampling (exp(-||cond-clip||^2/sigma))
             cond_command=np.array([cfg.vx, cfg.vy, cfg.wz], dtype=np.float32),     # commanded vel
             cond_slope=np.array([cfg.slope_x, cfg.slope_y], dtype=np.float32),     # world-frame slope grad
             cond_clearance=np.float32(cfg.clearance),
             cond_step_rise=np.float32(cfg.step_rise))

    # ---- sanity report + asserts ----
    # Diagonal pairing is a GAIT-PHASE property -> measure foot z in the BODY frame (rotate foot-rel-base
    # by inverse base rotation). World-frame z is corrupted by body climb AND pitch/roll (which oscillate
    # on yaw-on-slope clips), so body-frame isolates the true swing phase.
    foot_idx = [body_names.index(f"{lg}_foot") for lg in LEGS]

    def quat_inv_rot(q, v):  # rotate world vector v by inverse of quat q (wxyz) -> body frame
        w, x, y, z = q
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        return R.T @ v
    fw = np.zeros((N, 4))
    for i in range(N):
        for li in range(4):
            fw[i, li] = quat_inv_rot(brot[i, 0], bpos[i, foot_idx[li]] - bpos[i, 0])[2]
    fl, fr, rr = fw[:, 0], fw[:, 1], fw[:, 3]
    diag_flrr = np.corrcoef(fl, rr)[0, 1]; diag_flfr = np.corrcoef(fl, fr)[0, 1]
    lift = (fw.max() - fw.min()) * 100
    maxerr = max(errs.values())
    print(f"[{cfg.name}] frames={N} dur={N/cfg.fps:.1f}s ik_maxerr={maxerr:.2e} "
          f"terrain={cfg.terrain_type}")
    print(f"   base: x {bpos[0,0,0]:+.2f}->{bpos[-1,0,0]:+.2f}  y {bpos[0,0,1]:+.2f}->{bpos[-1,0,1]:+.2f}  "
          f"yaw {yaw[-1]:+.2f}rad  z {bpos[0,0,2]:.2f}->{bpos[-1,0,2]:.2f}")
    print(f"   foot z (body frame): swing={lift:.1f}cm  diag FL-RR={diag_flrr:+.2f} FL-FR={diag_flfr:+.2f}")
    assert maxerr < 1e-3, f"IK roundtrip error too high: {maxerr}"
    assert diag_flrr > 0.5, f"diagonal pair FL-RR not in phase: {diag_flrr}"
    assert diag_flfr < 0.0, f"FL-FR not anti-phase: {diag_flfr}"
    return out


# ---------------- gait library ----------------
# Two-layer design (per research: AMP 2104.02180 / ANYmal-Parkour 2306.14874 / blind-stairs 2402.06143):
#   Layer A = continuous-terrain locomotion (flat + slope), condition [vx,vy,wz,slope_x,slope_y].
# Real ROUGH_TERRAINS_CFG: slope_range (0.0,0.4) rad -> representative 0.25.
SLOPE = 0.25   # representative incline (rad ~14deg), mid-low of (0,0.4) since curriculum starts easy


def gait_library():
    return {
        # --- flat (6 directions) ---
        'fwd':   GaitCfg('fwd',   vx=+0.4),
        'back':  GaitCfg('back',  vx=-0.4),
        'left':  GaitCfg('left',  vy=+0.30),
        'right': GaitCfg('right', vy=-0.30),
        'yawl':  GaitCfg('yawl',  wz=+0.6),
        'yawr':  GaitCfg('yawr',  wz=-0.6),
        # --- slope (continuous; body aligns to surface) ---
        'uphill_fwd':    GaitCfg('uphill_fwd',    vx=+0.35, slope_x=+SLOPE, terrain_type='slope'),
        'downhill_fwd':  GaitCfg('downhill_fwd',  vx=+0.35, slope_x=-SLOPE, terrain_type='slope'),
        'uphill_yawl':   GaitCfg('uphill_yawl',   vx=+0.25, wz=+0.4, slope_x=+SLOPE, terrain_type='slope'),
        'uphill_yawr':   GaitCfg('uphill_yawr',   vx=+0.25, wz=-0.4, slope_x=+SLOPE, terrain_type='slope'),
        # --- cross-slope (asymmetric load, not discrete foothold; fwd/back) ---
        'cross_fwd':     GaitCfg('cross_fwd',     vx=+0.35, slope_y=+SLOPE, terrain_type='slope'),
        'cross_back':    GaitCfg('cross_back',    vx=-0.35, slope_y=+SLOPE, terrain_type='slope'),
    }


def flat_multispeed_library():
    """Multi-SPEED flat clips so the AMP style supports the commanded speed range (single-speed
    references penalize off-speed via joint_vel -> undershoot). cond_command (speed) drives matching."""
    lib = {}
    def add(d, **kw):
        lib[d] = GaitCfg(d, **kw)
    for v in (0.3, 0.6, 0.9):
        add(f"fwd_{int(v*100):03d}", vx=+v)
    # HIGH-SPEED fwd clips toward the 2 m/s target. At constant period the stride would exceed leg reach
    # (~0.72 m), so SHORTEN the period (higher stride frequency) + lift a bit more as speed rises -> the IK
    # stride stays reachable and the foot clears at speed. AMP needs these so the (raised) style reward has
    # an on-template reference for fast gaits instead of penalizing them.
    for v, T, clr in ((1.2, 0.50, 0.10), (1.6, 0.45, 0.11), (2.0, 0.40, 0.12)):
        add(f"fwd_{int(v*100):03d}", vx=+v, T=T, clearance=clr)
    for v in (0.3, 0.6):
        add(f"back_{int(v*100):03d}", vx=-v)
    for v in (0.25, 0.45):
        add(f"left_{int(v*100):03d}", vy=+v)
        add(f"right_{int(v*100):03d}", vy=-v)
    for v in (0.4, 0.8):
        add(f"yawl_{int(v*100):03d}", wz=+v)
        add(f"yawr_{int(v*100):03d}", wz=-v)
    return lib


# clip groups
FLAT6 = ['fwd', 'back', 'left', 'right', 'yawl', 'yawr']
SLOPE6 = ['uphill_fwd', 'downhill_fwd', 'uphill_yawl', 'uphill_yawr', 'cross_fwd', 'cross_back']
FLATMS = list(flat_multispeed_library())   # 13 multi-speed flat clips


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gait", default="layerA", help="name | flat6 | flatms | slope6 | layerA | all")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "clips"))
    args = ap.parse_args()
    lib = {**gait_library(), **flat_multispeed_library()}
    groups = {'flat6': FLAT6, 'flatms': FLATMS, 'slope6': SLOPE6, 'layerA': FLAT6 + SLOPE6,
              'all': FLATMS + SLOPE6}
    names = groups.get(args.gait, [args.gait])
    print(f"nominal: foot x={X0:.3f} stand_h={H0:.3f} base_z={BASE_Z}")
    for nm in names:
        generate(lib[nm], args.out)
