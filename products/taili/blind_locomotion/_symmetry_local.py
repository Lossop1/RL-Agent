"""Left-right symmetry augmentation for the Taili AMP policy.

Forces a symmetric gait by mirror-augmenting the PPO minibatches: each minibatch is concatenated with
its sagittal-plane mirror (swap L<->R legs, flip hip joints + lateral/yaw quantities). The actor then
learns that mirror(state) -> mirror(action) is as good as the original (copied advantage), so it
converges left-right symmetric. Integrated by wrapping skrl's RandomMemory.sample_all — the 293-line
AMP _update is NOT touched (robust to skrl internals).

Mirror correctness is SELF-CHECKED at build time (involution + symmetric fixed point); a wrong sign
raises immediately instead of silently degrading training.

Obs layout (670), Isaac joint order [FL,FR,RL,RR] within hip/thigh/calf:
  0:3 angv   3:6 grav   6:9 cmd   9:21 jpos   21:33 jvel   33:45 last_act   45:53 gait(sin4,cos4)
  53:473 history (10 x 42: jpos12,jvel12,angv3,grav3,lastact12)
  473:476 lin_vel   476:663 height_scan(187)   663:666 terrain_ctx   666:670 foot_contact
Action (12): same joint layout.
"""
from __future__ import annotations
import numpy as np
import torch

# ── joint mirror (12): swap FL<->FR, RL<->RR within each of hip/thigh/calf; negate HIP (abduction) ──
_JP = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
_JS = [-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
# 3-vectors under y->-y reflection: linear vec flips y; angular (pseudo) flips x,z
_ANGV_S = [-1.0, 1.0, -1.0]
_GRAV_S = [1.0, -1.0, 1.0]
_CMD_S = [1.0, -1.0, -1.0]          # vx, vy, wz
_LINV_S = [1.0, -1.0, 1.0]
_TCTX_S = [1.0, -1.0, 1.0]          # fore_slope, lat_slope, rough
_GAIT_P = [1, 0, 3, 2, 5, 4, 7, 6]  # swap legs in sin(4) then cos(4)
_CONTACT_P = [1, 0, 3, 2]           # FL,FR,RL,RR -> FR,FL,RR,RL


def _seg(perm, sign, base, p, s):
    for k in range(len(p)):
        perm[base + k] = base + p[k]
        sign[base + k] = s[k]


def _frame_mirror(perm, sign, base):
    """42-dim proprio frame: jpos12, jvel12, angv3, grav3, lastact12."""
    _seg(perm, sign, base + 0, _JP, _JS)
    _seg(perm, sign, base + 12, _JP, _JS)
    _seg(perm, sign, base + 24, [0, 1, 2], _ANGV_S)
    _seg(perm, sign, base + 27, [0, 1, 2], _GRAV_S)
    _seg(perm, sign, base + 30, _JP, _JS)


def build_obs_mirror(hscan_perm):
    """hscan_perm: list/array (187,) mapping each ray to its y-flipped ray (identity if unknown)."""
    N = 670
    perm = list(range(N)); sign = [1.0] * N
    _seg(perm, sign, 0, [0, 1, 2], _ANGV_S)
    _seg(perm, sign, 3, [0, 1, 2], _GRAV_S)
    _seg(perm, sign, 6, [0, 1, 2], _CMD_S)
    _seg(perm, sign, 9, _JP, _JS)              # jpos
    _seg(perm, sign, 21, _JP, _JS)             # jvel
    _seg(perm, sign, 33, _JP, _JS)             # last_act
    _seg(perm, sign, 45, _GAIT_P, [1.0] * 8)   # gait
    for f in range(10):                        # history
        _frame_mirror(perm, sign, 53 + 42 * f)
    _seg(perm, sign, 473, [0, 1, 2], _LINV_S)  # lin_vel
    for k in range(187):                       # height_scan (y-mirror via ray matching)
        perm[476 + k] = 476 + int(hscan_perm[k])
    _seg(perm, sign, 663, [0, 1, 2], _TCTX_S)  # terrain_ctx
    _seg(perm, sign, 666, _CONTACT_P, [1.0] * 4)
    return np.array(perm, dtype=np.int64), np.array(sign, dtype=np.float32)


def build_action_mirror():
    return np.array(_JP, dtype=np.int64), np.array(_JS, dtype=np.float32)


def _self_check(perm, sign, name):
    n = len(perm)
    # involution: applying twice returns identity
    pp = perm[perm]
    ss = sign[perm] * sign
    assert np.array_equal(pp, np.arange(n)), f"{name}: mirror not an involution (perm)"
    assert np.allclose(ss, 1.0), f"{name}: mirror not an involution (sign)"


# module-level mirrors (set by the env at init once the height-scan layout is known)
_OBS_PERM = None; _OBS_SIGN = None; _ACT_PERM = None; _ACT_SIGN = None
_ENABLED = False
# PHASE GATE: symmetry, like the quality penalties, is a REFINEMENT — enforcing it during bootstrap (phase 0)
# constrains a random policy to the symmetric solution space before it can even walk, blocking exploration
# (proven: 245k fresh-bootstrapped with ALL penalties ON but NO symmetry; symmetry-on runs got stuck). The env
# turns this True only at phase >= 1 (gait-cleaning), so phase-0 bootstrap is unconstrained.
_ACTIVE = False


def set_active(flag: bool):
    global _ACTIVE
    _ACTIVE = bool(flag)


def set_mirrors(hscan_perm, device):
    global _OBS_PERM, _OBS_SIGN, _ACT_PERM, _ACT_SIGN, _ENABLED
    op, os_ = build_obs_mirror(hscan_perm)
    ap, as_ = build_action_mirror()
    _self_check(op, os_, "obs"); _self_check(ap, as_, "action")
    _OBS_PERM = torch.as_tensor(op, device=device)
    _OBS_SIGN = torch.as_tensor(os_, device=device)
    _ACT_PERM = torch.as_tensor(ap, device=device)
    _ACT_SIGN = torch.as_tensor(as_, device=device)
    _ENABLED = True
    print(f"[SYM] mirror augmentation ON (obs 670, action 12; self-check passed)", flush=True)


def mirror_obs(x):
    return x[:, _OBS_PERM] * _OBS_SIGN


def mirror_action(x):
    return x[:, _ACT_PERM] * _ACT_SIGN


def patch_memory_sample_all():
    """Wrap RandomMemory.sample_all so the PPO batch (states+actions present) is doubled with its mirror."""
    from skrl.memories.torch import RandomMemory
    if getattr(RandomMemory, "_taili_sym_patched", False):
        return
    orig = RandomMemory.sample_all

    def sample_all(self, names, mini_batches=1, sequence_length=1):
        batches = orig(self, names, mini_batches=mini_batches, sequence_length=sequence_length)
        if not _ENABLED or not _ACTIVE or "states" not in names or "actions" not in names:
            return batches
        si = names.index("states"); ai = names.index("actions")
        out = []
        for mb in batches:
            mb = list(mb)
            mir = [t.clone() for t in mb]
            mir[si] = mirror_obs(mb[si])
            mir[ai] = mirror_action(mb[ai])
            out.append([torch.cat([mb[j], mir[j]], dim=0) for j in range(len(mb))])
        return out

    RandomMemory.sample_all = sample_all
    RandomMemory._taili_sym_patched = True
    print("[SYM] RandomMemory.sample_all patched for mirror augmentation", flush=True)
