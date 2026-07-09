# Taili Blind Locomotion — Strategy Decisions (`taili_strategy_decisions.md`)

> **Status.** This is the design-rationale contract for the reward, gates, and curriculum. It is
> the "why" behind `taili_core/taili_reward.py`, `taili_core/taili_curriculum.py`, and the
> `reward:` / `env.curriculum:` / `domain_randomization:` blocks of
> `taili_blind_config.yaml`. Acceptance thresholds live in [`taili_spec.md`](./taili_spec.md);
> this document explains the mechanisms that reach them. Numbered items (`#1…#8`, `3a`, `3b`,
> `φ0…φ4`) are the stable references the code comments cite.

---

## 1. Reward / Gate principles

1. **One reward source.** The training env and the reward-budget backend call the **same**
   `compute_reward_components(inputs, cfg)` (`taili_reward.py`). There is never a second formula;
   the budget suite tests the identical function, so reward behavior is unit-verifiable off-sim.
2. **Per-robot scalar reduction (`3b`).** Ordinary penalties reduce to a per-robot scalar by
   **mean** (per-joint / per-foot) or **fraction** (event class), **never a raw sum**. Raw sums
   make the penalty scale with joint/foot count and swamp the task reward.
3. **Gates gate; penalties shape.** A term is multiplied by the gate(s) that make it meaningful:
   `stable_motion_gate` (posture/collapse), `stand_gate` / `moving_gate` (command regime), and
   `quality_gate` (curriculum penalty ramp). Terms that must survive a brief collapse frame (slip
   window, terminal) are intentionally **not** multiplied by the current-frame gate.
4. **No degenerate optimum.** Every shaping term is checked so that "stop moving" / "shuffle in
   place" / "crawl" is **never** the safer policy. Positive anchors (gait clock, four-foot stand
   contact) are used where a pure penalty would create a stand-still trap.
5. **Bounded shaping.** Kernels are bounded `[0,1]`; quadratic penalties are clamped. Unbounded
   terms would let one channel dominate the KL-controlled update.

---

## 2. Spec → reward mapping

| Spec gate | Reward mechanism (`taili_reward.py`) |
|-----------|--------------------------------------|
| A1 linear tracking | `tracking_lin` (exp-kernel, σ = acceptance band) + `tracking_lin_far` (Laplacian far-field). |
| A2 yaw tracking | `tracking_yaw` (`yaw_cmd_gate` on `|cmd_z|>0.05`) + `tracking_yaw_far`. |
| A3/C stand | `stand` (stillness × default-pose × low-action) + `stand_contact` (four-foot planted) + `stand_far`. |
| A3 anti-drift | `off_axis` (near-zero-command axes only; **not** moving-gated so it also holds the quiet stand). |
| B1 landing | `landing_impact` (touchdown-masked, graded above 0.10 m/s). |
| B2 slip | `stance_slip` (settled-window, graded linear above `slip_free_speed`). |
| B3 clearance | `clearance_under` / `clearance_over` (swing-foot mean, band around target). |
| B4 symmetry | `gait_anchor` + `diagonal_contact` + `duty_balance` (achieved gait, since the equivariant model makes the *policy* symmetric but not the *achieved* gait). |
| D terrain | `terrain_progress` (dot(v, cmd)/|cmd|, floor-gated) + `wrong_dir` pressure. |
| F2 actuator | `torque_margin` (mean, >0.85 util) + `torque_saturation` (fraction of joints at limit). |
| B5/F3 smooth | `action_rate`, `base_vz`, `base_wxy`, `orient`. |
| terminal | `terminal_penalty` (`#4`: true terminal only; timeout is not penalized). |

---

## 3a. Gate & penalty design

- **Two-regime `stable_motion_gate`.** In the gray zone it is a **soft product** of smoothstep
  factors on height, tilt, and support instability, so penalties fade smoothly. At collapse or
  terminal (`h < h_gate_close`, `tilt > tilt_gate_close`, severe body collision, terminal window)
  it hard-zeros. Height is single-source: `h_ok = nominal_base_h − 0.05`,
  `h_gate_close = nominal_base_h − 0.10`.
- **Recovery gradient in the gate-closed region.** Collapse penalties keep a constant slope (not a
  saturated quadratic) so a fallen robot still has a gradient back toward upright.
- **Graded, not capped-quadratic, for skating/impact.** `stance_slip` and `landing_impact` are
  **linear above threshold up to a scale then saturated**. The earlier capped-quadratic had a
  zero-gradient region exactly in the failure regime (feet skating at 0.3–1.0 m/s, landings at
  0.5–1.5 m/s), so PPO had no local signal to improve — the single biggest historical plateau.

## 3b. Weights & phase schedule (φ0–φ4)

- Weights are **strategy-book starting values, then physeval-tuned**; the authoritative live
  values are the `reward:` block of the YAML (the dataclass defaults are only a fallback).
- The **penalty ramp** (`quality_gate`) scales late quality penalties in over
  `penalty_ramp_intervals` up to `penalty_budget_ratio_max`, so early training is not crushed by
  quality penalties before it can walk at all. `w_stance_slip_late`, `w_landing_impact_late`, and
  the `diagonal_contact` / `duty_balance` terms ride this ramp.

---

## 4. Implementation details (#1–#8)

- **#1 Tracking bandwidth = acceptance band.** The exp-kernel σ equals the A1/A2 band
  (`sigma_lin_abs 0.10`, `sigma_yaw 0.15`). Widening σ saturates the kernel below spec precision
  and removes the gradient needed to reach it; far-field pull comes from the separate Laplacian
  `*_far` kernels, so tightening σ does **not** starve the far gradient.
- **#2 Command sampling (φ0–φ1).** All four directions + stand train from step 0 (strategy-book
  φ0), so the command channel carries information from the first update and no direction hardens
  into an attractor. Yaw is an anti-symmetric, hard-to-explore behavior and gets a raised
  probability + a raised floor so gentle-yaw commands are not mis-classified as standing.
- **#4 Terminal ≠ timeout.** `terminal_penalty` fires only when `terminal_reason != "timeout"`.
  Penalizing timeouts would teach the robot to end episodes early.
- **#5 Terrain families (φ2).** `slope / rough / boxes / stairs_up / stairs_down`, per-family,
  per-env adaptive level; `stairs_down` treated as more dangerous; tracking relaxed to controlled
  progress (spec §3.6).
- **#6 Domain randomization (φ3).** `obs_noise` + `known_latency` are **deploy-faithful** and on
  from φ0 (fidelity, not difficulty). Difficulty DR ramps in dimension order
  `friction → mass → com → imu_bias → kp_kd → latency → actuator_mismatch`; Kp/Kd DR requires an
  actuator-sanity prerequisite.
- **#7 Push (φ4).** Body-frame Δv impulse, ≥1.0 m/s equiv., keep-command-in-recovery, ≤1.0 s
  recovery window (E1/E2). In the current v3 runtime push is folded into the DR schedule rather
  than run as a separate terminal phase.
- **#8 Direction pressure.** `off_axis` penalizes motion on near-zero-command axes; `wrong_dir`
  actively penalizes motion **against** the commanded linear/yaw sign. Tracking/progress alone
  only *withhold* reward for wrong-direction motion, leaving a hardened forward attractor
  cost-free under backward/yaw commands.

### Anomaly fixes referenced in code
- **anomaly #2 — far-field shaping.** The `*_far` Laplacian kernels give a nonzero gradient
  everywhere, complementing the acceptance-band Gaussians (which vanish beyond ~2σ), so the policy
  can climb toward the command from a poor initialization.
- **anomaly #5 — touchdown-only landing impact.** `landing_impact` is gated to the actual
  swing→stance transition (`touchdown_mask`), so it shapes landing velocity **without** punishing
  the swing vertical motion walking requires (the every-step `|foot_vz|` proxy caused a
  stand-still trap).

---

## 5. Curriculum schedule (`taili_curriculum.py` ↔ YAML `training_recipe`)

`taili_curriculum.py` documents the conceptual φ0–φ4 ladder (with push as a terminal phase); the
**runtime** ladder is the YAML `training_recipe.phases` (v3: `0` flat single-axis all-dirs →
`1` flat mixed → `2` +terrain/DR → `3` full envelope), with push folded into DR. When the two
differ, the YAML is authoritative for what actually trains; `taili_curriculum.py` carries the
spec targets (A5 envelope, D difficulties, E5 DR ranges) used for internal-consistency tests.

> **Known drift (tracked):** `taili_curriculum.py` constants (`COMMAND_PROPORTIONS_PHI0` yaw 0.20,
> `T_CMD_RANGE_S` 2–4 s) predate the YAML's φ0 yaw-bootstrap (yaw 0.40) and per-env resample
> (3–6 s). Treat `taili_curriculum.py` as the design ladder and the YAML as the live values.
