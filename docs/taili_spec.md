# Taili Blind Locomotion — Acceptance Specification (`taili_spec.md`)

> **Status.** This document is the single source of truth for what "Taili passes" means.
> It is the human-readable contract; the machine-enforced contract is
> `autotuner/blind_locomotion/acceptance_score.py` (pure, unit-testable, no sim). Every
> threshold below is the exact value that scorer enforces — when the two disagree, the scorer
> wins and this document is the bug. The scorer is what `physeval_blind.py` /
> `physeval_blind_e.py` feed measured statistics into, and what `acceptance_aggregate.py`
> merges into the final §2 verdict.
>
> Scope: a **blind** quadruped (`Taili`, ~39 kg, 12 actuated joints, weakest joint = thigh at
> 110 N·m) trained with AMP + PPO on IsaacLab/skrl and deployed with **no exteroception**.

---

## §1 — Measurement口径 (evaluation protocol)

All acceptance numbers are measured under the **deployment口径**, not the training口径:

1. **Blind.** The deployed policy sees only proprioception + command + the learned terrain
   latent; no height-scan / camera is available at deploy time. Acceptance is measured on the
   same observation the robot will actually have (`taili_spec §1`).
2. **Mean-action.** Evaluation rolls out the policy **mean action** (no exploration noise) —
   this is what runs on hardware. `physeval_blind.py` sets the policy to mean-action mode.
3. **Off-sim / off-policy scoring.** The simulator measures raw physical quantities
   (velocities, foot states, torques); the pass/fail thresholds live in `acceptance_score.py`
   so the "judge" is verifiable independently of the simulator.
4. **Statistic definitions.** Unless stated otherwise: `median` and `p90` are per-command-bucket
   order statistics of the per-step error; `p95` is the per-touchdown-event order statistic;
   slip / clearance are measured only on **settled** contact frames (touchdown and liftoff
   transition frames are excluded from slip; swing frames only for clearance).
5. **Non-strict bounds.** Every comparison uses the spec's `<=` / `>=` exactly;
   threshold-equal ⇒ **PASS**.

---

## §2 — 最终通过条件 (final acceptance)

A policy **PASSES** the benchmark iff **every hard-gate family is present AND ok** across the
**merged** results of all physeval runs (flat + each terrain + push + DR). A required family
that was **never evaluated** counts as **NOT passed** — coverage is never silently skipped
(`final_verdict()` in `acceptance_score.py`).

- **Hard families (all must pass):**
  `A1, A2, A3, A4, B1, B2, B3, B4, C, D1, D2, D3, D4, E1, E2, E3, E4, E5, F2`
- **Soft families (report-only, never gating):** `A5, B5, F1, F3`

The `[label]` suffix (e.g. `A1[fwd05]`) denotes a per-bucket sub-check; a family passes only if
**all** of its buckets pass.

---

## §3 — Acceptance battery

### §3.1 A — Command tracking & standing

| Gate | Name | Threshold (median **and** p90 unless noted) |
|------|------|---------------------------------------------|
| **A1** | Linear velocity tracking | `|v − cmd| ≤ max(0.10, 0.15·|cmd|)` m/s, for **both** median and p90, in every forward/back/lateral bucket. |
| **A2** | Yaw-rate tracking | `|ω_z − cmd_z| ≤ 0.15` rad/s, for both median and p90, in every yaw bucket. |
| **A3** | Clean stop / settle (立正) | `|v| ≤ 0.05` m/s **and** `|ω_z| ≤ 0.05` rad/s **and** settle time `≤ 1.0` s **and** four-foot duty `≥ 0.95` **and** upright fraction `≥ 0.99`. |
| **A4** | Mixed commands | Each component (vx, vy, ω_z) simultaneously in its A1/A2 band — **no** sacrificing vy/ω_z to serve vx. |
| **A5** *(soft)* | Velocity envelope | Robot tracks up to the envelope: forward **1.5** m/s, backward **0.8** m/s, lateral **0.7** m/s, yaw **1.5** rad/s. |

### §3.2 B — Gait quality

| Gate | Name | Threshold |
|------|------|-----------|
| **B1** | No hard landing / impact | touchdown vertical speed `p95 ≤ 0.10` m/s on **flat**, `≤ 0.20` m/s on **terrain**. |
| **B2** | No stance slip | settled-stance foot slip speed `p90 ≤ 0.05` m/s on **flat**, `≤ 0.10` m/s on **terrain**. |
| **B3** | Foot clearance | **flat:** swing-peak clearance ≈ 0.08 m, graded pass band `0.05–0.15` m. **terrain:** peak `≥ obstacle_h + 0.04` m margin (obstacle-relative; report-only where local obstacle height is unavailable). |
| **B4** | Symmetry | In every symmetric scene, L/R duty difference `Δduty ≤ 0.05` **and** clearance difference `Δclr ≤ 0.01` m. Family passes only if all symmetric scenes pass. |
| **B5** *(soft)* | Smooth action / torque | Action-rate and torque time-series are regular (no chatter); reported, not gating. |

### §3.3 C — Standing posture & recovery (立正)

**C** passes iff `A3` passes **and** standing L/R symmetry `Δduty ≤ 0.05` **and** at-rest upright
fraction `≥ 0.99` **and** at-rest four-foot duty `≥ 0.95`. Push-recovery to a clean stand is
covered by the E battery.

### §3.4 D — Terrain traversal

Terrain is scored as **controlled progress**, not flat-band tracking. Families:
`slope (≤20°)`, `rough (≤0.12 m noise)`, `boxes (≤0.25 m)`, `stairs_up (≤0.25 m step)`,
`stairs_down (≤0.25 m step)`. `stairs_down` is treated as **more dangerous** than `stairs_up`.

| Gate | Threshold |
|------|-----------|
| **D1–D4** | forward speed `≥ 0.30` m/s **and** fall rate `= 0` (controlled progress at each difficulty before it is raised). |
| **D4 (rough) additionally** | base-height drop `< 0.05` m. |

### §3.5 E — Robustness / disturbance

| Gate | Condition | Threshold |
|------|-----------|-----------|
| **E1** | Walking push (≥1.0 m/s equiv.) | no fall **and** recover `≤ 1.0` s **and** tracking resumes **and** no torque saturation. |
| **E2** | Standing push (≥1.0 m/s) | no fall **and** recover `≤ 1.0` s **and** returns to a clean stand. |
| **E3** | Low friction (μ ≥ 0.4) | stand + walk stable, no sustained slipping, no fall. |
| **E4** | Mass randomization | all `DR_HARD_GATES` still pass under the mass-randomization condition. |
| **E5** | Full domain randomization | all `DR_HARD_GATES` still pass under full DR. |

`DR_HARD_GATES = A1, A2, A3, B1, B2, B3, B4, C, D, E1, E2, E3, F2`. E4/E5 re-evaluate these
gates *under the DR condition*; a gate not evaluated on that battery is reported missing, **not**
silently passed.

**DR ranges (E5 target):** friction `0.4–1.4`, added mass `−5 … +20` kg, CoM offset `±0.05` m,
Kp/Kd `±40%`, plus IMU gyro/gravity bias and latency (deploy-faithful, always-on from φ0).

### §3.6 — Terrain-mode relaxations

On terrain the A1/A2 tracking band is **relaxed** in favor of controlled traversal (this section
is the anchor cited by `acceptance_score.score_D` and `physeval_blind`): B1 relaxes `0.10 → 0.20`,
B2 relaxes `0.05 → 0.10`, B3 becomes obstacle-relative, and D replaces flat tracking with the
forward-speed-floor + no-fall progress test above.

### §3.7 F — Actuator margin & smoothness

| Gate | Name | Threshold |
|------|------|-----------|
| **F2** | Actuator margin | 99.5-percentile `|τ|/limit ≤ 0.85` (worst joint) **and** peak `|τ|/limit ≤ 1.0` **and** clamp rate `= 0`. |
| **F1, F3** *(soft)* | Efficiency / smoothness | reported, not gating. |

---

## §4 — Mapping to the training runtime

The training mechanisms that target each gate, and where each is measured, are enumerated in
`autotuner/locomotion_console/spec_coverage.py` (the coverage ledger the console/LLM read).
The reward/gate/curriculum design that implements these targets is documented in
[`taili_strategy_decisions.md`](./taili_strategy_decisions.md). The single editable strategy
contract is `autotuner/blind_locomotion/taili_blind_config.yaml`; all runtime artifacts
(`agent.skrl.yaml`, `effective_config.yaml`) are generated from it per run.
