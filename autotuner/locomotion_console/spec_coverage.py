"""Taili spec coverage ledger for the console and LLM.

This is intentionally explicit.  The rows mirror taili_spec acceptance items; columns explain
what the current training runtime implements and what the evaluation stack can actually measure.
It is a read-only fact source for UI/LLM grounding, not a pass/fail substitute for diagnostics.
"""
from __future__ import annotations

import time
from pathlib import Path

from .schemas import SpecCoverageItem, SpecCoverageReport

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = _ROOT / "docs" / "taili_spec.md"


def build_spec_coverage_report() -> SpecCoverageReport:
    items = [
        SpecCoverageItem(
            id="A1",
            title="Linear velocity tracking",
            group="A/C command tracking and standing",
            training_mechanism="tracking_lin + far-field tracking, training sigma wider than acceptance band; phase gates use progress/tracking.",
            evaluation="physeval flat direction buckets measure commanded vx/vy tracking against spec thresholds.",
            status="complete",
            evidence=["reward.tracking_lin", "reward.tracking_lin_far", "diagnostics flat direction suites"],
        ),
        SpecCoverageItem(
            id="A2",
            title="Yaw tracking",
            group="A/C command tracking and standing",
            training_mechanism="Yaw reward is conditioned on |cmd_wz| > deadband to avoid standing bonus.",
            evaluation="physeval yaw bucket measures commanded angular velocity tracking.",
            status="complete",
            evidence=["conditional yaw tracking", "diagnostics yaw suite"],
        ),
        SpecCoverageItem(
            id="A3",
            title="Clean stop / settle",
            group="A/C command tracking and standing",
            training_mechanism="stand reward chain, off-axis anti-drift penalty, upright/height gates.",
            evaluation="stop-recovery / settle diagnostics report residual motion and pose stability.",
            status="complete",
            evidence=["reward.stand", "reward.off_axis", "diagnostic settle stage"],
        ),
        SpecCoverageItem(
            id="A4",
            title="Mixed commands",
            group="A/C command tracking and standing",
            training_mechanism="Mixed sampling is enabled after the early phase schedule.",
            evaluation="Current physeval direction suites do not expose a dedicated mixed-command bucket.",
            status="partial",
            gaps=["Add mixed vx+vy/yaw command buckets to payload diagnostics and report aggregation."],
            next_actions=["Add mixed diagnostic preset and per-stage metrics."],
            evidence=["curriculum command_mode=mixed", "diagnostics catalog currently direction-focused"],
        ),
        SpecCoverageItem(
            id="A5",
            title="Velocity envelope",
            group="A/C command tracking and standing",
            training_mechanism="Velocity curriculum ramps to the spec envelope for forward/back/lateral/yaw.",
            evaluation="Reports expose commanded envelope and achieved progress/tracking.",
            status="complete",
            evidence=["env.commands envelope", "curriculum velocity gates"],
        ),
        SpecCoverageItem(
            id="C",
            title="Standing posture and recovery",
            group="A/C command tracking and standing",
            training_mechanism="A3 standing chain plus symmetry/upright gates; push recovery belongs to E battery.",
            evaluation="E battery and settle diagnostics cover perturbation return-to-stand behavior.",
            status="complete",
            evidence=["stand reward", "upright gate", "E diagnostics battery"],
        ),
        SpecCoverageItem(
            id="B1",
            title="No hard landing / impact",
            group="B gait quality",
            training_mechanism="Touchdown-only landing impact penalty.",
            evaluation="Diagnostics currently count hard-impact bouts; spec p95 impact metric still needs direct alignment.",
            status="partial",
            gaps=["Expose p95 touchdown impact in diagnostics alongside bouts."],
            next_actions=["Add p95 impact calculation and definition to diagnostics.report."],
            evidence=["landing_impact telemetry", "hard_impact_bout events"],
        ),
        SpecCoverageItem(
            id="B2",
            title="No stance slip",
            group="B gait quality",
            training_mechanism="Settled-contact slip penalty and event pressure; touchdown/liftoff frames are excluded.",
            evaluation="Diagnostics report slip bouts and settled slip statistics.",
            status="complete",
            evidence=["settled stance slip", "high_slip_bout diagnostics"],
        ),
        SpecCoverageItem(
            id="B3",
            title="Foot clearance",
            group="B gait quality",
            training_mechanism="Flat clearance target is implemented; terrain-aware clearance depends on local_obstacle_h.",
            evaluation="Flat clearance is reported; obstacle-relative clearance is not fully wired.",
            status="partial",
            gaps=["Wire local_obstacle_h for terrain clearance band."],
            next_actions=["Feed terrain-relative obstacle height into reward and diagnostics."],
            evidence=["clearance_under/over telemetry", "terrain local height source pending"],
        ),
        SpecCoverageItem(
            id="B4",
            title="Symmetry",
            group="B gait quality",
            training_mechanism="Structurally equivariant model plus duty/diagonal quality metrics.",
            evaluation="Diagnostics expose duty spread / diagonal consistency; physical round-trip contract test should stay in CI.",
            status="partial",
            gaps=["Confirm physical round-trip symmetry test is included in the active test set."],
            next_actions=["Pin symmetry round-trip test in CI/payload tests."],
            evidence=["taili_core model symmetry tests", "diagnostic dduty/dclr outputs"],
        ),
        SpecCoverageItem(
            id="B5",
            title="Smooth action / torque regularity",
            group="B gait quality",
            training_mechanism="Light action-rate penalty and torque margin/saturation penalties.",
            evaluation="Telemetry and diagnostics report action/torque health.",
            status="complete",
            evidence=["action_rate", "torque_margin", "torque_saturation"],
        ),
        SpecCoverageItem(
            id="D1-D5",
            title="Terrain family",
            group="D terrain",
            training_mechanism="Terrain families, per-env level curriculum, and terrain-relative height sources are partially present.",
            evaluation="Terrain diagnostics cover flat and selected terrain presets, but several spec distinctions are missing.",
            status="partial",
            gaps=[
                "local_obstacle_h not wired for terrain clearance.",
                "body_collision signal is not wired into reward/termination.",
                "Terrain mode still uses global tracking weights; progress should dominate on terrain.",
                "Stair descent needs a separate dangerous-downstairs treatment.",
                "Before speed-up, diagonal airborne handling and velocity-dependent duty target need adjustment.",
            ],
            next_actions=[
                "Wire obstacle height and body collision.",
                "Add terrain-specific reward weighting.",
                "Split stair up/down diagnostics and curriculum.",
                "Make gait quality metrics speed-aware.",
            ],
            evidence=["terrain curriculum", "diagnostic terrain catalog", "telemetry terrain fields"],
        ),
        SpecCoverageItem(
            id="E1-E2",
            title="Push robustness and recovery",
            group="E robustness / DR",
            training_mechanism="Push is currently coupled to DR schedule; explicit recovery-window reward is not implemented.",
            evaluation="physeval blind E battery exists for push/recovery checks.",
            status="partial",
            gaps=["Separate push as a late independent curriculum axis.", "Add recovery-window reward if needed."],
            next_actions=["Decouple push from DR table before robustness phase."],
            evidence=["E diagnostics battery", "DR schedule"],
        ),
        SpecCoverageItem(
            id="E3",
            title="Friction randomization",
            group="E robustness / DR",
            training_mechanism="Friction range covers spec range 0.4-1.4.",
            evaluation="DR level and terrain diagnostics expose active DR conditions.",
            status="complete",
            evidence=["domain_randomization.friction_range_3"],
        ),
        SpecCoverageItem(
            id="E4",
            title="Mass randomization",
            group="E robustness / DR",
            training_mechanism="Mass randomization covers the configured -5% to +20% band.",
            evaluation="DR diagnostics expose active level; detailed mass value reporting can be expanded.",
            status="complete",
            evidence=["domain randomization mass band"],
        ),
    ]
    complete = sum(1 for item in items if item.status == "complete")
    partial = sum(1 for item in items if item.status == "partial")
    missing = sum(1 for item in items if item.status == "missing")
    return SpecCoverageReport(
        spec_path=str(_SPEC),
        generated_at=time.time(),
        legend={
            "complete": "mechanism and evaluation both exist",
            "partial": "mechanism exists but one training/evaluation corner is incomplete",
            "missing": "required mechanism/evaluation is absent",
        },
        summary=f"{complete} complete, {partial} partial, {missing} missing; terrain and mixed-command coverage are the main open gaps.",
        items=items,
    )
