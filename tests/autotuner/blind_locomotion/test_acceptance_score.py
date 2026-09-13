"""Threshold-lock tests for acceptance_score.py.

These pin every hard-gate boundary to the value documented in docs/taili_spec.md, so the
scorer (the machine-enforced contract) and the spec (the human contract) can never silently
drift. Pure Python — no torch, no sim — so they run in any CI.

Convention: for each gate we test one value that must PASS (at or inside the bound, since the
spec uses non-strict <=/>=) and one that must FAIL (just outside).
"""
import math

import pytest

from autotuner.blind_locomotion import acceptance_score as A


# ── A1 linear tracking: |v-cmd| <= max(0.10, 0.15*|cmd|) for median AND p90 ──────────────
def test_A1_abs_floor_boundary():
    # small command -> floor 0.10 dominates
    r = A.score_A1({"fwd03": (0.10, 0.10, 0.3)})   # 0.15*0.3=0.045 < 0.10 -> thr=0.10
    assert r["A1[fwd03]"]["ok"] is True
    r = A.score_A1({"fwd03": (0.10, 0.101, 0.3)})  # p90 just over
    assert r["A1[fwd03]"]["ok"] is False


def test_A1_relative_band_dominates_at_speed():
    # |cmd|=1.5 -> thr = 0.15*1.5 = 0.225 (relative band beats the 0.10 floor).
    # Avoid asserting on the FP-exact boundary (0.15*1.5 == 0.22499..): use inside/outside values.
    assert A.score_A1({"fwd15": (0.22, 0.22, 1.5)})["A1[fwd15]"]["ok"] is True
    assert A.score_A1({"fwd15": (0.20, 0.23, 1.5)})["A1[fwd15]"]["ok"] is False


def test_A1_median_and_p90_both_required():
    # median in-band but p90 out -> fail (both must hold)
    assert A.score_A1({"b": (0.05, 0.30, 0.3)})["A1[b]"]["ok"] is False


# ── A2 yaw tracking: |wz-cmd| <= 0.15 for median AND p90 ─────────────────────────────────
def test_A2_boundary():
    assert A.score_A2({"yawl": (0.15, 0.15)})["A2[yawl]"]["ok"] is True
    assert A.score_A2({"yawl": (0.15, 0.1501)})["A2[yawl]"]["ok"] is False


# ── A3 clean stop: v<=0.05, wz<=0.05, settle<=1.0, duty_min>=0.95, up>=0.99 ─────────────
def test_A3_all_conditions():
    ok = A.score_A3(speed=0.05, yaw=0.05, settle_s=1.0, duty_min=0.95, up=0.99)
    assert ok["A3"]["ok"] is True
    # each single violation fails
    assert A.score_A3(0.051, 0.05, 1.0, 0.95, 0.99)["A3"]["ok"] is False
    assert A.score_A3(0.05, 0.051, 1.0, 0.95, 0.99)["A3"]["ok"] is False
    assert A.score_A3(0.05, 0.05, 1.01, 0.95, 0.99)["A3"]["ok"] is False
    assert A.score_A3(0.05, 0.05, 1.0, 0.949, 0.99)["A3"]["ok"] is False
    assert A.score_A3(0.05, 0.05, 1.0, 0.95, 0.989)["A3"]["ok"] is False


# ── B1 touchdown impact: p95 <= 0.10 flat / 0.20 terrain ────────────────────────────────
def test_B1_flat_vs_terrain_threshold():
    assert A.score_B1(0.10, "flat")["B1"]["ok"] is True
    assert A.score_B1(0.11, "flat")["B1"]["ok"] is False
    assert A.score_B1(0.20, "rough")["B1"]["ok"] is True
    assert A.score_B1(0.21, "rough")["B1"]["ok"] is False


# ── B2 stance slip: p90 <= 0.05 flat / 0.10 terrain ─────────────────────────────────────
def test_B2_flat_vs_terrain_threshold():
    assert A.score_B2(0.05, "flat")["B2"]["ok"] is True
    assert A.score_B2(0.051, "flat")["B2"]["ok"] is False
    assert A.score_B2(0.10, "rough")["B2"]["ok"] is True
    assert A.score_B2(0.101, "rough")["B2"]["ok"] is False


# ── B3 clearance: flat band 0.05-0.15 (~0.08); terrain obstacle-relative + margin ───────
def test_B3_flat_lower_bound_only():
    # RECALIBRATED (0705): flat B3 is lower-bound-only — a BLIND policy correctly over-lifts to clear
    # unseen terrain (SOTA gates clearance lower-bound-only), so only ground clearance is required.
    assert A.score_B3(0.05, "flat")["B3"]["ok"] is True
    assert A.score_B3(0.15, "flat")["B3"]["ok"] is True
    assert A.score_B3(0.22, "flat")["B3"]["ok"] is True     # blind over-lift: PASS (was FAIL)
    assert A.score_B3(0.049, "flat")["B3"]["ok"] is False   # foot not clearing the ground: FAIL


def test_B3_terrain_obstacle_relative():
    # peak must clear obstacle + 0.04 margin
    assert A.score_B3(0.29, "boxes", obstacle_h=0.25, margin=0.04)["B3"]["ok"] is True
    assert A.score_B3(0.28, "boxes", obstacle_h=0.25, margin=0.04)["B3"]["ok"] is False
    # no obstacle height available -> report-only pass
    assert A.score_B3(0.02, "rough", obstacle_h=None)["B3"]["ok"] is True


# ── B4 symmetry: dduty <= 0.05, dclr <= 0.01 in every symmetric scene ──────────────────
def test_B4_symmetry_all_scenes():
    good = A.score_B4({"fwd": (0.05, 0.01), "back": (0.0, 0.0)})
    assert good["B4"]["ok"] is True
    bad = A.score_B4({"fwd": (0.05, 0.01), "back": (0.06, 0.0)})
    assert bad["B4"]["ok"] is False
    assert bad["B4[back]"]["ok"] is False


# ── C standing posture ──────────────────────────────────────────────────────────────────
def test_C_requires_A3_and_symmetry():
    assert A.score_C(a3_ok=True, stand_dd=0.05, up_rest=0.99, duty_min_rest=0.95)["C"]["ok"] is True
    assert A.score_C(False, 0.05, 0.99, 0.95)["C"]["ok"] is False
    assert A.score_C(True, 0.051, 0.99, 0.95)["C"]["ok"] is False
    assert A.score_C(True, 0.05, 0.989, 0.95)["C"]["ok"] is False


# ── D terrain: fwd>=0.30, fall=0, (rough) base_drop<0.05 ────────────────────────────────
def test_D_progress_and_fall():
    assert A.score_D(0.30, 0.0, 0.0, "slope")["D[slope]"]["ok"] is True
    assert A.score_D(0.29, 0.0, 0.0, "slope")["D[slope]"]["ok"] is False
    assert A.score_D(0.30, 0.01, 0.0, "slope")["D[slope]"]["ok"] is False


def test_D_rough_base_drop():
    assert A.score_D(0.30, 0.0, 0.049, "rough")["D[rough]"]["ok"] is True
    assert A.score_D(0.30, 0.0, 0.05, "rough")["D[rough]"]["ok"] is False  # strict <0.05


# ── E robustness ────────────────────────────────────────────────────────────────────────
def test_E1_walking_push():
    assert A.score_E1(0.0, 1.0, True, False)["E1"]["ok"] is True
    assert A.score_E1(0.0, 1.01, True, False)["E1"]["ok"] is False   # recovery too slow
    assert A.score_E1(0.0, 1.0, True, True)["E1"]["ok"] is False     # torque saturated
    assert A.score_E1(0.0, 1.0, False, False)["E1"]["ok"] is False   # tracking not resumed


def test_E2_standing_push():
    assert A.score_E2(0.0, 1.0, True)["E2"]["ok"] is True
    assert A.score_E2(0.01, 1.0, True)["E2"]["ok"] is False


def test_E3_low_friction():
    assert A.score_E3(0.4, True, True, 0.0)["E3"]["ok"] is True
    assert A.score_E3(0.39, True, True, 0.0)["E3"]["ok"] is False


def test_E_under_dr_requires_all_hard_gates_present():
    # A perfectly-passing subset that is missing gates must NOT pass (honest coverage).
    partial = {"A1[fwd]": {"ok": True}, "A2[yaw]": {"ok": True}}
    r = A.score_E_under_dr(partial, "E5")
    assert r["E5"]["ok"] is False
    assert "not evaluated" in r["E5"]["detail"]


# ── F2 actuator margin ──────────────────────────────────────────────────────────────────
def test_F2_boundaries():
    assert A.score_F2(0.85, 1.0, 0.0)["F2"]["ok"] is True
    assert A.score_F2(0.851, 1.0, 0.0)["F2"]["ok"] is False
    assert A.score_F2(0.85, 1.01, 0.0)["F2"]["ok"] is False
    assert A.score_F2(0.85, 1.0, 0.0001)["F2"]["ok"] is False


# ── §2 final verdict ────────────────────────────────────────────────────────────────────
def _all_hard_pass():
    res = {}
    for fam in A.FINAL_HARD_FAMILIES:
        res[fam] = {"ok": True, "detail": ""}
    return res


def test_final_verdict_all_present_and_ok_passes():
    v = A.final_verdict(_all_hard_pass())
    assert v["passed"] is True
    assert v["missing"] == [] and v["failed"] == []


def test_final_verdict_missing_family_fails():
    res = _all_hard_pass()
    del res["E5"]
    v = A.final_verdict(res)
    assert v["passed"] is False
    assert "E5" in v["missing"]


def test_final_verdict_failed_family_fails():
    res = _all_hard_pass()
    res["B2"] = {"ok": False, "detail": "slip too high"}
    v = A.final_verdict(res)
    assert v["passed"] is False
    assert "B2" in v["failed"]


def test_soft_families_never_gate():
    # Soft families absent -> still passes as long as hard families are all present+ok.
    v = A.final_verdict(_all_hard_pass())
    assert v["passed"] is True
    assert set(v["soft"].keys()) == set(A.SOFT_FAMILIES)


def test_hard_family_set_matches_spec_doc():
    # Lock the exact hard/soft partition documented in docs/taili_spec.md §2.
    assert A.FINAL_HARD_FAMILIES == (
        "A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C",
        "D", "E1", "E2", "E3", "E4", "E5", "F2",
    )
    assert A.SOFT_FAMILIES == ("A5", "B5", "F1", "F3")
    assert A.DR_HARD_GATES == ("A1", "A2", "A3", "B1", "B2", "B3", "B4", "C", "D", "E1", "E2", "E3", "F2")


def test_final_verdict_matches_emitted_D_terrain_keys():
    # Regression: score_D emits "D[slope]" etc. The D family must match those keys (not "D1").
    res = _all_hard_pass()
    # a realistic terrain sweep: per-terrain D keys, all passing
    for terr in ("slope", "rough", "boxes", "stairs", "stairs_down"):
        res[f"D[{terr}]"] = {"ok": True, "detail": ""}
    v = A.final_verdict(res)
    assert v["families"]["D"]["present"] is True
    assert v["families"]["D"]["ok"] is True
    assert "D" not in v["missing"]
    # one failing terrain sub-check fails the whole D family
    res["D[stairs_down]"] = {"ok": False, "detail": "fell on descent"}
    v2 = A.final_verdict(res)
    assert v2["passed"] is False and "D" in v2["failed"]


def test_all_hard_pass_helper_covers_D_via_terrain_keys():
    # _all_hard_pass sets a bare "D" gate; confirm the family resolves either way (bare or [terrain]).
    res = _all_hard_pass()
    assert A.final_verdict(res)["families"]["D"]["present"] is True
