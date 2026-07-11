"""The facts card must carry the exact code-derived numbers the agent kept failing
to answer (gate thresholds, penalty ramp, terrain levels), and must overlay live
per-run gate values when given."""
from autotuner.locomotion_console.curriculum_facts import build_facts_card


def test_card_contains_code_derived_gate_and_curriculum_facts():
    card = build_facts_card()
    # gate thresholds (A1 — the question that scored 0% cold)
    for token in ["0.60", "0.70", "0.50", "6.0", "0.05"]:
        assert token in card, token
    # penalty ramp (A2)
    assert "penalty_ramp_intervals=25" in card
    assert "0.04" in card                      # 1/25 per step
    # actual_vx (A3)
    assert "vb[:, 0].mean()" in card or "vb[:,0].mean()" in card
    # terrain progression (A4)
    assert "terrain_start_phase" in card and "降级" in card
    # terrain levels / stair heights (A5)
    assert "num_rows=10" in card
    assert "(0.05,0.15)" in card and "(0.08,0.30)" in card
    # the anti-fallacy steer (B)
    assert "不看绝对步数占比" in card


def test_live_gate_overlays_defaults_and_is_labelled():
    card = build_facts_card(live_gate={"progress_min": 0.65, "fall_max": 0.03, "terrain_min": 7.0})
    assert "本次 run 配置" in card
    assert "0.65" in card          # overlaid progress threshold
    assert "0.03" in card          # overlaid fall threshold
    assert "7.0" in card           # overlaid terrain threshold


def test_card_is_compact():
    # It is injected into every turn; keep it from bloating the prompt.
    assert len(build_facts_card()) < 1400
