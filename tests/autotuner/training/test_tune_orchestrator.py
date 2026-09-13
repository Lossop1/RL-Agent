"""The autonomous tuning brain: gap ranking, the gate→lever heuristic with its hard-won guard-rails
(skip the B2 metric artifact, cap F2, avoid regressed levers), and stall-recovering monitoring."""
from autotuner.training import tune_orchestrator as TO


_VERDICT = {"gates": {
    "A1[fwd05]": {"ok": True, "detail": "med=0.02 <= 0.10"},
    "A2[yaw04]": {"ok": False, "detail": "|wz-cmd| p90=0.279 <= 0.150"},
    "B2": {"ok": False, "detail": "stance slip p90=0.226 m/s <= 0.05"},
    "B3": {"ok": False, "detail": "swing peak clearance=0.221m ~0.08 (band .05-.15)"},
    "B4[back04]": {"ok": False, "detail": "dduty=0.174<=0.05 dclr=0.0051<=0.010"},
    "F2": {"ok": False, "detail": "99.5pct/limit=0.88<=0.85 (worst j8)"},
}}


def test_analyze_gaps_ranks_and_skips_artifact():
    gaps = TO.analyze_gaps(_VERDICT)
    fams = [g.family for g in gaps]
    assert "B2" not in fams                                  # metric artifact skipped (D3j)
    # worst normalized-margin first: B4 (0.174 vs 0.05 = 2.5x) beats F2 (0.88 vs 0.85 = 0.035x)
    assert fams[0] == "B4" and fams.index("B4") < fams.index("F2")


def test_propose_targets_worst_movable_gap():
    gaps = TO.analyze_gaps(_VERDICT)
    cur = {"w_diagonal_contact": 0.45, "w_duty_balance": 0.45, "w_torque_margin": 1.5,
           "w_tracking_yaw": 2.5, "w_yaw_far": 1.5}
    ch = TO.propose_change(gaps, cur, [])
    assert "w_diagonal_contact" in ch and ch["w_diagonal_contact"] > 0.45   # steps the symmetry lever


def test_f2_lever_capped_per_d3l():
    # w_torque_margin=2.2 deterministically hung training (D3l); the lever must cap at 1.8.
    f2_gaps = [g for g in TO.analyze_gaps(_VERDICT) if g.family == "F2"]
    assert TO.propose_change(f2_gaps, {"w_torque_margin": 1.8}, []) == {}


def test_regressed_lever_not_retried():
    gaps = TO.analyze_gaps(_VERDICT)
    cur = {"w_diagonal_contact": 0.45, "w_duty_balance": 0.45}
    ch = TO.propose_change(gaps, cur, [{"key": "w_diagonal_contact", "outcome": "regressed"}])
    assert "w_diagonal_contact" not in ch                    # falls through to the next lever


def test_score_counts_passing_instances():
    assert TO._score(_VERDICT) == 1                           # only A1[fwd05] passes here


def test_monitor_recovers_from_stall():
    class Mock(TO.TuningDriver):
        def __init__(self):
            self.seq = [100, 100, 100, 100, 200, 30001]; self.i = 0; self.restarts = 0
            self.log = lambda *a: None
        def latest_step(self, r):
            s = self.seq[min(self.i, len(self.seq) - 1)]; self.i += 1; return s
        def alive(self): return True
        def latest_checkpoint(self, r): return "/x/agent_100.pt"
        def launch(self, *a, **k): self.restarts += 1
    m = Mock()
    res = m.monitor_to_target("r", 30000, payload="p", checkpoint="c", num_envs=2048,
                              max_restarts=2, poll_s=0, sleep=lambda s: None)
    assert res["reached"] and m.restarts == 1                 # detected the stall, restarted once, reached
