"""Proof that the autonomous campaign COMPLETES a tuning task correctly (not just runs one):
it keeps a change that improves the score, ROLLS BACK one that regresses, and stops at convergence.
Driven entirely by a mock — no box, no GPU — so the decision logic is proven deterministically."""
from autotuner.training import tune_orchestrator as TO


class _MockDriver:
    """Stands in for TuningDriver: measure() returns a scripted sequence of verdicts; deploy/launch/
    monitor are no-ops that report success."""
    data_root = "/root/gpufree-data"

    def __init__(self, verdict_scores):
        self._scores = list(verdict_scores)   # scores measure() returns, in order
        self._i = 0
        self.deployed = []
        self.launched_ckpts = []
        self.closed = False

    def _verdict(self, score):
        # a verdict whose _score == `score`: `score` passing gates + one failing movable gate (B4)
        gates = {f"A1[g{i}]": {"ok": True, "detail": "ok"} for i in range(score)}
        gates["B4[back04]"] = {"ok": False, "detail": "dduty=0.174<=0.05"}
        return {"gates": gates}

    def measure(self, run, ckpt, terrains):
        s = self._scores[min(self._i, len(self._scores) - 1)]; self._i += 1
        return self._verdict(s)

    def apply_and_deploy(self, changes, note, stamp):
        self.deployed.append(changes); return f"/payload/{stamp}"

    def launch(self, payload, run_id, checkpoint, num_envs):
        self.launched_ckpts.append(checkpoint)

    def monitor_to_target(self, run_id, target, **k):
        return {"reached": True, "stalls": 0, "last_step": target, "checkpoint": f"/ck/{run_id}.pt"}

    def close(self): self.closed = True


def _run(monkeypatch, scores, max_iters=1):
    rolled = []
    monkeypatch.setattr(TO, "TuningDriver", lambda *a, **k: _MockDriver(scores))
    monkeypatch.setattr(TO, "propose_change", lambda gaps, cur, hist: {"w_diagonal_contact": 0.55})
    # avoid touching the real config/rollback file
    monkeypatch.setattr("autotuner.blind_locomotion.taili_blind_config.load_taili_blind_config", lambda: {})
    monkeypatch.setattr("autotuner.blind_locomotion.taili_blind_config.get_config_value", lambda c, k: 0.45)
    monkeypatch.setattr("autotuner.training.strategy_edit.rollback_last", lambda: rolled.append(1) or {"ok": True})
    camp = TO.Campaign(max_iters=max_iters)
    summary = TO.run_campaign(camp, "run0", "agent_0.pt", log=lambda *a: None,
                              stamp_fn=lambda: str(len(rolled)))
    return summary, rolled


def test_campaign_keeps_improvement(monkeypatch):
    # baseline 7 → iter1 measures 8 (improved) → kept, no rollback, best=8
    summary, rolled = _run(monkeypatch, [7, 8], max_iters=1)
    assert summary["best_score"] == 8
    assert rolled == []                                   # nothing rolled back
    assert summary["iterations"][0]["score"] == 8


def test_campaign_rolls_back_regression(monkeypatch):
    # baseline 7 → iter1 measures 6 (regressed) → rolled back, best stays 7
    summary, rolled = _run(monkeypatch, [7, 6], max_iters=1)
    assert summary["best_score"] == 7                     # regression not kept
    assert len(rolled) >= 1                                # rollback invoked


def test_campaign_resolves_bare_checkpoint_to_absolute_path(monkeypatch):
    # Regression: launch() got the bare name 'agent_15000.pt' → train_taili FileNotFoundError → the
    # campaign never stepped (0706). A bare from_checkpoint must resolve against the run dir.
    drivers = []
    def make_driver(*a, **k):
        d = _MockDriver([7, 8]); drivers.append(d); return d
    monkeypatch.setattr(TO, "TuningDriver", make_driver)
    monkeypatch.setattr(TO, "propose_change", lambda gaps, cur, hist: {"w_diagonal_contact": 0.55})
    monkeypatch.setattr("autotuner.blind_locomotion.taili_blind_config.load_taili_blind_config", lambda: {})
    monkeypatch.setattr("autotuner.blind_locomotion.taili_blind_config.get_config_value", lambda c, k: 0.45)
    monkeypatch.setattr("autotuner.training.strategy_edit.rollback_last", lambda: {"ok": True})
    TO.run_campaign(TO.Campaign(max_iters=1), "run0", "agent_15000.pt", log=lambda *a: None,
                    stamp_fn=lambda: "s1")
    assert drivers[0].launched_ckpts == ["/root/gpufree-data/taili_runs/run0/checkpoints/agent_15000.pt"]


def test_campaign_stops_at_convergence(monkeypatch):
    # once the only lever is exhausted (propose returns {}), the campaign stops cleanly
    def _run_noproposal(mp):
        mp.setattr(TO, "TuningDriver", lambda *a, **k: _MockDriver([7]))
        mp.setattr(TO, "propose_change", lambda *a: {})   # nothing to do
        mp.setattr("autotuner.blind_locomotion.taili_blind_config.load_taili_blind_config", lambda: {})
        mp.setattr("autotuner.blind_locomotion.taili_blind_config.get_config_value", lambda c, k: 0.45)
        return TO.run_campaign(TO.Campaign(max_iters=5), "r", "c.pt", log=lambda *a: None)
    summary = _run_noproposal(monkeypatch)
    assert summary["n_iters"] == 0 and summary["best_score"] == 7   # measured, found nothing productive, stopped


def test_campaign_stops_immediately_when_benchmark_passes(monkeypatch):
    # THE PRODUCT stopping criterion: as soon as a measured verdict PASSES the benchmark, the
    # campaign stops and delivers — no further tuning churn.
    class _PassDriver(_MockDriver):
        def measure(self, run, ckpt, terrains):
            v = super().measure(run, ckpt, terrains)
            if self._i > 1:                      # baseline fails, iter-1 re-measure PASSES
                v["passed"] = True
            return v
    monkeypatch.setattr(TO, "TuningDriver", lambda *a, **k: _PassDriver([7, 19]))
    monkeypatch.setattr(TO, "propose_change", lambda *a: {"w_diagonal_contact": 0.55})
    monkeypatch.setattr("autotuner.blind_locomotion.taili_blind_config.load_taili_blind_config", lambda: {})
    monkeypatch.setattr("autotuner.blind_locomotion.taili_blind_config.get_config_value", lambda c, k: 0.45)
    summary = TO.run_campaign(TO.Campaign(max_iters=5), "r", "c.pt", log=lambda *a: None,
                              stamp_fn=lambda: "s")
    assert summary["n_iters"] == 1                                   # stopped right after the pass
    assert summary["iterations"][0].get("benchmark_passed") is True
