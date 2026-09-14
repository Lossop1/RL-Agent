"""Pinned behaviour of early stopping: a decision is a pure function of the curve, the reading it
rested on survives the ledger, and a trial that stopped early can always say why.

Each group below pins one way of being quietly wrong:

* **Value reading** -- a rule that cannot tell "no reading" from "the reading is NaN" fires on
  the wrong thing, and one that reads a bool as a number measures something nobody chose.
* **Canonicalisation** -- this is measured, not reasoned about: the ledger's write path is
  ``model_dump(mode="json")`` (``trial_ledger.py:565``), which turns ``float('nan')`` into
  ``null`` on pydantic 2.12.5 while the text ``"nan"`` survives.  A NaN that reached the ledger
  as ``None`` makes the only rule that catches it stop being re-derivable, silently.
* **Gates** -- a rule with no stated activation point fires at a step its author never intended,
  and a fraction gate with no denominator must stay silent rather than invent one, because the
  shipped telemetry carries ``total_steps: null`` on every line.
* **The verdict** -- ``stop_reason`` is a stored field, so it can drift from the fields it
  summarises; the validator is what stops a ledger holding "reason says one thing, the one
  readable line says another".
* **The ledger layout** -- the single most expensive mistake available here.  A record written
  before the run header does not inconvenience one reader, it makes the root permanently
  unreadable to every later ``SearchLedger`` call (``hyperparameter_search.py:574-622``, raising
  at ``:587``/``:592``/``:597``/``:609``).  The first test in group D demonstrates the hazard
  actually happening and then shows the guard refusing it.
* **Idempotence** -- a watcher asked twice must not decide twice, and a curve recorded twice must
  not become a curve whose steps go 1, 2, 1, 2.
* **The second audit (group G)** -- every group above was green, and 22 mutations had already been
  killed, when a later pass found seven ways the audit layer could be *confident and wrong*: a
  correct record reported as broken, a forged record reported as verified, a curve silently
  emptied, a documented capability that did not exist.  None of them is a crash, which is exactly
  why a green suite did not catch them -- the module was never red, it was just wrong.  Group G
  pins each one, and each test in it was checked by re-applying the defect and confirming it goes
  red.
* **The guards that were promised and never written (group H)** -- the design named five checks
  that were load-bearing for its own arguments (the emitter still writes the sections the parser
  reads, the module stays inside its layer, a decision re-derives in a fresh interpreter, a pruned
  trial still counts against the sampler's budget), and the first four of them did not exist in
  any form.  Their absence was invisible to every other test here, because nothing tested the
  *design*, only the code.  Group H writes them and pins them the same way: each was checked by
  breaking the thing it guards and confirming the test goes red.
"""
from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from autotuner.research.config_injection import load_config_file
from autotuner.research.hyperparameter_sampler import SamplerStats, SearchPlan
from autotuner.research.hyperparameter_search import (
    SearchError,
    SearchLedger,
    SearchTracker,
    TrialObservation,
    TrialOutcome,
)
from autotuner.research.hyperparameter_space import load_search_space
from autotuner.research.research_ledger import content_hash
from autotuner.research.search_pruning import (
    DECISION_RECORD_TYPE,
    POLICY_RECORD_TYPE,
    FloorRule,
    JsonlTelemetryCurveSource,
    NaNRule,
    PruneAuditError,
    PruneDecisionRecord,
    PrunePolicyRecord,
    PruningPolicy,
    PruningTrialRunner,
    PruningWatcher,
    StallRule,
    TrackerPruningSink,
    TrialWorkspaceTelemetry,
    canonical_metric,
    canonical_metrics,
    format_stop_reason,
    is_nonfinite,
    metric_value,
    parse_telemetry_lines,
    parse_telemetry_payload,
    policy_for,
    policy_record,
    prune_summary,
    read_prune_stop_reason,
    reading_is_nulled,
    store_policy_record,
    verify_decision,
)
from autotuner.research.trial_ledger import TrialLedgerIntegrityError, TrialLedgerStore

REPO_ROOT = Path(__file__).resolve().parents[3]
_PRODUCT_DIR = REPO_ROOT / "products" / "taili" / "blind_locomotion"
SPACE_PATH = _PRODUCT_DIR / "hyperparameter_space.yaml"
CONFIG_PATH = _PRODUCT_DIR / "taili_blind_config.yaml"


# --- Fixtures over the real product files and a fake execution backend -----------------------


@lru_cache(maxsize=None)
def _taili_space():
    return load_search_space(yaml.safe_load(SPACE_PATH.read_text(encoding="utf-8")))


def _plan(*, seed: int = 7, budget: int = 1) -> SearchPlan:
    return SearchPlan.for_random(_taili_space(), seed=seed, budget=budget)


class FakeRunner:
    """A runner with no base class: ``TrialRunner`` is a Protocol and nothing checks its type."""

    def __init__(self, *, on_run: Any = None, outcome: TrialOutcome | None = None) -> None:
        self.on_run = on_run
        self.outcome = outcome if outcome is not None else TrialOutcome(status="succeeded", objective=1.5)
        self.requests: list[Any] = []

    def preflight(self, request: Any) -> None:
        pass

    def run(self, request: Any) -> TrialOutcome:
        self.requests.append(request)
        if self.on_run is not None:
            self.on_run(request)
        return self.outcome


def _tracker(tmp_path: Path, *, budget: int = 1, store: TrialLedgerStore | None = None, runner: Any = None, work_root: Path | None = None) -> SearchTracker:
    return SearchTracker(
        store=store if store is not None else TrialLedgerStore(tmp_path / "ledger"),
        space=_taili_space(),
        plan=_plan(budget=budget),
        source_config=load_config_file(CONFIG_PATH),
        source_config_path=CONFIG_PATH,
        work_root=work_root if work_root is not None else tmp_path / "work",
        runner=runner if runner is not None else FakeRunner(),
    )


def obs(step: int, metrics: dict[str, Any] | None = None, **kwargs: Any) -> TrialObservation:
    merged: dict[str, Any] = dict(metrics or {})
    merged.update(kwargs)
    return TrialObservation(step=step, metrics=merged)


def reward(step: int, total: Any, **extra: Any) -> TrialObservation:
    return obs(step, {"reward": {"total": total}, **extra})


class ListSource:
    """A scripted :class:`CurveSource`: the curve it returns is whatever the test set."""

    def __init__(self, curve: tuple[TrialObservation, ...] = ()) -> None:
        self.curve_value = curve
        self.calls: list[str] = []

    def curve(self, trial_id: str) -> tuple[TrialObservation, ...]:
        self.calls.append(trial_id)
        return self.curve_value


class RecordingSink:
    """A :class:`PruningSink` that only remembers, so watcher behaviour is observable alone."""

    def __init__(self, *, accepts: bool = True) -> None:
        self.accepts = accepts
        self.curves: list[tuple[str, tuple[TrialObservation, ...]]] = []
        self.verdicts: list[tuple[str, Any]] = []

    def record_curve(self, trial_id: str, observations: Any) -> None:
        self.curves.append((trial_id, tuple(observations)))

    def apply_verdict(self, trial_id: str, verdict: Any) -> bool:
        self.verdicts.append((trial_id, verdict))
        return self.accepts


# --- Group A: reading a value out of a curve point -------------------------------------------


def test_a_dotted_path_finds_a_nested_reading():
    assert metric_value({"reward": {"total": 3.5}}, "reward.total") == 3.5


def test_a_missing_reading_is_none_and_never_zero():
    """The distinction the whole rule set rests on: absent is not "measured as 0.0"."""
    assert metric_value({}, "reward.total") is None
    assert metric_value({"reward": {}}, "reward.total") is None
    assert metric_value({"reward": "not a mapping"}, "reward.total") is None
    assert metric_value({"reward": None}, "reward.total") is None


def test_a_boolean_is_not_a_metric():
    """``True`` is an ``int`` in Python; reading it as 1.0 makes a rule look alive while
    measuring a flag nobody meant to measure."""
    assert metric_value({"health": {"alive": True}}, "health.alive") is None


def test_a_number_that_is_not_a_finite_spelling_is_not_a_reading():
    assert metric_value({"reward": {"total": "junk"}}, "reward.total") is None
    assert metric_value({"reward": {"total": [1.0]}}, "reward.total") is None


def test_the_non_finite_spellings_read_back_as_floats():
    """The inverse of :func:`canonical_metric`, which is what makes a curve survive the ledger."""
    assert math.isnan(metric_value({"m": "nan"}, "m"))
    assert metric_value({"m": "inf"}, "m") == math.inf
    assert metric_value({"m": "-inf"}, "m") == -math.inf
    assert metric_value({"m": "Infinity"}, "m") == math.inf
    assert metric_value({"m": " NaN "}, "m") != metric_value({"m": " NaN "}, "m")  # NaN != NaN


def test_is_nonfinite_separates_no_reading_from_a_nan_reading():
    assert is_nonfinite(None) is False
    assert is_nonfinite(1.0) is False
    assert is_nonfinite(float("nan")) is True
    assert is_nonfinite(float("inf")) is True
    assert is_nonfinite(float("-inf")) is True


def test_canonical_metric_maps_each_case_to_its_storable_form():
    assert canonical_metric(None) is None
    assert canonical_metric(2) == 2.0
    assert canonical_metric(float("nan")) == "nan"
    assert canonical_metric(float("inf")) == "inf"
    assert canonical_metric(float("-inf")) == "-inf"


def test_a_nan_does_not_survive_the_ledgers_json_dump_but_its_text_does():
    """The measurement the canonical form exists for, taken on the installed pydantic.

    If this ever starts passing with the float form preserved, ``canonical_metric`` has become
    unnecessary -- and until then, a writer that skips it loses the reading that justified a
    decision, which is the only evidence there is.
    """
    from pydantic import BaseModel, ConfigDict

    class Holder(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        raw: Any = None

    dumped = Holder(raw=float("nan")).model_dump(mode="json")
    assert dumped["raw"] is None, "predicting pydantic's behaviour, not observing it"

    assert Holder(raw=canonical_metric(float("nan"))).model_dump(mode="json")["raw"] == "nan"
    # And the round trip back through the rule's own reader:
    assert math.isnan(metric_value({"raw": "nan"}, "raw"))


# --- Group B: the rules ----------------------------------------------------------------------


def test_a_nan_reading_fires_at_or_after_the_gate():
    policy = PruningPolicy(rules=(NaNRule(metric="health.terminal_rate", at_step=10),))
    before = [obs(5, {"health": {"terminal_rate": "nan"}})]
    assert policy.decide(before) is None, "the gate is step 10"

    at = [obs(10, {"health": {"terminal_rate": "nan"}})]
    verdict = policy.decide(at)
    assert verdict is not None and verdict.step == 10 and verdict.reason == "nan_metric"


def test_a_nan_reading_is_found_even_after_many_clean_steps():
    """The rule reads the whole curve on purpose: a NaN at one step never becomes untrue, and a
    fixed-size window would eventually forget the reading that justified the decision."""
    policy = PruningPolicy(rules=(NaNRule(metric="m", at_step=0),))
    curve = [obs(0, {"m": "nan"})] + [obs(step, {"m": 1.0}) for step in range(1, 500)]
    verdict = policy.decide(curve)
    assert verdict is not None and verdict.step == 0


def test_a_missing_reading_does_not_fire_the_nan_rule():
    """``None`` means "nothing was measured", which is not evidence of instability."""
    policy = PruningPolicy(rules=(NaNRule(metric="m", at_step=0),))
    assert policy.decide([obs(1, {"other": 1.0})]) is None


def test_a_verdict_never_carries_a_non_finite_reading():
    """A verdict is written to a ledger, and a non-finite float does not survive that."""
    policy = PruningPolicy(rules=(NaNRule(metric="m", at_step=0),))
    verdict = policy.decide([obs(1, {"m": "inf"})])
    assert verdict is not None
    assert verdict.observed is None
    assert "not a finite number" in verdict.detail


def test_a_floor_rule_waits_for_its_patience():
    policy = PruningPolicy(rules=(FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=3),))
    assert policy.decide([reward(1, 0.5), reward(2, 0.4)]) is None
    verdict = policy.decide([reward(1, 0.5), reward(2, 0.4), reward(3, 0.3)])
    assert verdict is not None
    assert verdict.matched_steps == (1, 2, 3)
    assert verdict.observed == 0.3


def test_a_good_reading_breaks_a_floor_rules_streak():
    policy = PruningPolicy(rules=(FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=2),))
    # One reading above the floor in the middle means the streak never reached two.
    curve = [reward(1, 0.5), reward(2, 2.0), reward(3, 0.4)]
    assert policy.decide(curve) is None


def test_a_ceiling_rule_watches_the_other_side():
    policy = PruningPolicy(
        rules=(FloorRule(metric="health.terminal_rate", at_step=0, threshold=0.9, mode="ceiling", patience=1),)
    )
    assert policy.decide([obs(1, {"health": {"terminal_rate": 0.5}})]) is None
    verdict = policy.decide([obs(1, {"health": {"terminal_rate": 0.95}})])
    assert verdict is not None and verdict.observed == 0.95


def test_missing_readings_neither_extend_nor_break_a_streak():
    """A gap is not evidence about the threshold, and NaNRule owns non-finite readings."""
    policy = PruningPolicy(rules=(FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=2),))
    curve = [reward(1, 0.5), obs(2, {"other": 1.0}), reward(3, 0.4)]
    verdict = policy.decide(curve)
    assert verdict is not None and verdict.matched_steps == (1, 3)


def test_a_stall_rule_needs_a_full_window():
    policy = PruningPolicy(rules=(StallRule(metric="reward.total", at_step=0, min_delta=0.01, window=3),))
    assert policy.decide([reward(1, 5.0), reward(2, 5.001)]) is None, "a plateau needs 3 points"


def test_a_stall_rule_fires_on_a_plateau():
    policy = PruningPolicy(rules=(StallRule(metric="reward.total", at_step=0, min_delta=0.01, window=3),))
    curve = [reward(1, 5.0), reward(2, 5.005), reward(3, 5.001)]
    verdict = policy.decide(curve)
    assert verdict is not None
    assert verdict.matched_steps == (1, 2, 3)
    assert verdict.reason == "metric_stall"


def test_a_stall_rule_does_not_fire_while_the_metric_is_still_moving():
    policy = PruningPolicy(rules=(StallRule(metric="reward.total", at_step=0, min_delta=0.01, window=3),))
    curve = [reward(1, 1.0), reward(2, 3.0), reward(3, 9.0)]
    assert policy.decide(curve) is None


def test_a_stall_rule_fires_only_when_the_span_exceeds_the_allowance():
    """The boundary is strict.  ``span == min_delta`` is still a stall; ``span > min_delta`` is
    movement.  Without this the comparison could drift either way and no test would notice."""
    policy = PruningPolicy(rules=(StallRule(metric="m", at_step=0, min_delta=1.0, window=2),))
    assert policy.decide([obs(1, {"m": 5.0}), obs(2, {"m": 6.0})]) is not None, "span == min_delta"
    assert policy.decide([obs(1, {"m": 5.0}), obs(2, {"m": 6.5})]) is None, "span > min_delta"


def test_a_stall_rules_window_slides_rather_than_accumulating():
    """The plateau is the *trailing* window: an early flat patch followed by a rise is not a stall."""
    policy = PruningPolicy(rules=(StallRule(metric="reward.total", at_step=0, min_delta=0.01, window=2),))
    curve = [reward(1, 5.0), reward(2, 5.0), reward(3, 50.0)]
    assert policy.decide(curve) is None


def test_a_floor_rule_refuses_an_impossible_threshold():
    with pytest.raises(ValidationError, match="finite"):
        FloorRule(metric="m", at_step=0, threshold=float("inf"))
    with pytest.raises(ValidationError, match="patience"):
        FloorRule(metric="m", at_step=0, threshold=1.0, patience=0)


def test_a_stall_rule_refuses_a_window_that_cannot_be_a_span():
    with pytest.raises(ValidationError, match="window"):
        StallRule(metric="m", at_step=0, min_delta=0.0, window=1)
    with pytest.raises(ValidationError, match="min_delta"):
        StallRule(metric="m", at_step=0, min_delta=-1.0, window=2)


# --- Group B2: the activation gate -----------------------------------------------------------


def test_a_rule_must_state_exactly_one_activation_point():
    """Two gates are two sources of truth for one number; none leaves "from when?" for the
    reader to guess out of the threshold."""
    with pytest.raises(ValidationError, match="exactly one"):
        FloorRule(metric="m", threshold=1.0)
    with pytest.raises(ValidationError, match="exactly one"):
        FloorRule(metric="m", at_step=0, at_total_fraction=0.5, threshold=1.0)


def test_a_negative_step_gate_is_refused():
    with pytest.raises(ValidationError, match="negative"):
        FloorRule(metric="m", at_step=-1, threshold=1.0)


def test_a_fraction_gate_must_be_a_fraction():
    for fraction in (0.0, -0.5, 1.5):
        with pytest.raises(ValidationError, match="fraction"):
            FloorRule(metric="m", at_total_fraction=fraction, threshold=1.0)


def test_a_fraction_gate_reads_the_runs_own_length():
    """The gate is ``ceil(fraction * total_steps)``, taken from the curve rather than assumed."""
    policy = PruningPolicy(
        rules=(FloorRule(metric="reward.total", at_total_fraction=0.5, threshold=100.0),)
    )
    below = [reward(499, 0.1, total_steps=1000)]
    assert policy.decide(below) is None, "the gate for half of 1000 is step 500"
    verdict = policy.decide([reward(500, 0.1, total_steps=1000)])
    assert verdict is not None and verdict.step == 500


def test_a_fraction_gate_rounds_the_first_eligible_step_upward():
    """With a length that does not divide evenly, rounding down would let the rule fire one step
    before the fraction it was asked for.  A length that divides evenly (1000) cannot tell the two
    apart, which is why this uses 999."""
    policy = PruningPolicy(
        rules=(FloorRule(metric="reward.total", at_total_fraction=0.5, threshold=100.0),)
    )
    assert policy.decide([reward(499, 0.1, total_steps=999)]) is None, "ceil(499.5) is 500"
    assert policy.decide([reward(500, 0.1, total_steps=999)]) is not None


def test_a_fraction_gate_stays_silent_when_the_run_has_no_stated_length():
    """Measured: the shipped telemetry carries ``total_steps: null`` on every one of its 4583
    lines when the run was not told its length.  Inventing a denominator would fire the rule at
    a step nobody chose, so the rule does not fire at all."""
    policy = PruningPolicy(
        rules=(FloorRule(metric="reward.total", at_total_fraction=0.5, threshold=100.0),)
    )
    assert policy.decide([reward(1, 0.1)]) is None, "no total_steps at all"
    # A fraction small enough that an INVENTED denominator would put the gate at step 1: without
    # this, "return None when there is no length" and "assume a length" behave identically here.
    tiny = PruningPolicy(
        rules=(FloorRule(metric="reward.total", at_total_fraction=0.001, threshold=100.0),)
    )
    assert tiny.decide([reward(1, 0.1)]) is None
    assert tiny.decide([reward(1, 0.1, total_steps=None)]) is None
    assert policy.decide([reward(1, 0.1, total_steps=None)]) is None, "null total_steps, as shipped"
    assert policy.decide([reward(1, 0.1, total_steps=0)]) is None, "a zero denominator"
    assert policy.decide([reward(1, 0.1, total_steps=-10)]) is None


# --- Group C: the policy ---------------------------------------------------------------------


def test_an_empty_policy_is_the_default_and_never_fires():
    """Empty by default is the safety story: ``finish`` cannot be undone
    (``hyperparameter_search.py:1121-1125``), so pruning has to be asked for."""
    policy = PruningPolicy()
    assert policy.rules == ()
    assert policy.decide([reward(1, -999.0)]) is None


def test_the_first_matching_rule_decides():
    """Order is the contract, so it is pinned: a stall rule placed first wins over a floor rule
    that would also have matched."""
    stall = StallRule(metric="reward.total", at_step=0, min_delta=0.01, window=2)
    floor = FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=2)
    curve = [reward(1, 0.5), reward(2, 0.5)]
    assert PruningPolicy(rules=(stall, floor)).decide(curve).reason == "metric_stall"
    assert PruningPolicy(rules=(floor, stall)).decide(curve).reason == "metric_floor"


def test_a_policy_fingerprint_tracks_content_and_not_spelling():
    """Law R3: the hash is over ``model_dump(mode="json")``, because hashing the model would let
    ``content_hash``'s own ``exclude_none`` erase an unset field from the digest."""
    one = PruningPolicy(rules=(FloorRule(metric="m", at_step=0, threshold=1.0),))
    same = PruningPolicy(rules=(FloorRule(metric="m", at_step=0, threshold=1.0),))
    other = PruningPolicy(rules=(FloorRule(metric="m", at_step=0, threshold=2.0),))
    assert one.fingerprint == same.fingerprint
    assert one.fingerprint != other.fingerprint
    assert one.fingerprint == content_hash(one.model_dump(mode="json"))


def test_a_policy_survives_a_round_trip_through_its_own_dump():
    policy = PruningPolicy(
        min_step=5,
        min_observations=2,
        rules=(
            NaNRule(metric="a.b", at_step=1),
            FloorRule(metric="c.d", at_total_fraction=0.25, threshold=0.5, mode="ceiling", patience=4),
            StallRule(metric="e.f", at_step=7, min_delta=0.1, window=5),
        ),
    )
    rebuilt = PruningPolicy.from_mapping(policy.model_dump(mode="json"))
    assert rebuilt == policy


def test_from_mapping_names_the_kinds_it_knows_instead_of_listing_branches():
    with pytest.raises(ValueError, match="nan_metric"):
        PruningPolicy.from_mapping({"rules": [{"kind": "no_such_rule", "metric": "m", "at_step": 0}]})


def test_from_mapping_refuses_a_rules_field_that_is_not_a_list():
    with pytest.raises(ValueError, match="must be a list"):
        PruningPolicy.from_mapping({"rules": "everything"})


def test_a_policy_refuses_a_non_positive_observation_floor():
    with pytest.raises(ValidationError, match="min_observations"):
        PruningPolicy(min_observations=0)
    with pytest.raises(ValidationError, match="min_step"):
        PruningPolicy(min_step=-1)


def test_min_observations_holds_the_policy_back_until_it_has_seen_enough():
    policy = PruningPolicy(
        min_observations=3,
        rules=(FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=1),),
    )
    assert policy.decide([reward(1, 0.1)]) is None
    assert policy.decide([reward(1, 0.1), reward(2, 0.1)]) is None
    assert policy.decide([reward(1, 0.1), reward(2, 0.1), reward(3, 0.1)]) is not None


def test_min_step_holds_the_policy_back_even_when_a_rule_would_fire():
    policy = PruningPolicy(
        min_step=10,
        rules=(FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=1),),
    )
    assert policy.decide([reward(9, 0.1)]) is None
    verdict = policy.decide([reward(9, 0.1), reward(10, 0.1)])
    # The step matters, not just the fact that something fired.  Asserting only ``is not None`` is
    # what let this test stay green while the rule was reading the observation from step 9 -- the
    # verdict it produced cited a step the policy was configured not to speak before.
    assert verdict is not None
    assert verdict.step == 10
    assert verdict.matched_steps == (10,)


def test_a_curve_that_steps_backwards_is_refused_rather_than_averaged_over():
    """Such a curve is a replay or two files concatenated, and a window statistic over it is a
    number that looks reasonable and means nothing."""
    policy = PruningPolicy(rules=(StallRule(metric="m", at_step=0, min_delta=0.0, window=2),))
    with pytest.raises(ValueError, match="steps back"):
        policy.decide([obs(5, {"m": 1.0}), obs(4, {"m": 1.0})])


def test_a_rule_refuses_a_field_it_does_not_understand():
    with pytest.raises(ValidationError):
        NaNRule(metric="m", at_step=0, threshhold=1.0)  # the misspelling must not be ignored


def test_a_rule_cannot_be_mutated_in_place():
    rule = FloorRule(metric="m", at_step=0, threshold=1.0)
    with pytest.raises(ValidationError):
        rule.threshold = 5.0


# --- Group C2: the stop reason text ----------------------------------------------------------


def test_a_stop_reason_reads_back_as_the_verdict_that_wrote_it():
    text = format_stop_reason("metric_floor", 12, "reward.total stayed below 1 for 3 observations")
    parsed = read_prune_stop_reason(text)
    assert parsed is not None
    assert parsed.reason == "metric_floor" and parsed.step == 12
    assert parsed.detail == "reward.total stayed below 1 for 3 observations"


def test_a_foreign_stop_reason_is_not_a_prune_reason():
    """The other terminal statuses and a hand-typed reason are ordinary, not errors: a reader
    that raised on them would make the ledger unreadable because one trial failed differently."""
    assert read_prune_stop_reason("") is None
    assert read_prune_stop_reason("loss floor crossed") is None
    assert read_prune_stop_reason("budget exhausted") is None


def test_a_prefixed_but_unknown_reason_is_refused():
    assert read_prune_stop_reason("prune:no_such_rule@3 something") is None


def test_a_prefixed_reason_without_a_step_is_refused():
    assert read_prune_stop_reason("prune:nan_metric something") is None


# --- Group C3: the verdict record ------------------------------------------------------------

def _verdict(**overrides: Any):
    from autotuner.research.search_pruning import PruneVerdict

    payload = {
        "reason": "metric_floor",
        "step": 4,
        "metric": "reward.total",
        "observed": 0.5,
        "threshold": 1.0,
        "detail": "reward.total stayed below 1 for 2 observations",
        "stop_reason": format_stop_reason("metric_floor", 4, "reward.total stayed below 1 for 2 observations"),
        "matched_steps": (3, 4),
    }
    payload.update(overrides)
    return PruneVerdict(**payload)


def test_a_verdict_stop_reason_must_describe_the_verdict():
    """Otherwise the ledger holds a record whose fields and whose one readable line disagree."""
    with pytest.raises(ValidationError, match="does not describe"):
        _verdict(stop_reason="prune:metric_stall@9 something else")


def test_the_stop_reason_in_a_verdict_is_a_stored_field_and_survives_the_json_dump():
    """A ``@computed_field`` would be written out and then refused by ``extra="forbid"`` on the
    way back in, so the record could not round-trip -- law R1."""
    dumped = _verdict().model_dump(mode="json")
    assert dumped["stop_reason"].startswith("prune:metric_floor@4")
    from autotuner.research.search_pruning import PruneVerdict

    assert PruneVerdict.model_validate(dumped).stop_reason == dumped["stop_reason"]


def test_a_verdict_refuses_a_non_finite_reading():
    with pytest.raises(ValidationError, match="finite"):
        _verdict(observed=float("nan"))


def test_a_verdict_refuses_a_blank_explanation():
    with pytest.raises(ValidationError, match="why"):
        _verdict(detail="   ")


def test_a_verdict_refuses_a_negative_step():
    with pytest.raises(ValidationError, match="negative"):
        _verdict(step=-1)


def test_a_verdict_must_name_the_metric_it_judged():
    with pytest.raises(ValidationError, match="metric"):
        _verdict(metric="")


def test_a_verdict_cites_a_step_it_actually_matched():
    with pytest.raises(ValidationError, match="not among"):
        _verdict(matched_steps=(1, 2))


# --- Group D: the audit layer, starting with the ledger-layout hazard ------------------------


def test_a_record_written_before_the_run_header_makes_the_ledger_unreadable(tmp_path):
    """The hazard this module's write guard exists for, demonstrated rather than described.

    ``_validate_layout`` raises ``unknown_layout`` for *any* non-empty ledger whose first event
    is not the ``search_run`` append (``hyperparameter_search.py:585-594``).  So the cost of
    writing one policy record too early is not one record: it is every later read of that root,
    including ``open_run`` itself, with no way to repair it.

    The first half shows the damage.  The second half shows the guard refusing to cause it.
    """
    store = TrialLedgerStore(tmp_path / "ledger")
    policy = PruningPolicy(rules=(NaNRule(metric="m", at_step=0),))

    store.append(POLICY_RECORD_TYPE, policy_record(policy), actor="test")
    with pytest.raises(TrialLedgerIntegrityError, match="unknown_layout"):
        SearchLedger(store).runs()

    guarded_store = TrialLedgerStore(tmp_path / "guarded")
    tracker = _tracker(tmp_path, store=guarded_store)
    with pytest.raises(SearchError, match="has not opened"):
        store_policy_record(guarded_store, policy, precondition=tracker._require_open)

    # Refused means not written, so the root is still usable.
    assert SearchLedger(guarded_store).runs() == ()
    tracker.open_run()
    store_policy_record(guarded_store, policy, precondition=tracker._require_open)
    assert SearchLedger(guarded_store).runs() != ()


def test_a_stored_policy_reads_back_byte_for_byte(tmp_path):
    store = TrialLedgerStore(tmp_path / "ledger")
    tracker = _tracker(tmp_path, store=store)
    tracker.open_run()
    policy = PruningPolicy(
        min_step=3,
        rules=(FloorRule(metric="reward.total", at_step=3, threshold=0.25, patience=2),),
    )
    store_policy_record(store, policy, precondition=tracker._require_open)
    assert policy_for(store, policy.fingerprint) == policy


def _event_count(store: TrialLedgerStore) -> int:
    """Appended events, counted raw.  ``records()`` folds by ``record_id``, so counting *that*
    cannot see a duplicate append of the same id -- it overwrites the same key."""
    return sum(
        1 for line in store.events_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def test_storing_the_same_policy_twice_appends_nothing_the_second_time(tmp_path):
    """The id is content-addressed, so a resumed search does not grow a duplicate.

    Counted in events, not in ``records()``.  The first version of this test counted folded
    records, which is vacuously true when a second append of the same id lands on the same key --
    mutation M16 removed the idempotence check entirely and the test still passed.
    """
    store = TrialLedgerStore(tmp_path / "ledger")
    tracker = _tracker(tmp_path, store=store)
    tracker.open_run()
    policy = PruningPolicy(rules=(NaNRule(metric="m", at_step=0),))
    first = store_policy_record(store, policy, precondition=tracker._require_open)
    after_first = _event_count(store)
    second = store_policy_record(store, policy, precondition=tracker._require_open)
    assert first.id == second.id
    assert _event_count(store) == after_first, "the second store appended an event"


def test_a_policy_record_refuses_to_carry_a_policy_it_does_not_hash_to():
    policy = PruningPolicy(rules=(NaNRule(metric="m", at_step=0),))
    record = policy_record(policy)
    with pytest.raises(ValidationError, match="hashes to"):
        PrunePolicyRecord(
            id=record.id,
            policy_fingerprint=record.policy_fingerprint,
            policy={"rules": [], "min_step": 0, "min_observations": 1},
        )


def test_an_unstored_policy_reads_back_as_none(tmp_path):
    store = TrialLedgerStore(tmp_path / "ledger")
    assert policy_for(store, "0" * 64) is None


def test_a_decision_record_refuses_a_stop_reason_it_does_not_describe():
    with pytest.raises(ValidationError, match="does not describe"):
        PruneDecisionRecord(
            id="prune-decision:t1",
            trial_id="t1",
            policy_fingerprint="0" * 64,
            reason="metric_floor",
            step=2,
            through_step=2,
            curve_len=2,
            metric="m",
            stop_reason=format_stop_reason("metric_stall", 2, "different"),
        )


def test_a_decision_record_refuses_a_reading_that_is_neither_finite_nor_canonical():
    with pytest.raises(ValidationError, match="observed text must be one of"):
        PruneDecisionRecord(
            id="prune-decision:t1",
            trial_id="t1",
            policy_fingerprint="0" * 64,
            reason="metric_floor",
            step=2,
            through_step=2,
            curve_len=2,
            metric="m",
            observed="eleventy",
            stop_reason=format_stop_reason("metric_floor", 2, "d"),
        )


def test_a_decision_record_cannot_cite_a_step_it_had_not_seen():
    with pytest.raises(ValidationError, match="later than"):
        PruneDecisionRecord(
            id="prune-decision:t1",
            trial_id="t1",
            policy_fingerprint="0" * 64,
            reason="metric_floor",
            step=9,
            through_step=2,
            curve_len=2,
            metric="m",
            stop_reason=format_stop_reason("metric_floor", 9, "d"),
        )


def test_verify_decision_re_derives_a_decision_from_the_ledger_alone():
    policy = PruningPolicy(rules=(FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=2),))
    curve = (reward(1, 0.5), reward(2, 0.4))
    verdict = policy.decide(curve)
    decision = PruneDecisionRecord(
        id="prune-decision:t1",
        trial_id="t1",
        policy_fingerprint=policy.fingerprint,
        reason=verdict.reason,
        step=verdict.step,
        through_step=2,
        curve_len=2,
        matched_steps=verdict.matched_steps,
        metric=verdict.metric,
        observed=verdict.observed,
        threshold=verdict.threshold,
        stop_reason=verdict.stop_reason,
    )
    assert verify_decision(decision, curve, policy) == "verified"


def test_verify_decision_is_loud_when_the_record_contradicts_its_own_curve():
    policy = PruningPolicy(rules=(FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=2),))
    curve = (reward(1, 0.5), reward(2, 0.4))
    decision = PruneDecisionRecord(
        id="prune-decision:t1",
        trial_id="t1",
        policy_fingerprint=policy.fingerprint,
        reason="metric_stall",
        step=2,
        through_step=2,
        curve_len=2,
        metric="reward.total",
        stop_reason=format_stop_reason("metric_stall", 2, "claimed a plateau"),
    )
    with pytest.raises(PruneAuditError) as raised:
        verify_decision(decision, curve, policy)
    # The message must name every field that disagrees, not just say "it disagrees": the record
    # claims a stall with no reading and no matched steps, and the curve really holds a floor
    # breach at 0.4 over steps 1 and 2.  A reader has to be able to see which claim is wrong.
    message = str(raised.value)
    assert "reason: recorded 'metric_stall', re-derived 'metric_floor'" in message
    assert "observed: recorded None, re-derived 0.4" in message
    assert "matched_steps: recorded (), re-derived (1, 2)" in message
    assert "threshold: recorded None, re-derived 1.0" in message


def test_verify_decision_reports_the_one_case_it_cannot_check_rather_than_inventing_a_verdict():
    """The R-4 normalisation loss, isolated: a decision that acted on a NaN whose reading the
    ledger turned into ``None``.  That is a writer having bypassed ``canonical_metric``, not a
    false record, so it is reported and not raised."""
    policy = PruningPolicy(rules=(NaNRule(metric="health.terminal_rate", at_step=0),))
    curve = (obs(1, {"health": {"terminal_rate": None}}),)
    decision = PruneDecisionRecord(
        id="prune-decision:t1",
        trial_id="t1",
        policy_fingerprint=policy.fingerprint,
        reason="nan_metric",
        step=1,
        through_step=1,
        curve_len=1,
        metric="health.terminal_rate",
        observed="nan",
        stop_reason=format_stop_reason("nan_metric", 1, "health.terminal_rate is not a finite number"),
    )
    assert policy.decide(curve) is None, "the reading is gone, so the rule cannot fire again"
    assert verify_decision(decision, curve, policy) == "unverifiable"


def test_verify_decision_raises_when_no_rule_matches_and_nothing_was_lost():
    """Only a *lost* non-finite reading earns "unverifiable".  A curve that simply holds no
    matching reading is a record contradicting its own evidence, and calling that unverifiable
    would make the audit blind to exactly the case it exists to catch."""
    policy = PruningPolicy(rules=(NaNRule(metric="health.terminal_rate", at_step=0),))
    curve = (obs(1, {"health": {"terminal_rate": 0.25}}),)  # a finite reading: nothing was lost
    decision = PruneDecisionRecord(
        id="prune-decision:t1",
        trial_id="t1",
        policy_fingerprint=policy.fingerprint,
        reason="nan_metric",
        step=1,
        through_step=1,
        curve_len=1,
        metric="health.terminal_rate",
        observed=0.25,
        stop_reason=format_stop_reason("nan_metric", 1, "claimed a non-finite reading"),
    )
    assert policy.decide(curve) is None
    with pytest.raises(PruneAuditError, match="matches no rule"):
        verify_decision(decision, curve, policy)


# --- Group E: the watcher and the sink --------------------------------------------------------


def test_a_watcher_decides_once_and_then_stops_looking():
    """Re-deciding the same evidence would let one rule end one trial twice and write a second
    decision for it."""
    policy = PruningPolicy(rules=(FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=2),))
    source = ListSource((reward(1, 0.5), reward(2, 0.4)))
    sink = RecordingSink()
    watcher = PruningWatcher(policy=policy, source=source, sink=sink)

    first = watcher.review("t1")
    assert first is not None
    looks_after_deciding = len(source.calls)
    second = watcher.review("t1")
    assert second is not None and second.stop_reason == first.stop_reason
    assert len(source.calls) == looks_after_deciding, "the source is not consulted again"
    assert len(sink.verdicts) == 1


def test_a_watcher_keeps_looking_while_no_rule_matches():
    policy = PruningPolicy(rules=(FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=2),))
    source = ListSource((reward(1, 0.5),))
    watcher = PruningWatcher(policy=policy, source=source, sink=RecordingSink())
    assert watcher.review("t1") is None

    source.curve_value = (reward(1, 0.5), reward(2, 0.4))
    assert watcher.review("t1") is not None


def test_a_watcher_records_only_the_observations_it_has_not_recorded_before():
    """``observe`` appends blindly, so forwarding the whole curve on every look would multiply
    the record's observations by the number of polls."""
    policy = PruningPolicy()  # no rules: nothing ever fires, so every look records
    source = ListSource((reward(1, 0.5),))
    sink = RecordingSink()
    watcher = PruningWatcher(policy=policy, source=source, sink=sink)

    watcher.review("t1")
    assert [len(observations) for _, observations in sink.curves] == [1]

    source.curve_value = (reward(1, 0.5), reward(2, 0.6), reward(3, 0.7))
    watcher.review("t1")
    assert [len(observations) for _, observations in sink.curves] == [1, 2]
    assert [observation.step for observation in sink.curves[-1][1]] == [2, 3]

    watcher.review("t1")
    assert len(sink.curves) == 2, "a look with nothing new records nothing"


def test_a_watcher_that_loses_the_race_stops_deciding():
    """A pruner racing a training run that just succeeded is ordinary, not exceptional."""
    policy = PruningPolicy(rules=(FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=1),))
    source = ListSource((reward(1, 0.5),))
    sink = RecordingSink(accepts=False)
    watcher = PruningWatcher(policy=policy, source=source, sink=sink)

    assert watcher.review("t1") is None, "the trial ended on its own first"
    assert watcher.review("t1") is None
    assert sink.verdicts != [], "the sink was asked"
    assert len(sink.verdicts) == 1, "and asked only once"


def test_the_sink_does_not_double_a_curve_another_writer_already_recorded(tmp_path):
    """Measured failure: without the step filter the curve becomes 1, 2, 1, 2, which this
    module's own validator then refuses and every window statistic would silently average."""
    store = TrialLedgerStore(tmp_path / "ledger")
    runner = FakeRunner()
    tracker = _tracker(tmp_path, store=store, runner=runner)
    sink = TrackerPruningSink(tracker=tracker, store=store, ledger=SearchLedger(store))
    seen: dict[str, Any] = {}

    def on_run(request: Any) -> None:
        tracker.observe(
            request.trial_id,
            TrialOutcome(status="running", observations=(reward(1, 0.5), reward(2, 0.5))),
        )
        seen["trial"] = request.trial_id

    runner.on_run = on_run
    tracker.run()

    record = SearchLedger(store).trial(seen["trial"])
    assert [observation.step for observation in record.observations] == [1, 2]
    sink.record_curve(seen["trial"], (reward(1, 0.5), reward(2, 0.5)))
    assert [observation.step for observation in record.observations] == [1, 2]


def test_pruning_a_running_trial_end_to_end_leaves_a_decision_that_verifies(tmp_path):
    """The whole path: the watcher sees a curve mid-run, the trial ends as ``pruned`` with a
    readable reason, the policy and the decision are both on disk, and the audit re-derives the
    decision from the ledger's own contents."""
    store = TrialLedgerStore(tmp_path / "ledger")
    policy = PruningPolicy(rules=(FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=2),))
    runner = FakeRunner()
    tracker = _tracker(tmp_path, store=store, runner=runner)
    read_back = SearchLedger(store)
    sink = TrackerPruningSink(tracker=tracker, store=store, ledger=read_back, policy=policy)

    class LedgerSource:
        def curve(self, trial_id: str) -> tuple[TrialObservation, ...]:
            record = read_back.trial(trial_id)
            return tuple(record.observations) if record is not None else ()

    watcher = PruningWatcher(policy=policy, source=LedgerSource(), sink=sink)
    seen: dict[str, Any] = {}

    def on_run(request: Any) -> None:
        tracker.observe(
            request.trial_id,
            TrialOutcome(status="running", observations=(reward(1, 0.5), reward(2, 0.45))),
        )
        seen["verdict"] = watcher.review(request.trial_id)

    runner.on_run = on_run
    tracker.run()

    assert seen["verdict"] is not None, "the floor rule should have fired"
    finished = read_back.run(tracker.run_ref)
    assert len(finished.trials) == 1
    trial = finished.trials[0]
    assert trial.status == "pruned"
    assert trial.stop_reason.startswith("prune:metric_floor@2 ")
    assert read_prune_stop_reason(trial.stop_reason).reason == "metric_floor"

    payload = store.latest(DECISION_RECORD_TYPE, f"prune-decision:{trial.id}")
    assert payload is not None, "the decision must be on disk, or nothing says why"
    decision = PruneDecisionRecord.model_validate(payload)
    assert decision.observed == 0.45
    assert decision.policy_fingerprint == policy.fingerprint
    assert policy_for(store, policy.fingerprint) == policy, "the policy is stored, not just cited"
    assert verify_decision(decision, trial.observations, policy) == "verified"

    # And the root is still readable: the audit writes did not cost the layout check.
    summary = prune_summary(read_back, store=store)
    assert summary.pruned_indices == (trial.index,)
    assert summary.verified == (trial.index,)
    assert summary.unexplained == ()


def test_the_sink_declines_a_trial_that_already_reached_an_ending(tmp_path):
    store = TrialLedgerStore(tmp_path / "ledger")
    runner = FakeRunner()
    tracker = _tracker(tmp_path, store=store, runner=runner)
    read_back = SearchLedger(store)
    policy = PruningPolicy(rules=(NaNRule(metric="m", at_step=0),))
    sink = TrackerPruningSink(tracker=tracker, store=store, ledger=read_back, policy=policy)
    runner.on_run = lambda request: None
    tracker.run()

    trial = read_back.trials()[0]
    assert trial.status == "succeeded"
    assert sink.apply_verdict(trial.id, policy.decide([obs(1, {"m": "nan"})])) is False
    assert store.latest(DECISION_RECORD_TYPE, f"prune-decision:{trial.id}") is None


def test_the_sink_records_what_cancellation_did_including_its_failure(tmp_path):
    """``cancel`` is best-effort and its outcome is a fact about the call, not a second verdict:
    a canceller that raised must not look like one that worked."""
    store = TrialLedgerStore(tmp_path / "ledger")
    runner = FakeRunner()
    tracker = _tracker(tmp_path, store=store, runner=runner)
    read_back = SearchLedger(store)
    policy = PruningPolicy(rules=(FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=1),))

    def explode(trial_id: str, reason: str) -> None:
        raise RuntimeError("no such process")

    sink = TrackerPruningSink(
        tracker=tracker, store=store, ledger=read_back, policy=policy, cancel=explode
    )
    seen: dict[str, Any] = {}

    def on_run(request: Any) -> None:
        tracker.observe(request.trial_id, TrialOutcome(status="running", observations=(reward(1, 0.5),)))
        seen["trial"] = request.trial_id
        sink.apply_verdict(request.trial_id, policy.decide([reward(1, 0.5)]))

    runner.on_run = on_run
    tracker.run()

    payload = store.latest(DECISION_RECORD_TYPE, f"prune-decision:{seen['trial']}")
    assert payload is not None
    decision = PruneDecisionRecord.model_validate(payload)
    assert decision.cancel_state.startswith("error: RuntimeError")
    assert "no such process" in decision.cancel_state


def test_the_sink_notes_a_cancellation_that_worked(tmp_path):
    store = TrialLedgerStore(tmp_path / "ledger")
    runner = FakeRunner()
    tracker = _tracker(tmp_path, store=store, runner=runner)
    read_back = SearchLedger(store)
    policy = PruningPolicy(rules=(FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=1),))
    cancelled: list[tuple[str, str]] = []
    sink = TrackerPruningSink(
        tracker=tracker,
        store=store,
        ledger=read_back,
        policy=policy,
        cancel=lambda trial_id, reason: cancelled.append((trial_id, reason)),
    )
    seen: dict[str, Any] = {}

    def on_run(request: Any) -> None:
        tracker.observe(request.trial_id, TrialOutcome(status="running", observations=(reward(1, 0.5),)))
        seen["trial"] = request.trial_id
        sink.apply_verdict(request.trial_id, policy.decide([reward(1, 0.5)]))

    runner.on_run = on_run
    tracker.run()

    assert cancelled and cancelled[0][0] == seen["trial"]
    payload = store.latest(DECISION_RECORD_TYPE, f"prune-decision:{seen['trial']}")
    assert PruneDecisionRecord.model_validate(payload).cancel_state == "requested"


def test_prune_summary_separates_pruning_from_losing(tmp_path):
    """Task 6 ranks on these numbers: "pruned at step 10" and "ran the whole way and scored
    badly" are different facts about a configuration."""
    store = TrialLedgerStore(tmp_path / "ledger")
    tracker = _tracker(tmp_path, budget=2, store=store)
    read_back = SearchLedger(store)
    policy = PruningPolicy(rules=(FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=1),))
    sink = TrackerPruningSink(tracker=tracker, store=store, ledger=read_back, policy=policy)

    runner = FakeRunner()
    tracker.runner = runner
    counter = {"n": 0}

    def on_run(request: Any) -> None:
        counter["n"] += 1
        if counter["n"] == 1:
            tracker.observe(request.trial_id, TrialOutcome(status="running", observations=(reward(1, 0.5),)))
            sink.apply_verdict(request.trial_id, policy.decide([reward(1, 0.5)]))

    runner.on_run = on_run
    tracker.run()

    summary = prune_summary(read_back, store=store)
    assert summary.pruned_indices == (1,)
    assert summary.scored_indices == (2,)
    assert summary.verified == (1,)
    assert summary.stopped_indices == ()
    assert summary.error == ""


def test_prune_summary_calls_a_pruned_trial_with_no_decision_unexplained(tmp_path):
    store = TrialLedgerStore(tmp_path / "ledger")
    runner = FakeRunner()
    tracker = _tracker(tmp_path, store=store, runner=runner)
    read_back = SearchLedger(store)
    seen: dict[str, Any] = {}

    def on_run(request: Any) -> None:
        seen["trial"] = request.trial_id
        tracker.finish(request.trial_id, "pruned", stop_reason="a human decided")

    runner.on_run = on_run
    tracker.run()

    summary = prune_summary(read_back, store=store)
    assert summary.pruned_indices == (1,)
    assert summary.unexplained == (1,), "nothing on disk says why it stopped"
    assert summary.verified == ()


def test_prune_summary_reports_a_broken_ledger_instead_of_raising(tmp_path):
    """Being able to look at a broken ledger is worth more than only being able to look at a
    healthy one -- and the report says what broke rather than claiming nothing was wrong."""
    store = TrialLedgerStore(tmp_path / "ledger")
    tracker = _tracker(tmp_path, store=store)
    tracker.open_run()
    store.events_path.write_text("{not json at all}\n", encoding="utf-8")

    summary = prune_summary(SearchLedger(store), store=store)
    assert summary.error != ""
    assert summary.pruned_indices == ()


# --- Group F: reading the running run's own output --------------------------------------------


def test_telemetry_payload_keeps_the_sections_where_a_rule_can_name_them():
    observation = parse_telemetry_payload(
        {
            "type": "train_tick",
            "step": 7,
            "total_steps": 1000,
            "reward": {"total": 3.5},
            "health": {"terminal_rate": 0.0},
            "command": {"vx": 1.0},
        }
    )
    assert observation is not None
    assert observation.step == 7
    assert observation.metrics["total_steps"] == 1000
    assert metric_value(observation.metrics, "reward.total") == 3.5
    assert metric_value(observation.metrics, "health.terminal_rate") == 0.0
    assert metric_value(observation.metrics, "command.vx") == 1.0


def test_a_null_total_steps_stays_null_rather_than_becoming_a_denominator():
    """The shipped payload's own shape: inventing a length would make a fraction gate fire at a
    step nobody chose."""
    observation = parse_telemetry_payload({"type": "train_tick", "step": 3, "total_steps": None})
    assert observation is not None and observation.metrics["total_steps"] is None


def test_a_line_that_is_not_a_tick_is_not_an_observation():
    assert parse_telemetry_payload({"type": "something_else", "step": 1}) is None
    assert parse_telemetry_payload({"type": "train_tick"}) is None, "no step"
    assert parse_telemetry_payload({"type": "train_tick", "step": -1}) is None
    assert parse_telemetry_payload({"type": "train_tick", "step": True}) is None


def test_a_non_finite_reading_in_the_payload_is_canonicalised_on_the_way_in():
    observation = parse_telemetry_payload(
        {"type": "train_tick", "step": 1, "reward": {"total": float("nan")}}
    )
    assert observation is not None
    assert observation.metrics["reward"]["total"] == "nan"
    assert is_nonfinite(metric_value(observation.metrics, "reward.total"))


def test_a_relayed_stat_line_and_its_reward_line_become_one_observation():
    parsed = parse_telemetry_lines("[TPSTAT] step=4 total=1000\n[TPREW] step=4 total=3.25\n")
    assert len(parsed) == 1
    assert parsed[0].step == 4
    assert parsed[0].metrics["total_steps"] == 1000
    assert metric_value(parsed[0].metrics, "reward.total") == 3.25


def test_a_reward_line_with_no_step_is_dropped_rather_than_attributed_to_the_previous_step():
    """The console parser guesses here; a guess is what makes a curve describe a step that never
    produced it."""
    parsed = parse_telemetry_lines("[TPSTAT] step=4 total=1000\n[TPREW] total=9.0\n")
    assert len(parsed) == 1
    assert metric_value(parsed[0].metrics, "reward.total") is None


def test_lines_that_are_neither_are_ignored():
    parsed = parse_telemetry_lines("some noise\nanother line\n[TPSTAT] step=2 total=10\n")
    assert [observation.step for observation in parsed] == [2]


def _write_jsonl(path: Path, *payloads: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload) + "\n")


def test_the_curve_source_reads_a_jsonl_file_incrementally(tmp_path):
    path = tmp_path / "t.telemetry.jsonl"
    _write_jsonl(path, {"type": "train_tick", "step": 1, "reward": {"total": 1.0}})
    source = JsonlTelemetryCurveSource(locate=lambda trial_id: path)
    assert [observation.step for observation in source.curve("t1")] == [1]

    _write_jsonl(path, {"type": "train_tick", "step": 2, "reward": {"total": 2.0}})
    assert [observation.step for observation in source.curve("t1")] == [1, 2]


def test_the_curve_source_leaves_a_half_written_line_for_the_next_look(tmp_path):
    """A read that races the writer must not invent an observation out of a partial line."""
    path = tmp_path / "t.telemetry.jsonl"
    _write_jsonl(path, {"type": "train_tick", "step": 1, "reward": {"total": 1.0}})
    source = JsonlTelemetryCurveSource(locate=lambda trial_id: path)
    assert [observation.step for observation in source.curve("t1")] == [1]

    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"type": "train_tick", "step": 2, "rew')
    assert [observation.step for observation in source.curve("t1")] == [1]

    with path.open("a", encoding="utf-8") as handle:
        handle.write('ard": {"total": 2.0}}\n')
    assert [observation.step for observation in source.curve("t1")] == [1, 2]


def test_the_curve_source_forgets_a_curve_whose_file_was_truncated(tmp_path):
    """Bytes before a truncation are not evidence of anything; starting over is the only reading
    that does not invent steps."""
    path = tmp_path / "t.telemetry.jsonl"
    _write_jsonl(
        path,
        {"type": "train_tick", "step": 1, "reward": {"total": 1.0}},
        {"type": "train_tick", "step": 2, "reward": {"total": 2.0}},
    )
    source = JsonlTelemetryCurveSource(locate=lambda trial_id: path)
    assert [observation.step for observation in source.curve("t1")] == [1, 2]

    path.write_text("", encoding="utf-8")
    _write_jsonl(path, {"type": "train_tick", "step": 9, "reward": {"total": 9.0}})
    assert [observation.step for observation in source.curve("t1")] == [9]


def test_the_curve_source_reads_a_relayed_text_log(tmp_path):
    path = tmp_path / "training.log"
    path.write_text("[TPSTAT] step=1 total=100\n[TPREW] step=1 total=2.5\n", encoding="utf-8")
    source = JsonlTelemetryCurveSource(locate=lambda trial_id: path)
    curve = source.curve("t1")
    assert len(curve) == 1 and metric_value(curve[0].metrics, "reward.total") == 2.5

    with path.open("a", encoding="utf-8") as handle:
        handle.write("[TPREW] step=2 total=3.5\n")
    assert len(source.curve("t1")) == 2


def test_the_curve_source_has_no_evidence_for_an_unknown_trial(tmp_path):
    source = JsonlTelemetryCurveSource(locate=lambda trial_id: None)
    assert source.curve("nope") == ()
    assert source.curve("nope") == ()


# --- Group F2: finding the file ------------------------------------------------------------------


def test_the_workspace_locator_reads_the_trial_directory_off_the_ledger(tmp_path):
    """``TrialRecord.run_dir`` is written when the trial is materialised, at ``pending``
    (``hyperparameter_search.py:1081``), so it is already on disk while the trial runs -- no
    directory-name guessing is needed or done."""
    store = TrialLedgerStore(tmp_path / "ledger")
    runner = FakeRunner()
    tracker = _tracker(tmp_path, store=store, runner=runner)
    read_back = SearchLedger(store)
    seen: dict[str, Any] = {}

    def on_run(request: Any) -> None:
        seen["trial"] = request.trial_id
        seen["workspace"] = request.workspace

    runner.on_run = on_run
    tracker.run()

    record = read_back.trial(seen["trial"])
    assert record.run_dir != "", "the run_dir is on the record before the trial ends"
    assert Path(record.run_dir) == Path(seen["workspace"])

    locate = TrialWorkspaceTelemetry(read_back)
    assert locate(seen["trial"]) is None, "nothing has been written there yet"

    (seen["workspace"] / "training.log").write_text("[TPSTAT] step=1 total=10\n", encoding="utf-8")
    assert locate(seen["trial"]).name == "training.log"

    # A JSONL beside it wins, because it is the machine-readable one.
    (seen["workspace"] / "run.telemetry.jsonl").write_text("{}\n", encoding="utf-8")
    assert locate(seen["trial"]).suffix == ".jsonl"


def test_the_workspace_locator_prefers_the_path_the_manifest_declares(tmp_path):
    store = TrialLedgerStore(tmp_path / "ledger")
    runner = FakeRunner()
    tracker = _tracker(tmp_path, store=store, runner=runner)
    read_back = SearchLedger(store)
    seen: dict[str, Any] = {}

    def on_run(request: Any) -> None:
        seen["trial"] = request.trial_id
        seen["workspace"] = request.workspace

    runner.on_run = on_run
    tracker.run()

    workspace = Path(seen["workspace"])
    elsewhere = workspace / "declared.jsonl"
    elsewhere.write_text("{}\n", encoding="utf-8")
    (workspace / "run.json").write_text(
        json.dumps({"paths": {"telemetry_jsonl": str(elsewhere)}}), encoding="utf-8"
    )
    (workspace / "run.telemetry.jsonl").write_text("{}\n", encoding="utf-8")
    assert TrialWorkspaceTelemetry(read_back)(seen["trial"]) == elsewhere


def test_the_workspace_locator_can_tell_the_relayed_log_from_the_trainers_own_log(tmp_path):
    """Two different files with confusingly similar names: ``training.log`` is the process's
    relayed stdout, ``train.log`` is the trainer's own."""
    store = TrialLedgerStore(tmp_path / "ledger")
    runner = FakeRunner()
    tracker = _tracker(tmp_path, store=store, runner=runner)
    read_back = SearchLedger(store)
    seen: dict[str, Any] = {}

    def on_run(request: Any) -> None:
        seen["trial"] = request.trial_id
        seen["workspace"] = request.workspace

    runner.on_run = on_run
    tracker.run()

    workspace = Path(seen["workspace"])
    (workspace / "train.log").write_text("no stats here\n", encoding="utf-8")
    assert TrialWorkspaceTelemetry(read_back)(seen["trial"]).name == "train.log"
    (workspace / "training.log").write_text("[TPSTAT] step=1 total=10\n", encoding="utf-8")
    assert TrialWorkspaceTelemetry(read_back)(seen["trial"]).name == "training.log"


def test_the_workspace_locator_returns_nothing_for_a_trial_it_does_not_know(tmp_path):
    store = TrialLedgerStore(tmp_path / "ledger")
    assert TrialWorkspaceTelemetry(SearchLedger(store))("no-such-trial") is None


# --- Group G: the defects a second audit pass found -------------------------------------------
#
# Every test here fails on the version of the module that already passed the first 89 cases, and
# they are grouped because they share a shape: each is a way the audit layer could be *confident
# and wrong*.  A correct record reported as broken, a forged record reported as verified, a curve
# silently emptied, a documented capability that did not exist.  None of these is a crash, which is
# exactly why the first pass missed them -- the module was never red, it was just wrong.


class _InnerRunner:
    """An inner ``TrialRunner`` whose duration and failure a runner test controls."""

    def __init__(self, *, delay: float = 0.0, error: BaseException | None = None) -> None:
        self.delay = delay
        self.error = error
        self.runs = 0

    def preflight(self, request: Any) -> None:
        pass

    def run(self, request: Any) -> TrialOutcome:
        self.runs += 1
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return TrialOutcome(status="succeeded", objective=2.0)


class _Request:
    def __init__(self, trial_id: str) -> None:
        self.trial_id = trial_id


def _watcher_over(sink: Any, curve: tuple[TrialObservation, ...]) -> PruningWatcher:
    return PruningWatcher(
        policy=PruningPolicy(
            rules=(FloorRule(metric="reward.total", at_step=0, threshold=5.0, patience=1),)
        ),
        source=ListSource(curve),
        sink=sink,
    )


def _prune_one_trial(tmp_path: Path, *, reading: Any, policy: PruningPolicy, name: str) -> tuple[TrialLedgerStore, SearchLedger, SearchTracker, Any]:
    """Drive the documented pipeline -- tracker, watcher, sink -- until one trial is pruned."""
    store = TrialLedgerStore(tmp_path / name)
    ledger = SearchLedger(store)
    tracker = _tracker(tmp_path, store=store, runner=FakeRunner())
    source = ListSource((reward(1, reading),))
    sink = TrackerPruningSink(tracker=tracker, store=store, ledger=ledger, policy=policy)
    seen: dict[str, Any] = {}

    def on_run(request: Any) -> None:
        seen["trial"] = request.trial_id
        PruningWatcher(policy=policy, source=source, sink=sink).review(request.trial_id)

    tracker.runner.on_run = on_run
    tracker.run()
    return store, ledger, tracker, ledger.trial(seen["trial"])


# --- G1: the writer must not be the one producing un-auditable records ---------------------


def test_canonical_metrics_rewrites_only_the_readings_the_ledger_cannot_keep():
    metrics = {
        "nan": float("nan"),
        "inf": float("inf"),
        "neg": float("-inf"),
        "finite": 1.5,
        "count": 3,
        "gate": True,
        "text": "nan",
        "nothing": None,
        "nested": {"deep": float("nan"), "kept": 2.0},
        "series": [float("nan"), 4.0],
    }
    assert canonical_metrics(metrics) == {
        "nan": "nan",
        "inf": "inf",
        "neg": "-inf",
        "finite": 1.5,
        "count": 3,
        "gate": True,
        "text": "nan",
        "nothing": None,
        "nested": {"deep": "nan", "kept": 2.0},
        "series": ["nan", 4.0],
    }
    # The string "nan" is left alone, so this is not idempotent-by-accident: re-canonicalising the
    # output changes nothing, and a reading that was already text is not rewritten into anything.
    assert canonical_metrics(canonical_metrics(metrics)) == canonical_metrics(metrics)


def test_the_reference_sink_stores_a_non_finite_reading_as_text_rather_than_as_null(tmp_path):
    """The module's own writer is the last place that may produce an unauditable record.

    ``model_dump(mode="json")`` turns ``float('nan')`` into ``None``, and a reading lost that way
    can never be re-derived -- so the audit has to call the decision "unverifiable", the outcome
    its docstring reserves for a writer *outside* this module having bypassed the guard.  Measured
    before the fix: the ledger held ``{'reward': {'total': None}}`` and the summary reported
    ``verified=() unverifiable=(1,)`` for a decision the module itself had made.
    """
    policy = PruningPolicy(rules=(NaNRule(metric="reward.total", at_step=0),))
    store, ledger, _, trial = _prune_one_trial(
        tmp_path, reading=float("nan"), policy=policy, name="ledger"
    )
    assert trial.status == "pruned"
    assert [observation.metrics for observation in trial.observations] == [
        {"reward": {"total": "nan"}}
    ]
    summary = prune_summary(ledger, store=store)
    assert summary.verified == (trial.index,)
    assert summary.unverifiable == ()
    assert summary.contradicted == ()


# --- G2: "the ledger dropped it" and "it was never there" are different facts --------------


def test_a_record_citing_a_metric_the_curve_never_held_is_a_contradiction(tmp_path):
    """Both cases make ``metric_value`` return ``None``, so only the *key* tells them apart.

    A metric that is present-but-null is a reading the ledger could not keep.  A metric that has no
    key at all is a record citing evidence the ledger does not contain.  Reading them as one case
    files a forged record in the benign bucket.
    """
    policy = PruningPolicy(rules=(NaNRule(metric="reward.total", at_step=0),))
    curve = (obs(1, {"health": {"terminal_rate": 0.1}}),)
    decision = PruneDecisionRecord(
        id="prune-decision:t1",
        trial_id="t1",
        policy_fingerprint=policy.fingerprint,
        reason="nan_metric",
        step=1,
        through_step=1,
        curve_len=1,
        metric="reward.total",
        observed="nan",
        stop_reason=format_stop_reason("nan_metric", 1, "invented"),
    )
    assert metric_value(curve[0].metrics, "reward.total") is None
    assert reading_is_nulled(curve[0].metrics, "reward.total") is False
    assert policy.decide(curve) is None
    with pytest.raises(PruneAuditError, match="matches no rule"):
        verify_decision(decision, curve, policy)


def test_a_reading_the_ledger_really_nulled_is_reported_rather_than_raised():
    """The same decision text against a curve where the key *is* there and holds ``None``."""
    policy = PruningPolicy(rules=(NaNRule(metric="reward.total", at_step=0),))
    curve = (obs(1, {"reward": {"total": None}}),)
    decision = PruneDecisionRecord(
        id="prune-decision:t1",
        trial_id="t1",
        policy_fingerprint=policy.fingerprint,
        reason="nan_metric",
        step=1,
        through_step=1,
        curve_len=1,
        metric="reward.total",
        observed="nan",
        stop_reason=format_stop_reason("nan_metric", 1, "recovered"),
    )
    assert reading_is_nulled(curve[0].metrics, "reward.total") is True
    assert verify_decision(decision, curve, policy) == "unverifiable"


# --- G3: re-derivation runs over the prefix the decision saw, not the finished curve --------


def _decision_for(curve: tuple[TrialObservation, ...], policy: PruningPolicy) -> PruneDecisionRecord:
    verdict = policy.decide(curve)
    assert verdict is not None
    return PruneDecisionRecord(
        id="prune-decision:t1",
        trial_id="t1",
        policy_fingerprint=policy.fingerprint,
        reason=verdict.reason,
        step=verdict.step,
        through_step=verdict.step,
        curve_len=len(curve),
        matched_steps=verdict.matched_steps,
        metric=verdict.metric,
        observed=verdict.observed,
        threshold=verdict.threshold,
        stop_reason=verdict.stop_reason,
    )


def test_a_correct_decision_stays_verified_after_the_record_grows_past_it():
    """A pruned trial's record keeps growing: task 4 appends the reading collected on the way out.

    That later point is evidence about the trial and none at all about the decision.  A trailing
    window rule answers differently over it, so re-deriving against the finished curve reports a
    correct record as a contradiction -- and the record is the only thing on disk that says why the
    trial was killed.
    """
    policy = PruningPolicy(
        rules=(StallRule(metric="reward.total", at_step=0, min_delta=0.01, window=2),)
    )
    seen = (reward(1, 5.0), reward(2, 5.0))
    decision = _decision_for(seen, policy)
    assert decision.curve_len == 2
    assert verify_decision(decision, seen, policy) == "verified"

    grown = seen + (reward(3, 50.0),)
    assert verify_decision(decision, grown, policy) == "verified"


def test_a_record_claiming_more_observations_than_the_ledger_keeps_is_a_contradiction():
    policy = PruningPolicy(
        rules=(FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=1),)
    )
    curve = (reward(1, 0.5), reward(2, 0.4))
    decision = _decision_for(curve, policy)
    overclaimed = PruneDecisionRecord.model_validate(
        {**decision.model_dump(mode="json"), "curve_len": 5}
    )
    with pytest.raises(PruneAuditError, match="not on disk"):
        verify_decision(overclaimed, curve, policy)


def test_a_record_whose_last_seen_step_the_curve_contradicts_is_caught():
    policy = PruningPolicy(
        rules=(FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=1),)
    )
    curve = (reward(1, 0.5), reward(2, 0.4))
    decision = _decision_for(curve, policy)
    wrong = PruneDecisionRecord.model_validate(
        {**decision.model_dump(mode="json"), "through_step": 9, "curve_len": 2}
    )
    with pytest.raises(PruneAuditError, match="saw up to step 9"):
        verify_decision(wrong, curve, policy)


def test_prune_summary_names_a_contradicting_record_rather_than_raising_out_of_the_ranking(tmp_path):
    """One bad record must not cost every other trial its verdict, and must not be hidden either."""
    policy = PruningPolicy(
        rules=(FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=1),)
    )
    store = TrialLedgerStore(tmp_path / "ledger")
    ledger = SearchLedger(store)
    tracker = _tracker(tmp_path, store=store, runner=FakeRunner())
    source = ListSource((reward(1, 0.25),))
    sink = TrackerPruningSink(tracker=tracker, store=store, ledger=ledger, policy=policy)
    seen: dict[str, Any] = {}

    def on_run(request: Any) -> None:
        trial_id = request.trial_id
        seen["trial"] = trial_id
        PruningWatcher(policy=policy, source=source, sink=sink).review(trial_id)

        real = store.latest(DECISION_RECORD_TYPE, f"prune-decision:{trial_id}")
        assert real is not None
        # The record the module wrote is auditable: re-deriving it from the curve agrees.
        assert (
            verify_decision(
                PruneDecisionRecord.model_validate(real),
                ledger.trial(trial_id).observations,
                policy,
            )
            == "verified"
        )
        # Supersede it with one field changed.  This has to happen while the search is open --
        # a finished search refuses every trial write, which is the guard working, not an
        # obstacle to work around.
        store.append(
            DECISION_RECORD_TYPE,
            PruneDecisionRecord.model_validate({**real, "observed": 0.99}),
            actor="test",
            event_type="supersede",
            precondition=tracker._require_open,
        )

    tracker.runner.on_run = on_run
    tracker.run()
    trial = ledger.trial(seen["trial"])
    assert trial.status == "pruned"

    summary = prune_summary(ledger, store=store)
    assert summary.contradicted == (trial.index,)
    assert summary.verified == ()
    assert summary.unverifiable == ()
    assert f"trial {trial.index}" in summary.contradiction_detail
    assert "observed" in summary.contradiction_detail


# --- G4: every field of the verdict is compared, not just the rule and the step --------------


@pytest.mark.parametrize(
    ("field", "forgery"),
    [
        ("metric", "health.terminal_rate"),
        ("observed", 999.0),
        ("threshold", 0.0),
        ("matched_steps", (2,)),
        ("stop_reason", format_stop_reason("metric_floor", 2, "a story the curve never tells")),
    ],
)
def test_a_forged_verdict_field_is_caught_even_when_the_rule_and_the_step_agree(field, forgery):
    """A record whose reading and threshold no rule could produce must not pass as audited."""
    policy = PruningPolicy(
        rules=(FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=2),)
    )
    curve = (reward(1, 0.5), reward(2, 0.4))
    decision = _decision_for(curve, policy)
    assert verify_decision(decision, curve, policy) == "verified"
    forged = PruneDecisionRecord.model_validate({**decision.model_dump(mode="json"), field: forgery})
    with pytest.raises(PruneAuditError, match=field):
        verify_decision(forged, curve, policy)


def test_the_forgery_message_names_the_recorded_and_the_re_derived_value():
    policy = PruningPolicy(
        rules=(FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=2),)
    )
    curve = (reward(1, 0.5), reward(2, 0.4))
    decision = _decision_for(curve, policy)
    forged = PruneDecisionRecord.model_validate(
        {**decision.model_dump(mode="json"), "observed": 999.0}
    )
    with pytest.raises(PruneAuditError) as raised:
        verify_decision(forged, curve, policy)
    assert "observed: recorded 999.0, re-derived 0.4" in str(raised.value)


# --- G5: a rotation the size test cannot see must not read as an empty curve -----------------


def _write_ticks(path: Path, steps: list[tuple[int, float]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for step, value in steps:
            handle.write(
                json.dumps({"type": "train_tick", "step": step, "reward": {"total": value}}) + "\n"
            )


def test_a_rotation_to_a_same_or_larger_file_is_noticed_rather_than_read_as_empty(tmp_path):
    """The size test alone misses the destructive case: a rewrite that is not smaller.

    The stale byte offset then lands mid-line, the fragment does not start with ``{``, so the whole
    JSONL is re-read through the text-log parser -- which understands none of it.  The curve comes
    back empty, the points already read are discarded, and the cursor parks at the new file's end,
    so the loss is permanent.  Measured before the guard: ``[1, 2, 3]`` -> ``[]`` -> ``[200]``.
    """
    telemetry = tmp_path / "t.telemetry.jsonl"
    source = JsonlTelemetryCurveSource(locate=lambda trial_id: telemetry)

    _write_ticks(telemetry, [(1, 2.0), (2, 2.0), (3, 2.0)])
    assert [observation.step for observation in source.curve("t1")] == [1, 2, 3]

    _write_ticks(telemetry, [(100 + index, 1.0) for index in range(10)])
    assert [observation.step for observation in source.curve("t1")] == [
        100 + index for index in range(10)
    ]

    with telemetry.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "train_tick", "step": 200, "reward": {"total": 1.0}}) + "\n")
    assert [observation.step for observation in source.curve("t1")] == [
        *[100 + index for index in range(10)],
        200,
    ]


def test_the_curve_source_leaves_a_torn_line_and_stops_exactly_where_it_parsed(tmp_path):
    """The cursor must stop where the parsed bytes end; the steps alone cannot show that.

    This test was originally named for the incremental read and asserted only the step list, which
    is not the same claim: a mutant that consumes the torn trailing line returns the *identical*
    curve while moving the cursor to ``size + 1``.  That overshoot invalidates the rotation
    fingerprint, so the next poll starts from zero and re-reads the whole file -- correct output,
    and the entire point of the class gone.  The cursor assertions are the ones that pin it.
    """
    telemetry = tmp_path / "t.telemetry.jsonl"
    source = JsonlTelemetryCurveSource(locate=lambda trial_id: telemetry)
    fragment = '{"type": "train_tick", "step": 4, "rew'

    _write_ticks(telemetry, [(1, 1.0), (2, 1.0)])
    assert [observation.step for observation in source.curve("t1")] == [1, 2]
    # The file ends on a newline, so nothing is left over: the cursor sits exactly at EOF.
    assert source._cursors["t1"][1] == telemetry.stat().st_size

    with telemetry.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "train_tick", "step": 3, "reward": {"total": 1.0}}) + "\n")
        handle.write(fragment)
    assert [observation.step for observation in source.curve("t1")] == [1, 2, 3]
    assert source._cursors["t1"][1] == telemetry.stat().st_size - len(fragment)

    with telemetry.open("a", encoding="utf-8") as handle:
        handle.write('ard": {"total": 1.0}}\n')
    assert [observation.step for observation in source.curve("t1")] == [1, 2, 3, 4]
    assert source._cursors["t1"][1] == telemetry.stat().st_size


def test_the_watcher_keeps_forwarding_after_the_source_starts_over():
    """Forwarding by count goes permanently silent after a reset; forwarding by step does not."""
    source = ListSource((reward(1, 1.0), reward(2, 1.0), reward(3, 1.0)))
    sink = RecordingSink()
    watcher = PruningWatcher(policy=PruningPolicy(), source=source, sink=sink)
    assert watcher.review("t1") is None
    assert sink.curves == [("t1", (reward(1, 1.0), reward(2, 1.0), reward(3, 1.0)))]

    # The file was rotated: the source now reports a shorter curve of different steps.  A count-based
    # cursor would ask for ``curve[3:]`` on a two-point curve and forward nothing, forever.
    source.curve_value = (reward(9, 1.0), reward(10, 1.0))
    assert watcher.review("t1") is None
    assert sink.curves[-1] == ("t1", (reward(9, 1.0), reward(10, 1.0)))

    # A step already forwarded is not forwarded twice, which is what keeps ``observe`` from
    # building the curve 1, 2, 1, 2 that this module's own validator refuses.
    source.curve_value = (reward(9, 1.0), reward(10, 1.0), reward(11, 1.0))
    assert watcher.review("t1") is None
    assert sink.curves[-1] == ("t1", (reward(11, 1.0),))


# --- G6: the pruning runner, which had no test at all ----------------------------------------


def test_the_pruning_runner_returns_the_inner_outcome_when_the_inner_runner_cooperates():
    inner = _InnerRunner()
    runner = PruningTrialRunner(
        runner=inner, watcher=_watcher_over(RecordingSink(), (reward(1, 9.0),)), interval_seconds=0.01
    )
    outcome = runner.run(_Request("t1"))
    assert outcome.status == "succeeded"
    assert outcome.objective == 2.0
    assert inner.runs == 1


def test_the_pruning_runner_reraises_when_the_inner_runner_dies():
    runner = PruningTrialRunner(
        runner=_InnerRunner(error=ValueError("training crashed")),
        watcher=_watcher_over(RecordingSink(), (reward(1, 9.0),)),
        interval_seconds=0.01,
    )
    with pytest.raises(ValueError, match="training crashed"):
        runner.run(_Request("t1"))
    assert [t for t in threading.enumerate() if t.name.startswith("prune-")] == []


def test_a_failing_poller_neither_abandons_the_worker_nor_swallows_its_failure():
    """``review`` raising is the ordinary failure mode, and the old code lost the training crash.

    Letting the poller's exception escape the loop meant ``thread.join()`` was never reached: the
    worker kept running unmanaged, and the inner runner's own exception was stored in a frame
    nobody returned to -- no traceback, and no ``threading.excepthook`` call either.
    """

    class PollerBreaks(RecordingSink):
        """A sink that fails the way the ledger does when the trial ended underneath it."""

        def apply_verdict(self, trial_id: str, verdict: Any) -> bool:
            raise SearchError("the trial finished underneath the poller")

    inner = _InnerRunner(delay=0.2, error=ValueError("training crashed"))
    runner = PruningTrialRunner(
        runner=inner,
        watcher=_watcher_over(PollerBreaks(), (reward(1, 0.1),)),
        interval_seconds=0.02,
    )
    with pytest.raises(SearchError, match="underneath the poller") as raised:
        runner.run(_Request("t1"))

    assert [t for t in threading.enumerate() if t.name.startswith("prune-")] == []
    notes = getattr(raised.value, "__notes__", [])
    assert any("training crashed" in note for note in notes), notes
    assert inner.runs == 1


def test_terminate_does_not_read_a_missing_process_handle_as_a_clean_stop():
    """``True`` means "observed to have ended"; a handle that was never given proves nothing."""
    runner = PruningTrialRunner(
        runner=_InnerRunner(),
        watcher=_watcher_over(RecordingSink(), (reward(1, 9.0),)),
        interval_seconds=1.0,
    )
    assert runner.terminate("t1", "prune:nan_metric@1") is False


def test_terminate_reports_true_only_for_a_process_it_observed_ended():
    class Exited:
        """A process handle that reports itself finished -- what ``terminate`` must count as ended."""

        def poll(self) -> int:
            return 0

        def kill(self) -> None:  # pragma: no cover - must never be reached
            raise AssertionError("an exited process must not be killed")

    runner = PruningTrialRunner(
        runner=_InnerRunner(),
        watcher=_watcher_over(RecordingSink(), (reward(1, 9.0),)),
        interval_seconds=1.0,
        sleep=lambda seconds: None,
    )
    assert runner.terminate("t1", "prune:nan_metric@1", Exited()) is True


def test_the_module_no_longer_carries_a_sentinel_it_never_writes():
    """A constant nothing produces is a promise nothing keeps; it was removed rather than left."""
    import autotuner.research.search_pruning as module

    assert not hasattr(module, "_SENTINEL")


# --- G7: min_step is a floor on the evidence, not a counter ---------------------------------


def test_min_step_excludes_the_evidence_a_rule_would_otherwise_fire_on():
    """A rule with ``at_step=0`` must not reach back past the policy's own floor.

    Measured before the fix: ``min_step=500`` still produced ``metric_floor at step 1``, so the
    trial was killed by exactly the early noise the floor was configured to exclude.
    """
    policy = PruningPolicy(
        min_step=500,
        rules=(FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=1),),
    )
    curve = (reward(1, 0.1),) + tuple(reward(step, 0.1) for step in (500, 501, 502))
    verdict = policy.decide(curve)
    assert verdict is not None
    assert verdict.step == 500
    assert verdict.matched_steps == (500,)
    assert all(step >= policy.min_step for step in verdict.matched_steps)


def test_min_step_can_hold_back_a_rule_entirely_when_nothing_clears_it():
    policy = PruningPolicy(
        min_step=500,
        rules=(FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=1),),
    )
    assert policy.decide((reward(1, 0.1), reward(2, 0.1))) is None


# --- Group H: the guards the design named and then never wrote -------------------------------
#
# Section 4 of the design document lists these among its test cases, and §9 reconciled only line
# and case *counts* -- so for a while the document presented five anti-drift guards as landed when
# none of them existed.  They are the load-bearing ones, which is why they are here now rather than
# deleted from the document: each pins a claim that is otherwise maintained by hand.

EMITTER = Path("products") / "taili" / "blind_locomotion" / "telemetry_emit.py"
CAPTURED_PAYLOAD = Path("output") / "current_balanced_telemetry.jsonl"

#: What the parser is allowed to reach for.  R-10 of the design: the research layer may import
#: ``.hyperparameter_search``, ``.trial_ledger`` and (for ``content_hash``) ``.research_ledger``,
#: and must never reach up into the console, the product tree, training or the adapter layer.
ALLOWED_RELATIVE_IMPORTS = {".hyperparameter_search", ".trial_ledger", ".research_ledger"}
FORBIDDEN_PREFIXES = (
    "autotuner.locomotion_console",
    "autotuner.training",
    "autotuner.adapter",
    "products",
)


def test_the_parser_still_names_the_sections_the_emitter_writes():
    """A rename in the emitter would empty the curve silently; only this reads both sides."""
    source = EMITTER.read_text(encoding="utf-8")
    assert '"type": "train_tick"' in source, (
        "the emitter no longer stamps train_tick, which is the parser's only discriminator"
    )
    for section in ("reward", "curriculum", "health", "command", "counters"):
        assert f'"{section}":' in source, f"the emitter no longer writes a {section!r} section"
    for key in ("step", "total_steps"):
        assert f'"{key}":' in source, f"the emitter no longer writes a {key!r} key"
    # The other direction: what the parser would produce from a line in that shape.
    parsed = parse_telemetry_payload(
        {
            "type": "train_tick",
            "step": 4,
            "total_steps": None,
            "reward": {"total": 1.5},
            "health": {"terminal_rate": 0.0},
            "curriculum": {"terrain_mean": 3.0},
            "command": {"cmd_vx": 0.4},
            "counters": {"falls": 0},
        }
    )
    assert parsed is not None
    assert metric_value(parsed.metrics, "reward.total") == 1.5
    assert metric_value(parsed.metrics, "health.terminal_rate") == 0.0
    assert metric_value(parsed.metrics, "curriculum.terrain_mean") == 3.0
    assert metric_value(parsed.metrics, "command.cmd_vx") == 0.4
    assert metric_value(parsed.metrics, "counters.falls") == 0


def test_the_parser_reads_the_keys_the_human_line_emitter_writes():
    """The fallback path reads ``step=``/``total=``; those are spelled in a second place."""
    source = EMITTER.read_text(encoding="utf-8")
    for marker in ("[TPSTAT]", "[TPREW]"):
        start = source.index(f'"{marker} "')
        block = source[start : source.index("]))", start)]
        assert '("step",' in block, f"{marker} no longer writes step=, so the parser drops every line"
        assert '("total",' in block, f"{marker} no longer writes total="
    parsed = parse_telemetry_lines(
        "[TPSTAT] step=100 total=1000 pct=50.00 fps=1.00\n"
        "[TPREW] step=100 total=2.5 tracking_lin=0.4\n"
    )
    assert [observation.step for observation in parsed] == [100]
    assert parsed[0].metrics["total_steps"] == 1000
    assert metric_value(parsed[0].metrics, "reward.total") == 2.5


def test_the_captured_payload_the_design_measured_still_reads_the_same_way():
    """Measured against a real capture rather than a hand-written dict -- when one is present.

    The file is the product's own output and is gitignored, so this skips rather than fails on a
    tree that never ran training.  What it pins cannot be pinned by a fixture: the shipped payload
    carries ``total_steps: null`` on every line, which is the reason a fraction gate must stay
    silent instead of inventing a denominator.
    """
    if not CAPTURED_PAYLOAD.exists():
        pytest.skip(f"{CAPTURED_PAYLOAD} is not on this tree (it is a product output, not a fixture)")
    with CAPTURED_PAYLOAD.open(encoding="utf-8") as handle:
        line = handle.readline()
    payload = json.loads(line)
    parsed = parse_telemetry_payload(payload)
    assert parsed is not None
    assert parsed.metrics["total_steps"] is None
    for path, expected in (
        ("reward.total", payload["reward"]["total"]),
        ("curriculum.terrain_mean", payload["curriculum"]["terrain_mean"]),
        ("health.terminal_rate", payload["health"]["terminal_rate"]),
        ("health.fall_rate", payload["health"]["fall_rate"]),
        ("reward.tracking_lin", payload["reward"]["tracking_lin"]),
    ):
        assert metric_value(parsed.metrics, path) == expected, path
        assert is_nonfinite(metric_value(parsed.metrics, path)) is False, path
    # ``total_steps`` is null on every line of the capture, so a fraction gate on a curve built
    # from it must not fire at all -- the behaviour the ``null`` exists to produce.
    gate = PruningPolicy(
        rules=(FloorRule(metric="reward.total", at_total_fraction=0.5, threshold=1e9, patience=1),)
    )
    assert gate.decide((parsed,)) is None


def test_the_module_imports_only_the_layers_it_is_allowed_to():
    """R-10, as an AST walk rather than a grep -- a grep cannot tell an import from a string."""
    source = Path("autotuner/research/search_pruning.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    relative: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(FORBIDDEN_PREFIXES), alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                name = "." * node.level + (node.module or "")
                relative.add(name)
                assert name in ALLOWED_RELATIVE_IMPORTS, f"{name} is outside the research layer"
            else:
                assert not (node.module or "").startswith(FORBIDDEN_PREFIXES), node.module
    assert relative == ALLOWED_RELATIVE_IMPORTS


def test_a_decision_re_derives_identically_in_a_fresh_interpreter():
    """``verify_decision`` must not lean on anything process-local.

    The design's claim is that a third party can recompute a decision *from the ledger alone*.  A
    same-process test cannot see that: a module-level cache, a memoised rule or an iteration-order
    dependence would all survive it.  This rebuilds the policy from its stored form in a new
    interpreter and compares every field.
    """
    policy = PruningPolicy(
        rules=(
            FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=2),
            NaNRule(metric="health.terminal_rate", at_step=0),
        ),
        min_observations=2,
    )
    curve = (reward(1, 0.5), reward(2, 0.4))
    verdict = policy.decide(curve)
    assert verdict is not None
    script = (
        "import json,sys;"
        "from autotuner.research.search_pruning import PruningPolicy,TrialObservation;"
        "policy=PruningPolicy.model_validate(json.loads(sys.argv[1]));"
        "curve=tuple(TrialObservation.model_validate(o) for o in json.loads(sys.argv[2]));"
        "v=policy.decide(curve);"
        "print(json.dumps(v.model_dump(mode='json') if v else None, sort_keys=True))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, policy.model_dump_json(), json.dumps(
            [observation.model_dump(mode="json") for observation in curve]
        )],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == verdict.model_dump(mode="json")


def test_a_pruned_trial_still_counts_against_the_sampler_proposals(tmp_path):
    """§1's budget invariant: pruning must not make a search look like it skipped a proposal.

    ``close_run`` refuses when ``trial_count != stats["proposals"]`` -- a pruned trial has to stay
    counted, or an honest search becomes unclosable and a dishonest one becomes closable.
    """
    policy = PruningPolicy(
        rules=(FloorRule(metric="reward.total", at_step=0, threshold=1.0, patience=1),)
    )
    store = TrialLedgerStore(tmp_path / "ledger")
    ledger = SearchLedger(store)
    tracker = _tracker(tmp_path, store=store, runner=FakeRunner())
    source = ListSource((reward(1, 0.25),))
    sink = TrackerPruningSink(tracker=tracker, store=store, ledger=ledger, policy=policy)

    def on_run(request: Any) -> None:
        PruningWatcher(policy=policy, source=source, sink=sink).review(request.trial_id)

    tracker.runner.on_run = on_run
    tracker.run()
    assert [record.status for record in ledger.trials()] == ["pruned"]

    closed = tracker.close_run(SamplerStats(proposals=1, attempts=1, exhausted=True))
    # The pruned trial is still a landed trial, so the counts agree and the close is allowed.
    # (The refusal when they disagree is task 4's own case, in test_hyperparameter_search.py.)
    assert len(ledger.trials()) == closed.stats["proposals"] == 1
    assert prune_summary(ledger, store=store).pruned_indices == (1,)
