"""Pinned behaviour of the post-search read: which trial won, what the sample is worth, and whether the exported file is the one that trial ran.

Every test here pins one way the answer can be **quietly wrong**, which is why each is worth its
own name:

* The read goes through ``SearchLedger.run()`` and not ``trials()``.  ``trials()`` returns records
  for a duplicate assignment key and for an index gap, so reading through it yields a *smaller,
  confident-looking* sample -- and the smaller sample usually has a different best trial.  The
  first three tests pin that the difference is real and that the downgrade is reported.
* A number that was never read is not zero.  ``recorded``/``proposed``/``exhausted`` are ``None``
  or the report says "unreadable"; a report that fills them with 0 turns "nothing was read" into
  "it read nothing", which is a different (and false) fact.
* The direction of the ranking is a required argument.  Nothing in the ledger records whether an
  objective is better larger or smaller, so a default would be wrong half the time and the wrong
  half exports the worst config under the label "best".
* The export is checked against the fingerprint the *trial record* has carried since it
  materialised, not against the injector's own report (which would agree with itself by
  construction).
* The receipt carries a pair of hashes rather than a ``verified: bool``, because a stored flag
  that is true whenever the record exists is not evidence.

The measurements behind the constructions (index gaps and duplicate keys are refused by
``close_run``, so a report describing one can only have come from a tampered or foreign ledger;
``SamplerStats.exhausted`` defaults to ``False``; an open run writes ``{}`` stats) are recorded in
``docs/P4.4_task6_design.md`` §11 and repeated next to the test that depends on them.
"""
from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml
from pydantic import ValidationError

from autotuner.research.config_injection import inject_into_file, load_config_file
from autotuner.research.hyperparameter_sampler import (
    SamplerStats,
    SearchPlan,
    build_sampler,
)
from autotuner.research.hyperparameter_search import (
    SearchLedger,
    SearchTracker,
    TrialOutcome,
    TrialRecord,
    TrialStatus,
    _supersede,
)
from autotuner.research.hyperparameter_space import SearchSpaceSchema, load_search_space
from autotuner.research.research_ledger import canonical_json, content_hash
from autotuner.research.search_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    COVERAGE_VERDICTS,
    EXPORT_SCHEMA_VERSION,
    MANIFEST_SUFFIX,
    Coverage,
    ExportArtifact,
    ExportRefusedError,
    RankingRule,
    Reading,
    SearchAnalysis,
    SearchAnalysisError,
    TrialCensus,
    UnrankedTrial,
    _coverage_verdict,
    analyze,
    export_best,
    verify_export,
)
from autotuner.research.search_pruning import metric_value, reading_is_nulled
from autotuner.research.trial_ledger import TrialLedgerStore

REPO_ROOT = Path(__file__).resolve().parents[3]
_PRODUCT_DIR = REPO_ROOT / "products" / "taili" / "blind_locomotion"
SPACE_PATH = _PRODUCT_DIR / "hyperparameter_space.yaml"
CONFIG_PATH = _PRODUCT_DIR / "taili_blind_config.yaml"

MODULE_PATH = REPO_ROOT / "autotuner" / "research" / "search_analysis.py"

#: The reading state vocabulary is a *refinement* of the pruning layer's three answers
#: (``metric_value``/``is_nonfinite``/``reading_is_nulled``): every state here must be
#: consistent with what those functions say, and the two extra states (``absent`` and
#: ``not_a_number``) exist only because they are two different sentences to an operator.
NON_VALUE_STATES = ("no_objective", "absent", "nulled", "not_a_number", "nonfinite")


# --- Fixtures: the shipped product files, and a writable copy of the config --------------------


@lru_cache(maxsize=None)
def _taili_space() -> SearchSpaceSchema:
    """The shipped search space, read from the product files."""
    return load_search_space(yaml.safe_load(SPACE_PATH.read_text(encoding="utf-8")))


@lru_cache(maxsize=None)
def _narrow_space() -> SearchSpaceSchema:
    """The shipped space with one parameter narrowed.

    A space that differs from the shipped one only in a value: the plan drawn from it is a
    different plan, so an export can be asked to write a winning assignment into the wrong
    space and must refuse.  Every key still exists in the config, so nothing else changes.
    """
    payload = yaml.safe_load(SPACE_PATH.read_text(encoding="utf-8"))
    for spec in payload["parameters"]:
        if spec["name"] == "skrl.agent.mini_batches":
            spec["high"] = 20
            return load_search_space(payload)
    raise AssertionError("the shipped space no longer declares skrl.agent.mini_batches")


@lru_cache(maxsize=None)
def _grid_space() -> SearchSpaceSchema:
    """A one-parameter, fully enumerable space over a key the shipped config really holds.

    The shipped space cannot be enumerated as a grid: ``kl_threshold`` has no ``grid_points``, and
    ``GridSampler`` refuses to invent a density for it.  This space exists so the two grid verdicts
    have a real producer rather than being reachable only by hand-written records.
    """
    return load_search_space(
        {
            "parameters": [
                {
                    "name": "skrl.agent.mini_batches",
                    "kind": "discrete",
                    "low": 4,
                    "high": 8,
                    "step": 4,
                    "default": 4,
                }
            ]
        }
    )


def _copy_config(tmp_path: Path, label: str) -> Path:
    """A writable copy of the shipped config, so a test may change or delete the source.

    Byte-for-byte, so the tracker's own check (the file on disk hashes like the mapping passed
    in) still passes and the recorded source fingerprint is the shipped config's.
    """
    target = tmp_path / f"source_{label}.yaml"
    target.write_bytes(CONFIG_PATH.read_bytes())
    return target


# --- Fixtures: a scripted execution backend ----------------------------------------------------


class ScriptedRunner:
    """A runner that answers each trial with the next scripted outcome.

    Unlike the other test module's ``FakeRunner``, the outcomes differ per trial -- a ranking test
    where every trial scores the same cannot tell "the best was exported" from "the first was".
    """

    def __init__(self, outcomes: list[TrialOutcome], *, preflight_error: BaseException | None = None) -> None:
        self.outcomes = list(outcomes)
        self.preflight_error = preflight_error
        self.preflights: list[Any] = []
        self.requests: list[Any] = []

    def preflight(self, request: Any) -> None:
        self.preflights.append(request)
        if self.preflight_error is not None:
            raise self.preflight_error

    def run(self, request: Any) -> TrialOutcome:
        self.requests.append(request)
        if len(self.requests) > len(self.outcomes):
            raise AssertionError(
                f"the plan proposed a trial {len(self.requests)} the test did not script: "
                f"only {len(self.outcomes)} outcomes were given"
            )
        return self.outcomes[len(self.requests) - 1]


class FailingRunner(ScriptedRunner):
    """A runner that dies on the Nth trial, after the trials before it were recorded."""

    def __init__(self, outcomes: list[TrialOutcome], *, fail_at: int) -> None:
        super().__init__(outcomes)
        self.fail_at = fail_at

    def run(self, request: Any) -> TrialOutcome:
        if len(self.requests) + 1 == self.fail_at:
            self.requests.append(request)
            raise RuntimeError("the backend died")
        return super().run(request)


def _outcome(
    *,
    status: TrialStatus = "succeeded",
    objective: float | None = 1.0,
    metrics: dict[str, Any] | None = None,
    message: str = "",
) -> TrialOutcome:
    return TrialOutcome(
        status=status,
        objective=objective,
        metrics=metrics or {},
        message=message or ("stopped by the pruner" if status in {"pruned", "stopped"} else ""),
    )


@dataclass(frozen=True)
class _Search:
    """One search on disk plus the two paths an analysis needs.

    ``ledger()`` builds a **fresh** reader every time: that is the analyst's situation (a report is
    written after the process that ran the search is gone), and it is what makes a test able to
    tamper with the file between reads and be believed.
    """

    tracker: SearchTracker
    store: TrialLedgerStore
    source: Path

    @property
    def run_ref(self) -> str:
        return self.tracker.run_ref

    def ledger(self) -> SearchLedger:
        return SearchLedger(self.store)

    def report(
        self, *, direction: str = "maximize", key: str = "objective", top_n: int = 20
    ) -> SearchAnalysis:
        return analyze(
            self.ledger(),
            self.run_ref,
            rule=RankingRule(key=key, direction=direction),  # type: ignore[arg-type]
            top_n=top_n,
        )

    def record(self, index: int) -> TrialRecord:
        """The ledger's own record for one trial index: what a comparison must be made against."""
        trials = {record.index: record for record in self.ledger().run(self.run_ref).trials}
        return trials[index]


def _search(
    tmp_path: Path,
    outcomes: list[TrialOutcome],
    *,
    space: SearchSpaceSchema | None = None,
    plan: SearchPlan | None = None,
    label: str = "search",
) -> _Search:
    """Run a whole search with a scripted runner and return the reader's view of it."""
    space = space if space is not None else _taili_space()
    plan = plan if plan is not None else SearchPlan.for_random(space, seed=7, budget=len(outcomes))
    source = _copy_config(tmp_path, label)
    store = TrialLedgerStore(tmp_path / f"ledger_{label}")
    tracker = SearchTracker(
        store=store,
        space=space,
        plan=plan,
        source_config=load_config_file(source),
        source_config_path=source,
        work_root=tmp_path / f"work_{label}",
        runner=ScriptedRunner(outcomes),
    )
    tracker.run()
    return _Search(tracker=tracker, store=store, source=source)


def _crashed_search(tmp_path: Path, outcomes: list[TrialOutcome], *, fail_at: int, label: str) -> _Search:
    """A search whose runner died mid-way: the trials before it hold real results and the run is open.

    ``run()`` records the failure and then re-raises, so nothing closes the run -- which is exactly
    the state an operator finds after a backend dies, and the one where "the sample has results but
    was not count-checked" is true rather than hypothetical.

    ``outcomes`` must hold one outcome per trial that ran, so ``fail_at - 1`` of them: the trial at
    ``fail_at`` is the one the runner dies on.
    """
    assert len(outcomes) == fail_at - 1, "the script and the failure point disagree"
    space = _taili_space()
    source = _copy_config(tmp_path, label)
    store = TrialLedgerStore(tmp_path / f"ledger_{label}")
    tracker = SearchTracker(
        store=store,
        space=space,
        plan=SearchPlan.for_random(space, seed=7, budget=fail_at),
        source_config=load_config_file(source),
        source_config_path=source,
        work_root=tmp_path / f"work_{label}",
        runner=FailingRunner(outcomes, fail_at=fail_at),
    )
    with pytest.raises(RuntimeError, match="the backend died"):
        tracker.run()
    return _Search(tracker=tracker, store=store, source=source)


def _hand_driven(
    tmp_path: Path,
    *,
    space: SearchSpaceSchema | None = None,
    plan: SearchPlan | None = None,
    label: str = "hand",
) -> _Search:
    """A tracker on a fresh root whose trials the test materialises itself.

    ``begin_trial`` is public and supported, so everything built here goes in through the front
    door; only the *closing* record is ever written by hand in the tests below, and each such test
    says why.

    ``plan`` defaults to a random plan with ``budget=2``.  It is a parameter because the two grid
    rows of the coverage verdict are only reachable through a grid plan, and a test that pins the
    order of those rows has to hand-drive a grid (``test_a_grid_closed_without_the_exhausted_flag_
    reports_unstated``).
    """
    space = space if space is not None else _taili_space()
    source = _copy_config(tmp_path, label)
    store = TrialLedgerStore(tmp_path / f"ledger_{label}")
    tracker = SearchTracker(
        store=store,
        space=space,
        plan=plan if plan is not None else SearchPlan.for_random(space, seed=7, budget=2),
        source_config=load_config_file(source),
        source_config_path=source,
        work_root=tmp_path / f"work_{label}",
        runner=ScriptedRunner([]),
    )
    return _Search(tracker=tracker, store=store, source=source)


def _assignments(tracker: SearchTracker) -> tuple[dict[str, Any], ...]:
    return build_sampler(tracker.space, tracker.plan).collect()


def _value_at(mapping: dict[str, Any], dotted: str) -> Any:
    """Follow one of the space's dotted parameter names into a nested mapping."""
    current: Any = mapping
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _materialise(
    case: _Search,
    index: int,
    assignment: dict[str, Any],
    status: TrialStatus = "succeeded",
    objective: float | None = None,
    attempt: int = 1,
) -> TrialRecord:
    """Materialise one trial and end it, without reading the run back.

    ``finish`` returns the record it wrote, so this helper never touches ``read()`` -- which matters
    for the tests below that deliberately build a ledger whose run-scoped checks *fail*, and where
    ``read()`` therefore raises.

    ``objective`` is written by a second append because ``finish`` takes none: in a real run the
    objective comes from the runner's outcome (``hyperparameter_search.py:1239``), and a caller
    ending a trial by hand has only this route to the same record shape.  A trial ended without one
    is a terminal trial with no score, which is a different test.
    """
    record = case.tracker.begin_trial(index, assignment, attempt=attempt)
    finished = case.tracker.finish(
        record.id, status, stop_reason="the pruner cut it short" if status == "pruned" else ""
    )
    if objective is not None:
        _append(case, _supersede(finished, objective=objective), "search_trial")
    return finished


def _append(case: _Search, record: Any, record_type: str) -> None:
    case.tracker.store.append(record_type, record, actor="test", event_type="supersede")


def _fake_close(
    case: _Search, *, proposals: int, trial_count: int, exhausted: bool = True
) -> None:
    """Write a closing record by hand.

    No code path in this repository produces one of these with a count that disagrees with the
    trials on disk (``close_run`` refuses, and the ``SearchRunRecord`` validator refuses to even
    build it).  A report that *describes* such a ledger therefore describes a tampered or foreign
    one, which is precisely the case a reader must survive: refusing to describe it helps nobody.
    """
    header = case.tracker.open_run()
    _append(
        case,
        _supersede(
            header,
            status="closed",
            stats={"proposals": proposals, "attempts": proposals, "exhausted": exhausted},
            trial_count=trial_count,
        ),
        "search_run",
    )


# --- A: the read, and how much of it actually ran ----------------------------------------------


def test_the_read_is_the_one_that_runs_the_run_scoped_checks(tmp_path):
    """``trials()`` returns records under a duplicate ``assignment_key``; ``run()`` does not.

    Both halves are asserted, because the module's whole read-side argument is that the two
    differ: ``trials()`` calls ``refresh()`` (chain + layout) but not ``_trials_of``, so reading a
    run through it yields a sample the run-scoped checks would have rejected -- smaller, and
    confidently enough shaped to rank.  Two indices drawn from one point is one trial recorded
    twice, and averaging that in as two observations is a silent lie about the sample size.
    """
    case = _hand_driven(tmp_path)
    assignments = _assignments(case.tracker)
    case.tracker.open_run()
    _materialise(case, 1, assignments[0], objective=1.5)
    _materialise(case, 2, assignments[0], objective=2.5)  # the same point, a second time

    unfiltered = case.ledger().trials(run_ref=case.run_ref)
    assert [record.index for record in unfiltered] == [1, 2]

    report = case.report()
    assert report.sample_trust == "layout_checked"
    assert "assignment_key" in report.error
    assert report.census.recorded == 2
    # The sample it ranks is the one the run-scoped checks would have refused: two rows, one point.
    assert report.ranked_count == 2
    assert len({row.assignment_key for row in report.ranked}) == 1
    assert any("cannot be used to pick a winner" in note for note in report.caveats)


def test_an_index_gap_is_reported_and_the_sample_is_downgraded(tmp_path):
    """An index that was never materialised is named, not smoothed over.

    ``begin_trial`` accepts any index >= 1, so an index gap is writable and only the read side
    refuses it.  The report must say which sample shape it holds: the trial count alone cannot
    distinguish "2 trials" from "2 of the 3 the plan drew".

    The report of a gapped ledger is also where ``SearchAnalysis`` nearly broke its own promise:
    the recorded trial indices are not ``1..N``, so a validator bounding a named index by the
    *number* of recorded trials raises on a ledger it exists to describe (measured).
    """
    case = _hand_driven(tmp_path)
    assignments = _assignments(case.tracker)
    case.tracker.open_run()
    _materialise(case, 2, assignments[1], objective=1.0)  # index 1 never materialised

    report = case.report()
    assert report.census.recorded == 1
    assert [row.index for row in report.ranked] == [2]
    assert report.sample_trust == "layout_checked"
    assert "indices" in report.error
    assert report.coverage.verdict == "never_closed"
    with pytest.raises(ExportRefusedError) as caught:
        export_best(
            report,
            ledger=case.ledger(),
            space=_taili_space(),
            output_path=tmp_path / "best.yaml",
        )
    assert any("count-checked" in reason for reason in caught.value.reasons)


def test_a_count_mismatch_is_reported_with_both_numbers(tmp_path):
    """The forged close counts 1 trial while the file holds 2, and the report says both numbers.

    "The counts disagree" without the two numbers sends an operator to read ``trials.jsonl`` by
    hand, which is how a wrong conclusion gets drawn a week later.
    """
    case = _hand_driven(tmp_path)
    assignments = _assignments(case.tracker)
    case.tracker.open_run()
    _materialise(case, 1, assignments[0], objective=1.0)
    _materialise(case, 2, assignments[1], objective=2.0)
    _fake_close(case, proposals=1, trial_count=1)

    report = case.report()
    assert report.sample_trust == "layout_checked"
    assert "1" in report.error and "2" in report.error
    # The report still describes and ranks the two trials it could read: refusing to describe them
    # would leave an operator with nothing at all, and the downgrade is what carries the warning.
    assert report.census.recorded == 2
    assert report.ranked_count == 2
    assert report.best_value == 2.0
    assert report.coverage.verdict == "drew_its_budget"


def test_a_broken_hash_chain_leaves_every_number_unknown_and_says_so(tmp_path):
    """A ledger that cannot be read at all is not a ledger with zero trials in it.

    The difference matters at the moment it is read: "0 trials recorded" is a fact someone will
    act on ("the search never ran"), and "nothing could be read" is a request to go and look.  The
    fallback read runs the same integrity checks, so both fail and the report says so.
    """
    case = _search(tmp_path, [_outcome(objective=1.0), _outcome(objective=2.0)])
    events = case.store.events_path
    lines = events.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 3
    events.write_text(
        "\n".join(lines[:1] + lines[2:]) + "\n", encoding="utf-8", newline="\n"
    )

    report = case.report()
    assert report.sample_trust == "unreadable"
    assert report.census.recorded == 0
    assert report.coverage.verdict == "unreadable"
    assert report.error
    assert report.caveats and "absent rather than zero" in report.caveats[0]


def test_an_unknown_run_ref_is_reported_not_raised(tmp_path):
    """A mistyped run name is a readable ledger with no such run, and the two sentences differ.

    Measured: the first version of this module raised a pydantic ``ValidationError`` out of
    ``analyze`` on exactly this input, because a missing header left the plan fingerprint blank
    while the trust said a read had happened.  ``analyze`` promises never to raise for a ledger
    fact; "the name is not in this ledger" is one.
    """
    case = _search(tmp_path, [_outcome(objective=1.0)])
    report = analyze(
        case.ledger(), "search:not_this_one", rule=RankingRule(direction="maximize")
    )
    assert report.sample_trust == "unreadable"
    assert report.coverage.run_status == "unreadable"
    assert "search:not_this_one" in report.coverage.detail
    assert report.run_ref == "search:not_this_one"


def test_the_trust_level_is_the_name_of_what_ran(tmp_path):
    """All three levels, on three ledgers that differ only in how far the read got.

    The middle level is the interesting one: the run-scoped checks did not run, and *why* they did
    not is in ``error``.  A boolean "complete" would print these three as "no", "no", "no".
    """
    finished = _search(tmp_path, [_outcome(objective=1.0), _outcome(objective=2.0)], label="ok")
    assert finished.report().sample_trust == "counts_checked"
    assert finished.report().error == ""

    open_run = _hand_driven(tmp_path, label="open")
    assignments = _assignments(open_run.tracker)
    open_run.tracker.open_run()
    _materialise(open_run, 1, assignments[0])
    assert open_run.report().sample_trust == "layout_checked"
    assert open_run.report().error == ""

    gap = _hand_driven(tmp_path, label="gap")
    gap_assignments = _assignments(gap.tracker)
    gap.tracker.open_run()
    _materialise(gap, 2, gap_assignments[1])
    assert gap.report().sample_trust == "layout_checked"
    assert gap.report().error != ""

    with pytest.raises(ValueError) as caught:
        analyze(finished.ledger(), finished.run_ref, rule=RankingRule(direction="maximize"), top_n=0)
    assert "top_n" in str(caught.value)


# --- B: the rule --------------------------------------------------------------------------------


def test_the_rule_has_no_default_direction():
    """A defaulted direction is wrong half the time, and the wrong half is invisible.

    The failure mode is not an exception: it is a report whose "best" is the worst configuration,
    exported with a receipt that says everything was checked.
    """
    with pytest.raises(ValidationError) as caught:
        RankingRule()  # type: ignore[call-arg]
    assert "direction" in str(caught.value)
    assert RankingRule(direction="maximize").key == "objective"


def test_a_blank_key_is_refused():
    with pytest.raises(ValidationError) as caught:
        RankingRule(key="   ", direction="minimize")
    assert "blank" in str(caught.value)


def test_two_directions_do_not_share_a_fingerprint():
    """The rule is part of the report's identity: a report maximising and one minimising are not the same report."""
    up = RankingRule(key="objective", direction="maximize")
    down = RankingRule(key="objective", direction="minimize")
    assert up.fingerprint() != down.fingerprint()
    assert up.describe() != down.describe()


def test_a_report_refuses_a_rule_fingerprint_that_is_not_its_rule(tmp_path):
    case = _search(tmp_path, [_outcome(objective=1.0)])
    payload = case.report().model_dump(mode="json")
    payload["rule_fingerprint"] = RankingRule(key="objective", direction="minimize").fingerprint()
    with pytest.raises(ValidationError) as caught:
        SearchAnalysis.model_validate(payload)
    assert "fingerprints to" in str(caught.value)


def test_the_rule_survives_its_json_round_trip():
    """Law R3: the fingerprint is over the JSON form, so the JSON form has to be able to come back."""
    rule = RankingRule(key="eval.return", direction="minimize")
    restored = RankingRule.model_validate(rule.model_dump(mode="json"))
    assert restored == rule
    assert restored.fingerprint() == rule.fingerprint()


# --- C: readings and the ranking ---------------------------------------------------------------


def test_none_is_never_coerced_into_a_score(tmp_path):
    """A trial with no objective is listed in ``unranked``, not ranked last with a zero.

    "No result" and "a bad result" are different facts about a trial, and the difference decides
    whether an operator reruns it or moves on.
    """
    case = _search(
        tmp_path,
        [_outcome(objective=None), _outcome(objective=2.0)],
    )
    report = case.report()
    assert report.ranked_count == 1
    assert [row.index for row in report.ranked] == [2]
    assert [(row.index, row.state) for row in report.unranked] == [(1, "no_objective")]
    assert "no objective" in report.unranked[0].detail
    assert any("not ranked last" in note for note in report.caveats)


def test_a_zero_objective_is_a_score_and_a_missing_one_is_not(tmp_path):
    case = _search(tmp_path, [_outcome(objective=0.0), _outcome(objective=None)])
    report = case.report()
    assert report.ranked[0].value == 0.0
    assert report.best_value == 0.0
    assert [row.state for row in report.unranked] == ["no_objective"]


def test_an_absent_metric_and_a_nulled_one_are_different_states(tmp_path):
    """``metric_value`` returns ``None`` for both, and folding them together loses the sentence.

    "The key was never in the payload" and "the value was dropped on the way to disk" contradict
    each other: the second says the trial *did* measure something, and it is the one an operator
    needs when they are asking why the plot is empty.
    """
    case = _search(
        tmp_path,
        [
            _outcome(objective=1.0, metrics={"eval": {"other": 1.0}}),
            _outcome(objective=1.0, metrics={"eval": {"return": None}}),
        ],
    )
    report = case.report(key="eval.return")
    states = {row.index: row.state for row in report.unranked}
    assert states == {1: "absent", 2: "nulled"}
    assert "no key at" in report.unranked[0].detail
    assert "null at" in report.unranked[1].detail


def test_a_canonical_nan_text_is_read_back_as_nonfinite(tmp_path):
    """The text the pruning layer stores is the text this layer must read back.

    A curve that makes the round trip through the ledger is still judged by the rule that produced
    it; a report that called ``"nan"`` a value would rank a trial that has no usable number.  The
    reason names the text that was read, next to the key it was read at, so an operator can tell
    *which* non-finite it was -- the three canonical texts ``"nan"``/``"inf"``/``"-inf"`` are three
    different training failures.
    """
    case = _search(tmp_path, [_outcome(objective=1.0, metrics={"eval": {"return": "nan"}})])
    report = case.report(key="eval.return")
    assert [(row.index, row.state) for row in report.unranked] == [(1, "nonfinite")]
    #: Pinned to the sentence, not to a substring: a reason that dropped the text would still
    #: contain the words "not a finite number", and the text is the part an operator needs.
    assert report.unranked[0].detail == "'eval.return' is nan, which is not a finite number"


def test_a_non_finite_float_reaches_the_reader_as_a_nulled_key(tmp_path):
    """Measured: pydantic's ``model_dump(mode="json")`` turns a non-finite float into ``None``.

    The ledger's write path is exactly that call, so a trial that finished with ``nan`` reads back
    as a key holding ``null`` -- which is why the pruning layer stores the *text* instead, and why
    this module has a separate state for the two.
    """
    case = _search(
        tmp_path,
        [_outcome(objective=1.0, metrics={"eval": {"return": float("nan")}})],
    )
    assert case.record(1).metrics["eval"]["return"] is None
    report = case.report(key="eval.return")
    assert [row.state for row in report.unranked] == ["nulled"]


def test_a_boolean_metric_is_not_a_number(tmp_path):
    """``True`` is an ``int`` in Python: a gate read as ``1.0`` looks like a working metric."""
    case = _search(tmp_path, [_outcome(objective=1.0, metrics={"eval": {"passed": True}})])
    report = case.report(key="eval.passed")
    assert [row.state for row in report.unranked] == ["not_a_number"]
    assert "True" in report.unranked[0].detail


def test_the_objective_key_reads_the_record_field_not_a_same_named_metric(tmp_path):
    """The field is the surrogate model's y; a ``metrics["objective"]`` is a different number.

    The metric here is the *better-looking* one, so a rule read the wrong way round would export
    the wrong trial and report a higher score while doing it.
    """
    case = _search(
        tmp_path,
        [
            _outcome(objective=1.0, metrics={"objective": 99.0}),
            _outcome(objective=2.0, metrics={}),
        ],
    )
    report = case.report()
    assert [row.value for row in report.ranked] == [2.0, 1.0]
    assert report.best_value == 2.0
    assert report.ranked[1].objective == 1.0


def test_every_reading_state_agrees_with_the_pruning_layer(tmp_path):
    """The vocabulary is a refinement, not a second opinion.

    The pruning layer answers three questions -- is there a number, is it finite, is the key there
    but null -- and this module's six states must never contradict them.  The two states the
    pruning layer folds together (``absent`` and ``not_a_number``) are the only place this layer
    says more; both are asserted to be states where the pruning layer answered "no reading here".
    """
    case = _search(
        tmp_path,
        [
            _outcome(objective=1.5, metrics={}),  # no such metric key at all
            _outcome(objective=1.0, metrics={"eval": {"return": "inf"}}),
            _outcome(objective=1.0, metrics={"eval": {"return": 2.5}}),
            _outcome(objective=1.0, metrics={"eval": {"return": None}}),
            _outcome(objective=1.0, metrics={"eval": {"return": True}}),
        ],
    )
    report = case.report(key="eval.return")
    states = {row.index: row.state for row in report.unranked}
    states.update({row.index: "value" for row in report.ranked})
    values = {(row.index): row.value for row in report.ranked}
    assert states == {1: "absent", 2: "nonfinite", 3: "value", 4: "nulled", 5: "not_a_number"}
    assert values[3] == 2.5

    # The pruning layer's own answers, for the four non-value readings:
    assert metric_value(case.record(1).metrics, "eval.return") is None
    assert not reading_is_nulled(case.record(1).metrics, "eval.return")
    assert math.isinf(metric_value(case.record(2).metrics, "eval.return") or 0.0)
    assert metric_value(case.record(3).metrics, "eval.return") == 2.5
    assert metric_value(case.record(4).metrics, "eval.return") is None
    assert reading_is_nulled(case.record(4).metrics, "eval.return")
    assert metric_value(case.record(5).metrics, "eval.return") is None
    assert not reading_is_nulled(case.record(5).metrics, "eval.return")
    # The two states the pruning layer does not distinguish are both "no reading" to it, and they
    # are told apart here by asking the other question (is the key there at all):
    assert set(states[index] for index in (1, 5)) == {"absent", "not_a_number"}
    assert set(states.values()) - set(NON_VALUE_STATES) == {"value"}


def test_the_best_is_the_highest_objective_and_minimising_reverses_it(tmp_path):
    case = _search(tmp_path, [_outcome(objective=1.0), _outcome(objective=3.5), _outcome(objective=2.0)])
    best = case.report()
    assert [row.index for row in best.ranked] == [2, 3, 1]
    assert best.best_value == 3.5
    assert best.best() is not None and best.best().index == 2

    worst = case.report(direction="minimize")
    assert [row.index for row in worst.ranked] == [1, 3, 2]
    assert worst.best_value == 1.0
    assert worst.rule_fingerprint != best.rule_fingerprint


def test_the_direction_reverses_the_order_and_not_the_rankable_set(tmp_path):
    """Both directions rank the same trials: a direction is not a filter."""
    case = _search(tmp_path, [_outcome(objective=1.0), _outcome(objective=None), _outcome(objective=2.0)])
    up, down = case.report(), case.report(direction="minimize")
    assert up.ranked_count == down.ranked_count == 2
    assert len(up.unranked) == len(down.unranked) == 1
    assert {row.index for row in up.ranked} == {row.index for row in down.ranked}


def test_a_tie_names_every_winner_rather_than_picking_one(tmp_path):
    """A tie is a fact about the sample, not something to break by keeping whichever came first."""
    case = _search(tmp_path, [_outcome(objective=3.5), _outcome(objective=1.0), _outcome(objective=3.5)])
    report = case.report()
    assert report.ties == (1, 3)
    assert report.tied_count == 2
    assert report.best_value == 3.5
    assert [row.index for row in report.ranked] == [1, 3, 2]
    assert any("tie for first place" in note for note in report.caveats)


def test_a_tie_is_counted_over_the_sample_and_not_over_the_stored_rows(tmp_path):
    """``top_n`` caps the rows and never the tie, so a one-row window onto a three-way tie must not read as a winner.

    Measured before the fix, with three trials tied at 0.7: ``top_n=1`` gave ``ties == (1,)`` and the
    tie caveat did not fire at all, while ``top_n=2`` announced "2 trials tie".  The size of the tie
    was an artefact of the row limit -- and at ``top_n=1`` the report named a champion the ledger does
    not have, which is what an export off that report would have handed to a training run.
    """
    case = _search(tmp_path, [_outcome(objective=0.7)] * 3)

    narrow = case.report(top_n=1)
    assert narrow.ranked_count == 3
    assert narrow.ties == (1,)
    assert narrow.tied_count == 3
    note = next(n for n in narrow.caveats if "tie for first place" in n)
    assert "3 trials tie for first place at 0.7" in note
    assert "only 1 of them are among the 1 rows stored here" in note
    assert "nothing in this ledger makes one of them the winner" in note

    wide = case.report(top_n=3)
    assert wide.tied_count == 3
    assert "all of them are stored here (indices 1, 2, 3)" in next(
        n for n in wide.caveats if "tie for first place" in n
    )


def test_a_reading_that_no_float_can_hold_is_unranked_rather_than_fatal(tmp_path):
    """A JSON integer has no upper bound and a float does; the gap is where ``analyze`` used to throw.

    ``metric_value`` ends in ``float(current)`` for any ``int`` (``search_pruning.py:108``), so a
    metrics bag holding ``10**400`` -- an exponent in a surrogate's likelihood, a loss that diverged
    -- raised ``OverflowError: int too large to convert to float`` straight out of a function whose
    docstring promises that a fact about the ledger never raises.  The reading is not "no number": it
    is a number this module cannot compare, which is what keeps it out of the ranking and into
    ``unranked``, where it is named rather than dropped.
    """
    case = _hand_driven(tmp_path)
    assignments = _assignments(case.tracker)
    case.tracker.open_run()
    record = case.tracker.begin_trial(1, assignments[0])
    finished = case.tracker.finish(record.id, "succeeded")
    _append(case, _supersede(finished, metrics={"loss": 10**400}), "search_trial")

    report = case.report(key="loss")
    assert report.ranked_count == 0
    assert [row.index for row in report.unranked] == [1]
    assert report.unranked[0].state == "nonfinite"
    #: The digits are the one part an operator cannot use: 401 of them would bury the fact.
    assert (
        report.unranked[0].detail
        == "'loss' is an integer of 401 digits, which is not a finite number"
    )


def test_a_retried_index_appears_once_with_its_last_attempt(tmp_path):
    """An index is one trial in the plan: two attempts are two attempts at one point, not two points.

    Counting both would double one point's weight in the sample, and the *first* attempt's score is
    the one a retry was made to replace -- so folding to the last is also the only order that
    reports the result the search decided to keep.
    """
    case = _hand_driven(tmp_path)
    assignments = _assignments(case.tracker)
    case.tracker.open_run()
    _materialise(case, 1, assignments[0], objective=1.0, attempt=1)
    _materialise(case, 1, assignments[0], objective=2.0, attempt=2)

    report = case.report()
    assert report.census.recorded == 1
    assert [row.index for row in report.ranked] == [1]
    assert report.ranked[0].attempt == 2
    assert report.best_value == 2.0


def test_a_pending_trial_is_counted_and_never_ranked(tmp_path):
    """Materialised but never trained: it is not a low score, it is a trial that did not happen."""
    case = _hand_driven(tmp_path)
    assignments = _assignments(case.tracker)
    case.tracker.open_run()
    case.tracker.begin_trial(1, assignments[0])

    report = case.report()
    assert report.census.pending == 1
    assert report.census.pending_indices == (1,)
    assert report.ranked_count == 0
    assert report.unranked == ()


def test_a_running_trial_is_counted_and_never_ranked(tmp_path):
    """Interrupted mid-training: the result is lost, which the report must not print as "low"."""
    case = _hand_driven(tmp_path)
    assignments = _assignments(case.tracker)
    case.tracker.open_run()
    record = case.tracker.begin_trial(1, assignments[0])
    _append(case, _supersede(record, status="running"), "search_trial")

    report = case.report()
    assert report.census.running == 1
    assert report.census.running_indices == (1,)
    assert report.ranked_count == 0
    assert any("lost rather than low" in note for note in report.caveats)


def test_a_pruned_trial_with_a_number_is_ranked_and_one_without_is_named(tmp_path):
    """Being stopped early is not the same as having no result: the number decides, not the status.

    A pruner stops a trial when the curve says the outcome is already decided, and that curve
    usually carries a score -- so dropping every pruned trial from the ranking would discard real
    results, while ranking a pruned trial that has none would invent one.
    """
    case = _search(
        tmp_path,
        [
            _outcome(status="pruned", objective=9.0, message="loss floor crossed"),
            _outcome(status="pruned", objective=None, message="no reading at the cut"),
        ],
    )
    report = case.report()
    assert [row.index for row in report.ranked] == [1]
    assert report.ranked[0].status == "pruned"
    assert [row.index for row in report.unranked] == [2]
    assert report.unranked[0].status == "pruned"
    assert report.unranked[0].state == "no_objective"
    assert any("pruned trial(s) carry a score" in note for note in report.caveats)


def test_the_top_n_caps_the_rows_and_not_the_sample_size(tmp_path):
    """``ranked_count`` is the sample size; ``len(ranked)`` is how much of it the report stores.

    Two different numbers, and printing only one of them is how "the best of 3" and "the best of
    300" end up reading the same.
    """
    case = _search(
        tmp_path,
        [_outcome(objective=1.0), _outcome(objective=2.0), _outcome(objective=3.0)],
    )
    report = case.report(top_n=1)
    assert report.ranked_count == 3
    # The stored row is the *best* one (trial index 3), and its rank is 1: rank is the position in
    # the ranking, index is the trial's place in the plan, and confusing the two is how a report
    # ends up naming the wrong trial.
    assert [row.rank for row in report.ranked] == [1]
    assert [row.index for row in report.ranked] == [3]
    assert report.best_value == 3.0
    assert any("only the best 1 rows are stored" in note for note in report.caveats)


# --- D: the coverage verdicts -------------------------------------------------------------------


def test_a_grid_that_walked_every_point_reports_walked_the_grid(tmp_path):
    """The only verdict that licenses "the search covered its space"."""
    space = _grid_space()
    case = _search(
        tmp_path,
        [_outcome(objective=1.0), _outcome(objective=2.0)],
        space=space,
        plan=SearchPlan.for_grid(space),
        label="grid",
    )
    report = case.report()
    assert report.coverage.verdict == "walked_the_grid"
    assert report.coverage.kind == "grid"
    assert report.coverage.exhausted is True
    assert report.coverage.proposed == 2
    assert report.sample_trust == "counts_checked"


def test_a_grid_stopped_by_its_budget_reports_budget_truncated_the_grid(tmp_path):
    """A budget that ends a grid early is not the same fact as a grid that ran out of points."""
    space = _grid_space()
    case = _search(
        tmp_path,
        [_outcome(objective=1.0)],
        space=space,
        plan=SearchPlan.for_grid(space, budget=1),
        label="cut",
    )
    report = case.report()
    assert report.coverage.verdict == "budget_truncated_the_grid"
    assert report.coverage.exhausted is False
    assert "never proposed" in report.coverage.detail


def test_a_random_plan_that_spent_its_budget_reports_drew_its_budget(tmp_path):
    """Exhausted for a non-grid plan says nothing about coverage, and the report says so twice."""
    case = _search(tmp_path, [_outcome(objective=1.0), _outcome(objective=2.0)])
    report = case.report()
    assert report.coverage.verdict == "drew_its_budget"
    assert report.coverage.kind == "random"
    assert any("not that it covered the space" in note for note in report.caveats)


def test_a_close_without_the_exhausted_flag_reports_unstated(tmp_path):
    """Closed without saying whether the stream ran out: known unknown, not an assumption.

    Measured: ``close_run({"proposals": N})`` writes exactly this (``SamplerStats.exhausted``
    defaults to ``False``, and a mapping without the key leaves the field at its default in the
    record's ``stats``), so the verdict has a real producer rather than only a hand-written one.
    """
    case = _hand_driven(tmp_path)
    assignments = _assignments(case.tracker)
    case.tracker.open_run()
    _materialise(case, 1, assignments[0])
    case.tracker.close_run({"proposals": 1})

    report = case.report()
    assert report.coverage.verdict == "unstated"
    assert report.coverage.exhausted is None
    assert report.sample_trust == "counts_checked"
    assert "was exhausted" in report.coverage.detail


def test_a_grid_closed_without_the_exhausted_flag_reports_unstated(tmp_path):
    """The same unstated verdict, but for a grid: it outranks the two grid rows below it.

    ``unstated`` is decided before the plan kind is looked at, and until this test no *grid* plan
    ever reached it -- ``_hand_driven`` defaulted to a random plan and the grid tests above all go
    through ``_search``, which closes with a real ``SamplerStats``.  Measured consequence: moving
    the unstated early return below the two grid rows keeps all 71 pre-existing tests green while
    turning this one red, and the wrong answer it produces is ``budget_truncated_the_grid`` -- a
    budget cut the ledger never recorded.
    """
    space = _grid_space()
    plan = SearchPlan.for_grid(space, budget=4)
    case = _hand_driven(tmp_path, space=space, plan=plan, label="gridhand")
    assignments = _assignments(case.tracker)
    assert len(assignments) > 1, "a one-point grid would not distinguish the two grid rows"
    case.tracker.open_run()
    _materialise(case, 1, assignments[0])
    case.tracker.close_run({"proposals": 1})

    report = case.report()
    assert report.coverage.kind == "grid"
    assert report.coverage.verdict == "unstated"
    assert report.coverage.exhausted is None
    assert "was exhausted" in report.coverage.detail
    assert "never proposed" not in report.coverage.detail


def test_the_default_sampler_stats_report_stream_ended_early(tmp_path):
    """The other half of the same measurement: the default stats are what a careless close writes."""
    case = _hand_driven(tmp_path)
    assignments = _assignments(case.tracker)
    case.tracker.open_run()
    _materialise(case, 1, assignments[0])
    case.tracker.close_run(SamplerStats(proposals=1, attempts=1))

    report = case.report()
    assert report.coverage.verdict == "stream_ended_early"
    assert report.coverage.exhausted is False
    assert report.coverage.kind == "random"


def test_an_open_run_reports_never_closed_with_unknowns_rather_than_zero(tmp_path):
    """``trial_count`` defaults to 0, so reporting it would turn "not recorded" into "recorded nothing".

    Measured: an open run's ``stats`` is ``{}`` (``open_run`` writes no statistics), so
    ``proposed`` and ``exhausted`` are ``None`` here -- and the verdict does not consult them.
    """
    case = _hand_driven(tmp_path)
    assignments = _assignments(case.tracker)
    case.tracker.open_run()
    _materialise(case, 1, assignments[0])

    report = case.report()
    assert report.coverage.verdict == "never_closed"
    assert report.coverage.declared_trial_count is None
    assert report.coverage.proposed is None
    assert report.coverage.exhausted is None
    assert report.census.recorded == 1
    assert report.sample_trust == "layout_checked"


def test_an_open_header_that_carries_stats_reports_them_rather_than_none(tmp_path):
    """The two rules for an open run differ: ``trial_count`` is mapped by status, the ``stats`` pair is read.

    Measured: the ledger accepts a header that is ``open`` and still carries ``stats`` (the record
    validator only demands ``proposals`` of a *closed* run), and the report then prints those
    numbers under ``never_closed`` rather than ``None``.  That is deliberate -- a recorded number
    is a fact and a substituted ``None`` is not -- and it is exactly what a later refactor would
    erase by writing "open -> None" for the whole trio.
    """
    case = _hand_driven(tmp_path)
    assignments = _assignments(case.tracker)
    case.tracker.open_run()
    _materialise(case, 1, assignments[0])
    header = case.tracker.store.latest("search_run", f"run:{case.run_ref}")
    assert header is not None
    _append(
        case,
        {**header, "status": "open", "stats": {"proposals": 5, "attempts": 5, "exhausted": True}},
        "search_run",
    )

    report = case.report()
    assert report.coverage.verdict == "never_closed"
    assert report.coverage.proposed == 5
    assert report.coverage.exhausted is True
    #: The one field that *is* mapped by status: ``trial_count`` still holds its placeholder 0,
    #: and reporting that would turn "not recorded" into "recorded nothing".
    assert report.coverage.declared_trial_count is None
    assert "never checked" in report.coverage.detail


def test_an_abandoned_header_reports_abandoned_and_unconfirmed_counts(tmp_path):
    """No writer in this repository produces ``status="abandoned"`` (measured by grep); the reader must still place it.

    An abandoned run's counts are on disk and were written deliberately, so they are read and
    labelled unconfirmed -- which is not the same as an open run's placeholder zero.
    """
    case = _hand_driven(tmp_path)
    assignments = _assignments(case.tracker)
    case.tracker.open_run()
    _materialise(case, 1, assignments[0])
    _append(case, _supersede(case.tracker.open_run(), status="abandoned", trial_count=1), "search_run")

    report = case.report()
    assert report.coverage.verdict == "abandoned"
    assert report.coverage.run_status == "abandoned"
    assert report.coverage.closing_present is False
    assert report.coverage.declared_trial_count == 1
    assert "recorded but not confirmed" in report.coverage.detail


def test_a_coverage_record_cannot_contradict_its_own_fields():
    """The verdict is recomputed by the validator, so a record cannot claim a coverage it does not have."""
    with pytest.raises(ValidationError) as caught:
        Coverage(run_status="closed", kind="grid", closing_present=True, exhausted=False, verdict="walked_the_grid")
    assert "contradicts its own fields" in str(caught.value)
    with pytest.raises(ValidationError) as caught:
        Coverage(run_status="open", closing_present=True, verdict="never_closed")
    assert "closing record exists exactly when" in str(caught.value)


def test_the_coverage_table_has_no_hole():
    """Every permutation of the four inputs lands on a declared verdict, and every verdict is reachable.

    The second half is what stops a table from growing a row nothing can produce: a verdict that
    no input yields is a sentence in the report that will never be printed, and a missing one is
    an input that raises instead of being described.
    """
    statuses = ("open", "closed", "abandoned", "unreadable")
    kinds = ("grid", "random", "bayesian", "")
    flags = (True, False, None)
    produced = set()
    for status in statuses:
        for kind in kinds:
            for exhausted in flags:
                for closing in (True, False):
                    verdict = _coverage_verdict(status, kind, exhausted, closing)
                    assert verdict in COVERAGE_VERDICTS
                    produced.add(verdict)
    assert produced == set(COVERAGE_VERDICTS)


def test_a_plan_kind_this_version_does_not_know_takes_the_non_grid_rows(tmp_path):
    """A third ``SamplerKind`` would be refused by the record validator before any reader saw it.

    Measured: ``SearchRunRecord._validate_run`` re-validates the plan it carries
    (``hyperparameter_search.py:315``), so a ledger holding ``kind="bayesian"`` reads back as
    ``unreadable`` -- the second half of this test pins that, because the first half's tolerance is
    about not adding a *second* place that has to know the vocabulary, not about being able to read
    such a ledger.
    """
    assert _coverage_verdict("closed", "bayesian", True, True) == "drew_its_budget"
    assert _coverage_verdict("closed", "bayesian", False, True) == "stream_ended_early"

    case = _hand_driven(tmp_path)
    assignments = _assignments(case.tracker)
    case.tracker.open_run()
    _materialise(case, 1, assignments[0], objective=1.0)
    header = case.tracker.store.latest("search_run", f"run:{case.run_ref}")
    assert header is not None
    _append(case, {**header, "plan": {**header["plan"], "kind": "bayesian"}}, "search_run")
    report = case.report()
    assert report.sample_trust == "unreadable"
    assert "bayesian" in report.error
    assert report.coverage.verdict == "unreadable"


# --- E: the census, and the ledger that will not read --------------------------------------------


def test_the_census_parts_must_add_up_to_the_recorded_trials():
    """A trial in no bucket is a trial the report lost."""
    good = TrialCensus(recorded=2, succeeded=1, failed=1, pending_indices=(), running_indices=())
    assert good.terminal() == 2
    with pytest.raises(ValidationError) as caught:
        TrialCensus(recorded=3, succeeded=1, failed=1)
    assert "add up to 2" in str(caught.value)
    with pytest.raises(ValidationError):
        TrialCensus(recorded=1, pending=1)  # counted but not named
    with pytest.raises(ValidationError):
        TrialCensus(recorded=2, pending=2, pending_indices=(2, 1))


def test_no_trial_vanishes_from_the_report(tmp_path):
    """Ranked + unranked + pending + running == recorded, checked on several shapes at once.

    The invariant is checked by the record itself, so this test is about the *producer*: every
    trial the census counted is reachable from the ranking or from one of the named buckets.
    """
    shapes: list[list[TrialOutcome]] = [
        [_outcome(objective=1.0)],
        [_outcome(objective=1.0), _outcome(objective=None)],
        [_outcome(objective=None), _outcome(objective=None)],
        [_outcome(objective=1.0), _outcome(objective=2.0, status="pruned")],
    ]
    for position, outcomes in enumerate(shapes):
        case = _search(tmp_path, outcomes, label=f"shape{position}")
        report = case.report()
        assert report.ranked_count + len(report.unranked) == report.census.terminal()
        assert report.census.recorded == len(outcomes)
        named = [row.index for row in report.unranked]
        assert len(set(named)) == len(named)

    pending = _hand_driven(tmp_path, label="pending")
    assignments = _assignments(pending.tracker)
    pending.tracker.open_run()
    pending.tracker.begin_trial(1, assignments[0])
    materialised = pending.tracker.begin_trial(2, assignments[1])
    pending.tracker.finish(materialised.id, "succeeded", stop_reason="")
    report = pending.report()
    assert report.census.recorded == 2
    assert report.ranked_count + len(report.unranked) + report.census.pending == 2


def test_a_read_failure_that_is_not_a_search_error_still_produces_a_report(tmp_path):
    """``analyze`` catches ``Exception``, not ``SearchError``: an unreadable file is a ledger fact too.

    The narrow catch would be prettier, and it would send an operator whose disk hiccupped to parse
    ``trials.jsonl`` by hand -- which is how a wrong conclusion gets drawn a week later.  The
    exception's own name is in the report so the fact is not lost either.
    """
    case = _search(tmp_path, [_outcome(objective=1.0)])

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise OSError("the disk went away")

    case.tracker.store.events = explode  # type: ignore[method-assign]
    report = case.report()
    assert report.sample_trust == "unreadable"
    assert "OSError" in report.error
    assert "the disk went away" in report.error


def test_the_census_counts_running_and_failed_separately(tmp_path):
    """"Lost when the process died" and "the run failed" are different facts, and merging them corrupts both."""
    case = _hand_driven(tmp_path)
    assignments = _assignments(case.tracker)
    case.tracker.open_run()
    record = case.tracker.begin_trial(1, assignments[0])
    _append(case, _supersede(record, status="running"), "search_trial")
    second = case.tracker.begin_trial(2, assignments[1])
    _append(case, _supersede(second, status="failed", error="RuntimeError: the backend died", finished_at=time.time()), "search_trial")

    report = case.report()
    assert (report.census.running, report.census.failed) == (1, 1)
    assert report.census.running_indices == (1,)
    assert any("lost rather than low" in note for note in report.caveats)


# --- F: the export -------------------------------------------------------------------------------


def test_the_exported_config_carries_the_winning_assignment(tmp_path):
    """The winner's values at the winner's paths, and only there.

    Asserted parameter by parameter against the assignment the ranked row carries: a report that
    ranked trial 2 but exported trial 1's config would still produce a *valid* config file, and
    every fingerprint check in the module would agree with it, because the file it wrote would
    genuinely be the config the record says trial 1 ran.
    """
    case = _search(
        tmp_path,
        [_outcome(objective=1.0), _outcome(objective=3.5), _outcome(objective=2.0)],
    )
    report = case.report()
    winner = report.best()
    assert winner is not None and winner.index == 2 and winner.value == 3.5

    destination = tmp_path / "out" / "best.yaml"
    destination.parent.mkdir()
    artifact = export_best(
        report, ledger=case.ledger(), space=_taili_space(), output_path=destination
    )
    exported = load_config_file(destination)
    source = load_config_file(case.source)
    assert exported != source  # the injection changed something, or nothing was injected
    assert artifact.attempt == winner.attempt
    assert artifact.coverage_verdict == "drew_its_budget"
    assert artifact.sample_trust == "counts_checked"
    assert artifact.assignment == winner.assignment

    moved = case.record(winner.index).injection["moved_paths"]
    assert moved, "the winning trial recorded no injected path, so this test checks nothing"
    for name in moved:
        assert _value_at(exported, name) == _value_at(winner.assignment, name)
    # And it is not the loser's config: at least one moved path carries a different value than it
    # does in the trial that came second.
    loser = case.record(1)
    assert any(
        _value_at(exported, name) != _value_at(loser.assignment, name)
        for name in moved
        if _value_at(loser.assignment, name) is not None
    )


def test_the_export_is_the_config_the_winning_trial_ran(tmp_path):
    """Two independent anchors: the trial's own ``injected_fingerprint`` and the file it left behind.

    The injector's report is not evidence on its own -- it is computed by the same call that writes
    the file, so it agrees with itself by construction.  The record's fingerprint was written when
    the trial materialised, and the trial's config file is a third copy of the same claim.
    """
    case = _search(tmp_path, [_outcome(objective=1.0), _outcome(objective=2.0)])
    report = case.report()
    winner = report.best()
    assert winner is not None

    destination = tmp_path / "out" / "winner.yaml"
    destination.parent.mkdir()
    artifact = export_best(
        report, ledger=case.ledger(), space=_taili_space(), output_path=destination
    )

    assert artifact.expected_fingerprint == case.record(winner.index).injected_fingerprint
    assert artifact.output_fingerprint == artifact.expected_fingerprint
    assert artifact.compared_against_trial_file is True
    assert load_config_file(destination) == load_config_file(Path(artifact.trial_config_path))
    assert artifact.verified() is True
    assert verify_export(artifact) == "verified"

    # The same injector, called directly, produces the same config: this export adds no writer.
    direct = tmp_path / "out" / "direct.yaml"
    inject_into_file(case.source, direct, _taili_space(), dict(winner.assignment))
    assert load_config_file(direct) == load_config_file(destination)
    assert content_hash(load_config_file(direct)) == content_hash(load_config_file(destination))


def test_the_export_refuses_when_the_source_config_changed_since_the_run(tmp_path):
    """The guard that survives someone editing the config and re-exporting.

    The exported file would be a configuration the search never explored, and it would carry a
    receipt claiming the winning trial ran it.
    """
    case = _search(tmp_path, [_outcome(objective=1.0)])
    report = case.report()
    # A real key, not a comment: the hash is over the parsed tree, so a comment would not move it.
    case.source.write_text(
        case.source.read_text(encoding="utf-8") + "trainer:\n  seed: 1234\n", encoding="utf-8"
    )

    destination = tmp_path / "out" / "best.yaml"
    destination.parent.mkdir()
    with pytest.raises(ExportRefusedError) as caught:
        export_best(report, ledger=case.ledger(), space=_taili_space(), output_path=destination)
    assert any("has changed since this run started" in reason for reason in caught.value.reasons)
    assert not destination.exists()


def test_the_export_refuses_a_space_that_is_not_the_plans_space(tmp_path):
    """An assignment is only meaningful inside the space it was drawn from."""
    case = _search(tmp_path, [_outcome(objective=1.0)])
    report = case.report()
    destination = tmp_path / "out" / "best.yaml"
    destination.parent.mkdir()
    with pytest.raises(ExportRefusedError) as caught:
        export_best(report, ledger=case.ledger(), space=_narrow_space(), output_path=destination)
    assert any("the run was drawn from" in reason for reason in caught.value.reasons)
    assert not destination.exists()


def test_the_export_refuses_an_analysis_the_ledger_contradicts(tmp_path):
    """The report is re-checked against the ledger at export time, field by field.

    A report is a document: it can be edited, or produced by a different version, or describe a
    trial the ledger no longer holds.  "This is the config the winning trial ran" has exactly two
    truth values, and the export is where that claim is made.

    The field tampered with here is ``attempt``, and it is the interesting one: changing it keeps
    the record self-consistent (nothing else references it), and it changes *which training run*
    the exported config is claimed to come from.  A check on the trial id alone would pass.
    """
    case = _search(tmp_path, [_outcome(objective=1.0), _outcome(objective=2.0)])
    payload = case.report().model_dump(mode="json")
    assert payload["ranked"][0]["attempt"] == 1
    payload["ranked"][0]["attempt"] = 2
    forged = SearchAnalysis.model_validate(payload)

    destination = tmp_path / "out" / "best.yaml"
    destination.parent.mkdir()
    with pytest.raises(ExportRefusedError) as caught:
        export_best(forged, ledger=case.ledger(), space=_taili_space(), output_path=destination)
    assert any("the ranked row and the ledger disagree" in reason for reason in caught.value.reasons)
    assert not destination.exists()


def test_the_export_refuses_a_sample_that_was_not_count_checked(tmp_path):
    """A report that describes a run is not a report that can pick a winner, and the two are separated.

    The run here is the state a dead backend leaves behind: one finished trial with a real score
    and one failed, no closing record.  There *is* a row to export -- which is the point, because
    the refusal is not "nothing to export" but "these counts were never checked".
    """
    case = _crashed_search(tmp_path, [_outcome(objective=2.0)], fail_at=2, label="crash")
    report = case.report()
    assert report.ranked_count == 1
    assert report.best_value == 2.0
    assert report.coverage.verdict == "never_closed"
    assert report.census.failed == 1

    destination = tmp_path / "out" / "best.yaml"
    destination.parent.mkdir()
    with pytest.raises(ExportRefusedError) as caught:
        export_best(report, ledger=case.ledger(), space=_taili_space(), output_path=destination)
    assert any("not count-checked" in reason for reason in caught.value.reasons)
    assert not destination.exists()


def test_the_export_refuses_a_rank_beyond_the_stored_rows(tmp_path):
    """``rank`` indexes the *stored* rows, and the remedy is a larger ``top_n`` -- said in the message.

    Measured: with the refusal collection ordered *after* this bound, exporting from an unclosed run
    raised ``ValueError: rank 1 is outside the 0 rows`` -- true, but it hid the reason and escaped a
    caller that catches the documented refusal.
    """
    case = _search(tmp_path, [_outcome(objective=1.0), _outcome(objective=2.0)])
    report = case.report(top_n=1)
    destination = tmp_path / "out" / "best.yaml"
    destination.parent.mkdir()

    with pytest.raises(ValueError) as caught:
        export_best(
            report, ledger=case.ledger(), space=_taili_space(), output_path=destination, rank=2
        )
    assert "top_n" in str(caught.value)
    with pytest.raises(ValueError) as caught:
        export_best(
            report, ledger=case.ledger(), space=_taili_space(), output_path=destination, rank=0
        )
    assert "at least 1" in str(caught.value)
    assert not destination.exists()


def test_the_export_refuses_its_own_source_and_the_trials_config(tmp_path):
    """Writing over either of the two files the receipt is checked against destroys the evidence."""
    case = _search(tmp_path, [_outcome(objective=1.0), _outcome(objective=2.0)])
    report = case.report()
    winner = report.best()
    assert winner is not None

    with pytest.raises(ExportRefusedError) as caught:
        export_best(report, ledger=case.ledger(), space=_taili_space(), output_path=case.source)
    assert any("the run's own source config" in reason for reason in caught.value.reasons)

    trial_config = Path(case.record(winner.index).config_path)
    assert trial_config.is_file()
    before = trial_config.read_bytes()
    with pytest.raises(ExportRefusedError) as caught:
        export_best(report, ledger=case.ledger(), space=_taili_space(), output_path=trial_config)
    assert any("winning trial's own config" in reason for reason in caught.value.reasons)
    assert trial_config.read_bytes() == before


def test_the_export_refuses_an_occupied_output_or_receipt_path(tmp_path):
    """An occupied path may be a config someone is training with right now."""
    case = _search(tmp_path, [_outcome(objective=1.0)])
    report = case.report()
    destination = tmp_path / "out" / "best.yaml"
    destination.parent.mkdir()
    destination.write_text("not a config\n", encoding="utf-8")
    with pytest.raises(ExportRefusedError) as caught:
        export_best(report, ledger=case.ledger(), space=_taili_space(), output_path=destination)
    assert any("already exists" in reason for reason in caught.value.reasons)

    destination.unlink()
    manifest = Path(f"{destination}{MANIFEST_SUFFIX}")
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ExportRefusedError) as caught:
        export_best(report, ledger=case.ledger(), space=_taili_space(), output_path=destination)
    assert any("receipt path already exists" in reason for reason in caught.value.reasons)
    assert not destination.exists()


def test_the_export_refuses_a_missing_source_config(tmp_path):
    """A gone source is a refusal, not an OSError: the question "which file did this start from" has no answer."""
    case = _search(tmp_path, [_outcome(objective=1.0)])
    report = case.report()
    case.source.unlink()
    destination = tmp_path / "out" / "best.yaml"
    destination.parent.mkdir()
    with pytest.raises(ExportRefusedError) as caught:
        export_best(report, ledger=case.ledger(), space=_taili_space(), output_path=destination)
    assert any("is gone" in reason for reason in caught.value.reasons)
    assert not destination.exists()


def test_a_refusal_after_the_write_removes_the_partial_config(tmp_path):
    """A caller that sees the exception must never have to wonder whether an artifact is on disk.

    The trial's own config is edited behind the ledger's back, which is exactly the difference the
    third check exists to catch: the injector reproduces the recorded fingerprint, and the file the
    trial left behind no longer matches it.
    """
    case = _search(tmp_path, [_outcome(objective=1.0), _outcome(objective=2.0)])
    report = case.report()
    winner = report.best()
    assert winner is not None
    trial_config = Path(case.record(winner.index).config_path)
    trial_config.write_text(
        trial_config.read_text(encoding="utf-8") + "trainer:\n  seed: 999\n", encoding="utf-8"
    )

    destination = tmp_path / "out" / "best.yaml"
    destination.parent.mkdir()
    with pytest.raises(ExportRefusedError) as caught:
        export_best(report, ledger=case.ledger(), space=_taili_space(), output_path=destination)
    assert any("differs from the one trial" in reason for reason in caught.value.reasons)
    assert not destination.exists()


def test_an_export_refuses_a_config_that_does_not_reproduce_the_recorded_fingerprint(tmp_path):
    """The last check before the artifact is built: is what I wrote what the trial actually ran?

    Every other check compares the export against something *outside* the record -- the source
    config's hash, the trial's own file on disk, the ledger's copy of the row.  This one compares
    the injector's output against the record's own ``injected_fingerprint``, and it is the only
    check that still works when the trial's workspace has been deleted, which is the normal end
    state of a search someone has since cleaned up.

    No public route writes a record whose fingerprint cannot be reproduced from its own
    ``assignment``: ``begin_trial`` computes it from the injection it just performed.  So the
    ledger is forged here -- a superseding trial record with the same id, assignment, objective
    and status and the same *shape* of fingerprint, differing only in its value, which is what a
    ledger written by a different injector version (or a different source config root) would hold.
    Appended before ``close_run`` because a trial record cannot be appended after it.
    """
    case = _hand_driven(tmp_path, label="fpguard")
    assignments = _assignments(case.tracker)
    case.tracker.open_run()
    record = _materialise(case, 1, assignments[0], objective=1.0)
    assert record.injected_fingerprint != "0" * 64, "the forgery below would be a no-op"
    #: ``_materialise`` returns the record ``finish`` wrote; the objective arrived in a second
    #: append, so the forgery is built on that second record rather than on the returned one --
    #: superseding the returned one would silently drop the objective and the export would refuse
    #: for the wrong reason.
    scored = _supersede(record, objective=1.0)
    _append(case, _supersede(scored, injected_fingerprint="0" * 64), "search_trial")
    case.tracker.close_run({"proposals": 1})

    report = case.report()
    destination = tmp_path / "out" / "best.yaml"
    destination.parent.mkdir()
    with pytest.raises(ExportRefusedError) as caught:
        export_best(report, ledger=case.ledger(), space=_taili_space(), output_path=destination)
    assert any("is not the one the winning trial ran" in reason for reason in caught.value.reasons)
    assert not destination.exists()


def test_a_receipt_that_cannot_be_written_removes_the_config_too(tmp_path):
    """The pair is the artifact: a config without its receipt is the thing this module exists to prevent."""
    case = _search(tmp_path, [_outcome(objective=1.0)])
    report = case.report()
    destination = tmp_path / "out" / "best.yaml"
    destination.parent.mkdir()
    # A file name the filesystem cannot hold: the staged write fails before the receipt exists.
    unwritable = destination.parent / ("x" * 300)

    with pytest.raises(OSError):
        export_best(
            report,
            ledger=case.ledger(),
            space=_taili_space(),
            output_path=destination,
            manifest_path=unwritable,
        )
    assert not destination.exists()


def test_the_export_refuses_a_receipt_path_that_would_overwrite_its_own_evidence(tmp_path):
    """The receipt is written last and by rename, so a receipt path that *is* another file does not collide -- it replaces it.

    Measured before the fix, with the receipt path equal to the output path: the call returned
    normally and the file at the output path was the receipt JSON (its first bytes were
    ``{"analysis":``), so ``artifact.verified()`` was ``False`` from that moment onward and only
    ``verify_export`` days later noticed.  The config was gone and the export had reported success.
    """
    case = _search(tmp_path, [_outcome(objective=1.0), _outcome(objective=2.0)])
    report = case.report()
    winner = report.best()
    assert winner is not None
    destination = tmp_path / "out" / "best.yaml"
    destination.parent.mkdir()
    trial_config = Path(case.record(winner.index).config_path)
    assert trial_config.is_file()

    for receipt, fragment in (
        (destination, "the receipt path and the output path are the same file"),
        (case.source, "the run's own source config"),
        (trial_config, "the winning trial's own config"),
    ):
        with pytest.raises(ExportRefusedError) as caught:
            export_best(
                report,
                ledger=case.ledger(),
                space=_taili_space(),
                output_path=destination,
                manifest_path=receipt,
            )
        assert any(fragment in reason for reason in caught.value.reasons), fragment
    assert not destination.exists()


def test_the_export_refuses_a_receipt_directory_that_does_not_exist(tmp_path):
    """A missing receipt directory is a refusal, like a missing output directory -- not a FileNotFoundError.

    Measured before the fix: it escaped as ``FileNotFoundError`` from the staged write inside
    ``_write_manifest``, so a caller that caught the documented split caught nothing.  The config had
    already been written at that point, which made the exception the only signal that this call was
    not a success.
    """
    case = _search(tmp_path, [_outcome(objective=1.0)])
    report = case.report()
    destination = tmp_path / "out" / "best.yaml"
    destination.parent.mkdir()

    with pytest.raises(ExportRefusedError) as caught:
        export_best(
            report,
            ledger=case.ledger(),
            space=_taili_space(),
            output_path=destination,
            manifest_path=tmp_path / "nowhere" / "receipt.json",
        )
    assert any("receipt's directory does not exist" in reason for reason in caught.value.reasons)
    assert not destination.exists()


def test_a_config_that_is_no_longer_yaml_is_a_refusal_and_not_a_parser_error(tmp_path):
    """``yaml.YAMLError`` is neither an ``OSError`` nor a ``ValueError``, so all three reads have to name it.

    A source config someone edited into invalid YAML is the exact case the source-fingerprint check
    exists for, and a tampered export is the case ``verify_export`` exists for.  Measured before the
    fix: all three escaped as ``yaml.parser.ParserError`` -- out of ``export_best``, out of
    ``verify_export``, and out of ``ExportArtifact.verified()``, a predicate whose whole contract is
    that ``False`` means "not verified" rather than "cannot tell".
    """
    case = _search(tmp_path, [_outcome(objective=1.0), _outcome(objective=2.0)])
    report = case.report()
    destination = tmp_path / "out" / "best.yaml"
    destination.parent.mkdir()
    artifact = export_best(
        report, ledger=case.ledger(), space=_taili_space(), output_path=destination
    )
    assert artifact.verified() is True

    destination.write_text("trainer: [unclosed\n", encoding="utf-8")
    assert artifact.verified() is False
    with pytest.raises(ExportRefusedError) as caught:
        verify_export(artifact)
    assert "cannot be read" in str(caught.value)

    #: And the source config, which ``export_best`` reads for itself before it writes anything.
    elsewhere = tmp_path / "elsewhere.yaml"
    case.source.write_text("trainer: [unclosed\n", encoding="utf-8")
    with pytest.raises(ExportRefusedError) as caught:
        export_best(report, ledger=case.ledger(), space=_taili_space(), output_path=elsewhere)
    assert any("source config cannot be read" in reason for reason in caught.value.reasons)
    assert not elsewhere.exists()


def test_a_value_error_raised_after_the_injector_wrote_removes_the_config(tmp_path, monkeypatch):
    """The injector writes and *then* reads back, so its own ``ValueError`` can arrive with a config already on disk.

    ``inject_into_file`` is ``inject_assignment`` -> ``write_injection`` -> ``load_config_file`` of the
    file it just wrote -> a fingerprint comparison that raises ``ValueError`` (``config_injection.py``;
    the read-back is what makes the injector trustworthy, so it cannot move before the write).
    ``export_best``'s cleanup branch said "the injector failed before writing, so there is nothing to
    clean up" and re-raised without touching the file -- leaving behind the one artifact this module
    exists to prevent.  The injector is replaced rather than provoked because what is under test is
    *this* module's cleanup, not the injector's paranoia.
    """
    import autotuner.research.search_analysis as module

    case = _search(tmp_path, [_outcome(objective=1.0)])
    report = case.report()
    destination = tmp_path / "out" / "best.yaml"
    destination.parent.mkdir()

    def write_then_refuse(source_path, target, space, assignment):
        Path(target).write_text("trainer:\n  seed: 1\n", encoding="utf-8")
        raise ValueError("the written config does not carry the injection")

    monkeypatch.setattr(module, "inject_into_file", write_then_refuse)
    with pytest.raises(ValueError, match="does not carry the injection"):
        export_best(
            report, ledger=case.ledger(), space=_taili_space(), output_path=destination
        )
    assert not destination.exists()


def test_the_receipt_carries_the_size_of_the_tie_it_exported(tmp_path):
    """The receipt is read days later by someone who has no report: "this scored the best value" needs how many others did too."""
    case = _search(tmp_path, [_outcome(objective=2.0)] * 3)
    report = case.report(top_n=1)
    assert report.tied_count == 3
    destination = tmp_path / "out" / "best.yaml"
    destination.parent.mkdir()
    artifact = export_best(
        report, ledger=case.ledger(), space=_taili_space(), output_path=destination
    )
    assert artifact.tied_count == 3
    payload = json.loads(Path(artifact.manifest_path).read_text(encoding="utf-8"))
    assert payload["artifact"]["tied_count"] == 3


def test_the_receipt_is_written_beside_the_config_and_reads_back(tmp_path):
    """The receipt explains the config it sits next to, and it carries the report it came from."""
    case = _search(tmp_path, [_outcome(objective=1.0), _outcome(objective=2.0)])
    report = case.report()
    destination = tmp_path / "out" / "best.yaml"
    destination.parent.mkdir()
    artifact = export_best(
        report, ledger=case.ledger(), space=_taili_space(), output_path=destination
    )
    manifest = Path(f"{destination}{MANIFEST_SUFFIX}")
    assert Path(artifact.manifest_path) == manifest
    assert artifact.schema_version == EXPORT_SCHEMA_VERSION

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert set(payload) == {"artifact", "analysis"}
    assert ExportArtifact.model_validate(payload["artifact"]) == artifact
    assert payload["analysis"]["schema_version"] == ANALYSIS_SCHEMA_VERSION
    assert payload["analysis"]["rule_fingerprint"] == report.rule_fingerprint
    assert artifact.rule.describe() == report.rule.describe()


def test_verify_export_re_reads_the_file_and_notices_a_changed_one(tmp_path):
    """The check a third party runs days later with the receipt and nothing else."""
    case = _search(tmp_path, [_outcome(objective=1.0)])
    report = case.report()
    destination = tmp_path / "out" / "best.yaml"
    destination.parent.mkdir()
    artifact = export_best(
        report, ledger=case.ledger(), space=_taili_space(), output_path=destination
    )
    assert verify_export(artifact) == "verified"
    assert artifact.verified() is True

    destination.write_text(
        destination.read_text(encoding="utf-8") + "trainer:\n  seed: 4242\n", encoding="utf-8"
    )
    assert artifact.verified() is False
    with pytest.raises(ExportRefusedError) as caught:
        verify_export(artifact)
    assert "is not that config" in str(caught.value)

    destination.unlink()
    assert artifact.verified() is False
    with pytest.raises(ExportRefusedError) as caught:
        verify_export(artifact)
    assert "is not there" in str(caught.value)


def test_exporting_adds_nothing_to_the_ledger(tmp_path):
    """A report about a search is not part of the record of what the search did."""
    case = _search(tmp_path, [_outcome(objective=1.0), _outcome(objective=2.0)])
    report = case.report()
    before = case.store.events_path.read_bytes()
    destination = tmp_path / "out" / "best.yaml"
    destination.parent.mkdir()
    export_best(report, ledger=case.ledger(), space=_taili_space(), output_path=destination)
    assert case.store.events_path.read_bytes() == before


# --- G: the shape of the module itself ------------------------------------------------------------


_REDERIVE_SCRIPT = """import sys
sys.path.insert(0, {repo})
from autotuner.research.hyperparameter_search import SearchLedger
from autotuner.research.research_ledger import content_hash
from autotuner.research.search_analysis import RankingRule, analyze
from autotuner.research.trial_ledger import TrialLedgerStore

report = analyze(
    SearchLedger(TrialLedgerStore({ledger})),
    {run_ref},
    rule=RankingRule(key="objective", direction="maximize"),
    top_n=20,
)
print(content_hash(report.model_dump(mode="json")))
"""


def test_the_analysis_re_derives_identically_in_a_fresh_interpreter(tmp_path):
    """A report holds no wall clock, so two readers a week apart agree on every byte.

    That is a property of the record shape, not of the analysis code: adding ``generated_at`` would
    make the report unreproducible and, worse, would make two reports of one search look like two
    different reports.
    """
    case = _search(tmp_path, [_outcome(objective=1.0), _outcome(objective=2.0)])
    here = content_hash(case.report().model_dump(mode="json"))
    script = _REDERIVE_SCRIPT.format(
        repo=repr(str(REPO_ROOT)), ledger=repr(str(case.store.root)), run_ref=repr(case.run_ref)
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == here


#: ``config/repository_structure.toml:54-62``, the seven packages ``autotuner.research`` may not
#: depend on.  Repeated here rather than imported because the point of the test is that *this*
#: module's AST is checked against the list, not that the repository checker agrees with itself.
FORBIDDEN_PREFIXES = (
    "autotuner.locomotion_console",
    "autotuner.blind_locomotion",
    "autotuner.taili_core",
    "autotuner.taili_ops",
    "autotuner.training",
    "autotuner.adapter",
    "products",
)

#: Exactly what the module imports today.  Narrower than the design's allowlist in two places --
#: ``.trial_ledger`` and ``.hyperparameter_sampler`` are not imported at all, because the header's
#: plan is read as the plain mapping the record stores and nothing here mints or replays a plan.
ALLOWED_IMPORTS = {
    "__future__",
    "math",
    "os",
    "tempfile",
    "pathlib",
    "typing",
    #: The YAML parser, for the same reason ``.config_injection`` has it: the receipt is checked
    #: by re-parsing the config, and ``yaml.YAMLError`` is neither an ``OSError`` nor a
    #: ``ValueError``, so a refusal to catch it by name would leak a parser error to a caller
    #: documented to see one exception type.
    "yaml",
    "pydantic",
    ".config_injection",
    ".hyperparameter_search",
    ".hyperparameter_space",
    ".research_ledger",
    ".search_pruning",
}


def test_the_module_stays_inside_the_layers_it_may_import():
    """The AST, not the prose: an allowlist, because a blocklist only refuses the names someone thought of.

    A dependency on any of the seven would invert the layering -- the report would be able to reach
    the console, the product tree or the trainer -- and a report module that can reach the console
    is a report module someone will eventually make print itself.
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * (node.level or 0) + (node.module or ""))
    assert imported <= ALLOWED_IMPORTS, f"unexpected imports: {sorted(imported - ALLOWED_IMPORTS)}"
    for forbidden in FORBIDDEN_PREFIXES:
        assert not any(name.startswith(forbidden) for name in imported), forbidden


def test_the_module_borrows_only_the_reading_helpers_from_the_pruning_layer():
    """``prune_summary`` reads the ledger itself, and could audit a different sample than the one ranked here.

    Only the three pure helpers that define what a reading *is* are shared, so the two layers can
    never disagree about that -- and this module can never quietly substitute the pruner's sample
    for the run's.
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    borrowed: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "search_pruning":
            borrowed.update(alias.name for alias in node.names)
    assert borrowed == {"canonical_metric", "is_nonfinite", "metric_value", "reading_is_nulled"}


def test_the_module_uses_no_model_copy_and_no_computed_field():
    """Laws R1 and R2, checked on the AST rather than on the prose.

    ``model_copy(update=...)`` runs no validators, so a record built with it can be written to disk
    self-contradictory and read back completely normally; ``@computed_field`` would enter
    ``model_dump``, so the dict read back would not validate.  Both are invisible in review and
    fatal to the ledger's round trip, which is why they are checked mechanically.
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "model_copy":
            offenders.append("model_copy")
        if isinstance(node, ast.Name) and node.id == "computed_field":
            offenders.append("computed_field")
        if isinstance(node, ast.Attribute) and node.attr == "computed_field":
            offenders.append("computed_field")
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                if isinstance(target, ast.Name) and target.id == "computed_field":
                    offenders.append("computed_field decorator")
    assert offenders == []


def test_every_record_round_trips_through_its_json_form(tmp_path):
    """Law R4: the records are JSON-native, so a report can be stored, diffed and re-read.

    ``exclude_none=True`` is applied by the research ledger's own dump to models and drops fields
    whose value is ``None`` -- a record that needed such a field would come back different from the
    one that went in.  Every record here is asserted to survive the round trip unchanged.
    """
    case = _search(tmp_path, [_outcome(objective=1.0), _outcome(objective=None)])
    report = case.report()
    records: Iterator[Any] = iter(
        [
            report.rule,
            report.coverage,
            report.census,
            report,
            *report.ranked,
            *report.unranked,
        ]
    )
    for record in records:
        restored = type(record).model_validate(record.model_dump(mode="json"))
        assert restored == record
        canonical_json(restored.model_dump(mode="json"))

    reading = Reading(index=1, trial_id="trial:x:0001", key="objective", state="value", value=1.0)
    assert Reading.model_validate(reading.model_dump(mode="json")) == reading
    census = TrialCensus(recorded=1, succeeded=1)
    assert TrialCensus.model_validate(census.model_dump(mode="json")) == census
    unranked = UnrankedTrial(
        index=1, trial_id="trial:x:0001", status="succeeded", state="no_objective", detail="none"
    )
    assert UnrankedTrial.model_validate(unranked.model_dump(mode="json")) == unranked

    with pytest.raises(ValidationError):
        Reading(index=1, trial_id="trial:x:0001", key="objective", state="value", value=None)
    with pytest.raises(ValidationError):
        Reading(index=1, trial_id="trial:x:0001", key="objective", state="absent", value=1.0)


def test_a_reading_never_carries_a_non_finite_number():
    """A reading that is a value must be finite: a stored ``nan`` is a reading no reader can re-derive."""
    with pytest.raises(ValidationError) as caught:
        Reading(index=1, trial_id="trial:x:0001", key="objective", state="value", value=float("inf"))
    assert "must be finite" in str(caught.value)


def test_the_error_type_is_a_search_error():
    """A caller that already handles task 4's failures keeps working: the refusal is a ``SearchError``."""
    from autotuner.research.hyperparameter_search import SearchError

    assert issubclass(SearchAnalysisError, SearchError)
    assert issubclass(ExportRefusedError, SearchAnalysisError)
    error = ExportRefusedError(["one", "two"])
    assert error.reasons == ("one", "two")
    assert "one; two" in str(error)
