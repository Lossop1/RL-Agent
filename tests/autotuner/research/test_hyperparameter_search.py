"""Pinned behaviour of the search-record contract: a record is either self-consistent or cannot be written at all, a search is either complete or reads back as "not finished", and what a crash leaves on disk is decidable.

Each of these properties corresponds to one way of **quietly going wrong**, which is why
each is worth pinning on its own:

* If a status transition bypasses the validators (pydantic v2's ``model_copy(update=...)``
  runs no validators at all), the disk ends up holding a self-contradictory "pruned but no
  stop_reason" record that reads back completely normally.  This is the regression guard for
  rule R2: the source must not contain a single ``model_copy`` call.
* A record's fields corroborate each other (the plan and its fingerprint, the injection and
  its fingerprint, the status and the fields it must carry).  Any mismatch must raise **when
  the record is constructed**, not surface only when task 6 analyses the results and discovers
  "this trial never actually ran".
* A search "finished" means "there is a close record".  Not closed means not finished -- a
  search left behind by a crash must never read back like a completed search.
* A preflight failure must have zero side effects (no directory, no trial record, no GPU),
  otherwise one global misconfiguration writes every trial as failed and the search then
  "ends normally".
* An execution failure must be recorded first and raised second: raising alone leaves nobody
  able to tell, after the process exits, why one trial is missing; recording alone makes the
  caller treat a trial that never ran as one that finished.
* The "proposals" in ``close`` and the number of trials on disk come from two different code
  paths, and their equality is the only anchor for "nothing was silently truncated".
* Rerunning a killed index uses a new directory, so it never collides; the directory from
  before the crash is still on disk with its contents not overwritten -- the precise meaning
  of "not rewritten".
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, get_args

import pytest
import yaml
from pydantic import ValidationError

from autotuner.research import hyperparameter_search as search_module
from autotuner.research.config_injection import load_config_file
from autotuner.research.hyperparameter_sampler import (
    Sampler,
    SamplerKind,
    SamplerStats,
    SearchPlan,
    assignment_key,
    build_sampler,
)
from autotuner.research.hyperparameter_search import (
    NON_REPLAYABLE_SAMPLER_KINDS,
    REPLAYABLE_SAMPLER_KINDS,
    SearchError,
    SearchIntegrityError,
    SearchLedger,
    SearchPreflightError,
    SearchRunRecord,
    SearchTracker,
    TrialObservation,
    TrialObservationSource,
    TrialOutcome,
    TrialRecord,
    _supersede,
)
from autotuner.research.hyperparameter_space import SearchSpaceSchema, load_search_space
from autotuner.research.research_ledger import content_hash
from autotuner.research.trial_ledger import TrialLedgerIntegrityError, TrialLedgerStore

REPO_ROOT = Path(__file__).resolve().parents[3]
_PRODUCT_DIR = REPO_ROOT / "products" / "taili" / "blind_locomotion"
SPACE_PATH = _PRODUCT_DIR / "hyperparameter_space.yaml"
CONFIG_PATH = _PRODUCT_DIR / "taili_blind_config.yaml"

SEARCH_MODULE_PATH = REPO_ROOT / "autotuner" / "research" / "hyperparameter_search.py"
LEDGER_MODULE_PATH = REPO_ROOT / "autotuner" / "research" / "trial_ledger.py"


# --- Real product files and a fake execution backend -----------------------------


@lru_cache(maxsize=None)
def _taili_space() -> SearchSpaceSchema:
    """The shipped search space, read from the product files (the tests do not invent a space, except for one deliberate change)."""
    return load_search_space(yaml.safe_load(SPACE_PATH.read_text(encoding="utf-8")))


def _taili_config() -> dict:
    """The shipped training config, the source file that gets injected."""
    return load_config_file(CONFIG_PATH)


def _plan(*, seed: int = 7, budget: int = 2) -> SearchPlan:
    return SearchPlan.for_random(_taili_space(), seed=seed, budget=budget)


class FakeRunner:
    """An ordinary class with no base class and no registration.

    ``TrialRunner`` is a ``typing.Protocol``, and ``SearchTracker`` performs no ``isinstance``
    check: structurally satisfying "has preflight and run" is enough, and forcing inheritance
    would only push callers to inherit a class they do not need.
    """

    def __init__(
        self,
        *,
        outcome: TrialOutcome | None = None,
        preflight_error: BaseException | None = None,
        run_error: BaseException | None = None,
        on_run: Callable[[Any], None] | None = None,
    ) -> None:
        self.outcome = (
            outcome if outcome is not None else TrialOutcome(status="succeeded", objective=1.5)
        )
        self.preflight_error = preflight_error
        self.run_error = run_error
        #: Called once per trial, while the tracker holds it as ``running``: the seam an
        #: outside decision (a pruner's ``finish``, an operator's ``observe``) needs in order
        #: to land in the middle of training rather than after it.
        self.on_run = on_run
        self.preflights: list[Any] = []
        self.requests: list[Any] = []

    def preflight(self, request: Any) -> None:
        self.preflights.append(request)
        if self.preflight_error is not None:
            raise self.preflight_error

    def run(self, request: Any) -> TrialOutcome:
        self.requests.append(request)
        if self.on_run is not None:
            self.on_run(request)
        if self.run_error is not None:
            raise self.run_error
        return self.outcome


def _tracker(
    tmp_path: Path,
    *,
    budget: int = 2,
    seed: int = 7,
    runner: FakeRunner | None = None,
    store: TrialLedgerStore | None = None,
    work_root: Path | None = None,
    plan: SearchPlan | None = None,
    sampler: Sampler | None = None,
) -> SearchTracker:
    """A write facade wired to the real space and the real config; store/work_root/sampler can be supplied by the caller."""
    space = _taili_space()
    plan = plan if plan is not None else _plan(seed=seed, budget=budget)
    return SearchTracker(
        store=store if store is not None else TrialLedgerStore(tmp_path / "ledger"),
        space=space,
        plan=plan,
        source_config=_taili_config(),
        source_config_path=CONFIG_PATH,
        work_root=work_root if work_root is not None else tmp_path / "work",
        runner=runner if runner is not None else FakeRunner(),
        sampler=sampler,
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _injection() -> tuple[dict[str, Any], str]:
    """An injection report stored as opaque JSON, plus its fingerprint."""
    payload: dict[str, Any] = {
        "space_fingerprint": _taili_space().fingerprint(),
        "source_fingerprint": "source",
        "injected_fingerprint": "injected",
        "values": [{"name": "skrl.agent.mini_batches", "value": 16, "changed": True}],
        "moved_paths": ["skrl.agent.mini_batches"],
    }
    return payload, content_hash(payload)


def _trial_payload(**changes: Any) -> dict[str, Any]:
    """The JSON shape of a valid TrialRecord; the caller uses ``changes`` to build versions that violate an invariant."""
    space = _taili_space()
    plan = _plan()
    injection, injection_fingerprint = _injection()
    payload: dict[str, Any] = {
        "id": "trial:search:test:0001",
        "run_ref": "search:test",
        "index": 1,
        "attempt": 1,
        "history_len": 0,
        "assignment": {"skrl": {"agent": {"mini_batches": 16}}},
        "assignment_key": '{"skrl":{"agent":{"mini_batches":16}}}',
        "plan": plan.model_dump(mode="json"),
        "plan_fingerprint": plan.fingerprint(),
        "space_fingerprint": space.fingerprint(),
        "status": "succeeded",
        # Just strings inside the record; this file touches no file under either of these two paths.
        "run_dir": "/tmp/trial",
        "config_path": "/tmp/trial/trial_config.yaml",
        "injection": injection,
        "injection_fingerprint": injection_fingerprint,
        "source_fingerprint": "source",
        "injected_fingerprint": "injected",
        "started_at": 1.0,
        "finished_at": 2.0,
    }
    payload.update(changes)
    return payload


def _run_payload(**changes: Any) -> dict[str, Any]:
    """The JSON shape of a valid SearchRunRecord."""
    plan = _plan()
    payload: dict[str, Any] = {
        "id": "run:search:test",
        "run_ref": "search:test",
        "plan": plan.model_dump(mode="json"),
        "plan_fingerprint": plan.fingerprint(),
        "space_fingerprint": plan.space_fingerprint,
        "sampling_order": list(plan.sampling_order),
        "source_config": str(CONFIG_PATH),
        "source_config_fingerprint": "source",
        "budget": plan.budget,
        "max_in_flight": 1,
        "status": "open",
    }
    payload.update(changes)
    return payload


# --- A12: R2 -- transitions must rerun the validators ----------------------------


def test_a_transition_that_breaks_an_invariant_is_refused():
    """``pruned`` must carry a stop_reason, so a transition from ``succeeded`` cannot be bypassed by editing a field.

    This is the positive proof of rule R2: the single entry point ``_supersede`` reruns every
    invariant, so a record that "looks successful" must be refused on the spot once changed to
    ``pruned``, rather than landing on disk as a self-contradictory record.
    """
    succeeded = TrialRecord.model_validate(_trial_payload())
    assert succeeded.status == "succeeded"

    with pytest.raises(ValidationError) as caught:
        _supersede(succeeded, status="pruned")
    # The error says "pruned must record why it stopped", which is the stop_reason invariant.
    assert "why it stopped" in str(caught.value)


def test_model_copy_update_is_not_used():
    """The source must not contain ``model_copy`` (an AST check, not a bare grep).

    ``model_copy(update=...)`` runs no validators at all and is the one bypass around R2.  A
    bare grep would also flag the "why it is forbidden" explanation in the module docstring,
    which that module's own law block keeps (it says the laws must still hold whenever the
    file is edited), so the criterion lives on the syntax tree: the number of nodes that are
    an attribute access or a name spelled ``model_copy`` must be 0.
    """
    offenders: list[str] = []
    for path in (SEARCH_MODULE_PATH, LEDGER_MODULE_PATH):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "model_copy":
                offenders.append(f"{path.name}:{node.lineno} {node.attr}")
            if isinstance(node, ast.Name) and node.id == "model_copy":
                offenders.append(f"{path.name}:{node.lineno} {node.id}")
    assert offenders == []


# --- A13: plan and fingerprint agree with each other -----------------------------


def test_a_record_whose_plan_fingerprint_does_not_match_its_plan_is_refused():
    """The plan in a record is the required input for replay, and the identity it claims for itself must match it.

    The two come from different sources (one is filled in by whoever wrote the record, the
    other is computed from the plan's contents), so this check catches when "the search the
    record says it is" and "the search the record carries" are not the same search.
    """
    with pytest.raises(ValidationError) as caught:
        TrialRecord.model_validate(_trial_payload(plan_fingerprint="0" * 64))
    assert "fingerprints to" in str(caught.value)

    with pytest.raises(ValidationError) as run_caught:
        SearchRunRecord.model_validate(_run_payload(plan_fingerprint="0" * 64))
    assert "fingerprints to" in str(run_caught.value)


# --- A14: injection is opaque and self-consistent --------------------------------


def test_a_record_whose_injection_fingerprint_does_not_match_its_injection_is_refused():
    """The injection report is stored as opaque JSON, so its integrity can only be guaranteed by its hash.

    The report has no path field of its own and must not be validated as an ``InjectionReport``
    (its computed_field would collide with extra=forbid), so "the injection in the record is the
    one that was injected" can only rest on this equality.
    """
    with pytest.raises(ValidationError) as caught:
        TrialRecord.model_validate(_trial_payload(injection_fingerprint="0" * 64))
    assert "hashes to" in str(caught.value)


def test_the_injection_is_stored_as_a_plain_dict():
    """The injection is a plain dict inside the record, and equals its fingerprint (rule R4)."""
    record = TrialRecord.model_validate(_trial_payload())
    assert isinstance(record.injection, dict)
    assert content_hash(record.injection) == record.injection_fingerprint


# --- A15: conditional fields of terminal states ----------------------------------


def test_a_succeeded_trial_requires_a_run_dir():
    with pytest.raises(ValidationError) as caught:
        TrialRecord.model_validate(_trial_payload(status="succeeded", run_dir=""))
    assert "directory it ran in" in str(caught.value)


def test_a_failed_trial_requires_an_error():
    with pytest.raises(ValidationError) as caught:
        TrialRecord.model_validate(_trial_payload(status="failed", error=""))
    assert "why it failed" in str(caught.value)


def test_a_pruned_trial_requires_a_stop_reason():
    with pytest.raises(ValidationError) as caught:
        TrialRecord.model_validate(_trial_payload(status="pruned", stop_reason=""))
    assert "why it stopped" in str(caught.value)


def test_a_running_trial_requires_a_config_path():
    """Nothing can run without having been injected with a config: ``running`` must also carry the on-disk path of that config."""
    with pytest.raises(ValidationError) as caught:
        TrialRecord.model_validate(_trial_payload(status="running", config_path=""))
    assert "config it was injected into" in str(caught.value)


# --- A16: conditions for closing -------------------------------------------------


def test_a_closed_run_requires_sampler_stats():
    """A closed run with no stats cannot prove to itself "the grid was walked to the end or was cut off by the budget"."""
    with pytest.raises(ValidationError) as caught:
        SearchRunRecord.model_validate(
            _run_payload(status="closed", stats={}, trial_count=2)
        )
    assert "proposals" in str(caught.value)

    # stats without proposals fails the same way: exhausted is the answer to that question.
    with pytest.raises(ValidationError) as incomplete:
        SearchRunRecord.model_validate(
            _run_payload(status="closed", stats={"exhausted": True}, trial_count=2)
        )
    assert "proposals" in str(incomplete.value)


def test_a_closed_run_requires_trial_count_to_equal_proposals():
    with pytest.raises(ValidationError) as caught:
        SearchRunRecord.model_validate(
            _run_payload(status="closed", stats={"proposals": 2, "exhausted": True}, trial_count=1)
        )
    assert "skipped" in str(caught.value)


# --- A17: timestamps --------------------------------------------------------------


def test_a_terminal_record_requires_finished_at():
    with pytest.raises(ValidationError) as caught:
        TrialRecord.model_validate(_trial_payload(status="succeeded", finished_at=0.0))
    assert "when it finished" in str(caught.value)

    record = TrialRecord.model_validate(_trial_payload())
    assert record.duration_s() == record.finished_at - record.started_at
    assert record.duration_s() == 1.0


# --- A18: no derived fields in a record ------------------------------------------


def test_derived_values_are_not_stored():
    """Derived quantities are ordinary methods (rule R1): once one enters ``model_dump``, the dict read back can no longer validate."""
    record = TrialRecord.model_validate(_trial_payload())
    assert "duration_s" not in record.model_dump(mode="json")
    assert "duration_s" not in TrialRecord.model_fields


# --- A19/A20/A21: round trip, injection into files, zero hardcoding ---------------


def test_a_trial_round_trips(tmp_path):
    """Round trip: written down, read back, equal field by field; the injected values really reached the config file in the trial directory."""
    runner = FakeRunner()
    tracker = _tracker(tmp_path, budget=2, runner=runner)
    run = tracker.run()

    assert len(run.trials) == 2
    assert run.complete is True
    assert [record.index for record in run.trials] == [1, 2]
    for record in run.trials:
        assert record.assignment_key == assignment_key(record.assignment)

    # Read back from disk after writing: the two read paths and run()'s return value must be the same thing (graft A).
    assert tracker.read() == run
    assert SearchLedger(tracker.store).run(tracker.run_ref) == run

    for record in run.trials:
        # The fingerprint double-checked before writing must equal the hash of "the file actually written" (A20).
        assert record.injection["injected_fingerprint"] == content_hash(
            load_config_file(record.config_path)
        )
        assert record.injection["moved_paths"]
        assert isinstance(record.injection["moved_paths"], list)


def test_the_search_modules_hardcode_no_product_paths():
    """The two new modules must not contain a product directory, a product name, a framework name, or an output directory inside the repo.

    The space and the config are parsed and passed in by the caller, and paths enter records
    only as strings; the moment a module hardcodes a product path it stops being a reusable
    research-layer component, and "which product this search runs" is no longer passed by
    anyone.
    """
    completed = subprocess.run(
        [
            "grep",
            "-nE",
            r"products/|taili|blind_locomotion|skrl\.|output/",
            str(SEARCH_MODULE_PATH),
            str(LEDGER_MODULE_PATH),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_every_written_path_stays_under_the_caller_root(tmp_path):
    """Every path a session writes stays under the work_root / ledger_root the caller supplied.

    Neither root has a default: once a default path exists somebody will depend on it, and then
    a second ledger grown on the same machine no longer has an authoritative copy.
    """
    tracker = _tracker(tmp_path, budget=2)
    run = tracker.run()

    work_root = tracker.work_root.resolve()
    ledger_root = tracker.store.root.resolve()
    assert work_root.is_relative_to(tmp_path.resolve())
    for record in run.trials:
        assert Path(record.run_dir).resolve().is_relative_to(work_root)
        assert Path(record.config_path).resolve().is_relative_to(work_root)
    assert tracker.store.events_path.resolve().is_relative_to(ledger_root)
    assert tracker.store.events_path.parent == ledger_root


def test_max_seconds_is_absent_unless_the_caller_gives_one(tmp_path):
    """When the caller passes no max_seconds, the runner receives ``None`` (no time limit).

    The module never invents a default number of seconds of its own: the scheduler's hardcoded
    3600.0 is the ready-made lesson that "the no-limit the caller thinks it has" and "the
    no-limit it actually has" are not the same thing.
    """
    runner = FakeRunner()
    tracker = _tracker(tmp_path, budget=1, runner=runner)
    tracker.run()

    assert runner.requests
    assert runner.requests[0].max_seconds is None
    assert runner.preflights[0].max_seconds is None


# --- A22/A23: the seam handed to task 7 -------------------------------------------


def test_the_ledger_satisfies_the_observation_protocol(tmp_path):
    """``SearchLedger`` is itself the integration point for task 7 and needs no wrapper.

    The methods share the same name and signature, so ``isinstance`` holds; both fingerprints
    must match, because a cross-space observation would teach the agent model a wrong mapping,
    and a wrong mapping never announces itself.
    """
    tracker = _tracker(tmp_path, budget=2)
    run = tracker.run()
    ledger = SearchLedger(tracker.store)

    assert isinstance(ledger, TrialObservationSource) is True
    found = ledger.observations(
        plan_fingerprint=run.header.plan_fingerprint,
        space_fingerprint=run.header.space_fingerprint,
    )
    assert isinstance(found, tuple)
    assert [record.index for record in found] == [1, 2]
    assert all(
        record.plan_fingerprint == run.header.plan_fingerprint
        and record.space_fingerprint == run.header.space_fingerprint
        for record in found
    )
    assert (
        ledger.observations(
            plan_fingerprint=run.header.plan_fingerprint, space_fingerprint="0" * 64
        )
        == ()
    )
    assert (
        ledger.observations(
            plan_fingerprint="1" * 64, space_fingerprint=run.header.space_fingerprint
        )
        == ()
    )


def test_the_replayable_and_non_replayable_kinds_partition_the_vocabulary():
    """Every sampler kind is decided about exactly once: in one set, and not in the other.

    This replaces an assertion that read ``REPLAYABLE_SAMPLER_KINDS == frozenset(get_args(
    SamplerKind))``.  That could only hold while every kind was replayable, and the docstring
    directly above it already said an adaptive sampler would have to be registered as "not
    replayable" -- the assertion contradicted its own prose the whole time it passed.  A partition
    says what was meant: nothing falls outside both sets (a kind nobody decided about), and nothing
    falls inside both (a kind whose replayability claim is a coin flip).
    """
    kinds = frozenset(get_args(SamplerKind))
    assert REPLAYABLE_SAMPLER_KINDS & NON_REPLAYABLE_SAMPLER_KINDS == frozenset()
    assert REPLAYABLE_SAMPLER_KINDS | NON_REPLAYABLE_SAMPLER_KINDS == kinds


# --- A24: directory names are unique, a crash never collides ----------------------


def test_a_rerun_never_collides_with_an_orphan_directory(tmp_path):
    """A same-named directory left by an earlier crash must not block this trial.

    The directory name carries a random token, so rerunning the same index uses a new directory
    and ``exist_ok=False`` cannot collide with the orphan the earlier run left behind; with a
    deterministic directory name, recovery would die outright on ``FileExistsError``.
    """
    tracker = _tracker(tmp_path, budget=2)
    trials_root = tracker.work_root / "trials"
    stale = trials_root / "trial-0001-a1-deadbeef"
    stale.mkdir(parents=True)

    run = tracker.run()

    assert run.complete is True
    assert [record.index for record in run.trials] == [1, 2]
    assert all(Path(record.run_dir) != stale for record in run.trials)
    assert stale.is_dir()


def test_orphan_directories_are_reported(tmp_path):
    """Directories referenced by no record must be listable -- it is a diagnosis, not an automatic cleanup list.

    A crash can happen in the narrow window of "directory created, record not yet written", or
    an old directory can be left behind after a trial is rerun; in both cases the config and the
    half-written log inside are evidence, and deleting them automatically amounts to judging on
    someone else's behalf that they are worthless.
    """
    tracker = _tracker(tmp_path, budget=1)
    run = tracker.run()
    orphan = tracker.work_root / "trials" / "trial-0007-a3-cafebabe"
    orphan.mkdir(parents=True)

    reported = {path.resolve() for path in SearchLedger(tracker.store).orphan_dirs()}

    assert orphan.resolve() in reported
    assert Path(run.trials[0].run_dir).resolve() not in reported


# --- A25: after kill -9 ------------------------------------------------------------


_KILL_SCRIPT = """import os, sys
sys.path.insert(0, {repo})
from pathlib import Path
import yaml
from autotuner.research.config_injection import load_config_file
from autotuner.research.hyperparameter_sampler import SearchPlan
from autotuner.research.hyperparameter_space import load_search_space
from autotuner.research.hyperparameter_search import SearchTracker
from autotuner.research.trial_ledger import TrialLedgerStore


class Killer:
    def preflight(self, request):
        pass

    def run(self, request):
        os._exit(9)


space = load_search_space(yaml.safe_load(Path({space}).read_text(encoding='utf-8')))
plan = SearchPlan.for_random(space, seed=7, budget=2)
tracker = SearchTracker(
    store=TrialLedgerStore({ledger}),
    space=space,
    plan=plan,
    source_config=load_config_file({config}),
    source_config_path={config},
    work_root={work},
    runner=Killer(),
)
tracker.run()
"""


def test_a_killed_trial_is_reported_not_faked(tmp_path):
    """The process dies mid-trial: what remains is a ``running`` record, not a guessed result.

    ``os._exit`` is used rather than an exception: an exception would be caught by
    ``except BaseException`` and write a failed record, which is "an explainable failure", not
    "the process vanished and the result is lost forever".  By default there is no automatic
    rerun (``retry_interrupted=False``) -- a rerun costs another card, and spending it twice is
    irreversible; it reruns only when explicitly asked, and then into a new directory, while the
    pre-crash directory together with its config file is still on disk, not overwritten.
    """
    ledger_root = tmp_path / "ledger"
    work_root = tmp_path / "work"
    script = _KILL_SCRIPT.format(
        repo=repr(str(REPO_ROOT)),
        space=repr(str(SPACE_PATH)),
        config=repr(str(CONFIG_PATH)),
        ledger=repr(str(ledger_root)),
        work=repr(str(work_root)),
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
    assert completed.returncode == 9, completed.stderr

    store = TrialLedgerStore(ledger_root)
    ledger = SearchLedger(store)
    run_ref = ledger.runs()[0].run_ref
    crashed = ledger.run(run_ref)

    assert crashed.complete is False
    assert crashed.closing is None
    assert crashed.interrupted_indices() == (1,)
    assert crashed.terminal_trials() == ()
    assert [record.status for record in crashed.trials] == ["running"]

    crashed_dir = Path(crashed.trials[0].run_dir)
    crashed_config = crashed_dir / "trial_config.yaml"
    crashed_digest = _digest(crashed_config)

    # By default it only reports and does not rerun automatically (§11.3 item 5): a rerun costs another card, and spending it twice is irreversible.
    default = inspect.signature(SearchTracker.run).parameters["retry_interrupted"].default
    assert default is False

    # Explicit resume: attempt increments, a new directory is used, and the pre-crash directory stays on disk untouched.
    resumed = _tracker(
        tmp_path, budget=2, runner=FakeRunner(), store=store, work_root=work_root
    )
    finished = resumed.run(retry_interrupted=True)

    assert finished.complete is True
    assert finished.closing is not None
    assert [record.index for record in finished.trials] == [1, 2]
    assert finished.trials[0].attempt == 2
    assert Path(finished.trials[0].run_dir) != crashed_dir
    assert crashed_dir.is_dir()
    assert _digest(crashed_config) == crashed_digest
    assert crashed_dir.resolve() in {
        path.resolve() for path in SearchLedger(store).orphan_dirs()
    }


# --- A26: a preflight failure has no side effects ---------------------------------


def test_a_preflight_failure_leaves_no_trace(tmp_path):
    """A preflight failure must raise synchronously with zero side effects: no directory, no trial record, no GPU.

    Written as "record a failed and carry on", one global misconfiguration would record every
    trial as failed and the search would then "end normally" -- the worst kind of silent failure.
    """
    runner = FakeRunner(preflight_error=SearchPreflightError("no candidate artifact: /missing"))
    tracker = _tracker(tmp_path, budget=1, runner=runner)

    with pytest.raises(SearchPreflightError):
        tracker.run()

    assert tracker.store.records("search_trial") == {}
    trials_root = tracker.work_root / "trials"
    assert not trials_root.exists() or list(trials_root.iterdir()) == []
    assert len(runner.requests) == 0
    assert len(runner.preflights) == 1

    # The run header is written but not closed: not closed means not finished.
    run = tracker.read()
    assert run.complete is False
    assert run.closing is None
    assert run.header.status == "open"


# --- A27: an execution failure is recorded then raised -----------------------------


def test_a_runner_exception_is_recorded_then_raised(tmp_path):
    """Training raises: the exception is still raised to the caller, and an explainable failed record is left on disk at the same time."""
    runner = FakeRunner(run_error=RuntimeError("boom"))
    tracker = _tracker(tmp_path, budget=1, runner=runner)

    with pytest.raises(RuntimeError, match="boom"):
        tracker.run()

    run = tracker.read()
    assert run.complete is False
    assert len(run.trials) == 1
    last = run.trials[-1]
    assert last.status == "failed"
    assert "boom" in last.error
    assert "RuntimeError" in last.error
    assert last.finished_at > 0


# --- A28: no silent truncation ------------------------------------------------------


def test_closing_with_an_unrecorded_proposal_is_refused(tmp_path):
    """The write-side anchor: when the sampler said 2 trials but only 1 is on disk, close_run must refuse."""
    tracker = _tracker(tmp_path, budget=1)
    tracker.open_run()
    assignment = build_sampler(tracker.space, tracker.plan).collect()[0]
    record = tracker.begin_trial(1, assignment)
    tracker.finish(record.id, "succeeded")

    with pytest.raises(SearchError) as caught:
        tracker.close_run(SamplerStats(proposals=2, attempts=2, exhausted=True))
    assert "proposed" in str(caught.value)
    assert tracker.read().complete is False


def test_a_faked_closing_record_is_refused_on_read(tmp_path):
    """The read-side anchor: the two numbers come from different code paths, and a mismatch means "a trial was silently swallowed".

    A close line is written straight into the file (the write side does not validate the
    payload -- that is exactly the entry point being guarded against), and reading it back must
    raise count_mismatch rather than treating this ledger as a completed search.
    """
    tracker = _tracker(tmp_path, budget=2)
    tracker.open_run()
    assignments = build_sampler(tracker.space, tracker.plan).collect()
    for index, assignment in enumerate(assignments, start=1):
        record = tracker.begin_trial(index, assignment)
        tracker.finish(record.id, "succeeded")

    header = tracker.open_run()
    faked = _supersede(
        header, status="closed", stats={"proposals": 1, "exhausted": True}, trial_count=1
    )
    tracker.store.append("search_run", faked, actor="test", event_type="supersede")

    with pytest.raises(TrialLedgerIntegrityError) as caught:
        tracker.read()
    assert caught.value.code == "count_mismatch"


def test_an_unclosed_run_is_incomplete(tmp_path):
    """Not closed means not finished: a run that died mid-way leaves no "search that looks complete" behind."""
    tracker = _tracker(tmp_path, budget=1)
    tracker.open_run()

    run = tracker.read()
    assert run.complete is False
    assert run.closing is None
    assert run.header.status == "open"


# --- A29: layout checks --------------------------------------------------------------


def test_a_second_run_record_is_refused(tmp_path):
    """One root carries exactly one search: a second search_run line is a hard error, not "the last one wins"."""
    tracker = _tracker(tmp_path, budget=1)
    tracker.run()
    header = tracker.store.latest("search_run", f"run:{tracker.run_ref}")
    assert header is not None
    tracker.store.append(
        "search_run",
        {**header, "id": "run:search:other", "run_ref": "search:other"},
        actor="test",
    )

    with pytest.raises(TrialLedgerIntegrityError) as caught:
        tracker.read()
    assert caught.value.code == "unknown_layout"


def test_a_trial_after_close_is_refused(tmp_path):
    """Appending a trial after close = a search that already ended grows one more result."""
    tracker = _tracker(tmp_path, budget=1)
    tracker.run()
    trial_id = tracker.read().trials[0].id
    payload = tracker.store.latest("search_trial", trial_id)
    assert payload is not None
    tracker.store.append(
        "search_trial",
        {**payload, "id": f"trial:{tracker.run_ref}:0002", "index": 2},
        actor="test",
        event_type="supersede",
    )

    with pytest.raises(TrialLedgerIntegrityError) as caught:
        SearchLedger(tracker.store).run(tracker.run_ref)
    assert caught.value.code == "trial_after_close"


def test_a_duplicate_close_is_refused(tmp_path):
    """A duplicate close gives "when did this search end" two answers."""
    tracker = _tracker(tmp_path, budget=1)
    tracker.run()
    header = tracker.open_run()
    tracker.store.append(
        "search_run",
        _supersede(
            header, status="closed", stats={"proposals": 1, "exhausted": True}, trial_count=1
        ),
        actor="test",
        event_type="supersede",
    )

    with pytest.raises(TrialLedgerIntegrityError) as caught:
        tracker.read()
    assert caught.value.code == "duplicate_close"


# --- A30: idempotency and refusal ----------------------------------------------------


def test_open_run_is_idempotent(tmp_path):
    """Opening again with the same run_ref and fingerprint only reuses, it does not append: reconnecting after a crash is the routine path, not a new mechanism."""
    tracker = _tracker(tmp_path, budget=1)
    first = tracker.open_run()
    second = tracker.open_run()

    assert first.id == second.id
    assert first.run_ref == second.run_ref
    events = tracker.store.events(verify=True)
    assert [event.record_type for event in events] == ["search_run"]
    assert len(events) == 1


def test_a_different_plan_on_the_same_root_is_refused(tmp_path):
    """Swapping in a different plan (a different seed) on the same root must raise, not merge two searches into one."""
    tracker = _tracker(tmp_path, budget=1)
    tracker.open_run()
    other = _tracker(
        tmp_path, seed=8, budget=1, store=tracker.store, work_root=tracker.work_root
    )

    with pytest.raises(SearchIntegrityError):
        other.open_run()
    assert len(tracker.store.records("search_run")) == 1


def test_running_close_run_twice_adds_no_event(tmp_path):
    """A second ``run()`` writes not one event: terminal indexes are skipped and close is idempotent."""
    tracker = _tracker(tmp_path, budget=1)
    first = tracker.run()
    before = len(tracker.store.events(verify=True))

    second = tracker.run()

    assert second == first
    assert second.complete is True
    assert len(tracker.store.events(verify=True)) == before
    closes = [
        event
        for event in tracker.store.events(verify=True)
        if event.record_type == "search_run" and event.event_type != "append"
    ]
    assert len(closes) == 1


# --- A31: replay comparison ------------------------------------------------------------


def test_a_recorded_trial_replays_to_the_same_assignment_keys(tmp_path):
    """The recorded sequence and the sequence drawn today from the same space and the same plan are identical, item by item."""
    tracker = _tracker(tmp_path, budget=2)
    tracker.run()

    report = SearchLedger(tracker.store).replay_check(
        _taili_space(), run_ref=tracker.run_ref
    )

    assert report.plan_match is True
    assert report.space_match is True
    assert report.replayable is True
    assert report.mismatched == ()
    assert report.reason == ""


def test_replay_check_notices_a_changed_space(tmp_path):
    """A space that was changed (here, one default) is no longer the same space: the comparison must say so, not run anyway."""
    tracker = _tracker(tmp_path, budget=2)
    tracker.run()
    data = yaml.safe_load(SPACE_PATH.read_text(encoding="utf-8"))
    for spec in data["parameters"]:
        if spec["name"] == "skrl.agent.discount_factor":
            spec["default"] = 0.98
    changed = load_search_space(data)
    assert changed.fingerprint() != _taili_space().fingerprint()

    report = SearchLedger(tracker.store).replay_check(changed, run_ref=tracker.run_ref)

    assert report.space_match is False
    assert report.plan_match is True
    assert report.mismatched == ()
    assert report.reason != ""


def test_a_non_replayable_kind_is_not_compared(tmp_path, monkeypatch):
    """A kind not registered as replayable only reports "not compared": it does not falsely report drift and does not raise.

    The direction is fail-safe: forgetting to register an adaptive sampler only costs one
    comparison, and never mistakes "the points it draws differ each time" for space drift.
    """
    tracker = _tracker(tmp_path, budget=2)
    tracker.run()
    monkeypatch.setattr(search_module, "REPLAYABLE_SAMPLER_KINDS", frozenset({"grid"}))

    report = SearchLedger(tracker.store).replay_check(_taili_space(), run_ref=tracker.run_ref)

    assert report.replayable is False
    assert report.reason != ""
    assert report.mismatched == ()
    assert report.space_match is True


def test_replay_check_notices_a_tampered_assignment(tmp_path):
    """The positive control for the comparison: a record whose assignment was rewritten outside the tracker must be reported as a mismatch.

    The clean case is the one the suite already pins, and an implementation that reports
    "nothing mismatched" whatever it reads survives it.  The rewritten record is
    self-consistent -- its assignment and the key it claims for that assignment agree, and its
    plan and space fingerprints are the run's -- so no other check on the read side can see
    that it is no longer the point the plan drew, and the comparison is the only thing standing
    between a corrupted record and a search report that reads as healthy.
    """
    tracker = _tracker(tmp_path, budget=1)
    tracker.open_run()
    assignment = build_sampler(tracker.space, tracker.plan).collect()[0]
    materialised = tracker.begin_trial(1, assignment)
    tracker.finish(materialised.id, "succeeded")
    recorded = SearchLedger(tracker.store).trial(materialised.id)
    assert recorded is not None

    # A different, but equally legal, point of the same space: mini_batches steps by 4 and
    # both 4 and 8 divide every rollouts value the space allows.
    tampered = copy.deepcopy(recorded.assignment)
    drawn = tampered["skrl"]["agent"]["mini_batches"]
    tampered["skrl"]["agent"]["mini_batches"] = 4 if drawn != 4 else 8
    assert assignment_key(tampered) != recorded.assignment_key
    tracker.store.append(
        "search_trial",
        {
            **recorded.model_dump(mode="json"),
            "assignment": tampered,
            "assignment_key": assignment_key(tampered),
        },
        actor="test",
        event_type="supersede",
    )
    # The rewrite is coherent: the record reads back with the key it claims for its own
    # assignment, so nothing but a replay of the plan can tell that it moved.
    rewritten = SearchLedger(tracker.store).trial(materialised.id)
    assert rewritten.assignment_key == assignment_key(rewritten.assignment)
    assert rewritten.assignment != recorded.assignment

    report = SearchLedger(tracker.store).replay_check(_taili_space(), run_ref=tracker.run_ref)

    assert report.plan_match is True
    assert report.space_match is True
    assert report.replayable is True
    assert report.mismatched == (1,)
    assert report.reason != ""


# --- A32: batch semantics are auditable ----------------------------------------------


def test_history_len_records_the_batch_semantics(tmp_path):
    """The i-th assignment is chosen after seeing ``history_len`` useful observations.

    Only trials that are ``is_observation`` (terminal and with an objective) count: a trial
    without a y is not an (x, y) sample, and counting it would turn "how many previous results
    was this chosen after seeing" into a lie.
    """
    tracker = _tracker(tmp_path, budget=3, runner=FakeRunner())
    run = tracker.run()

    assert [record.history_len for record in run.trials] == [0, 1, 2]
    assert run.header.max_in_flight == 1
    assert all(record.status == "succeeded" for record in run.trials)


# --- A33: a closed search refuses further trial writes --------------------------------


def test_a_closed_search_refuses_a_retry_of_a_recorded_index(tmp_path):
    """``close_run`` tells a caller with unfinished trials to rerun them with ``retry=()``; following that advice on a closed search must cost nothing.

    A trial event appended after the closing record does not lose one line: from then on every
    read of this root fails with ``trial_after_close`` and nothing can repair it, because the
    close cannot be re-opened and the extra trial cannot be removed -- and by the time the line
    is written, the GPU time that produced it is already spent.  The refusal therefore has to
    happen before the directory is made and before the runner is entered, and the ledger
    afterwards has to still read back as the finished search it was.
    """
    tracker = _tracker(tmp_path, budget=2)
    finished = tracker.run()
    assert finished.complete is True
    bytes_before = tracker.store.events_path.read_bytes()
    events_before = tracker.store.events(verify=True)

    # A caller reconnecting to the closed search: the routine path after a crash, and the one
    # close_run's own error message sends it down.
    resumed = _tracker(tmp_path, budget=2, store=tracker.store, work_root=tracker.work_root)
    with pytest.raises(SearchError):
        resumed.run(retry=(1,))

    assert tracker.store.events(verify=True) == events_before
    assert tracker.store.events_path.read_bytes() == bytes_before

    # The part that matters: the ledger still reads, and it is still the closed search.
    reopened = SearchLedger(tracker.store).run(tracker.run_ref)
    assert reopened.complete is True
    assert reopened.closing is not None
    assert [record.index for record in reopened.trials] == [1, 2]
    assert [record.attempt for record in reopened.trials] == [1, 1]
    last = tracker.store.events(verify=True)[-1]
    assert last.record_type == "search_run"
    assert last.event_type == "supersede"


def test_a_closed_search_refuses_an_interrupted_retry(tmp_path):
    """``retry_interrupted=True`` is the other half of close_run's advice, and it must be refused by the same rule.

    The ledger here is closed while one trial is still ``running`` -- a shape the write side
    itself refuses to produce, written straight into the file as the read-side layout checks
    exist to catch.  It is what a search that closed before a resume could look like to a
    reconnecting process, and resuming it would append after the close, so the tracker has to
    refuse before it spends anything.
    """
    tracker = _tracker(tmp_path, budget=1)
    tracker.open_run()
    assignment = build_sampler(tracker.space, tracker.plan).collect()[0]
    materialised = tracker.begin_trial(1, assignment)
    assert materialised.status == "pending"
    tracker.store.append(
        "search_trial",
        _supersede(materialised, status="running"),
        actor="test",
        event_type="supersede",
    )
    tracker.store.append(
        "search_run",
        _supersede(
            tracker.open_run(),
            status="closed",
            stats={"proposals": 1, "exhausted": True},
            trial_count=1,
        ),
        actor="test",
        event_type="supersede",
    )
    assert SearchLedger(tracker.store).run(tracker.run_ref).interrupted_indices() == (1,)

    resumed = _tracker(tmp_path, budget=1, store=tracker.store, work_root=tracker.work_root)
    bytes_before = tracker.store.events_path.read_bytes()
    with pytest.raises(SearchError):
        resumed.run(retry_interrupted=True)

    assert tracker.store.events_path.read_bytes() == bytes_before
    reopened = SearchLedger(tracker.store).run(tracker.run_ref)
    assert reopened.complete is True
    assert [record.status for record in reopened.trials] == ["running"]


# --- A34: an injected sampler is only usable while it is unstarted --------------------


def test_reusing_an_advanced_sampler_across_two_runs_is_refused(tmp_path):
    """A sampler carries its own stream position, so a second run with the same object binds the *next* stretch of the stream to indices 1..N.

    Every assignment the second run records is individually legal and every read-side check
    passes, so what is silently broken is the reproducibility the sampler interface promises:
    "index 1 holds the plan's first assignment" is false, and nothing on disk says so.  A
    rebuilt sampler is right for the one this class derives from the plan, and wrong for one it
    was handed, so the second run refuses and names the count that exposes the position.
    """
    plan = _plan(budget=2)
    sampler = build_sampler(_taili_space(), plan)
    first = _tracker(tmp_path / "first", plan=plan, sampler=sampler)
    assert first.run().complete is True
    assert sampler.stats.proposals == 2

    second = _tracker(tmp_path / "second", plan=plan, sampler=sampler)
    with pytest.raises(SearchError) as caught:
        second.run()
    assert "2 assignment" in str(caught.value)
    assert second.store.records("search_trial") == {}


def test_a_caller_injected_sampler_that_has_proposed_nothing_is_used(tmp_path):
    """The accepted half of the same rule: an unstarted sampler is the caller's own stream, and the tracker must draw from it rather than from a sampler of its own.

    Both are derived from the same plan, so a substitution would be invisible in the record --
    the only evidence is whose counter moved.
    """
    plan = _plan(budget=2)
    sampler = build_sampler(_taili_space(), plan)
    assert sampler.stats.proposals == 0

    tracker = _tracker(tmp_path, plan=plan, sampler=sampler)
    run = tracker.run()

    assert run.complete is True
    assert [record.index for record in run.trials] == [1, 2]
    assert sampler.stats.proposals == 2


# --- A35: a decision made outside run() survives it -----------------------------------


def _resume_after_a_failed_attempt(
    tmp_path: Path, runner: FakeRunner
) -> SearchTracker:
    """A search left open by a failed first attempt, plus a tracker ready to retry index 1.

    The failed run leaves index 1 terminal (``failed``) and the search unclosed, which is the
    state in which ``retry=`` is meaningful: without the retry the index would be skipped, and
    with it a *second* attempt is entered over a record that already exists.
    """
    failing = FakeRunner(run_error=RuntimeError("the first attempt never started"))
    tracker = _tracker(tmp_path, budget=2, runner=failing)
    with pytest.raises(RuntimeError):
        tracker.run()
    assert [record.status for record in tracker.read().trials] == ["failed"]
    return _tracker(
        tmp_path,
        budget=2,
        runner=runner,
        store=tracker.store,
        work_root=tracker.work_root,
    )


def test_an_outside_finish_during_a_retry_is_not_overwritten(tmp_path):
    """An outside decision that landed while a trial was training must survive that trial's own result being written.

    The tracker ends the trial on top of the record it read back *after* the runner returned.
    Built from the pending snapshot it captured before the runner was entered, the ending would
    be ``succeeded`` -- written over the ``pruned`` an operator or a pruner had already decided
    on -- so the ledger would hold a different outcome than the one that actually happened, and
    nothing in it would show the contradiction.
    """
    runner = FakeRunner()
    resumed = _resume_after_a_failed_attempt(tmp_path, runner)

    def outside_decision(request):
        if request.index != 1:
            return
        resumed.observe(
            request.trial_id,
            TrialOutcome(
                status="running",
                observations=(TrialObservation(step=10, metrics={"loss": 0.5}),),
            ),
        )
        resumed.finish(request.trial_id, "pruned", stop_reason="the loss floor was crossed")

    runner.on_run = outside_decision
    finished = resumed.run(retry=(1,))

    assert finished.complete is True
    decided = next(record for record in finished.trials if record.index == 1)
    assert decided.status == "pruned"
    assert decided.stop_reason == "the loss floor was crossed"
    # The outcome the runner returned is not the verdict, and the curve it collected on the
    # way out is still a fact about the trial.
    assert decided.objective is None
    assert [observation.step for observation in decided.observations] == [10]


def test_an_outside_observation_during_a_retry_is_kept(tmp_path):
    """A stretch of the curve appended while a trial was running is still in the record once that trial ends.

    The observation is written by an outside call and the ending by the tracker afterwards;
    ending the trial from the pending snapshot would drop the stretch, and the ledger would
    hold a trial with fewer observations than it collected -- invisible, because the record
    that remains is perfectly self-consistent.
    """
    runner = FakeRunner()
    resumed = _resume_after_a_failed_attempt(tmp_path, runner)

    def outside_observation(request):
        if request.index != 1:
            return
        resumed.observe(
            request.trial_id,
            TrialOutcome(
                status="running",
                observations=(TrialObservation(step=10, metrics={"loss": 0.5}),),
            ),
        )

    runner.on_run = outside_observation
    finished = resumed.run(retry=(1,))

    assert finished.complete is True
    recorded = next(record for record in finished.trials if record.index == 1)
    assert recorded.status == "succeeded"
    assert recorded.objective == 1.5
    assert [observation.step for observation in recorded.observations] == [10]


# --- A36: closing and finishing refuse an unfinished state ----------------------------


def test_closing_over_an_unfinished_trial_is_refused(tmp_path):
    """Equal counts are not enough: one trial with no result makes a closed search read exactly like a finished one.

    This is the second half of the write-side anchor.  With "proposals == recorded" alone, a
    search holding a trial that never finished closes successfully and reads back as complete
    -- which is the one thing this module must never produce.
    """
    tracker = _tracker(tmp_path, budget=1)
    tracker.open_run()
    assignment = build_sampler(tracker.space, tracker.plan).collect()[0]
    materialised = tracker.begin_trial(1, assignment)
    assert materialised.status == "pending"

    with pytest.raises(SearchError) as pending_caught:
        tracker.close_run(SamplerStats(proposals=1, attempts=1, exhausted=True))
    assert "have not finished" in str(pending_caught.value)
    assert tracker.read().complete is False

    tracker.store.append(
        "search_trial",
        _supersede(materialised, status="running"),
        actor="test",
        event_type="supersede",
    )
    with pytest.raises(SearchError) as running_caught:
        tracker.close_run(SamplerStats(proposals=1, attempts=1, exhausted=True))
    assert "have not finished" in str(running_caught.value)
    assert tracker.read().complete is False


def test_finishing_a_trial_twice_is_refused(tmp_path):
    """A pruner racing a training run that completes must not be able to end one trial twice.

    "Pruned at step 10" and "it went on to finish successfully" cannot both be true of one
    trial; the first decision stands and the second is refused rather than written, so the
    contradiction never reaches the disk.
    """
    tracker = _tracker(tmp_path, budget=1)
    tracker.open_run()
    assignment = build_sampler(tracker.space, tracker.plan).collect()[0]
    materialised = tracker.begin_trial(1, assignment)
    tracker.store.append(
        "search_trial",
        _supersede(materialised, status="running"),
        actor="test",
        event_type="supersede",
    )

    first = tracker.finish(materialised.id, "pruned", stop_reason="the loss floor was crossed")
    assert first.status == "pruned"
    events_before = tracker.store.events(verify=True)

    with pytest.raises(SearchError) as caught:
        tracker.finish(materialised.id, "succeeded")
    assert "already finished as 'pruned'" in str(caught.value)
    assert tracker.store.events(verify=True) == events_before
    assert tracker.read().trials[0].status == "pruned"


# --- A37: append introduces an id, supersede rewrites it ------------------------------


def test_a_new_trial_event_is_an_append_and_a_rewrite_supersedes(tmp_path):
    """An id is introduced exactly once with ``append`` and every later version of it is a ``supersede``.

    The fold is what the distinction buys: a reader can tell "this trial grew a new version"
    from "a second trial was written under the same name", and the closing record -- a
    supersede of the run header -- from a second search opening in the same root.
    """
    tracker = _tracker(tmp_path, budget=2)
    tracker.run()
    events = tracker.store.events(verify=True)

    trial_ids = {event.record_id for event in events if event.record_type == "search_trial"}
    assert len(trial_ids) == 2
    for record_id in trial_ids:
        kinds = [
            event.event_type
            for event in events
            if event.record_type == "search_trial" and event.record_id == record_id
        ]
        # pending -> running -> terminal, and the first of them is the introduction.
        assert kinds == ["append", "supersede", "supersede"], record_id

    run_kinds = [
        event.event_type for event in events if event.record_type == "search_run"
    ]
    assert run_kinds == ["append", "supersede"]


def test_a_retried_index_supersedes_instead_of_appending_again(tmp_path):
    """A retry writes the same record id a second time, so its new events are supersedes even when what they carry is ``pending``.

    The decision belongs to "has this id been written before", computed from the ledger when a
    tracker reconnects, rather than to the call site: a retry entered as a plain append would
    introduce an id the file already holds.
    """
    resumed = _resume_after_a_failed_attempt(tmp_path, FakeRunner())
    finished = resumed.run(retry=(1,))

    assert finished.complete is True
    trial_id = finished.trials[0].id
    kinds = [
        event.event_type
        for event in resumed.store.events(verify=True)
        if event.record_type == "search_trial" and event.record_id == trial_id
    ]
    assert kinds[0] == "append"
    assert kinds.count("append") == 1
    assert kinds[1:] == ["supersede"] * (len(kinds) - 1)


# --- A38: the fold refuses a sequence that is not 1..N --------------------------------


def test_a_gap_in_the_trial_indices_is_refused(tmp_path):
    """The indices of one run are exactly 1..N: a gap means a trial was dropped, or two were recorded under one index.

    Without the check the run reads back normally, and the missing assignment looks like a
    point the plan never drew.
    """
    tracker = _tracker(tmp_path, budget=2)
    tracker.open_run()
    assignments = build_sampler(tracker.space, tracker.plan).collect()
    for index, assignment in ((1, assignments[0]), (3, assignments[1])):
        record = tracker.begin_trial(index, assignment)
        tracker.finish(record.id, "succeeded")

    with pytest.raises(SearchIntegrityError) as caught:
        SearchLedger(tracker.store).run(tracker.run_ref)
    assert "recorded trial indices" in str(caught.value)
    assert "(1, 3)" in str(caught.value)


def test_two_trials_sharing_one_assignment_are_refused(tmp_path):
    """Two indices drawn from one point are two records of one trial, and the assignment key is the identity a replay compares.

    Accepting them would let one point be counted as two results, which is exactly what
    "nothing was silently truncated" is anchored against on the other side.
    """
    tracker = _tracker(tmp_path, budget=2)
    tracker.open_run()
    assignment = build_sampler(tracker.space, tracker.plan).collect()[0]
    for index in (1, 2):
        record = tracker.begin_trial(index, assignment)
        tracker.finish(record.id, "succeeded")

    with pytest.raises(SearchIntegrityError) as caught:
        SearchLedger(tracker.store).run(tracker.run_ref)
    assert "same assignment_key" in str(caught.value)


# --- A39: a pending trial is resumable, an interrupted one is not ---------------------


def test_a_pending_trial_is_resumed_without_being_asked(tmp_path):
    """A trial that was materialised but never started is rerun by a plain ``run()``: repeating it repeats nothing.

    ``pending`` means the directory and the record exist and training never began, so there is
    no result to preserve -- unlike an interrupted trial, which is rerun only when
    ``retry_interrupted`` says so, because that one spends a card a second time.
    """
    tracker = _tracker(tmp_path, budget=2)
    tracker.open_run()
    assignment = build_sampler(tracker.space, tracker.plan).collect()[0]
    materialised = tracker.begin_trial(1, assignment)
    assert materialised.status == "pending"

    resumed = _tracker(tmp_path, budget=2, store=tracker.store, work_root=tracker.work_root)
    finished = resumed.run()

    assert finished.complete is True
    assert [record.index for record in finished.trials] == [1, 2]
    assert finished.trials[0].attempt == 2
    assert Path(finished.trials[0].run_dir) != Path(materialised.run_dir)


# --- A40: field ranges and identities a record carries --------------------------------


def test_index_and_attempt_start_at_one_and_history_len_is_not_negative():
    """An index is a position in the plan's sequence and an attempt counts tries, so neither has a zero, and history_len counts observations.

    A record is either self-consistent or cannot be built at all: an index of 0 would make "the
    Nth assignment of the sequence" unanswerable and nothing downstream could detect it.
    """
    with pytest.raises(ValidationError) as index_caught:
        TrialRecord.model_validate(_trial_payload(index=0))
    assert "index starts at 1" in str(index_caught.value)

    with pytest.raises(ValidationError) as attempt_caught:
        TrialRecord.model_validate(_trial_payload(attempt=0))
    assert "attempt starts at 1" in str(attempt_caught.value)

    with pytest.raises(ValidationError) as history_caught:
        TrialRecord.model_validate(_trial_payload(history_len=-1))
    assert "history_len cannot be negative" in str(history_caught.value)


def test_a_blank_run_ref_and_a_blank_assignment_key_are_refused():
    """Both fields are identities: the run a record belongs to and the point it was drawn from.

    Blank, they do not fail loudly -- they silently join every other blank one, attributing a
    trial to no run or deduplicating it against every keyless record.
    """
    with pytest.raises(ValidationError) as trial_caught:
        TrialRecord.model_validate(_trial_payload(run_ref=""))
    assert "run_ref must not be blank" in str(trial_caught.value)

    with pytest.raises(ValidationError) as key_caught:
        TrialRecord.model_validate(_trial_payload(assignment_key=""))
    assert "assignment_key must not be blank" in str(key_caught.value)

    with pytest.raises(ValidationError) as run_caught:
        SearchRunRecord.model_validate(_run_payload(run_ref=""))
    assert "run_ref must not be blank" in str(run_caught.value)


def test_a_non_finite_objective_is_refused():
    """The objective is the surrogate's y, and inf or nan is a number that compares false with everything, including itself.

    Refusing it when the record is built is the only place where it can be told apart from a
    real score; once on disk it enters the ranking as an ordinary float.
    """
    for bad in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValidationError) as caught:
            TrialRecord.model_validate(_trial_payload(objective=bad))
        assert "objective must be a finite number" in str(caught.value)


def test_a_space_fingerprint_that_does_not_match_its_plan_is_refused():
    """The space fingerprint claims which space a record was drawn from, and the plan it carries states its own.

    The two come from different sources, so a disagreement describes a search that could never
    have happened -- and replaying that plan would draw against the wrong space.
    """
    with pytest.raises(ValidationError) as trial_caught:
        TrialRecord.model_validate(_trial_payload(space_fingerprint="0" * 64))
    assert "was drawn from" in str(trial_caught.value)

    with pytest.raises(ValidationError) as run_caught:
        SearchRunRecord.model_validate(_run_payload(space_fingerprint="0" * 64))
    assert "was drawn from" in str(run_caught.value)


def test_a_record_carrying_an_injection_must_carry_both_config_fingerprints():
    """An injection is opaque JSON, so it can only be traced back to a config through the two fingerprints beside it.

    The injection itself is never validated again (its ``computed_field`` would collide with
    ``extra=forbid``), so dropping either fingerprint leaves a record whose config nobody can
    identify -- and the record validates just as happily with the value missing.
    """
    with pytest.raises(ValidationError) as source_caught:
        TrialRecord.model_validate(_trial_payload(source_fingerprint=""))
    assert "source and injected" in str(source_caught.value)

    with pytest.raises(ValidationError) as injected_caught:
        TrialRecord.model_validate(_trial_payload(injected_fingerprint=""))
    assert "source and injected" in str(injected_caught.value)


# --- A42: the state before a trial write is read from the ledger, not remembered -----
# (A33 is taken at line 1007; this case was added after the design's list stopped.)


def test_a_trial_cannot_be_written_without_opening_the_run(tmp_path):
    """A trial event in a root with no header belongs to no search, and no reader -- including ``open_run`` itself -- can ever make sense of it again.

    ``begin_trial`` is public and is a supported way in: a caller that materialises trials by
    hand never passes through ``run``, so "open the run first" cannot live in the caller's
    discipline.  The refusal has to come before the directory is made and before the runner is
    entered, because either of those is a side effect that outlives the failure.
    """
    tracker = _tracker(tmp_path)
    assignment = build_sampler(tracker.space, tracker.plan).collect()[0]

    with pytest.raises(SearchError) as caught:
        tracker.begin_trial(1, assignment)
    assert "has not opened" in str(caught.value)

    assert tracker.runner.preflights == []
    assert not tracker.work_root.exists()
    assert not tracker.store.events_path.exists()


def test_a_fresh_tracker_on_a_closed_search_refuses_a_trial_write(tmp_path):
    """The close is a fact about the ledger, so a tracker built *after* the close has to find out from the ledger.

    Reconnecting to a finished search is the routine path -- ``run()`` itself ends by reading
    the root back -- and the tracker it builds has never called ``open_run``, so a flag set
    there cannot be what says "closed".  The trap this closes is written into ``close_run``'s
    own error message: it tells a caller with unfinished trials to rerun them with ``retry=()``
    or ``retry_interrupted=True``, and doing that on a closed search would spend the GPU and
    then append a trial after the closing record, bricking the root.
    """
    tracker = _tracker(tmp_path, budget=2)
    assert tracker.run().complete is True
    closed_events = tracker.store.events(verify=True)
    workspaces = sorted(path.name for path in (tracker.work_root / "trials").iterdir())

    resumed = _tracker(tmp_path, budget=2, store=tracker.store, work_root=tracker.work_root)
    assignment = build_sampler(resumed.space, resumed.plan).collect()[0]

    with pytest.raises(SearchError) as caught:
        resumed.begin_trial(1, assignment)
    assert "has not opened" in str(caught.value)

    # Reconnecting the honest way; the refusal then comes from the close, still before
    # anything is written.  (Which of the two mechanisms supplies that verdict here is not
    # what this case pins: with ``open_run``'s header line deleted the tail re-read reaches
    # the same answer, and vice versa.  The case that distinguishes them -- and so the one
    # that pins the tail re-read -- is the opened-before-another-closed test below.)
    resumed.open_run()
    with pytest.raises(SearchError) as closed_caught:
        resumed.begin_trial(2, assignment)
    assert "closed" in str(closed_caught.value)

    assert resumed.runner.preflights == []
    assert sorted(path.name for path in (tracker.work_root / "trials").iterdir()) == workspaces
    assert tracker.store.events(verify=True) == closed_events


def test_a_tracker_that_opened_before_another_closed_the_search_is_refused(tmp_path):
    """This tracker opened while the search was open; the close happened afterwards, elsewhere, and it never heard about it.

    The local flag records what ``open_run`` saw and nothing more.  A pruner's driver, an
    operator, or a second tracker in the same process can close the search in between, and
    the trial write that follows would be the last one the root ever accepts.  Re-reading the
    ledger's last event is what turns "what this object remembers" into "what is true now" --
    and it is a tail read, so the correction costs one line rather than the whole search.
    """
    first = _tracker(tmp_path, budget=2)
    first.open_run()  # opened, and open at this moment
    assert first._closed is False
    assignment = build_sampler(first.space, first.plan).collect()[0]

    # Somebody else, over the same root, runs the search to its end and closes it.
    other = _tracker(tmp_path, budget=2, store=first.store, work_root=first.work_root)
    assert other.run().complete is True
    assert first._closed is False, "the flag is exactly as stale as the case describes"

    with pytest.raises(SearchError) as caught:
        first.begin_trial(1, assignment)
    assert "closed" in str(caught.value)

    # Nothing was written, and the finished search still reads back as finished.
    assert first.read().complete is True
    assert first.store.events(verify=True) == other.store.events(verify=True)


# --- A41: the guard is inside the write, and a close is sealed only when it can be read ---


def test_a_close_that_would_seal_a_gap_in_the_indices_is_refused(tmp_path):
    """A close is the one write that cannot be taken back, so it runs the read side's checks first.

    ``close_run`` refuses every later trial -- that is the whole point of A33 -- which makes a
    sealed root whose trial indices are not ``1..N`` unreadable *for good*: the missing index can
    never be materialised afterwards, and every read of the ledger then fails on the fold, not on
    the counts.  The indices are legal all the way in (``begin_trial`` takes any index >= 1 and
    the record only checks that much), so the check has to be here.

    What refusing buys is visible in the second half: the root is still readable, and the very
    same ``close_run`` succeeds once the missing trial exists -- which is the difference between
    "a search that needs one more trial" and "a search nobody can ever read".
    """
    tracker = _tracker(tmp_path, budget=2)
    tracker.open_run()
    assignments = build_sampler(tracker.space, tracker.plan).collect()
    record = tracker.begin_trial(2, assignments[1])  # index 1 never materialised
    tracker.finish(record.id, "succeeded")

    with pytest.raises(SearchError) as caught:
        tracker.close_run(SamplerStats(proposals=1, exhausted=False))
    assert "not 1..1" in str(caught.value)

    assert SearchLedger(tracker.store).runs()  # the root still reads

    repaired = tracker.begin_trial(1, assignments[0])
    tracker.finish(repaired.id, "succeeded")
    closed = tracker.close_run(SamplerStats(proposals=2, exhausted=False))

    assert closed.status == "closed"
    assert SearchLedger(tracker.store).run(tracker.run_ref).complete is True


def test_a_close_that_would_seal_two_trials_with_one_assignment_key_is_refused(tmp_path):
    """The same trap on the other read-side check: two indices drawn from one point are one trial recorded twice.

    The fold refuses it because the assignment key is what a replay compares, and counting one
    point twice would make the search's own result a fiction.  Refusing the close keeps the root
    repairable: the second index can still be superseded with the point it should have been --
    which is only possible while the search is open.
    """
    tracker = _tracker(tmp_path, budget=2)
    tracker.open_run()
    assignments = build_sampler(tracker.space, tracker.plan).collect()
    for index in (1, 2):
        record = tracker.begin_trial(index, assignments[0])
        tracker.finish(record.id, "succeeded")

    with pytest.raises(SearchError) as caught:
        tracker.close_run(SamplerStats(proposals=2, exhausted=False))
    assert "same assignment_key" in str(caught.value)

    repaired = tracker.begin_trial(2, assignments[1])
    tracker.finish(repaired.id, "succeeded")
    closed = tracker.close_run(SamplerStats(proposals=2, exhausted=False))

    assert closed.status == "closed"
    finished = SearchLedger(tracker.store).run(tracker.run_ref)
    assert [record.assignment_key for record in finished.trials] == [
        assignment_key(assignments[0]),
        assignment_key(assignments[1]),
    ]


def test_a_close_that_lands_before_the_write_is_still_seen_by_the_guard(tmp_path, monkeypatch):
    """The guard is the append's precondition, not a check in front of it, so a close that lands in between is seen.

    The window is real, and nothing exotic is needed to stand in it: a search with every trial
    terminal can be closed by anybody at any moment, and a tracker whose write is already on its
    way has no idea.  It is arranged here at the store's ``append`` -- after the guard has
    already passed once, at the top of ``begin_trial`` -- so the ordering is the only difference
    between this test passing and the root being bricked: an event landing after the closing
    record makes every later read fail, and nothing can repair it, because the close cannot be
    re-opened and the extra trial cannot be removed.

    The hook is a legal close through the public API, not a hand-written line: a second tracker
    over the same root, with the same plan, counting exactly the trials that are on disk.
    """
    tracker = _tracker(tmp_path, budget=2)
    tracker.open_run()
    assignments = build_sampler(tracker.space, tracker.plan).collect()
    record = tracker.begin_trial(1, assignments[0])
    tracker.finish(record.id, "succeeded")

    closer = _tracker(tmp_path, budget=2, store=tracker.store, work_root=tracker.work_root)
    real_append = TrialLedgerStore.append
    landed: list[bool] = []

    def append_after_a_close(self, record_type, *args, **kwargs):
        if record_type == "search_trial" and not landed:
            landed.append(True)
            closer.close_run(SamplerStats(proposals=1, exhausted=False))
        return real_append(self, record_type, *args, **kwargs)

    monkeypatch.setattr(TrialLedgerStore, "append", append_after_a_close)

    with pytest.raises(SearchError) as caught:
        tracker.begin_trial(2, assignments[1])
    assert "closed" in str(caught.value)
    assert landed == [True], "the close has to have landed, or this test proves nothing"

    reopened = SearchLedger(tracker.store).run(tracker.run_ref)
    assert reopened.complete is True
    assert [item.index for item in reopened.trials] == [1]


def test_a_close_that_lands_once_the_append_has_started_is_still_seen_by_the_guard(
    tmp_path, monkeypatch
):
    """The guard is run from inside the write lock, so a close that lands after the append began is still seen.

    The test above stands in the window *in front of* the call, which the guard sees whether
    or not it holds the lock.  This one stands in the window the lock itself defines: the close
    is injected the moment the trial's write tries to take the lock, so the only thing that
    decides the outcome is whether the guard runs before that point or after it.  A guard
    called next to the write rather than with it -- the shape ``_append_trial`` exists to
    avoid -- passes here, the trial lands behind the closing record, and every later read of
    the root raises ``trial_after_close`` with nothing able to repair it.

    Nothing exotic is needed to be that second writer, either: ``closer`` is another tracker
    over the same root, and it closes the search through the public API.
    """
    tracker = _tracker(tmp_path, budget=2)
    tracker.open_run()
    assignments = build_sampler(tracker.space, tracker.plan).collect()
    record = tracker.begin_trial(1, assignments[0])
    tracker.finish(record.id, "succeeded")

    closer = _tracker(tmp_path, budget=2, store=tracker.store, work_root=tracker.work_root)
    real_lock = TrialLedgerStore._write_lock
    landed: list[bool] = []

    def lock_after_a_close(self):
        if not landed:
            landed.append(True)
            closer.close_run(SamplerStats(proposals=1, exhausted=False))
        return real_lock(self)

    monkeypatch.setattr(TrialLedgerStore, "_write_lock", lock_after_a_close)

    with pytest.raises(SearchError) as caught:
        tracker.begin_trial(2, assignments[1])
    assert "closed" in str(caught.value)
    assert landed == [True], "the close has to have landed, or this test proves nothing"

    reopened = SearchLedger(tracker.store).run(tracker.run_ref)
    assert reopened.complete is True
    assert [item.index for item in reopened.trials] == [1]


def test_a_header_superseded_away_from_open_still_ends_the_search(tmp_path):
    """The guard decides "closed" from the event's type, the way the read side does, not from the payload's status.

    ``SearchRunRecord`` legally carries ``abandoned``, and the read side has already ruled on what
    it means: every ``search_run`` event that is not the opening append is the close, and nothing
    may follow it (``_validate_layout``).  A guard that compared the status *string* against
    ``closed`` would read such a run as open, accept the next trial write, and leave a root that
    fails ``trial_after_close`` on every read.  No writer in this module produces that record
    today; an operator, or a later task, can.

    The record written here is otherwise complete -- it counts the trials that are on disk -- so
    the root reads back normally and the only thing under test is the guard's predicate.
    """
    tracker = _tracker(tmp_path, budget=2)
    header = tracker.open_run()
    assignments = build_sampler(tracker.space, tracker.plan).collect()
    record = tracker.begin_trial(1, assignments[0])
    tracker.finish(record.id, "succeeded")
    tracker._append(
        "search_run",
        _supersede(
            header,
            status="abandoned",
            stats={"proposals": 1, "exhausted": False},
            trial_count=1,
        ),
        event_type="supersede",
    )
    events = tracker.store.events(verify=True)

    with pytest.raises(SearchError) as caught:
        tracker.begin_trial(2, assignments[1])
    assert "closed" in str(caught.value)

    assert tracker.store.events(verify=True) == events
    # The root still reads, and holds the one trial: the abandoned header is a close to the
    # *layout* check -- which is what makes a trial written after it fatal -- while
    # ``SearchRun.complete`` keys on the strict ``closed`` status, so an abandoned run reads as
    # an unfinished one.  That asymmetry is the read side's own business; what this case pins is
    # that nothing was written after the header.
    reopened = SearchLedger(tracker.store).run(tracker.run_ref)
    assert [record.index for record in reopened.trials] == [1]
    assert reopened.complete is False


def test_a_trial_write_into_a_ledger_whose_header_vanished_is_an_integrity_error(tmp_path):
    """``_opened`` is this object's memory; the file is the fact, and a deletion out of band is only visible in the file.

    An empty ledger is a legal state -- ``open_run`` writes its first line into one -- so
    ``last_event() is None`` is not by itself an error.  It is an error *here*, because this
    tracker has already opened the run: reading "no last event" as "no close" would let the next
    trial land as a headerless event, and a root like that fails ``unknown_layout`` for every
    reader, including ``open_run`` itself, so it could not be repaired by opening the run again.
    """
    tracker = _tracker(tmp_path, budget=2)
    tracker.open_run()
    assignments = build_sampler(tracker.space, tracker.plan).collect()
    record = tracker.begin_trial(1, assignments[0])
    tracker.finish(record.id, "succeeded")
    workspaces = sorted(path.name for path in (tracker.work_root / "trials").iterdir())

    tracker.store.events_path.unlink()  # a reset script, an editor saving an empty file

    with pytest.raises(SearchIntegrityError) as caught:
        tracker.begin_trial(2, assignments[1])
    assert "holds no events at all" in str(caught.value)

    assert not tracker.store.events_path.exists()
    assert sorted(path.name for path in (tracker.work_root / "trials").iterdir()) == workspaces


def test_a_pruner_that_ends_a_trial_while_the_runner_dies_keeps_its_verdict(tmp_path):
    """A cancelled runner usually dies on the way out; that is a fact about the call, not a second verdict on the trial.

    ``finish`` refuses to overwrite a terminal trial and ``run``'s success path honours that, but
    the error path has to as well, or the ledger ends up holding a record that says "pruned" in
    its ``stop_reason`` and "failed, the backend died" in its status -- self-contradictory, and
    with nothing on disk showing that the pruner and the exception disagreed.  It would also move
    the trial between the two buckets a search report counts, which is a change to the search's
    own result.

    The exception is raised either way: the caller is the one who has to know the runner blew up.
    The control in the second half is the same crash without the pruner, which must still be
    recorded as a failure -- otherwise "keep the earlier verdict" would have become "never record
    a crash at all".
    """
    pruned_root = tmp_path / "pruned"
    pruned_store = TrialLedgerStore(pruned_root / "ledger")
    pruned_work = pruned_root / "work"
    pruner = _tracker(pruned_root, budget=1, store=pruned_store, work_root=pruned_work)

    pruner.open_run()  # the pruner is a writer on this root, so it opens the run like any other

    def prune(request: Any) -> None:
        pruner.finish(request.trial_id, "pruned", stop_reason="loss floor crossed")

    crashed = _tracker(
        pruned_root,
        budget=1,
        store=pruned_store,
        work_root=pruned_work,
        runner=FakeRunner(run_error=RuntimeError("backend died"), on_run=prune),
    )

    with pytest.raises(RuntimeError, match="backend died"):
        crashed.run()

    finished = SearchLedger(pruned_store).run(crashed.run_ref)
    assert [record.status for record in finished.trials] == ["pruned"]
    assert finished.trials[0].stop_reason == "loss floor crossed"
    assert finished.trials[0].error == ""
    assert finished.complete is False

    crashed_root = tmp_path / "crashed"
    bare = _tracker(
        crashed_root, budget=1, runner=FakeRunner(run_error=RuntimeError("backend died"))
    )

    with pytest.raises(RuntimeError, match="backend died"):
        bare.run()

    recorded = SearchLedger(bare.store).run(bare.run_ref)
    assert [record.status for record in recorded.trials] == ["failed"]
    assert "RuntimeError: backend died" in recorded.trials[0].error
