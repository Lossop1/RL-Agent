"""The record contract of one hyperparameter search: the write facade, the read views, and the seams for tasks 5/6/7.

Task 4's output is the single read-side entry point for the three steps that follow it.
It records "which hyperparameters were chosen, which config file they were injected
into, which directory the trial ran in, what came out, and why it stopped" as records
that can be judged after a crash, read back from another process, and replayed later.
It does not run training, decide when to stop, or rank anything.

Two layers, and the line between them must not blur:

* ``trial_ledger`` is a general persistence engine (append, lock, chain check, torn-tail
  self-heal) with not one line of hyperparameter code, testable on its own without a
  space or a sampler.
* This module is the search semantics: what a record looks like, when it is written, and
  which status counts as terminal.  It deliberately does **not** reuse
  ``ResearchLedgerStore``: that store's append is O(file length) and its lock is
  in-process only, while a search naturally produces concurrent writes.

Four implementation laws, each backed by a measurement.  They must still hold whenever
this file is edited:

* **R1 no ``@computed_field``; derived values are plain methods.**  A computed field is
  written out by ``model_dump(mode="json")`` and then refused by ``extra="forbid"`` on the way
  back in, and the ledger's read path necessarily requires that every record can be validated
  again.  The measurement is ``InjectedValue``'s ``changed`` (``config_injection.py:183``),
  which sits *inside* the report rather than on it: a report holding seven values raises
  ``7 validation errors for InjectionReport``, one per ``values.N.changed``, while the
  report's own six fields round-trip unchanged.
* **R2 no ``model_copy(update=...)``.**  pydantic v2's ``model_copy`` runs no validator:
  ``M(a=0).model_copy(update={"a": 5})`` quietly fabricates a record that contradicts
  itself.  Every state transition goes through :func:`_supersede`, which re-runs every
  invariant.
* **R3 hash the output of ``model_dump(mode="json")``.**  ``content_hash`` handed a
  ``BaseModel`` goes through its own ``exclude_none``, and fields whose value is ``None``
  are dropped.
* **R4 record fields are all JSON-native; never put a ``BaseModel`` into a plain dict and
  hash that.**  An ``InjectionReport`` is stored as **opaque JSON**, never validated as an
  ``InjectionReport`` again (that would hit R1's road), and its integrity rests on
  ``content_hash(injection) == injection_fingerprint``.

Five tightenings relative to the design document, each with its reason, written down here
so they are not mistaken for drift:

1. In ``TrialRecord``'s invariants, a non-blank ``injection_fingerprint`` is only required
   for the four statuses that mean "this really ran" (``running`` / ``succeeded`` /
   ``pruned`` / ``stopped``); ``failed`` is not among them.  The design listed ``failed``
   too, and that makes "the config could not be injected" record impossible to write
   -- which is exactly the failure most worth recording, because it happens before any GPU
   is spent and, if it were only raised, nobody would know after the process exits why a
   trial is missing.
2. ``close_run`` refuses to close a search while any trial is still ``running`` or
   ``pending``.  The design asked only that "proposals == recorded".  A search whose counts
   match but which has one trial that never finished **reads exactly like a finished one**,
   and that is the failure this module exists to prevent.
3. A trial write requires that this tracker has opened its run, and re-checks the close
   against the ledger's last event rather than trusting its own flag.  The design lists
   ``open_run`` and ``begin_trial`` as independent entry points; making the header a
   precondition costs one tail read per write and closes three ways to leave a root that
   no reader can ever open again (see :meth:`SearchTracker._require_open`).
4. ``run`` refuses a caller-injected sampler that has already proposed assignments.  A
   sampler carries its own stream position: reused across two runs, it binds proposals
   ``1..N`` to the first search and repeats them under the second search's indices, which
   silently breaks the reproducibility the sampler interface promises.
5. ``run``'s terminal write reads the trial's *current* record instead of writing from the
   snapshot taken before the trial ran.  An ``observe`` or ``finish`` that landed while the
   runner was working is a decision somebody made; writing over it from a stale snapshot
   records an outcome that did not happen.
"""
from __future__ import annotations

import math
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .config_injection import inject_into_file, load_config_file
from .hyperparameter_sampler import (
    Sampler,
    SamplerStats,
    SearchPlan,
    assignment_key,
    build_sampler,
    replay,
)
from .hyperparameter_space import SearchSpaceSchema
from .research_ledger import ArtifactRef, LedgerRecord, content_hash
from .trial_ledger import TrialEvent, TrialLedgerIntegrityError, TrialLedgerStore


SEARCH_SCHEMA_VERSION = "rl-agent.hyperparameter-search/v1"

#: Every trial directory of one search lands under this subdirectory; the directory name
#: carries its own random token, see ``begin_trial``.
TRIAL_WORKSPACE_DIRNAME = "trials"

#: The config file written into a trial's directory after injection.  The name carries no
#: framework section (say ``agent.skrl``): that is the shape of a product config, not
#: something this module should know (zero hardcoding).
TRIAL_CONFIG_FILENAME = "trial_config.yaml"

TrialStatus = Literal["pending", "running", "succeeded", "failed", "pruned", "stopped"]

#: The statuses a trial can no longer move out of.  ``interrupted`` is not in
#: ``TrialStatus``: it describes how the ledger looks right now, not a fact some write
#: decided, so the read side derives it.
TERMINAL_TRIAL_STATUSES: frozenset[str] = frozenset({"succeeded", "failed", "pruned", "stopped"})

#: Which sampler kinds have a sequence that can be compared against a replay.  An adaptive
#: sampler picks each next point from the observations so far, so replaying it against the
#: current space necessarily differs from the record and comparing it would only produce false
#: drift alarms.  Forgetting to register a kind fails safe (no comparison means no false
#: alarm), and acceptance case A23's vocabulary test fails whenever ``SamplerKind`` changes,
#: forcing an explicit decision.
REPLAYABLE_SAMPLER_KINDS: frozenset[str] = frozenset({"grid", "random"})

#: The complement, named rather than left as "whatever is not above".  Two named sets that a
#: test proves partition ``SamplerKind`` say more than one set and a subtraction: a kind added
#: to neither is a kind nobody decided about, and the partition test fails on it.
#:
#: ``bayesian`` is here because its sequence is a function of the trials already run, not of
#: the plan alone.  That is a statement about replayability, not a defect: the sequence is
#: still reproducible from the ledger, just not from ``(space, plan)``.
NON_REPLAYABLE_SAMPLER_KINDS: frozenset[str] = frozenset({"bayesian"})


class SearchError(Exception):
    """Orchestration went wrong: an illegal status, a call out of order, missing statistics."""


class SearchPreflightError(SearchError):
    """This trial cannot run, discovered before anything with a side effect happened.

    It is its own type because the remedy is the opposite of the usual one: this class of
    failure contracts for **zero side effects** -- no record, no directory, no GPU.  An
    adapter puts its scheduler's pre-checks (say "the candidate artifact must be a file")
    here, raised synchronously, rather than letting them become an asynchronous FAILED
    state inside a scheduling thread.
    """


class SearchIntegrityError(SearchError):
    """The ledger does not match this search: a different run_ref/plan/space, or records that contradict each other."""


class TrialObservation(BaseModel):
    """One intermediate observation: a snapshot of the metrics at training step ``step``.

    Task 5's early-stopping curve lands here.  The slot exists in the record already so
    that landing task 5 does not have to change the record shape tasks 5 and 6 depend on.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_observation(self) -> "TrialObservation":
        if self.step < 0:
            raise ValueError(f"an observation step cannot be negative, got {self.step}")
        return self


class TrialRequest(BaseModel):
    """One trial as handed to a ``TrialRunner``: what to run, where, and for how long at most."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trial_id: str
    index: int
    attempt: int
    plan_fingerprint: str
    space_fingerprint: str
    assignment_key: str
    assignment: dict[str, Any]
    workspace: Path
    config_path: Path
    max_seconds: float | None = None

    @model_validator(mode="after")
    def _validate_request(self) -> "TrialRequest":
        if not self.trial_id.strip():
            raise ValueError("trial_id must not be blank")
        if self.index < 1:
            raise ValueError(f"a trial index starts at 1, got {self.index}")
        if self.attempt < 1:
            raise ValueError(f"a trial attempt starts at 1, got {self.attempt}")
        if self.max_seconds is not None and self.max_seconds <= 0:
            raise ValueError(f"max_seconds must be positive when given, got {self.max_seconds}")
        if self.workspace.resolve() not in self.config_path.resolve().parents:
            # If two trials shared one config file, the later write would alter the earlier
            # one's evidence, and both lines would still look perfectly normal.
            raise ValueError(
                f"the config {self.config_path} must live inside the trial workspace {self.workspace}"
            )
        return self


class TrialOutcome(BaseModel):
    """The facts after one trial has run.

    ``objective`` being ``None`` means this trial produced no usable scalar score
    (aligning with ``coordinator.EvaluatorResult.score``).  It is the surrogate model's y,
    rather than something guessed out of ``metrics`` -- a wrong guess is invisible.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: TrialStatus
    objective: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    observations: tuple[TrialObservation, ...] = ()
    message: str = ""
    backend_ref: str = ""
    log_path: str = ""

    @model_validator(mode="after")
    def _validate_outcome(self) -> "TrialOutcome":
        if self.objective is not None and not math.isfinite(self.objective):
            raise ValueError(f"objective must be a finite number, got {self.objective}")
        return self


@runtime_checkable
class TrialRunner(Protocol):
    """The **only** injected hop between the search and an execution backend.

    Structural conformance is enough: ``SearchTracker`` does no ``isinstance`` check and
    gives a runner no base class.  Checking would only push callers into inheriting a class
    they do not need, when all they have to satisfy is "it has these two methods".
    """

    def preflight(self, request: TrialRequest) -> None:
        """Validate synchronously before spending anything; raise on failure, with no side effects."""
        ...

    def run(self, request: TrialRequest) -> TrialOutcome:
        """Run one trial to completion, blocking, and return the facts."""
        ...


@runtime_checkable
class CancellableTrialRunner(TrialRunner, Protocol):
    """A runner that can be stopped mid-trial (task 5's loss-cutting lever).  The core path never calls it."""

    def cancel(self, trial_id: str, reason: str) -> None:
        """Ask the backend to stop ``trial_id``; whether it stopped is the backend's honest answer, not a promise made here."""
        ...


@runtime_checkable
class TrialObservationSource(Protocol):
    """Task 7's seam: whoever supplies (hyperparameters, objective) pairs to a surrogate model.

    The name and the signature match :meth:`SearchLedger.observations`, so ``SearchLedger``
    satisfies it directly (acceptance case A22 proves that with ``isinstance``).  Both
    fingerprints are required: an observation from a different space would teach the
    surrogate the wrong mapping, and a wrong mapping does not announce itself.
    """

    def observations(
        self, *, plan_fingerprint: str, space_fingerprint: str
    ) -> tuple["TrialRecord", ...]:
        """The trial records that match both fingerprints, in trial order."""
        ...


_RecordT = TypeVar("_RecordT", bound=BaseModel)


def _supersede(previous: _RecordT, **changes: Any) -> _RecordT:
    """Build a new record on top of ``previous`` and **re-run every invariant**.

    Never ``model_copy(update=...)``: pydantic v2's ``model_copy`` runs no validator, so a
    record that says ``pruned`` with an empty ``stop_reason`` gets fabricated quietly, and
    everyone who reads it afterwards has to notice on their own that an invariant broke.
    Every state transition goes through here.
    """
    payload = {**previous.model_dump(mode="json"), **changes}
    return type(previous).model_validate(payload)


def _fingerprint_of(plan: dict[str, Any]) -> str:
    """The identity of the plan a record carries.  An illegal plan raises from its own validator."""
    return SearchPlan.model_validate(plan).fingerprint()


class SearchRunRecord(LedgerRecord):
    """One per search.

    ``plan`` stores the **whole** plan (``model_dump(mode="json")``) and not only its
    fingerprint: ``replay(space, plan)`` needs the plan object, and the fingerprint is only
    its identity (``SearchPlan`` holds no parameter domains and no cursor).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SEARCH_SCHEMA_VERSION
    run_ref: str
    plan: dict[str, Any]
    plan_fingerprint: str
    space_fingerprint: str
    sampling_order: tuple[str, ...]
    source_config: str
    source_config_fingerprint: str
    space_artifact: ArtifactRef | None = None
    baseline_ref: str = ""
    budget: int | None = None
    max_in_flight: int = 1
    stats: dict[str, Any] = Field(default_factory=dict)
    trial_count: int = 0
    status: Literal["open", "closed", "abandoned"] = "open"

    @model_validator(mode="after")
    def _validate_run(self) -> "SearchRunRecord":
        for name in ("run_ref", "plan_fingerprint", "space_fingerprint"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be blank")
        plan = SearchPlan.model_validate(self.plan)
        expected = plan.fingerprint()
        if self.plan_fingerprint != expected:
            raise ValueError(
                f"the run cites plan {self.plan_fingerprint} but the plan it carries "
                f"fingerprints to {expected}"
            )
        if self.space_fingerprint != plan.space_fingerprint:
            raise ValueError(
                f"the run cites space {self.space_fingerprint} but its plan was drawn from "
                f"{plan.space_fingerprint}"
            )
        if tuple(self.sampling_order) != tuple(plan.sampling_order):
            # A space's fingerprint ignores declaration order; the sequence a replay
            # produces does not, so a differing order is a different sequence.
            raise ValueError(
                f"the run records the order {tuple(self.sampling_order)} but its plan draws in "
                f"{tuple(plan.sampling_order)}"
            )
        if self.budget != plan.budget:
            raise ValueError(f"the run records budget {self.budget} but its plan holds {plan.budget}")
        if self.max_in_flight < 1:
            raise ValueError(f"max_in_flight must be at least 1, got {self.max_in_flight}")
        if self.status == "closed":
            if "proposals" not in self.stats:
                raise ValueError(
                    "a closed search must record the sampler statistics, including 'proposals': "
                    "without them the record cannot show whether the grid finished or a budget cut it short"
                )
            if self.trial_count != self.stats["proposals"]:
                raise ValueError(
                    f"the run recorded {self.trial_count} trials but the sampler proposed "
                    f"{self.stats['proposals']}: closing now would report a search that skipped "
                    f"some of what it proposed"
                )
        return self


class TrialRecord(LedgerRecord):
    """One per trial, written several times as it is superseded (``pending`` -> ``running`` -> terminal).

    ``index`` is this trial's position in the plan's sampling sequence.  The plan does
    **not** carry it (``budget`` is only an upper bound), and without the index no
    observation can be mapped back to the Nth assignment of the sequence -- which is what
    "replay is recovery" would need.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SEARCH_SCHEMA_VERSION
    run_ref: str
    index: int
    attempt: int = 1
    history_len: int = 0
    assignment: dict[str, Any]
    assignment_key: str
    plan: dict[str, Any]
    plan_fingerprint: str
    space_fingerprint: str
    status: TrialStatus = "pending"
    run_dir: str = ""
    config_path: str = ""
    injection: dict[str, Any] = Field(default_factory=dict)
    injection_fingerprint: str = ""
    source_fingerprint: str = ""
    injected_fingerprint: str = ""
    backend_ref: str = ""
    log_path: str = ""
    objective: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    observations: tuple[TrialObservation, ...] = ()
    stop_reason: str = ""
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0

    def duration_s(self) -> float:
        """How long this trial took.  A plain method rather than ``@computed_field`` (law R1):

        a ``computed_field`` would enter ``model_dump``, and then the dict read back would
        not validate -- which is the ground the ledger stands on.  The price is that it
        has to be written ``record.duration_s()``.
        """
        return self.finished_at - self.started_at

    @model_validator(mode="after")
    def _validate_trial(self) -> "TrialRecord":
        if self.index < 1:
            raise ValueError(f"a trial index starts at 1, got {self.index}")
        if self.attempt < 1:
            raise ValueError(f"a trial attempt starts at 1, got {self.attempt}")
        if self.history_len < 0:
            raise ValueError(f"history_len cannot be negative, got {self.history_len}")
        for name in ("run_ref", "assignment_key"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be blank")
        plan = SearchPlan.model_validate(self.plan)
        expected = plan.fingerprint()
        if self.plan_fingerprint != expected:
            raise ValueError(
                f"the trial cites plan {self.plan_fingerprint} but the plan it carries "
                f"fingerprints to {expected}"
            )
        if self.space_fingerprint != plan.space_fingerprint:
            raise ValueError(
                f"the trial cites space {self.space_fingerprint} but its plan was drawn from "
                f"{plan.space_fingerprint}"
            )
        if self.status in {"running", "succeeded", "failed", "pruned", "stopped"} and not self.config_path.strip():
            raise ValueError(
                f"a trial that reached {self.status} must record the config it was injected into"
            )
        if not self.injection and not self.injection_fingerprint.strip() and self.status != "failed":
            raise ValueError(
                f"a trial that reached {self.status} must record the injection that produced "
                f"its config"
            )
        if self.injection:
            actual = content_hash(self.injection)
            if actual != self.injection_fingerprint:
                raise ValueError(
                    f"the trial cites injection {self.injection_fingerprint} but the injection it "
                    f"carries hashes to {actual}"
                )
            if not self.source_fingerprint.strip() or not self.injected_fingerprint.strip():
                raise ValueError(
                    "a record carrying an injection must also carry the source and injected "
                    "config fingerprints, or the injection cannot be traced back to a config"
                )
        if self.objective is not None and not math.isfinite(self.objective):
            raise ValueError(f"objective must be a finite number, got {self.objective}")
        if self.status == "succeeded" and not self.run_dir.strip():
            raise ValueError("a succeeded trial must record the directory it ran in")
        if self.status == "failed" and not self.error.strip():
            raise ValueError("a failed trial must record why it failed")
        if self.status in {"pruned", "stopped"} and not self.stop_reason.strip():
            raise ValueError(f"a {self.status} trial must record why it stopped")
        if self.status in TERMINAL_TRIAL_STATUSES and self.finished_at <= 0:
            raise ValueError(f"a {self.status} trial must record when it finished")
        return self


class SearchRun(BaseModel):
    """One search as seen from the read side at one moment.

    ``header`` is the **latest** version of this run's record: once closed it carries
    ``stats`` and ``trial_count``.  ``closing`` is that same content, non-empty only when
    the run has closed -- which name you use depends on whether you are asking "how did
    this search go" or "is it over".
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    header: SearchRunRecord
    closing: SearchRunRecord | None = None
    trials: tuple[TrialRecord, ...] = ()
    #: Unclosed means incomplete.  Stored as a field rather than computed (law R1), and the
    #: validator keeps it equal to ``closing is not None`` -- a redundant derived field that
    #: could disagree is not redundancy but a second source of truth.
    complete: bool = False

    @model_validator(mode="after")
    def _validate_completeness(self) -> "SearchRun":
        if self.complete != (self.closing is not None):
            raise ValueError("complete must be true exactly when the run carries a closing record")
        return self

    def interrupted_indices(self) -> tuple[int, ...]:
        """The trials that have a record but no ending: the process died in the middle of them.

        This is a read-side derivation (``interrupted`` is not a ``TrialStatus``), and it
        **does not guess the result**: the result of a killed trial is lost for good, and
        anything reconstructed from a training log is a guess, not a record.
        """
        return tuple(record.index for record in self.trials if record.status == "running")

    def terminal_trials(self) -> tuple[TrialRecord, ...]:
        """The trials that ran to an ending."""
        return tuple(record for record in self.trials if record.status in TERMINAL_TRIAL_STATUSES)


class ReplayReport(BaseModel):
    """The result of comparing the sampled sequence in the record against what the same space draws today."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_ref: str
    plan_match: bool
    space_match: bool
    replayable: bool
    mismatched: tuple[int, ...] = ()
    reason: str = ""


def is_observation(record: TrialRecord) -> bool:
    """Whether this trial can serve as one sample for a surrogate model: terminal, with a usable scalar objective.

    A ready-made criterion for task 7 rather than a decision made on its behalf -- whether
    ``pruned`` / ``failed`` trials belong in the training set is the surrogate's own
    choice.  This answers only "is there a y".
    """
    return record.status in TERMINAL_TRIAL_STATUSES and record.objective is not None


def default_run_ref(plan: SearchPlan, space: SearchSpaceSchema) -> str:
    """A run name decided by the plan alone: a second run of one plan folds into the same session.

    The space is taken only so that "this plan and this space are not a pair" is reported
    here, rather than when the first assignment is injected.
    """
    if plan.space_fingerprint != space.fingerprint():
        raise ValueError(
            f"the plan was drawn from space {plan.space_fingerprint}, which is not "
            f"{space.fingerprint()}"
        )
    return f"search:{plan.fingerprint()[:16]}"


class SearchLedger:
    """The read side: folds the event log into search runs, trials, and replay conclusions.

    ``SearchLedger`` offers ``observations`` with the same name and signature as
    ``TrialObservationSource``, so it **is** task 7's seam and needs no wrapper.

    Every read first runs the integrity checks (chain, sequence, torn tail) and then the
    layout checks (one search per root, nothing appended after a close, counts that agree).
    Refusing to return beats returning a search that looks complete and was in fact
    truncated.
    """

    def __init__(self, store: TrialLedgerStore) -> None:
        self.store = store
        self._stamp: tuple[int, int] | None = None
        self._events: tuple[TrialEvent, ...] = ()

    def _stamp_now(self) -> tuple[int, int]:
        try:
            stat = self.store.events_path.stat()
        except FileNotFoundError:
            return (0, 0)
        return (stat.st_size, stat.st_mtime_ns)

    def refresh(self) -> None:
        """Re-read only when the file moved; otherwise reuse what was parsed and verified.

        The stamp is taken *before* the read, and that is the stamp that gets recorded.  If
        the file changes during the read, the current stamp will differ from it and the
        next read happens as usual; taking the stamp after the read could pair old data
        with a new stamp and make the next read think nothing changed.  The most that costs
        is one extra read, in the safe direction.
        """
        stamp = self._stamp_now()
        if stamp == self._stamp:
            return
        events = self.store.events(verify=True)
        self._validate_layout(events)
        self._events = events
        self._stamp = stamp

    def _validate_layout(self, events: tuple[TrialEvent, ...]) -> None:
        """Layout rules: one root holds exactly one search, and the line order is fixed.

        An empty ledger is not an error: a directory nothing has been written to yet and
        one whose first line was deleted are different things, and ``open_run`` has to be
        able to make the first write into an empty directory.
        """
        if not events:
            return
        headers = [position for position, event in enumerate(events) if event.record_type == "search_run"]
        appends = [position for position in headers if events[position].event_type == "append"]
        if not appends:
            raise TrialLedgerIntegrityError(
                "unknown_layout",
                "the ledger holds trial events but no search_run record to explain them",
            )
        if appends[0] != 0:
            raise TrialLedgerIntegrityError(
                "unknown_layout",
                f"the first event is {events[0].record_type}, not the search run that opened the ledger",
            )
        if len(appends) > 1:
            raise TrialLedgerIntegrityError(
                "unknown_layout",
                f"{len(appends)} search_run records were appended; one ledger root carries exactly "
                f"one search",
            )
        closes = [position for position in headers if events[position].event_type != "append"]
        if len(closes) > 1:
            raise TrialLedgerIntegrityError(
                "duplicate_close", f"this search was closed {len(closes)} times"
            )
        if closes:
            if events[closes[0]].record_id != events[appends[0]].record_id:
                raise TrialLedgerIntegrityError(
                    "unknown_layout",
                    f"the closing event names record {events[closes[0]].record_id}, but the ledger "
                    f"was opened by {events[appends[0]].record_id}",
                )
            after = [
                position
                for position, event in enumerate(events)
                if event.record_type == "search_trial" and position > closes[0]
            ]
            if after:
                raise TrialLedgerIntegrityError(
                    "trial_after_close",
                    f"{len(after)} trial events were appended after the search was closed",
                )

    def _folded_trials(self) -> tuple[TrialRecord, ...]:
        """Fold by ``record_id``, keeping the last event of each id, ordered by trial index."""
        latest: dict[str, TrialEvent] = {}
        for event in self._events:
            if event.record_type == "search_trial":
                latest[event.record_id] = event
        records: list[TrialRecord] = []
        for event in latest.values():
            try:
                records.append(TrialRecord.model_validate(event.payload))
            except ValidationError as error:
                raise SearchIntegrityError(
                    f"the trial {event.record_id} does not describe a trial record: {error}"
                ) from error
        return tuple(sorted(records, key=lambda record: record.index))

    def _run_records(self) -> tuple[SearchRunRecord, ...]:
        latest: dict[str, TrialEvent] = {}
        for event in self._events:
            if event.record_type == "search_run":
                latest[event.record_id] = event
        records: list[SearchRunRecord] = []
        for event in latest.values():
            try:
                records.append(SearchRunRecord.model_validate(event.payload))
            except ValidationError as error:
                raise SearchIntegrityError(
                    f"the search run {event.record_id} does not describe a run record: {error}"
                ) from error
        return tuple(records)

    def _trials_of(self, header: SearchRunRecord) -> tuple[TrialRecord, ...]:
        """This search's trials, checking on the way that none went missing and none came from elsewhere."""
        trials = self._folded_trials()
        for record in trials:
            if record.run_ref != header.run_ref:
                raise SearchIntegrityError(
                    f"trial {record.id} belongs to run {record.run_ref!r}, but this ledger holds "
                    f"{header.run_ref!r}"
                )
            if (
                record.plan_fingerprint != header.plan_fingerprint
                or record.space_fingerprint != header.space_fingerprint
            ):
                raise SearchIntegrityError(
                    f"trial {record.id} cites plan {record.plan_fingerprint} / space "
                    f"{record.space_fingerprint}, which is not this run's "
                    f"{header.plan_fingerprint} / {header.space_fingerprint}: two searches have been "
                    f"written into one root"
                )
        indices = tuple(record.index for record in trials)
        if indices != tuple(range(1, len(trials) + 1)):
            raise SearchIntegrityError(
                f"the recorded trial indices are {indices}, not 1..{len(trials)}: a trial was "
                f"dropped from the run or recorded twice"
            )
        keys = [record.assignment_key for record in trials]
        if len(set(keys)) != len(keys):
            raise SearchIntegrityError("two trials of one run carry the same assignment_key")
        return trials

    def runs(self) -> tuple[SearchRunRecord, ...]:
        """The searches in this root, each record at its latest version."""
        self.refresh()
        return self._run_records()

    def run(self, run_ref: str) -> SearchRun:
        """The full read view of one search, with the count checks run before it is returned."""
        self.refresh()
        headers = [record for record in self._run_records() if record.run_ref == run_ref]
        if not headers:
            raise SearchIntegrityError(f"this ledger holds no search run named {run_ref!r}")
        header = headers[0]
        closing = header if header.status == "closed" else None
        trials = self._trials_of(header)
        if closing is not None:
            self._check_counts(closing, trials)
        return SearchRun(
            header=header, closing=closing, trials=trials, complete=closing is not None
        )

    @staticmethod
    def _check_counts(closing: SearchRunRecord, trials: tuple[TrialRecord, ...]) -> None:
        """``proposals == recorded``, exactly.

        The two numbers come from **different** code paths (one from the number of trials
        read back, one from a counter the sampler increments on every yield), which is what
        makes their agreement informative.  ``close_run`` checks the same invariant on the
        write side; the same trick appears in ``write_injection`` recomputing a hash and in
        ``inject_into_file`` comparing the two reports it wrote and read.
        """
        if closing.trial_count != len(trials):
            raise TrialLedgerIntegrityError(
                "count_mismatch",
                f"the closing record counts {closing.trial_count} trials but the file holds "
                f"{len(trials)}",
            )
        proposed = closing.stats.get("proposals")
        if proposed != len(trials):
            raise TrialLedgerIntegrityError(
                "count_mismatch",
                f"the sampler proposed {proposed} trials but the file holds {len(trials)}",
            )

    def trials(
        self,
        *,
        run_ref: str = "",
        plan_fingerprint: str = "",
        status: TrialStatus | None = None,
    ) -> tuple[TrialRecord, ...]:
        """Filter trials.  This is a filter and does not run ``run()``'s layout checks."""
        self.refresh()
        found = self._folded_trials()
        if run_ref:
            found = tuple(record for record in found if record.run_ref == run_ref)
        if plan_fingerprint:
            found = tuple(record for record in found if record.plan_fingerprint == plan_fingerprint)
        if status is not None:
            found = tuple(record for record in found if record.status == status)
        return found

    def terminal_trials(self, *, run_ref: str = "") -> tuple[TrialRecord, ...]:
        """The trials that ran to an ending (task 6's input when it ranks them)."""
        return tuple(
            record for record in self.trials(run_ref=run_ref) if record.status in TERMINAL_TRIAL_STATUSES
        )

    def trial(self, trial_id: str) -> TrialRecord | None:
        """The latest version of one trial, or ``None`` when there is none."""
        self.refresh()
        for record in self._folded_trials():
            if record.id == trial_id:
                return record
        return None

    def observations(
        self, *, plan_fingerprint: str, space_fingerprint: str
    ) -> tuple[TrialRecord, ...]:
        """Task 7's seam: the trial records that match both fingerprints.

        What comes back is the **raw nested assignment**, not a feature vector: everything
        an encoding needs lives in the space (``space.spec(name)``), and copying it into the
        record layer would let the surrogate's input space drift away from the real sampling
        space with nobody noticing.  Nothing is filtered by status either; ``is_observation``
        is the caller's ready-made criterion.
        """
        self.refresh()
        return tuple(
            record
            for record in self._folded_trials()
            if record.plan_fingerprint == plan_fingerprint
            and record.space_fingerprint == space_fingerprint
        )

    def replay_check(self, space: SearchSpaceSchema, *, run_ref: str) -> ReplayReport:
        """Replay the recorded plan against today's space and compare it assignment by assignment.

        Only samplers registered in :data:`REPLAYABLE_SAMPLER_KINDS` are compared: an
        adaptive sampler picks each point from the observations before it, so replaying it
        against the current space necessarily differs from the record and comparing it would
        only produce false drift alarms.
        """
        run = self.run(run_ref)
        plan = SearchPlan.model_validate(run.header.plan)
        # plan_match is guaranteed by SearchRunRecord's validator.  It is reported anyway so
        # the conclusion stands on its own: a reader of this report alone need not go and
        # confirm that the record is self-consistent.
        plan_match = plan.fingerprint() == run.header.plan_fingerprint
        space_match = space.fingerprint() == run.header.space_fingerprint
        replayable = plan.kind in REPLAYABLE_SAMPLER_KINDS
        reasons: list[str] = []
        if not plan_match:
            reasons.append(
                f"the run cites plan {run.header.plan_fingerprint} but the plan it carries "
                f"fingerprints to {plan.fingerprint()}"
            )
        if not space_match:
            reasons.append(
                f"the run was drawn from space {run.header.space_fingerprint}, not {space.fingerprint()}"
            )
        mismatched: tuple[int, ...] = ()
        if not replayable:
            reasons.append(
                f"a {plan.kind} sampler is not in REPLAYABLE_SAMPLER_KINDS, so the recorded "
                f"sequence was not compared"
            )
        elif space_match:
            expected = replay(space, plan)
            mismatched = tuple(
                record.index
                for position, record in enumerate(run.trials)
                if position >= len(expected)
                or assignment_key(expected[position]) != record.assignment_key
            )
            if mismatched:
                reasons.append(
                    f"{len(mismatched)} recorded trials do not match the sequence this space and "
                    f"plan draw today"
                )
        return ReplayReport(
            run_ref=run_ref,
            plan_match=plan_match,
            space_match=space_match,
            replayable=replayable,
            mismatched=mismatched,
            reason="; ".join(reasons),
        )

    def orphan_dirs(self, *, run_ref: str = "") -> tuple[Path, ...]:
        """Directories under a trial root that no record references.

        Two sources, and the report cannot tell them apart (no line in the record points at
        the directory any more):

        * a process that died in the narrow window between "directory created" and "record
          appended";
        * a trial that was rerun: the new attempt uses a new directory (design section 7.3,
          mechanism 5), which leaves the old one unreferenced.  That old directory holds a
          config and possibly a half-written training log, which is exactly what one wants
          to keep when rerunning.

        So this is a **diagnostic**, not a cleanup list.  Deleting them automatically would
        be deciding on someone's behalf that the log in them is worthless.  To tie an orphan
        back to a trial, read the config file inside it: the injected values are there.

        The answer is scoped to **this ledger's** records.  ``work_root`` is the caller's to
        choose and nothing stops two searches from writing their trials into one of them; when
        that happens, each search reports the other's live trial directories too, because no
        line of this ledger mentions them.  The remedy is a ``work_root`` per search, which is
        what the caller normally wants anyway -- the same directory holding two searches'
        directories also holds two searches' logs and checkpoints, with nothing in the names
        to tell them apart.
        """
        self.refresh()
        referenced: set[Path] = set()
        roots: set[Path] = set()
        for record in self._folded_trials():
            if run_ref and record.run_ref != run_ref:
                continue
            if record.run_dir.strip():
                workspace = Path(record.run_dir)
            elif record.config_path.strip():
                workspace = Path(record.config_path).parent
            else:
                continue
            referenced.add(workspace.resolve())
            roots.add(workspace.parent.resolve())
        found: list[Path] = []
        for root in sorted(roots):
            if not root.is_dir():
                continue
            for entry in sorted(root.iterdir()):
                if entry.is_dir() and entry.resolve() not in referenced:
                    found.append(entry)
        return tuple(found)


class SearchTracker:
    """The write facade: it turns every trial of one search into a judgeable event on the ledger.

    It does not run training (that hop is injected as ``runner``), does not decide when to
    stop, and does not rank.  What it guarantees is that every step that ran left a record,
    and that nothing which did not run is recorded as having run.

    One ``run()`` writes each trial three times (``pending`` -> ``running`` -> terminal),
    and the three states each say one thing when a crash is investigated: a ``pending`` with
    no ``running`` means the trial was materialised but training never started; a ``running``
    with no ending means training did start and the result is unknown.
    """

    def __init__(
        self,
        *,
        store: TrialLedgerStore,
        space: SearchSpaceSchema,
        plan: SearchPlan,
        source_config: Mapping[str, Any],
        source_config_path: str | os.PathLike[str],
        work_root: str | os.PathLike[str],
        runner: TrialRunner,
        run_ref: str | None = None,
        space_artifact: ArtifactRef | None = None,
        sampler: Sampler | None = None,
        max_seconds: float | None = None,
        max_in_flight: int = 1,
        actor: str = "search_tracker",
    ) -> None:
        if plan.space_fingerprint != space.fingerprint():
            raise SearchError(
                f"the plan was drawn from space {plan.space_fingerprint}, not {space.fingerprint()}"
            )
        if not str(work_root).strip():
            # Trial directories must have an owner.  A default would let a search scatter
            # training artifacts somewhere nobody expects, while the path in the record
            # looks perfectly normal.
            raise SearchError("work_root is required: this tracker has no default place for trial files")
        if max_in_flight < 1:
            raise SearchError(f"max_in_flight must be at least 1, got {max_in_flight}")
        if max_seconds is not None and max_seconds <= 0:
            raise SearchError(f"max_seconds must be positive when given, got {max_seconds}")
        parsed = load_config_file(source_config_path)
        try:
            on_disk = content_hash(parsed)
            in_memory = content_hash(dict(source_config))
        except TypeError as error:
            raise SearchError(f"a source config holds a value with no JSON form: {error}") from error
        if on_disk != in_memory:
            # The injection reads from disk, the record is computed from memory.  When the
            # two are not the same thing, the config the record describes and the one the
            # training run actually read are two different configs -- and they look alike.
            raise SearchError(
                f"the config parsed from {source_config_path} is not the one passed in: "
                f"the file hashes to {on_disk}, the mapping to {in_memory}"
            )
        self.store = store
        self.space = space
        self.plan = plan
        self.source_config_path = Path(source_config_path)
        self.work_root = Path(work_root)
        self.runner = runner
        self.space_artifact = space_artifact
        self.max_seconds = max_seconds
        self.max_in_flight = max_in_flight
        self.actor = actor
        self._sampler = sampler
        self._source_fingerprint = in_memory
        self.run_ref = run_ref.strip() if run_ref and run_ref.strip() else default_run_ref(plan, space)
        self._ledger = SearchLedger(store)
        self._written_trial_ids: set[str] = set()
        self._seeded = False
        #: Whether this tracker has established its run header (or matched an existing one).
        #: A trial event must belong to a header: written into a root without one, every
        #: later read of that root fails with ``unknown_layout``, and nothing here can remove
        #: the line.  Only :meth:`open_run` sets this.
        self._opened = False
        #: Whether this search has already been closed.  Kept locally, and re-checked against
        #: the ledger's last event before a write rather than asked of the ledger on every
        #: write: a full re-read per trial event would make one search cost a square of its
        #: ledger length.  The tail read is O(1), so the re-check is affordable; the flag is
        #: what makes the common case free.
        self._closed = False

    def open_run(self) -> SearchRunRecord:
        """Write (or reuse) this search's run header.  Idempotent.

        When a record already exists it is compared on ``run_ref`` + plan fingerprint +
        space fingerprint, and any difference raises.  Writing two searches into one root
        would mean "how many trials did this search run" never has a single answer.

        Reconnecting to a closed search is not an error here -- reading it back is the
        ordinary path, and ``run()`` legitimately ends by reading -- but it does set
        :attr:`_closed` from the header, so this object knows the search is finished before
        it tries to write.  That flag is *a* source for the verdict, not the only one:
        :meth:`_require_open` reads the ledger's own last event, so deleting the assignment
        here changes nothing about what a write is allowed to do (mutation-verified: the
        fresh-tracker case still refuses).  The case that tells the two apart is a tracker
        that opened while the search was still open, which only the ledger can answer.
        """
        existing = self._ledger.runs()
        if existing:
            header = existing[0]
            if header.run_ref != self.run_ref:
                raise SearchIntegrityError(
                    f"this ledger already holds the search {header.run_ref!r}; open a different "
                    f"root for {self.run_ref!r}"
                )
            if header.plan_fingerprint != self.plan.fingerprint():
                raise SearchIntegrityError(
                    f"this ledger holds plan {header.plan_fingerprint}, not {self.plan.fingerprint()}"
                )
            if header.space_fingerprint != self.space.fingerprint():
                raise SearchIntegrityError(
                    f"this ledger holds space {header.space_fingerprint}, not {self.space.fingerprint()}"
                )
            self._closed = header.status == "closed"
            self._opened = True
            return header
        record = SearchRunRecord(
            id=f"run:{self.run_ref}",
            run_ref=self.run_ref,
            plan=self.plan.model_dump(mode="json"),
            plan_fingerprint=self.plan.fingerprint(),
            space_fingerprint=self.space.fingerprint(),
            sampling_order=self.plan.sampling_order,
            space_artifact=self.space_artifact,
            source_config=str(self.source_config_path),
            source_config_fingerprint=self._source_fingerprint,
            budget=self.plan.budget,
            max_in_flight=self.max_in_flight,
            status="open",
        )
        self._append("search_run", record)
        self._opened = True
        return record

    def begin_trial(
        self,
        index: int,
        assignment: Mapping[str, Any],
        *,
        attempt: int = 1,
        history_len: int = 0,
    ) -> TrialRecord:
        """Materialise one trial: ask the runner whether it can run, then make the directory, inject, and write the ``pending`` record.

        ``preflight`` comes first, so when it fails none of the rest happens -- no
        directory, no record, no GPU.  An adapter puts checks like "the candidate artifact
        must be a file" there rather than letting them become an asynchronous failure the
        caller cannot see inside a scheduling thread.

        A closed search is refused here, at the top, before any of that: see
        :meth:`_require_open`.
        """
        self._require_open()
        self.space.validate_assignment(dict(assignment), require_complete=True)
        key = assignment_key(assignment)
        # The directory name carries a random token: rerunning the same index after a crash
        # uses a new directory, so it can never collide with the orphan the last run left
        # (``exist_ok=False`` makes a collision a hard error).
        token = uuid.uuid4().hex[:8]
        workspace = (
            self.work_root / TRIAL_WORKSPACE_DIRNAME / f"trial-{index:04d}-a{attempt}-{token}"
        )
        config_path = workspace / TRIAL_CONFIG_FILENAME
        trial_id = self._trial_id(index)
        request = TrialRequest(
            trial_id=trial_id,
            index=index,
            attempt=attempt,
            plan_fingerprint=self.plan.fingerprint(),
            space_fingerprint=self.space.fingerprint(),
            assignment_key=key,
            assignment=dict(assignment),
            workspace=workspace,
            config_path=config_path,
            max_seconds=self.max_seconds,
        )
        self.runner.preflight(request)
        workspace.mkdir(parents=True, exist_ok=False)
        try:
            report = inject_into_file(self.source_config_path, config_path, self.space, assignment)
        except BaseException as error:
            self._append_unmaterialised(request, error)
            raise
        record = TrialRecord(
            id=trial_id,
            run_ref=self.run_ref,
            index=index,
            attempt=attempt,
            history_len=history_len,
            assignment=dict(assignment),
            assignment_key=key,
            plan=self.plan.model_dump(mode="json"),
            plan_fingerprint=self.plan.fingerprint(),
            space_fingerprint=self.space.fingerprint(),
            status="pending",
            run_dir=str(workspace),
            config_path=str(config_path),
            # The injection report is stored as opaque JSON (law R4): the computed field on
            # the nested InjectedValue makes the round trip back through model_validate fail,
            # so its integrity rests on injection_fingerprint rather than on "it is still
            # legal when read back".
            injection=report.model_dump(mode="json"),
            injection_fingerprint=report.fingerprint(),
            source_fingerprint=report.source_fingerprint,
            injected_fingerprint=report.injected_fingerprint,
            started_at=time.time(),
        )
        self._append_trial(record)
        return record

    def observe(self, trial_id: str, outcome: TrialOutcome) -> TrialRecord:
        """Append an intermediate observation to a trial that is still running (task 5's curve slot).

        Only a running trial has a curve to extend.  A trial that already reached an ending
        has its result decided, and appending observations would only make the record's
        ``objective`` and its curve contradict each other, when both are "the record".
        """
        previous = self._require_trial(trial_id)
        if previous.status != "running":
            raise SearchError(
                f"trial {trial_id} is {previous.status!r}, so it has no running curve to extend"
            )
        if not outcome.observations:
            raise SearchError(f"observe() was given no observations for {trial_id}")
        updated = _supersede(previous, observations=previous.observations + outcome.observations)
        self._append_trial(updated)
        return updated

    def finish(
        self, trial_id: str, status: TrialStatus, *, stop_reason: str = "", error: str = ""
    ) -> TrialRecord:
        """Let an outside decision end a still-running trial (task 5's pruning)."""
        previous = self._require_trial(trial_id)
        if status not in TERMINAL_TRIAL_STATUSES:
            raise SearchError(f"{status!r} is not a terminal trial status")
        if previous.status in TERMINAL_TRIAL_STATUSES:
            # If the race between a pruner and a training run completing let the latter
            # overwrite the former, the ledger would hold "a trial pruned after ten steps
            # that finished successfully" with nothing to show the contradiction.
            raise SearchError(f"trial {trial_id} already finished as {previous.status!r}")
        updated = _supersede(
            previous, status=status, stop_reason=stop_reason, error=error, finished_at=time.time()
        )
        self._append_trial(updated)
        return updated

    def run(self, *, retry_interrupted: bool = False, retry: tuple[int, ...] = ()) -> SearchRun:
        """Run every trial the plan proposes, close the search when it can be closed, and read it back from disk.

        ``retry_interrupted`` defaults to ``False``: a trial the crash interrupted is
        reported as unfinished by default rather than automatically run again.  Rerunning on
        its own would spend another GPU without anyone knowing, and a repeated spend is
        irreversible; reporting it only adds one piece of information someone has to deal
        with.

        A search that has already closed cannot be resumed in place: see
        :meth:`_require_open`.  ``begin_trial`` refuses, and it refuses *before* the run
        directory is made and before the runner is entered, so a closed search costs nothing
        when someone asks it to run a trial again.
        """
        if self.plan.kind in NON_REPLAYABLE_SAMPLER_KINDS and self.max_in_flight > 1:
            # An adaptive sampler's next point is a function of the results already recorded.
            # This loop runs trials one at a time whatever ``max_in_flight`` says, so the
            # sequence in the ledger would still be adaptive -- but the header would declare
            # a batch size that the adaptive property cannot survive once anyone acts on it
            # (within a batch, every point is drawn before any of them has a result).  A
            # reader cannot tell the two apart, so the combination is refused rather than
            # recorded with a caveat.  Refused before ``open_run``, so nothing is written.
            raise SearchError(
                f"this plan is adaptive ({self.plan.kind}) and the tracker was built with "
                f"max_in_flight={self.max_in_flight}: an adaptive sampler draws its next point "
                f"from the trials already finished, which a batch destroys -- every point in a "
                f"batch is drawn before any of them has a result. Run it with max_in_flight=1, "
                f"or use a plan whose sequence does not depend on the observations"
            )
        self.open_run()
        if self._sampler is not None and self._sampler.stats.proposals:
            # A sampler handed in by the caller is stateful: ``proposals()`` advances it and
            # increments ``stats.proposals`` as it yields.  Reusing one across two calls
            # would draw the *next* stretch of the stream and bind it to indices 1..N, so the
            # record would claim index 1 held an assignment the plan only ever drew at index
            # 4 -- and no read-side check can see that, because every assignment in the
            # record is individually legal.  Rebuilding is right for the sampler this class
            # owns (it is derived from the plan alone), and wrong for one it was given.
            raise SearchError(
                f"the injected sampler has already proposed "
                f"{self._sampler.stats.proposals} assignment(s), so its stream no longer "
                f"starts at index 1; the next proposals it yields are not the plan's "
                f"assignments 1..N and recording them under those indices would be a lie no "
                f"reader can detect. Pass a fresh sampler, or leave sampler=None so the "
                f"tracker builds one from the plan."
            )
        sampler = self._sampler if self._sampler is not None else build_sampler(self.space, self.plan)
        for index, assignment in enumerate(sampler.proposals(), start=1):
            previous = self._ledger.trial(self._trial_id(index))
            attempt = self._next_attempt(
                previous, index, retry=retry, retry_interrupted=retry_interrupted
            )
            if attempt is None:
                continue
            record = self.begin_trial(
                index,
                assignment,
                attempt=attempt,
                history_len=self._history_len(),
            )
            request = self._request_for(record)
            self._append_trial(_supersede(record, status="running"))
            try:
                outcome = self.runner.run(request)
            except BaseException as error:
                # Record first, then raise.  Raising alone means that once the process
                # exits nobody knows why this trial is missing; recording alone means the
                # caller takes a trial that never ran for a finished one.
                #
                # The version to build on is read back rather than taken from ``record``:
                # while the runner was working, ``observe`` may have appended a stretch of
                # the curve, and writing on top of the pending snapshot would drop it.
                #
                # Unless somebody outside already ended the trial, which is exactly what a
                # pruner does: it decides the trial is done, and the runner it cancels often
                # dies on the way out rather than returning.  Writing ``failed`` over that
                # would leave a record that says "pruned at step 10" in its stop_reason and
                # "failed, the backend died" in its status, with nothing on disk showing the
                # two decisions disagreed -- and it would change the answer the search
                # reports, since task 6 counts prunes and failures differently.  The verdict
                # that already stands is about the trial; the exception is about this call,
                # and it is the caller who needs to hear it, so it is raised either way.
                current = self._require_trial(record.id)
                if current.status not in TERMINAL_TRIAL_STATUSES:
                    self._append_trial(
                        _supersede(
                            current,
                            status="failed",
                            error=f"{type(error).__name__}: {error}",
                            finished_at=time.time(),
                        )
                    )
                raise
            current = self._require_trial(record.id)
            if current.status in TERMINAL_TRIAL_STATUSES:
                # Somebody outside already ended this trial while training was still running:
                # task 5's pruner, or an operator calling finish().  That decision stands.
                # "Pruned at step 10" and "it went on to finish successfully" cannot both be
                # true of one trial, and letting this write overwrite the earlier one would
                # leave the ledger holding the second with nothing to show the contradiction.
                # The one thing the later outcome may still add is the curve it collected on
                # its way out, which is a fact about the trial rather than a verdict on it.
                if outcome.observations:
                    self._append_trial(
                        _supersede(
                            current, observations=current.observations + outcome.observations
                        )
                    )
                continue
            if outcome.status not in TERMINAL_TRIAL_STATUSES:
                failure = SearchError(
                    f"the runner returned {outcome.status!r} for {record.id}, which is not a "
                    f"terminal trial status"
                )
                self._append_trial(
                    _supersede(current, status="failed", error=str(failure), finished_at=time.time())
                )
                raise failure
            self._append_trial(
                _supersede(
                    current,
                    status=outcome.status,
                    objective=outcome.objective,
                    metrics=dict(outcome.metrics),
                    observations=current.observations + outcome.observations,
                    stop_reason=outcome.message if outcome.status in {"pruned", "stopped"} else "",
                    error=outcome.message if outcome.status == "failed" else "",
                    backend_ref=outcome.backend_ref,
                    log_path=outcome.log_path,
                    finished_at=time.time(),
                )
            )
        recorded = self._ledger.trials(run_ref=self.run_ref)
        if any(record.status not in TERMINAL_TRIAL_STATUSES for record in recorded):
            # Not closed, and no exception either: this search genuinely did not finish, so
            # ``complete`` being False is the answer itself and interrupted_indices() names
            # which trials it was.  Closing would produce a ledger that "looks finished",
            # which is the one thing this module must never produce.
            return self.read()
        self.close_run(sampler.stats)
        return self.read()

    def close_run(self, stats: SamplerStats | Mapping[str, Any]) -> SearchRunRecord:
        """Close this search.  Idempotent: already closed returns that version without appending an event.

        "How many trials did the sampler propose" must equal "how many landed on disk", and
        a mismatch raises.  That is the write-side anchor for "no silent truncation": a
        search that recorded 30 trials while claiming 50 reads exactly like one that really
        ran 50.  A trial still ``running`` or ``pending`` makes it refuse as well: equal
        counts with one trial that has no result are the same "looks finished".

        Every check the read side will run is run here first -- see
        :meth:`_require_sealable` -- because a close is the one write that cannot be taken
        back: it refuses every later trial, so a root sealed with a fatally-shaped trial set
        is sealed for good.
        """
        header = self.open_run()
        if header.status == "closed":
            return header
        counted = self._as_stats(stats)
        trials = self._ledger.trials(run_ref=self.run_ref)
        self._require_sealable(trials, counted)
        closed = _supersede(header, status="closed", stats=counted, trial_count=len(trials))
        # The trials above were read before the write lock was taken, so they are checked
        # again from inside it.  A trial that lands in between is the case that matters: the
        # file would hold one more trial than this record counts, and every later read of the
        # root then raises count_mismatch -- permanently, since the search is now closed and
        # the count cannot be rewritten.
        self._append(
            "search_run",
            closed,
            event_type="supersede",
            precondition=lambda: self._require_sealable(
                self._ledger.trials(run_ref=self.run_ref), counted
            ),
        )
        self._closed = True
        return closed

    def _require_sealable(
        self, trials: tuple[TrialRecord, ...], counted: Mapping[str, Any]
    ) -> None:
        """Everything a *readable* close needs, checked against the trials as they are at the moment of the write.

        Called twice by :meth:`close_run`: once before the closing record is built, so a
        caller gets the refusal without taking the write lock, and once from inside the lock
        as the append's precondition.  The second call is the one that decides, and it is
        worth its duplicate read: these are the read side's own checks (`_trials_of` on the
        indices and the assignment keys, `_check_counts` on the two counts, the layout on
        what may follow a close), and a close that fails one of them does not merely write a
        wrong number -- it makes the root unreadable for good, because the close is precisely
        what stops the missing index from ever being materialised.  A refusal here costs the
        caller one call; a sealed bad root costs the search.
        """
        indices = tuple(record.index for record in trials)
        if indices != tuple(range(1, len(trials) + 1)):
            raise SearchError(
                f"the recorded trial indices are {indices}, not 1..{len(trials)}: a trial was "
                f"not materialised, and closing now would seal a run whose missing index can "
                f"never be written afterwards (a closed search refuses every later trial). "
                f"Materialise it with begin_trial first."
            )
        keys = [record.assignment_key for record in trials]
        if len(set(keys)) != len(keys):
            raise SearchError(
                "two trials of this run carry the same assignment_key, which no reader of this "
                "ledger accepts; closing now would make that permanent. Two indices drawn from "
                "one point are one trial recorded twice."
            )
        if counted["proposals"] != len(trials):
            raise SearchError(
                f"the sampler proposed {counted['proposals']} trials but {len(trials)} were "
                f"recorded; closing now would report a search that skipped "
                f"{counted['proposals'] - len(trials)} of them"
            )
        unfinished = ", ".join(
            f"{record.index} ({record.status})"
            for record in trials
            if record.status not in TERMINAL_TRIAL_STATUSES
        )
        if unfinished:
            raise SearchError(
                f"trials {unfinished} have not finished; closing now would report a complete search "
                f"over trials whose result is unknown. Give them an outcome with finish(), or rerun "
                f"them with retry=() or retry_interrupted=True"
            )

    def read(self) -> SearchRun:
        """Read this search back from disk.

        ``run()`` ends with it: parsing the file back runs the integrity checks as a side
        effect of every run, so the moment the write format and the read format drift apart
        is the first run, not the moment someone analyses the results.
        """
        return self._ledger.run(self.run_ref)

    def _trial_id(self, index: int) -> str:
        """A deterministic record id: rerunning the same index after a crash supersedes it rather than writing another one."""
        return f"trial:{self.run_ref}:{index:04d}"

    def _next_attempt(
        self,
        previous: TrialRecord | None,
        index: int,
        *,
        retry: tuple[int, ...],
        retry_interrupted: bool,
    ) -> int | None:
        """Which attempt this trial is on; ``None`` means skip it this time.

        When a record already exists, the four statuses mean four different things: terminal
        ones are done; ``pending`` was materialised but training never started (rerunning
        repeats nothing); ``running`` may be sitting in a crash; and anything listed
        explicitly in ``retry`` is rerun whatever it says.
        """
        if previous is None:
            return 1
        if index in retry:
            return previous.attempt + 1
        if previous.status == "pending":
            return previous.attempt + 1
        if previous.status == "running":
            return previous.attempt + 1 if retry_interrupted else None
        return None

    def _history_len(self) -> int:
        """How many usable observations existed when the next assignment was chosen.

        Only ``is_observation`` trials count: a trial with no objective is not an (x, y)
        sample, and counting it would make "the i-th trial was chosen after seeing k
        results" false.
        """
        return sum(
            1
            for record in self._ledger.observations(
                plan_fingerprint=self.plan.fingerprint(),
                space_fingerprint=self.space.fingerprint(),
            )
            if is_observation(record)
        )

    def _request_for(self, record: TrialRecord) -> TrialRequest:
        """Rebuild this trial's request from the record already on disk.

        Rebuilt rather than kept alongside, so that "what the runner received" and "what the
        ledger recorded" are the same thing by construction: every field of the request
        comes from that record.
        """
        return TrialRequest(
            trial_id=record.id,
            index=record.index,
            attempt=record.attempt,
            plan_fingerprint=record.plan_fingerprint,
            space_fingerprint=record.space_fingerprint,
            assignment_key=record.assignment_key,
            assignment=dict(record.assignment),
            workspace=Path(record.run_dir),
            config_path=Path(record.config_path),
            max_seconds=self.max_seconds,
        )

    def _append_unmaterialised(self, request: TrialRequest, error: BaseException) -> None:
        """An injection failure leaves a record too, and only then is the exception raised.

        This is the one case where ``failed`` is allowed to carry no
        ``injection_fingerprint`` (tightening 1 at the top of the module): the config was
        never written, so any fingerprint filled in here could only be invented.
        """
        now = time.time()
        self._append_trial(
            TrialRecord(
                id=request.trial_id,
                run_ref=self.run_ref,
                index=request.index,
                attempt=request.attempt,
                assignment=dict(request.assignment),
                assignment_key=request.assignment_key,
                plan=self.plan.model_dump(mode="json"),
                plan_fingerprint=self.plan.fingerprint(),
                space_fingerprint=self.space.fingerprint(),
                status="failed",
                run_dir=str(request.workspace),
                config_path=str(request.config_path),
                error=f"the config could not be injected: {type(error).__name__}: {error}",
                started_at=now,
                finished_at=now,
            )
        )

    def _as_stats(self, stats: SamplerStats | Mapping[str, Any]) -> dict[str, Any]:
        counted = asdict(stats) if isinstance(stats, SamplerStats) else dict(stats)
        if "proposals" not in counted:
            raise SearchError(
                "the sampler statistics must carry 'proposals': without it a closed search cannot "
                "show that it recorded every trial it proposed"
            )
        return counted

    def _maybe_seed_written(self) -> None:
        """Before the first trial write, collect the trial ids already on disk.

        With them, "is this the first write or a supersede" need not re-read the ledger on
        every write -- which would make one search cost a square of its file length.
        """
        if self._seeded:
            return
        self._written_trial_ids = {record.id for record in self._ledger.trials()}
        self._seeded = True

    def _append(
        self,
        record_type: str,
        record: BaseModel,
        *,
        event_type: Literal["append", "supersede"] = "append",
        precondition: Callable[[], None] | None = None,
    ) -> None:
        self.store.append(
            record_type,
            record,
            actor=self.actor,
            event_type=event_type,
            precondition=precondition,
        )

    def _require_open(self) -> None:
        """Refuse to write a trial into a root that could not read it back.

        The ledger's line order is ``search_run`` -> trials -> the closing ``search_run``, and
        the read side treats a violation as fatal (``unknown_layout``, ``trial_after_close``).
        That check is right -- a search that grew a result after it finished has no single
        answer to "what did it find" -- but it means one misplaced append does not lose one
        record, it makes the whole root unreadable for good, and nothing can repair it: the
        close cannot be re-opened, and the extra trial cannot be removed.

        So the write side refuses instead, on all three ways in:

        1. **This tracker has not opened its run.**  A trial event with no header to explain
           it fails ``unknown_layout`` for every later reader, including ``open_run`` itself,
           so it cannot even be repaired by opening the run afterwards.  Refusing here is
           what stops a caller who materialises trials by hand (``begin_trial`` without
           ``run``) from writing one into an empty root or into another search's root.
        2. **The search on this root is closed.**  The trap this closes is a real one:
           ``close_run`` tells a caller with unfinished trials to rerun them with ``retry=()``
           or ``retry_interrupted=True``, and doing that *after* the search closed would
           otherwise run the trial, spend the GPU, and only then brick the ledger.
        3. **A different search owns this root**, which ``open_run`` already refuses; calling
           it is what sets (1), so a tracker whose ``run_ref`` does not match fails there.

        How the second one is decided matters as much as that it is.  The close is read from
        the ledger's last event rather than trusted from the flag, because the flag is set by
        :meth:`open_run` and a *fresh* tracker -- the pattern the resume path itself uses --
        would otherwise start life believing an already-closed search is open.  And it is
        decided by the event's **type**, matching the read side's own predicate
        (``_validate_layout``: a ``search_run`` event that is not an ``append`` *is* the
        close), not by the payload's ``status`` field: a run header superseded to some other
        status -- ``abandoned`` is a legal value, and an operator or a later task may well
        write one -- reads as "open" under a status comparison while the read side has already
        ruled that nothing may follow it.  One predicate, in two places, is one predicate
        fewer to keep in step.

        There is no fourth case in which this check is merely advisory.  It runs inside the
        store's write lock, because :meth:`_append_trial` hands it to ``append`` as the
        append's precondition: a check made next to the write instead of with it can be
        overtaken -- by another process, and just as well by a second tracker in this one --
        and the event that lands anyway costs the root rather than the record.  What the
        design's §7.4 boundary still leaves out of scope is a *concurrent writer* racing for
        the same trial: the lock makes each write atomic, not the sequence of them.
        """
        if not self._opened:
            raise SearchError(
                f"this tracker has not opened the search {self.run_ref!r}, so a trial written "
                f"now would have no run header to belong to: call open_run() first (run() "
                f"does).  A trial event in a root with no search_run record fails "
                f"unknown_layout for every later reader, and cannot be removed afterwards."
            )
        last = self.store.last_event()
        if last is None:
            # ``_opened`` is only ever set by open_run, which either writes the header or
            # finds one, so an empty ledger here means the header was removed or the file was
            # truncated while this search was in progress.  Writing the trial anyway produces
            # exactly the headerless event this method exists to prevent, and the flag cannot
            # see the divergence -- only the file can.
            raise SearchIntegrityError(
                f"the ledger {self.store.events_path} holds no events at all, though this "
                f"tracker opened the search {self.run_ref!r}: the run header is gone, so a "
                f"trial written now would have no search_run record to belong to and would "
                f"fail unknown_layout for every later reader. Restore the file or start a "
                f"fresh root."
            )
        if last.record_type == "search_run":
            self._closed = last.event_type != "append"
        if self._closed:
            raise SearchError(
                f"the search {self.run_ref!r} on this ledger is closed, so no trial can be "
                f"written to it: a search_run event that is not the opening append ends the "
                f"search whatever status it carries, and a trial event after it makes every "
                f"later read of this ledger fail. A finished search is not resumable in place "
                f"-- start another search in a fresh ledger root."
            )

    def _append_trial(self, record: TrialRecord) -> None:
        """Write one trial event.

        The first write is an ``append`` and every later one is a ``supersede``, decided by
        whether this id has been written before rather than by the call site: a retry writes
        the same id a second time, and that one is a supersede in meaning even when what it
        writes is ``pending``.

        :meth:`_require_open` is handed to ``append`` as its precondition rather than merely
        called here.  Called here it would be a check performed before the write, and between
        the two a second writer can close the search -- in this process just as much as in
        another one -- which leaves this event landing after the closing record, where the read
        side refuses the whole root for good.  Inside the lock there is no between.
        """
        self._maybe_seed_written()
        event_type: Literal["append", "supersede"] = (
            "supersede" if record.id in self._written_trial_ids else "append"
        )
        self._append(
            "search_trial", record, event_type=event_type, precondition=self._require_open
        )
        self._written_trial_ids.add(record.id)

    def _require_trial(self, trial_id: str) -> TrialRecord:
        record = self._ledger.trial(trial_id)
        if record is None:
            raise SearchError(f"this ledger holds no trial named {trial_id!r}")
        return record


__all__ = [
    "CancellableTrialRunner",
    "REPLAYABLE_SAMPLER_KINDS",
    "ReplayReport",
    "SEARCH_SCHEMA_VERSION",
    "SearchError",
    "SearchIntegrityError",
    "SearchLedger",
    "SearchPreflightError",
    "SearchRun",
    "SearchRunRecord",
    "SearchTracker",
    "TERMINAL_TRIAL_STATUSES",
    "TRIAL_CONFIG_FILENAME",
    "TRIAL_WORKSPACE_DIRNAME",
    "TrialObservation",
    "TrialObservationSource",
    "TrialOutcome",
    "TrialRecord",
    "TrialRequest",
    "TrialRunner",
    "TrialStatus",
    "default_run_ref",
    "is_observation",
]
