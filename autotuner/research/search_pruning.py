"""Deciding that a running trial is hopeless, stopping it, and recording what the decision was based on.

A search spends most of its wall clock on candidates that were never going to win.  This module
is the decision that ends one early, so the budget buys more candidates instead of more steps of
a hopeless one.  Three things it deliberately is **not**:

* Not a scheduler and not a loop.  :meth:`PruningWatcher.review` is one reproducible call -- no
  sleep, no clock, no thread -- so a test can drive it with a scripted curve and assert "it fired
  on the third look" without waiting for anything.  The polling loop lives in
  :class:`PruningTrialRunner`, which is the only thing here that owns a thread.
* Not a change to task 4.  The verdict lands through the seams that already exist
  (``SearchTracker.observe`` / ``SearchTracker.finish``), and a decision that justified itself is
  written as its own record type through the ledger's generic append.  Nothing in
  ``hyperparameter_search.py`` or ``trial_ledger.py`` is edited to make room for this.
* Not a reimplementation of ``products/taili/ops/early_stop.py``.  That rule set's three field
  names have no producer anywhere in this repository (measured: the shipped telemetry carries
  neither ``health.episode_length_mean`` nor a ``reward.mean``), so inheriting its thresholds
  would be shipping dead code with a new name.  The metric names here are configurable dotted
  paths, and the shipped ones are checked against the real payload.

Two things about the measurement stream shape the whole design:

* The decision is a **pure function of the curve**.  ``PruningPolicy.decide`` reads nothing but
  the observations it is handed, which is what makes "the rule that fired" re-derivable later.
* Non-finite readings must be canonicalised **before** they are written.  The ledger persists with
  ``model_dump(mode="json")`` (``trial_ledger.py:565``), which turns ``float('nan')`` into
  ``null`` -- measured on pydantic 2.12.5: ``{'total': nan}`` becomes ``{'total': None}`` while
  the string ``"nan"`` survives.  A NaN that reached the ledger as ``None`` would not fire the
  rule on the way back, so the one rule that catches it would silently stop being re-derivable.
  :func:`canonical_metric` is that guard, and it is the reason ``metric_value`` accepts the
  strings back.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, model_validator

from .hyperparameter_search import (
    TERMINAL_TRIAL_STATUSES,
    SearchLedger,
    SearchTracker,
    TrialObservation,
    TrialOutcome,
    TrialRecord,
)
from .research_ledger import content_hash
from .trial_ledger import TrialLedgerStore

PRUNE_SCHEMA_VERSION = "rl-agent.search-pruning/v1"
STOP_REASON_PREFIX = "prune:"

PruneReason = Literal["nan_metric", "metric_floor", "metric_stall"]

#: The rule kinds, in the order the platform's own failure modes matter: an unstable run first,
#: then one that is measurably worse than plausible, then one that stopped improving.
PRUNE_REASONS: tuple[PruneReason, ...] = ("nan_metric", "metric_floor", "metric_stall")

POLICY_RECORD_TYPE = "prune_policy"
DECISION_RECORD_TYPE = "prune_decision"

#: A non-finite reading is stored as this text so it survives the ledger's JSON dump.  A plan's
#: verdict must never *contain* one of these -- see ``PruneVerdict``'s validator.
NON_FINITE_TEXT: dict[str, float] = {
    "nan": math.nan,
    "-nan": math.nan,
    "inf": math.inf,
    "+inf": math.inf,
    "infinity": math.inf,
    "+infinity": math.inf,
    "-inf": -math.inf,
    "-infinity": -math.inf,
}

_CANONICAL_TEXT: dict[str, str] = {"nan": "nan", "inf": "inf", "-inf": "-inf"}


# --- Reading a value out of a curve point ---------------------------------------------------


def metric_value(metrics: Mapping[str, Any], path: str) -> float | None:
    """Follow a dotted ``path`` through nested mappings and return a number, or ``None``.

    ``None`` means "there is no reading here": the key is absent, it holds a string that is not a
    non-finite spelling, it holds a bool, or it holds something that is not a number at all.  It
    does **not** mean zero, and it is not the same as "the reading is NaN" -- that case returns
    ``float('nan')``, which is exactly the distinction ``NaNRule`` depends on.

    The two dictionaries above are inverses: :func:`canonical_metric` turns a non-finite float into
    the text it is stored as, and this function turns that text back into the float, so a curve can
    make the round trip through the ledger and still be judged by the rule that produced it.
    """
    current: Any = metrics
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    if isinstance(current, bool):
        # A gate is not a metric.  ``True`` is an ``int`` in Python, and a rule reading it as 1.0
        # would look like a working rule while measuring something nobody meant to measure.
        return None
    if isinstance(current, (int, float)):
        return float(current)
    if isinstance(current, str):
        return NON_FINITE_TEXT.get(current.strip().lower())
    return None


def is_nonfinite(value: float | None) -> bool:
    """Is this reading a number that is not finite?  ``None`` -- no reading -- is **not** NaN."""
    return value is not None and not math.isfinite(value)


def canonical_metric(value: float | None) -> float | str | None:
    """The form a reading must be stored in: a finite float, or the text ``"nan"``/``"inf"``/``"-inf"``.

    Everything that writes a curve into the ledger goes through this, because the ledger's
    ``model_dump(mode="json")`` silently turns a non-finite float into ``null`` and the rule that
    would have fired on it never fires again.
    """
    if value is None:
        return None
    if math.isfinite(value):
        return float(value)
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


# --- The rules ------------------------------------------------------------------------------


class _RuleBase(BaseModel):
    """Shared shape of every rule: a metric to watch, and **exactly one** statement of when.

    Two gates would be two sources of truth for one number, and no gate at all leaves the question
    "from when?" unanswered in the record -- the reader would have to guess it from the threshold,
    which is how a rule ends up firing at a step its author never intended.  So a rule must name
    one.  ``at_step`` is the one to use; ``at_total_fraction`` exists for a run asked to stop "in
    the first half" without knowing how long the run is, and it never guesses a denominator: with
    no ``total_steps`` in the curve it does not fire at all.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    at_step: int | None = None
    at_total_fraction: float | None = None

    @model_validator(mode="after")
    def _validate_gate(self) -> "_RuleBase":
        chose = [self.at_step is not None, self.at_total_fraction is not None]
        if sum(chose) != 1:
            raise ValueError(
                "a rule names exactly one of at_step / at_total_fraction, so that its activation "
                "point has a single source of truth"
            )
        if self.at_step is not None and self.at_step < 0:
            raise ValueError(f"at_step cannot be negative, got {self.at_step}")
        if self.at_total_fraction is not None and not 0 < self.at_total_fraction <= 1:
            raise ValueError(
                f"at_total_fraction is a fraction of the run, so it must be in (0, 1], got "
                f"{self.at_total_fraction}"
            )
        return self

    def _gate_step(self, curve: Sequence[TrialObservation]) -> int | None:
        """The first step this rule may fire at, or ``None`` when that cannot be known yet."""
        if self.at_step is not None:
            return self.at_step
        total = _total_steps(curve)
        if total is None:
            return None
        return int(math.ceil(float(self.at_total_fraction) * total))

    def _evaluate(self, curve: Sequence[TrialObservation]) -> "_Match | None":
        raise NotImplementedError


class NaNRule(_RuleBase):
    """A non-finite reading of ``metric`` at or after the gate ends the trial.

    This is the only rule that looks at the whole curve rather than a trailing window: a NaN at
    one step is a fact that never becomes untrue, so a fixed-size cache would eventually forget
    the very reading that justified the decision.
    """

    kind: Literal["nan_metric"] = "nan_metric"
    metric: str

    def _evaluate(self, curve: Sequence[TrialObservation]) -> "_Match | None":
        gate = self._gate_step(curve)
        if gate is None:
            return None
        for observation in curve:
            if observation.step < gate:
                continue
            reading = metric_value(observation.metrics, self.metric)
            if is_nonfinite(reading):
                return _Match(
                    step=observation.step,
                    matched_steps=(observation.step,),
                    # The reading itself is deliberately NOT carried: a verdict's ``observed`` is
                    # a finite number or absent, so that every verdict is legal JSON.
                    observed=None,
                    detail=f"{self.metric} is not a finite number at step {observation.step}",
                )
        return None


class FloorRule(_RuleBase):
    """``patience`` consecutive readings beyond ``threshold`` end the trial.

    ``patience`` is what keeps one noisy point from deciding.  Readings that are missing or
    non-finite neither count nor break the streak -- they are not evidence about the threshold,
    and NaNRule owns them.
    """

    kind: Literal["metric_floor"] = "metric_floor"
    metric: str
    threshold: float
    mode: Literal["floor", "ceiling"] = "floor"
    patience: int = 1

    @model_validator(mode="after")
    def _validate_patience(self) -> "FloorRule":
        if self.patience < 1:
            raise ValueError(f"patience counts observations, so it starts at 1, got {self.patience}")
        if not math.isfinite(self.threshold):
            raise ValueError(f"threshold must be a finite number, got {self.threshold}")
        return self

    def _breaches(self, reading: float) -> bool:
        return reading < self.threshold if self.mode == "floor" else reading > self.threshold

    def _evaluate(self, curve: Sequence[TrialObservation]) -> "_Match | None":
        gate = self._gate_step(curve)
        if gate is None:
            return None
        streak: list[int] = []
        last_reading: float | None = None
        for observation in curve:
            if observation.step < gate:
                continue
            reading = metric_value(observation.metrics, self.metric)
            if reading is None or not math.isfinite(reading):
                continue
            if self._breaches(reading):
                streak.append(observation.step)
                last_reading = reading
                if len(streak) >= self.patience:
                    compared = "below" if self.mode == "floor" else "above"
                    return _Match(
                        step=observation.step,
                        matched_steps=tuple(streak[-self.patience :]),
                        observed=last_reading,
                        # ``:g`` rather than ``repr``: this line is read by a person, and
                        # "reaching 0.30000000000000004" buries the one number they came for.
                        detail=(
                            f"{self.metric} stayed {compared} {self.threshold:g} for "
                            f"{self.patience} observations, reaching {last_reading:g} at step "
                            f"{observation.step}"
                        ),
                        threshold=self.threshold,
                    )
            else:
                streak.clear()
                last_reading = None
        return None


class StallRule(_RuleBase):
    """A trailing window whose readings moved by at most ``min_delta`` ends the trial.

    This is the general form of "it has stopped improving".  Only observations with a usable
    reading count toward the window; a window that is not yet full does not fire, because a
    plateau cannot be measured from fewer points than the window asks for.
    """

    kind: Literal["metric_stall"] = "metric_stall"
    metric: str
    min_delta: float
    window: int

    @model_validator(mode="after")
    def _validate_window(self) -> "StallRule":
        if self.window < 2:
            raise ValueError(
                f"a stall is a span across at least two observations, got window={self.window}"
            )
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError(f"min_delta must be a finite non-negative number, got {self.min_delta}")
        return self

    def _evaluate(self, curve: Sequence[TrialObservation]) -> "_Match | None":
        gate = self._gate_step(curve)
        if gate is None:
            return None
        usable = [
            (observation.step, reading)
            for observation in curve
            if observation.step >= gate
            for reading in (metric_value(observation.metrics, self.metric),)
            if reading is not None and math.isfinite(reading)
        ]
        if len(usable) < self.window:
            return None
        trailing = usable[-self.window :]
        readings = [reading for _, reading in trailing]
        span = max(readings) - min(readings)
        if span > self.min_delta:
            return None
        last_step = trailing[-1][0]
        return _Match(
            step=last_step,
            matched_steps=tuple(step for step, _ in trailing),
            observed=trailing[-1][1],
            detail=(
                f"{self.metric} moved {span:g} over the last {self.window} observations, at most "
                f"the allowed {self.min_delta:g}, ending at step {last_step}"
            ),
            threshold=self.min_delta,
        )


PruningRule = NaNRule | FloorRule | StallRule

_RULE_FOR_KIND: dict[str, type[_RuleBase]] = {
    "nan_metric": NaNRule,
    "metric_floor": FloorRule,
    "metric_stall": StallRule,
}


class _Match(BaseModel):
    """What a rule found: where, on what evidence, in what words.  Internal to the module."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int
    matched_steps: tuple[int, ...]
    observed: float | None = None
    threshold: float | None = None
    detail: str


def _total_steps(curve: Sequence[TrialObservation]) -> float | None:
    """The run's total step count as the curve's latest point knows it, or ``None``.

    ``None`` is the honest answer for the shipped telemetry: the captured payload carries
    ``total_steps: null`` on every line when ``TAILI_TOTAL_STEPS`` is unset, so a fraction gate
    has no denominator to work from and stays silent rather than inventing one.
    """
    if not curve:
        return None
    value = metric_value(curve[-1].metrics, "total_steps")
    if value is None or not math.isfinite(value) or value <= 0:
        return None
    return value


# --- The verdict ----------------------------------------------------------------------------


def format_stop_reason(reason: PruneReason, step: int, detail: str) -> str:
    """The one-line form written into ``TrialRecord.stop_reason``, which task 6 reads back."""
    return f"{STOP_REASON_PREFIX}{reason}@{step} {detail}"


class PruneVerdict(BaseModel):
    """The decision itself, and everything it rests on.

    ``stop_reason`` is a stored field rather than a ``@computed_field`` (law R1): a computed field
    is written out by ``model_dump(mode="json")`` and then refused by ``extra="forbid"`` on the way
    back in, so a record carrying one cannot round-trip -- which is the ground the ledger stands on.
    The same pattern as ``SearchRun.complete`` (``hyperparameter_search.py:474``).  The price is the
    validator below, which keeps the stored text equal to the fields it summarises.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: PruneReason
    step: int
    metric: str
    observed: float | None = None
    threshold: float | None = None
    detail: str
    stop_reason: str
    matched_steps: tuple[int, ...] = ()

    @model_validator(mode="after")
    def _validate_verdict(self) -> "PruneVerdict":
        if self.step < 0:
            raise ValueError(f"a verdict step cannot be negative, got {self.step}")
        if not self.detail.strip():
            raise ValueError("a verdict must say why it was reached")
        if not self.metric.strip():
            raise ValueError("a verdict must name the metric it judged")
        for name in ("observed", "threshold"):
            value = getattr(self, name)
            # A non-finite reading never enters the verdict: it cannot survive
            # ``model_dump(mode="json")``, and a verdict is written to a ledger.
            if value is not None and not math.isfinite(value):
                raise ValueError(
                    f"{name} must be a finite number or absent, got {value!r}; a non-finite "
                    f"reading is recorded as the text 'nan'/'inf' in the decision record instead"
                )
        if self.step not in self.matched_steps and self.matched_steps:
            raise ValueError(
                f"the verdict cites step {self.step}, which is not among the observations it "
                f"matched {list(self.matched_steps)}"
            )
        expected = format_stop_reason(self.reason, self.step, self.detail)
        if self.stop_reason != expected:
            # Without this, a ledger can hold a verdict whose ``reason`` says one thing and whose
            # one readable line says another, and every reader has to notice on their own.
            raise ValueError(
                f"stop_reason {self.stop_reason!r} does not describe this verdict; expected "
                f"{expected!r}"
            )
        return self


class PruneStopReason(BaseModel):
    """A parsed ``stop_reason``: the machine-readable half of "why this trial stopped"."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: PruneReason
    step: int
    detail: str


_STOP_REASON_PATTERN = re.compile(
    rf"^{re.escape(STOP_REASON_PREFIX)}(?P<reason>[a-z_]+)@(?P<step>\d+) (?P<detail>.+)$"
)


def read_prune_stop_reason(text: str) -> PruneStopReason | None:
    """Read back what this module wrote, or ``None`` for any other text.

    ``None`` is the ordinary answer for the other three terminal statuses and for a stop_reason a
    human typed; a reader that raised on those would make the whole ledger unreadable because one
    trial failed for a different reason.
    """
    match = _STOP_REASON_PATTERN.match(text)
    if match is None:
        return None
    reason = match.group("reason")
    if reason not in PRUNE_REASONS:
        return None
    return PruneStopReason(
        reason=reason, step=int(match.group("step")), detail=match.group("detail")
    )


# --- The policy -----------------------------------------------------------------------------


def _validated_curve(curve: Sequence[TrialObservation]) -> tuple[TrialObservation, ...]:
    """The curve as a tuple, refusing one whose steps go backwards.

    A curve that steps back is the product of a replay or of two files concatenated.  Window
    statistics over it would produce a number that looks perfectly reasonable and means nothing,
    so it fails loudly here instead.
    """
    checked = tuple(curve)
    previous = -1
    for observation in checked:
        if observation.step < previous:
            raise ValueError(
                f"the curve steps back from {previous} to {observation.step}; a rule evaluated "
                f"over a replayed or concatenated curve would report a window that never happened"
            )
        previous = observation.step
    return checked


class PruningPolicy(BaseModel):
    """An **ordered** set of rules: the first one that matches decides.  Empty by default.

    Empty by default is the whole safety story.  A rule that fires on a threshold nobody chose
    kills a trial that cannot be un-killed -- ``finish`` refuses to overwrite a terminal record
    (``hyperparameter_search.py:1121-1125``), so undoing it would mean a new trial, not a repair.
    Pruning therefore has to be asked for.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_step: int = 0
    min_observations: int = 1
    rules: tuple[PruningRule, ...] = ()

    @model_validator(mode="after")
    def _validate_policy(self) -> "PruningPolicy":
        if self.min_step < 0:
            raise ValueError(f"min_step cannot be negative, got {self.min_step}")
        if self.min_observations < 1:
            raise ValueError(
                f"min_observations is a count of observations, so it starts at 1, got "
                f"{self.min_observations}"
            )
        return self

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PruningPolicy":
        """Build a policy from plain data -- a YAML document, or a record read back from the ledger.

        The rule kind is looked up in a table rather than handed to pydantic's discriminator, so
        an unknown kind produces a message naming the kinds that exist.  A discriminator's own
        error lists every branch it tried, which buries the one fact the caller needs.
        """
        payload = dict(data)
        raw_rules = payload.get("rules", ())
        if not isinstance(raw_rules, (list, tuple)):
            raise ValueError(f"rules must be a list, got {type(raw_rules).__name__}")
        rules: list[_RuleBase] = []
        for position, raw in enumerate(raw_rules):
            if not isinstance(raw, Mapping):
                raise ValueError(f"rule {position} must be a mapping, got {type(raw).__name__}")
            kind = str(raw.get("kind", ""))
            model = _RULE_FOR_KIND.get(kind)
            if model is None:
                raise ValueError(
                    f"rule {position} names kind {kind!r}; the kinds this module knows are "
                    f"{sorted(_RULE_FOR_KIND)}"
                )
            rules.append(model.model_validate(dict(raw)))
        return cls.model_validate({**payload, "rules": tuple(rules)})

    @property
    def fingerprint(self) -> str:
        """Content address of the policy: ``content_hash`` over ``model_dump(mode="json")``.

        Hashing the dump rather than the model is law R3 -- ``content_hash`` handed a model would
        apply its own ``exclude_none``, and a policy with an unset ``min_step`` would hash the same
        as one with a different one.
        """
        return content_hash(self.model_dump(mode="json"))

    def decide(self, curve: Sequence[TrialObservation]) -> PruneVerdict | None:
        """Judge one curve.  ``None`` means "no rule matched", not "the trial is fine".

        The whole curve is re-read on every call rather than a delta kept in memory: the rules are
        stateless, so the same curve produces the same verdict however many times it is asked,
        which is what makes a decision re-derivable from the ledger alone.
        """
        checked = _validated_curve(curve)
        if len(checked) < self.min_observations:
            return None
        eligible = tuple(
            observation
            for observation in checked
            # The gate is judged per rule as well; this one only says the policy as a whole is not
            # to speak before it has seen enough of the run.
            if observation.step >= self.min_step
        )
        if len(eligible) < self.min_observations:
            return None
        for rule in self.rules:
            match = rule._evaluate(checked)
            if match is not None:
                return PruneVerdict(
                    reason=rule.kind,  # type: ignore[arg-type]
                    step=match.step,
                    metric=rule.metric,
                    observed=match.observed,
                    threshold=match.threshold,
                    detail=match.detail,
                    stop_reason=format_stop_reason(rule.kind, match.step, match.detail),  # type: ignore[arg-type]
                    matched_steps=match.matched_steps,
                )
        return None


# --- The audit layer ------------------------------------------------------------------------


class PrunePolicyRecord(BaseModel):
    """The policy itself, stored content-addressed so the decision stays checkable without it.

    Storing only a fingerprint would leave a later reader knowing *which* policy decided and not
    what it said, which is the same dead end as storing only the verdict.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    policy_fingerprint: str
    policy: dict[str, Any]
    schema_version: str = PRUNE_SCHEMA_VERSION

    @model_validator(mode="after")
    def _validate_policy_record(self) -> "PrunePolicyRecord":
        actual = content_hash(self.policy)
        if actual != self.policy_fingerprint:
            raise ValueError(
                f"this policy record cites {self.policy_fingerprint} but the policy it carries "
                f"hashes to {actual}"
            )
        return self


def policy_record(policy: PruningPolicy) -> PrunePolicyRecord:
    """Wrap a policy for storage.  The id is derived, so two spellings cannot both claim it."""
    fingerprint = policy.fingerprint
    return PrunePolicyRecord(
        id=f"prune-policy:{fingerprint}",
        policy_fingerprint=fingerprint,
        policy=policy.model_dump(mode="json"),
    )


def store_policy_record(
    store: TrialLedgerStore,
    policy: PruningPolicy,
    *,
    precondition: Callable[[], None] | None,
    actor: str = "search_pruning",
) -> PrunePolicyRecord:
    """Store a policy, but only through a ``precondition`` that keeps the ledger readable.

    **Measured, not inferred** (this is the one thing in this module that a test found rather than
    a reading): writing any record into an empty search ledger makes that ledger permanently
    unreadable.  ``_validate_layout`` (``hyperparameter_search.py:585-594``) raises
    ``unknown_layout`` for *any* non-empty ledger whose first event is not the ``search_run``
    append, so a policy stored before the run header does not inconvenience one reader -- every
    later ``SearchLedger`` call on that root, including ``open_run`` itself, fails for good.

    ``precondition`` is the same hook task 4 uses (``hyperparameter_search.py:1577`` passes
    ``self._require_open``), and a write that fails it never reaches the file.  The parameter is
    deliberately **required to be passed explicitly rather than defaulted to None**: a default
    would make the dangerous form the terse one.

    Idempotent, because the id is content-addressed: storing the same policy twice appends nothing
    the second time, so a resumed search does not grow a duplicate.
    """
    record = policy_record(policy)
    if store.latest(POLICY_RECORD_TYPE, record.id) is not None:
        return record
    store.append(POLICY_RECORD_TYPE, record, actor=actor, precondition=precondition)
    return record


def policy_for(store: TrialLedgerStore, fingerprint: str) -> PruningPolicy | None:
    """Read a stored policy back.  ``None`` when nothing was ever stored under that fingerprint."""
    payload = store.latest(POLICY_RECORD_TYPE, f"prune-policy:{fingerprint}")
    if payload is None:
        return None
    return PruningPolicy.from_mapping(PrunePolicyRecord.model_validate(payload).policy)


class PruneDecisionRecord(BaseModel):
    """One decision's evidence: the rule, the step, the curve it saw, and the reading it acted on.

    Written **before** ``finish``.  If the order were reversed and the write failed, the ledger
    would hold a trial that stopped for no recorded reason -- the exact state this record exists to
    prevent.  The other order leaves at worst a decision for a trial that is still running, which
    a reader can see and a writer can retry.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    trial_id: str
    policy_fingerprint: str
    reason: PruneReason
    step: int
    through_step: int
    curve_len: int
    matched_steps: tuple[int, ...] = ()
    metric: str
    observed: float | str | None = None
    threshold: float | None = None
    stop_reason: str
    cancel_state: str = "not_attempted"
    schema_version: str = PRUNE_SCHEMA_VERSION

    @model_validator(mode="after")
    def _validate_decision(self) -> "PruneDecisionRecord":
        if self.step < 0 or self.through_step < 0 or self.curve_len < 0:
            raise ValueError("steps and lengths in a decision cannot be negative")
        if self.step > self.through_step:
            raise ValueError(
                f"the decision cites step {self.step}, later than the last observation it saw "
                f"({self.through_step})"
            )
        if self.observed is not None and isinstance(self.observed, float):
            if not math.isfinite(self.observed):
                raise ValueError(
                    f"observed must be a finite number or the text 'nan'/'inf', got {self.observed!r}"
                )
        if isinstance(self.observed, str) and self.observed not in _CANONICAL_TEXT.values():
            raise ValueError(
                f"observed text must be one of {sorted(_CANONICAL_TEXT.values())}, got "
                f"{self.observed!r}"
            )
        # Checked through the parser, not by rebuilding the text: the decision record carries the
        # same line the trial record does, and the point is that a reader who parses it gets this
        # record's reason and step back.
        parsed = read_prune_stop_reason(self.stop_reason)
        if parsed is None or parsed.reason != self.reason or parsed.step != self.step:
            raise ValueError(
                f"stop_reason {self.stop_reason!r} does not describe this decision "
                f"({self.reason} at step {self.step})"
            )
        return self


class PruneAuditError(Exception):
    """A decision that the ledger contradicts: the curve re-derives a different verdict.

    Distinct from "unverifiable".  This one means the record and the evidence disagree, which is a
    defect; a reading lost to the ledger's JSON normalisation is not a defect and is reported by
    :func:`verify_decision`'s return value instead.
    """


def _lost_to_normalisation(
    decision: PruneDecisionRecord, curve: Sequence[TrialObservation]
) -> bool:
    """Was the reading this decision acted on turned into ``None`` by the ledger's JSON dump?"""
    if not isinstance(decision.observed, str) or decision.observed not in _CANONICAL_TEXT.values():
        return False
    for observation in curve:
        if observation.step != decision.step:
            continue
        return metric_value(observation.metrics, decision.metric) is None
    return False


def verify_decision(
    decision: PruneDecisionRecord,
    curve: Sequence[TrialObservation],
    policy: PruningPolicy,
) -> Literal["verified", "unverifiable"]:
    """Re-derive a decision from the ledger's own contents and compare.

    Returns ``"verified"`` when the curve and the policy produce the recorded verdict again, and
    ``"unverifiable"`` in one specific case: the reading the rule fired on was a non-finite float
    written straight to the ledger without :func:`canonical_metric`, so it came back as ``None``
    and the rule cannot fire a second time.  That is a caller having bypassed the writer, not a
    false record, so it is reported rather than raised.

    Anything else that disagrees raises :class:`PruneAuditError` with the re-derived values.

    A curve whose steps go backwards raises ``ValueError`` out of the policy's own validator
    rather than being reported as a mismatch: such a curve cannot be re-derived against at all,
    and calling that "the record disagrees with itself" would name the wrong defect.
    """
    recomputed = policy.decide(curve)
    if recomputed is None:
        if _lost_to_normalisation(decision, curve):
            return "unverifiable"
        raise PruneAuditError(
            f"the decision says {decision.reason} fired at step {decision.step}, but re-running "
            f"{decision.policy_fingerprint[:12]} over the recorded curve matches no rule"
        )
    if (recomputed.reason, recomputed.step, recomputed.matched_steps) != (
        decision.reason,
        decision.step,
        decision.matched_steps,
    ):
        raise PruneAuditError(
            f"the decision says {decision.reason} at step {decision.step} on "
            f"{list(decision.matched_steps)}, but the recorded curve re-derives "
            f"{recomputed.reason} at step {recomputed.step} on {list(recomputed.matched_steps)}"
        )
    return "verified"


class PruneSummary(BaseModel):
    """How the trials of one search ended, with pruning kept apart from losing.

    Task 6 needs the difference: "pruned at step 10" and "ran the whole way and scored badly" are
    different facts about a configuration, and a ranking that averages them together is wrong in a
    way no one can see.  ``error`` exists because "never raises" must not mean "says nothing" --
    ``trial_ledger.summary()`` reports a broken chain rather than hiding it, and so does this.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    pruned_indices: tuple[int, ...] = ()
    stopped_indices: tuple[int, ...] = ()
    scored_indices: tuple[int, ...] = ()
    running_indices: tuple[int, ...] = ()
    verified: tuple[int, ...] = ()
    unverifiable: tuple[int, ...] = ()
    unexplained: tuple[int, ...] = ()
    error: str = ""


def prune_summary(
    ledger: SearchLedger, *, store: TrialLedgerStore, run_ref: str = ""
) -> PruneSummary:
    """Read a search's endings off the ledger and check every pruned one against its own evidence.

    Three outcomes for a pruned trial, and they are kept apart because they call for different
    action:

    * ``verified`` -- the recorded curve and the recorded policy re-derive the recorded verdict.
    * ``unverifiable`` -- the reading the rule fired on was a non-finite float that reached the
      ledger without :func:`canonical_metric`, so it came back as ``None`` and the rule cannot fire
      again.  A writer bypassed the guard; the record is not false.
    * ``unexplained`` -- no decision record, or the policy it cites was never stored.  The trial
      stopped early and nothing on disk says why.

    On a ledger that cannot be read at all, ``error`` is set and nothing raises: a search whose
    results are still legible one field at a time is worth more than one that can only be read
    while healthy.  A ``PruneAuditError`` is **not** caught -- a decision record contradicting its
    own curve is a defect, and a summary that quietly left it out would be the exact failure this
    layer exists to prevent.  Task 6 ranks on these numbers; it should not have to wonder whether
    the ones it was handed were checked.
    """
    try:
        trials: tuple[TrialRecord, ...] = (
            ledger.trials(run_ref=run_ref) if run_ref else ledger.trials()
        )
    except Exception as error:  # noqa: BLE001 - the point of this function is to stay readable
        return PruneSummary(error=f"{type(error).__name__}: {error}")

    pruned: list[int] = []
    stopped: list[int] = []
    scored: list[int] = []
    running: list[int] = []
    verified: list[int] = []
    unverifiable: list[int] = []
    unexplained: list[int] = []
    for record in trials:
        if record.status == "pruned":
            pruned.append(record.index)
        elif record.status == "stopped":
            stopped.append(record.index)
        elif record.status in TERMINAL_TRIAL_STATUSES:
            scored.append(record.index)
        else:
            running.append(record.index)

    for record in trials:
        if record.status != "pruned":
            continue
        payload = store.latest(DECISION_RECORD_TYPE, f"prune-decision:{record.id}")
        if payload is None:
            unexplained.append(record.index)
            continue
        decision = PruneDecisionRecord.model_validate(payload)
        policy = policy_for(store, decision.policy_fingerprint)
        if policy is None:
            unexplained.append(record.index)
            continue
        if verify_decision(decision, record.observations, policy) == "verified":
            verified.append(record.index)
        else:
            unverifiable.append(record.index)

    return PruneSummary(
        pruned_indices=tuple(sorted(pruned)),
        stopped_indices=tuple(sorted(stopped)),
        scored_indices=tuple(sorted(scored)),
        running_indices=tuple(sorted(running)),
        verified=tuple(sorted(verified)),
        unverifiable=tuple(sorted(unverifiable)),
        unexplained=tuple(sorted(unexplained)),
    )


# --- The seams ------------------------------------------------------------------------------


@runtime_checkable
class CurveSource(Protocol):
    """Where a running trial's curve comes from.  Injected, so the policy never touches a file."""

    def curve(self, trial_id: str) -> tuple[TrialObservation, ...]:
        """Every observation of this trial so far, in non-decreasing step order.

        An unknown trial or a file that cannot be read yields an empty tuple: "no evidence yet" is
        the ordinary state at the start of every trial, not an error.
        """
        ...


@runtime_checkable
class PruningSink(Protocol):
    """Where a verdict goes.  Split from the source so the two can be tested apart."""

    def record_curve(self, trial_id: str, observations: Sequence[TrialObservation]) -> None:
        """Add observations that have not been recorded yet.  May be called with none."""
        ...

    def apply_verdict(self, trial_id: str, verdict: PruneVerdict) -> bool:
        """Record a verdict.  ``True`` when it landed; ``False`` when the trial ended on its own
        first, which is the ordinary race between a pruner and a training run finishing."""
        ...


class PruningWatcher:
    """One reproducible look at one trial: pull the curve, record what is new, decide, apply.

    There is no loop and no clock here on purpose.  ``review()`` returns the same way whether it is
    called ten times a second or once at the end, so the interesting behaviour -- "does it fire,
    and on which observation" -- is testable from a scripted source with no time passing at all.
    """

    def __init__(self, *, policy: PruningPolicy, source: CurveSource, sink: PruningSink) -> None:
        self._policy = policy
        self._source = source
        self._sink = sink
        self._verdicts: dict[str, PruneVerdict] = {}
        self._declined: set[str] = set()
        self._recorded: dict[str, int] = {}

    @property
    def verdicts(self) -> tuple[PruneVerdict, ...]:
        """Every verdict this watcher has landed, in the order it landed them."""
        return tuple(self._verdicts.values())

    def review(self, trial_id: str) -> PruneVerdict | None:
        """Three states, and they are kept apart on purpose:

        * **landed** -- the same verdict is returned on every later call, without touching the
          source or the sink again.  Re-pulling would let a rule that has already ended a trial
          fire a second time on the same evidence and write a second decision.
        * **declined** -- the trial ended on its own before the verdict landed.  ``None`` from then
          on, again without touching the source: there is nothing left to decide.
        * **not yet** -- the trial is still running and no rule matched.  Each call re-pulls and
          re-judges, which is what makes a later call able to fire.
        """
        if trial_id in self._verdicts:
            return self._verdicts[trial_id]
        if trial_id in self._declined:
            return None
        curve = tuple(self._source.curve(trial_id))
        already = self._recorded.get(trial_id, 0)
        fresh = curve[already:]
        if fresh:
            # Only the new part is forwarded.  ``SearchTracker.observe`` appends blindly
            # (``hyperparameter_search.py:1110``), so sending the whole curve again on every look
            # would multiply the record's observations by the number of polls.
            self._sink.record_curve(trial_id, fresh)
            self._recorded[trial_id] = len(curve)
        verdict = self._policy.decide(curve)
        if verdict is None:
            return None
        if not self._sink.apply_verdict(trial_id, verdict):
            self._declined.add(trial_id)
            return None
        self._verdicts[trial_id] = verdict
        return verdict


# --- Reference adapters (validated against fakes only) --------------------------------------


def parse_telemetry_payload(payload: Mapping[str, Any]) -> TrialObservation | None:
    """One line of the telemetry JSONL to one observation.  Pure; ``None`` for anything else.

    The four nested sections are kept as nested mappings so a rule can name a metric the same way
    the payload spells it -- ``reward.total``, ``health.terminal_rate``.  ``total_steps`` is read
    but never invented: the shipped payload carries ``null`` there when the run was not told its
    length, and a fabricated denominator would make a fraction gate fire at a step nobody chose.
    """
    if payload.get("type") != "train_tick":
        return None
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, (int, float)) or step < 0:
        return None
    metrics: dict[str, Any] = {"total_steps": payload.get("total_steps")}
    for section in ("reward", "curriculum", "health", "command", "counters"):
        value = payload.get(section)
        if isinstance(value, Mapping):
            metrics[section] = {
                key: canonical_metric(reading)
                for key, reading in value.items()
                if isinstance(reading, (int, float)) and not isinstance(reading, bool)
            }
    return TrialObservation(step=int(step), metrics=metrics)


_STAT_LINE = re.compile(r"^\[TPSTAT\]\s+(?P<body>.*)$")
_REW_LINE = re.compile(r"^\[TPREW\]\s+(?P<body>.*)$")


def _kv_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in body.split():
        if "=" not in token:
            continue
        key, _, value = token.partition("=")
        fields[key] = value
    return fields


def parse_telemetry_lines(text: str) -> tuple[TrialObservation, ...]:
    """Read the relayed ``[TPSTAT]``/``[TPREW]`` lines into observations.  Pure.

    This is the fallback for a workspace where only the process's own stdout was captured -- which
    is the common case, because ``CommandExperimentBackend`` redirects the child's stdout into
    ``training.log`` (``research_supervisor.py:214-217``) and that is where the ``[TPREW]`` lines
    physically land.  A ``[TPREW]`` line without a ``step=`` is dropped rather than attributed to
    the previous step: the console parser guesses there (``telemetry.py:199`` and its two
    siblings), and a guess is what makes a curve describe a step that never produced it.
    """
    by_step: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    for raw in text.splitlines():
        stat = _STAT_LINE.match(raw)
        rew = _REW_LINE.match(raw)
        if stat is None and rew is None:
            continue
        fields = _kv_fields((stat or rew).group("body"))
        raw_step = fields.get("step")
        if raw_step is None:
            continue
        try:
            step = int(raw_step)
        except ValueError:
            continue
        if step not in by_step:
            by_step[step] = {"total_steps": None, "reward": {}}
            order.append(step)
        bucket = by_step[step]
        if stat is not None:
            total = fields.get("total")
            if total is not None:
                try:
                    bucket["total_steps"] = int(float(total))
                except ValueError:
                    pass
        else:
            total = fields.get("total")
            if total is not None:
                try:
                    bucket["reward"]["total"] = canonical_metric(float(total))
                except ValueError:
                    pass
    observations = [
        TrialObservation(step=step, metrics=by_step[step]) for step in order
    ]
    return tuple(observations)


class TrialWorkspaceTelemetry:
    """Find the file a trial's metrics are landing in.  **Not validated against real training.**

    The trial's directory comes from the ledger record, not from a guessed naming scheme:
    ``TrialRecord.run_dir`` is written when the trial is materialised, at ``pending``
    (``hyperparameter_search.py:1081``), so it is already on disk while the trial is running.

    Four places are tried, most specific first, and the difference between the last two is real:
    ``training.log`` is the process's relayed stdout (``research_supervisor.py:214``), while
    ``train.log`` is the trainer's own human-readable log (``telemetry_emit.py:41``/``:88``).  They
    are two different files that share a confusingly similar name.
    """

    def __init__(self, ledger: SearchLedger) -> None:
        self._ledger = ledger

    def __call__(self, trial_id: str) -> Path | None:
        record = self._ledger.trial(trial_id)
        if record is None or not record.run_dir.strip():
            return None
        workspace = Path(record.run_dir)
        if not workspace.is_dir():
            return None
        manifest = workspace / "run.json"
        if manifest.is_file():
            try:
                paths = json.loads(manifest.read_text(encoding="utf-8")).get("paths") or {}
            except (OSError, ValueError):
                paths = {}
            declared = paths.get("telemetry_jsonl")
            if isinstance(declared, str) and declared.strip():
                candidate = Path(declared)
                if candidate.is_file():
                    return candidate
        jsonl = sorted(workspace.glob("*.telemetry.jsonl"))
        if jsonl:
            return jsonl[-1]
        for name in ("training.log", "train.log"):
            candidate = workspace / name
            if candidate.is_file():
                return candidate
        return None


class JsonlTelemetryCurveSource:
    """The reference :class:`CurveSource` over a file that is still being appended to.

    Two modes, chosen once per file by looking at the first non-blank line: JSON lines are read
    incrementally from a byte cursor, while ``[TPSTAT]``/``[TPREW]`` text is re-read whole and
    handed to :func:`parse_telemetry_lines`.  The second is O(file) per call, which is honest about
    what it costs: a growing text log has no stable record boundary to resume from, and pretending
    otherwise would mean a second, subtly different parser of the same grammar.
    """

    def __init__(self, *, locate: Callable[[str], Path | None]) -> None:
        self._locate = locate
        self._cursors: dict[str, tuple[Path, int, tuple[TrialObservation, ...]]] = {}

    def curve(self, trial_id: str) -> tuple[TrialObservation, ...]:
        path = self._locate(trial_id)
        if path is None:
            return ()
        known = self._cursors.get(trial_id)
        offset, observations = (known[1], known[2]) if known is not None and known[0] == path else (0, ())
        try:
            size = path.stat().st_size
        except OSError:
            return observations
        if size < offset:
            # The file was truncated or rotated.  Starting over is the only reading that does not
            # invent steps: the bytes before the cut are no longer evidence of anything.
            offset, observations = 0, ()
        if size == offset:
            return observations
        data = self._read_from(path, offset)
        if data is None:
            return observations
        head = self._first_content(data)
        if head is None:
            self._cursors[trial_id] = (path, size, observations)
            return observations
        if head.lstrip().startswith(b"{"):
            consumed, added = self._read_json_lines(data)
            merged = observations + added
            self._cursors[trial_id] = (path, offset + consumed, merged)
            return merged
        text = self._read_text(path)
        if text is None:
            return observations
        parsed = parse_telemetry_lines(text)
        self._cursors[trial_id] = (path, size, parsed)
        return parsed

    @staticmethod
    def _read_from(path: Path, offset: int) -> bytes | None:
        try:
            with path.open("rb") as handle:
                handle.seek(offset)
                return handle.read()
        except OSError:
            return None

    @staticmethod
    def _read_text(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    @staticmethod
    def _first_content(data: bytes) -> bytes | None:
        for raw in data.split(b"\n"):
            if raw.strip():
                return raw
        return None

    @staticmethod
    def _read_json_lines(data: bytes) -> tuple[int, tuple[TrialObservation, ...]]:
        """Parse the complete lines in ``data``; a trailing partial line is left for next time."""
        consumed = 0
        observations: list[TrialObservation] = []
        pieces = data.split(b"\n")
        for raw in pieces[:-1]:
            consumed += len(raw) + 1
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                # A line that is not JSON in a JSON-lines file is a torn write or a foreign line.
                # Dropping it loses one point; guessing at it would put a wrong step in the curve.
                continue
            if isinstance(payload, Mapping):
                observation = parse_telemetry_payload(payload)
                if observation is not None:
                    observations.append(observation)
        return consumed, tuple(observations)


class TrackerPruningSink:
    """The reference :class:`PruningSink`: verdicts land through task 4's write facade.

    The order inside :meth:`apply_verdict` is load-bearing, and each step is there to close a
    specific hole:

    1. read the trial back and stop if it is already terminal -- ``finish`` would raise otherwise
       (``hyperparameter_search.py:1121-1125``), and a pruner racing a training run that just
       succeeded is ordinary, not exceptional;
    2. write the decision record -- it is the only thing that will say why, so it must be on disk
       before the status that makes the question unanswerable;
    3. ``finish`` the trial as ``pruned``;
    4. cancel the underlying run, if a canceller was injected, and record what that returned.

    Writing the decision before ``finish`` means a crash between the two leaves a decision for a
    trial that is still running.  Visible, and retryable.  The other order leaves a ``pruned``
    trial with nothing to explain it, which is the state this whole module exists to avoid.
    """

    def __init__(
        self,
        *,
        tracker: SearchTracker,
        store: TrialLedgerStore,
        ledger: SearchLedger,
        policy: PruningPolicy | None = None,
        cancel: Callable[[str, str], None] | None = None,
        actor: str = "search_pruning",
    ) -> None:
        self._tracker = tracker
        self._store = store
        self._ledger = ledger
        self._policy = policy
        self._policy_fingerprint = policy.fingerprint if policy is not None else ""
        self._policy_stored = False
        self._cancel = cancel
        self._actor = actor

    def _ensure_policy_record(self) -> None:
        """Store the policy once, guarded, before the first decision cites it.

        Guarded by ``_require_open`` -- the tracker's own check, reached for as a private name on
        purpose: it is the only precondition in the repository that means "this write cannot brick
        the ledger", and it is the same object task 4 hands to its own appends.  A policy written
        before the run header would make the root permanently unreadable (see
        :func:`store_policy_record`), so the ordering is enforced here rather than left as advice.
        """
        if self._policy is None or self._policy_stored:
            return
        store_policy_record(
            self._store,
            self._policy,
            precondition=self._tracker._require_open,
            actor=self._actor,
        )
        self._policy_stored = True

    def record_curve(self, trial_id: str, observations: Sequence[TrialObservation]) -> None:
        """Append observations the ledger does not already hold, and nothing else.

        **Idempotent by step, and that is load-bearing rather than tidy.**  ``observe`` appends
        blindly (``hyperparameter_search.py:1110``), so a caller who records a curve the ledger
        already has -- a resumed search, a second watcher, a test that wrote one directly -- gets
        the whole curve twice.  The result is not a duplicate that looks like one: it is a curve
        whose steps go 1, 2, 1, 2, which every reader of a window statistic would then compute
        over, and which this module's own curve validator refuses outright.  A step is a step, so
        filtering on it makes this writer safe regardless of who else wrote.

        ``observe`` also refuses an empty payload and refuses a trial that is not running; both
        refusals are right, so this returns quietly instead of making a caller expect an error.
        """
        if not observations:
            return
        record = self._ledger.trial(trial_id)
        if record is None or record.status != "running":
            return
        known = {observation.step for observation in record.observations}
        fresh: list[TrialObservation] = []
        for observation in observations:
            if observation.step in known:
                continue
            known.add(observation.step)
            fresh.append(observation)
        if not fresh:
            return
        self._tracker.observe(trial_id, TrialOutcome(status="running", observations=tuple(fresh)))

    def apply_verdict(self, trial_id: str, verdict: PruneVerdict) -> bool:
        record = self._ledger.trial(trial_id)
        if record is None or record.status in TERMINAL_TRIAL_STATUSES:
            return False
        self._ensure_policy_record()
        decision = self._decision(trial_id, verdict, record)
        self._store.append(
            DECISION_RECORD_TYPE,
            decision,
            actor=self._actor,
            # In the lock, next to the write: a check made alongside it can be overtaken, and the
            # event that lands anyway costs the whole ledger root rather than one record.
            precondition=self._tracker._require_open,
        )
        self._tracker.finish(trial_id, "pruned", stop_reason=verdict.stop_reason)
        if self._cancel is not None:
            state = "requested"
            try:
                self._cancel(trial_id, verdict.stop_reason)
            except BaseException as error:  # noqa: BLE001 - the reason is recorded, not swallowed
                state = f"error: {type(error).__name__}: {error}"
            self._store.append(
                DECISION_RECORD_TYPE,
                decision.model_validate({**decision.model_dump(mode="json"), "cancel_state": state}),
                actor=self._actor,
                event_type="supersede",
                precondition=self._tracker._require_open,
            )
        return True

    def _decision(
        self, trial_id: str, verdict: PruneVerdict, record: TrialRecord
    ) -> PruneDecisionRecord:
        curve_len = len(record.observations)
        through_step = record.observations[-1].step if record.observations else 0
        observed: float | str | None = verdict.observed
        if observed is None and verdict.reason == "nan_metric":
            # The one rule whose reading cannot be carried as a number.  Storing the canonical text
            # is what lets a reader tell "this decision acted on a NaN" from "this decision recorded
            # no reading", which are different claims.
            observed = "nan"
        return PruneDecisionRecord(
            id=f"prune-decision:{trial_id}",
            trial_id=trial_id,
            policy_fingerprint=self._policy_fingerprint,
            reason=verdict.reason,
            step=verdict.step,
            through_step=max(through_step, verdict.step),
            curve_len=curve_len,
            matched_steps=verdict.matched_steps,
            metric=verdict.metric,
            observed=observed,
            threshold=verdict.threshold,
            stop_reason=verdict.stop_reason,
        )


_SENTINEL = "TAILI_RUN_FINISHED"


class PruningTrialRunner:
    """A reference :class:`TrialRunner` that watches the trial it is running.  **Not validated
    against real training**: the thread, the cadence, and the cancellation below have only ever
    been driven by fakes.

    The inner runner blocks in a worker thread while this one polls, because ``run()`` is
    synchronous by contract (``hyperparameter_search.py:233``) and a pruner that waited for the
    run to end before deciding would be deciding nothing.

    Termination is proven rather than assumed: the worker's end sentinel is the inner runner
    returning, so if the deadline passes without it the process is killed **and the sentinel's
    absence from the captured log is the evidence that it really was still running**.  That check
    is offline and reproducible; "we asked it to stop" is not evidence of anything.
    """

    def __init__(
        self,
        *,
        runner: Any,
        watcher: PruningWatcher,
        interval_seconds: float = 10.0,
        terminate_timeout_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError(f"interval_seconds must be positive, got {interval_seconds}")
        if terminate_timeout_s <= 0:
            raise ValueError(f"terminate_timeout_s must be positive, got {terminate_timeout_s}")
        self._runner = runner
        self._watcher = watcher
        self._interval = interval_seconds
        self._terminate_timeout_s = terminate_timeout_s
        self._sleep = sleep

    def preflight(self, request: Any) -> None:
        self._runner.preflight(request)

    def run(self, request: Any) -> TrialOutcome:
        outcome: dict[str, Any] = {}
        failure: dict[str, BaseException] = {}
        finished = threading.Event()

        def worker() -> None:
            try:
                outcome["value"] = self._runner.run(request)
            except BaseException as error:  # noqa: BLE001 - re-raised in the caller's thread
                failure["value"] = error
            finally:
                # The sentinel: set in ``finally`` so it means "the worker is done", not "the
                # worker succeeded".  A poller that read success into it would stop watching a
                # run that had already died with an exception.
                finished.set()

        thread = threading.Thread(target=worker, name=f"prune-{request.trial_id}", daemon=True)
        thread.start()
        while not finished.wait(self._interval):
            self._watcher.review(request.trial_id)
        thread.join()
        if "value" in failure:
            raise failure["value"]
        return outcome["value"]

    def terminate(self, trial_id: str, reason: str, process: Any | None = None) -> bool:
        """Best-effort stop of an inner run that has not ended by its deadline.

        Returns whether the sentinel was observed.  ``True`` means the inner run ended.  ``False``
        means the deadline passed and whatever ``process`` was given had to be killed -- which the
        caller should treat as "unknown how far it got", not as a clean stop.
        """
        if process is None:
            return True
        deadline = time.monotonic() + self._terminate_timeout_s
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return True
            self._sleep(min(0.5, self._terminate_timeout_s))
        process.kill()
        return process.poll() is not None
