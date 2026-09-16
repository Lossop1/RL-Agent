"""Ranking a finished search and exporting the configuration that won.

Task 4 records what a search did.  This module answers the three questions an operator asks
*after* it stops, and each of them has a failure mode that looks like success:

* **Which trial won?**  Ranking needs a direction, and nothing in the ledger records one
  (measured: no field on ``TrialOutcome`` / ``TrialRecord`` / ``SearchRunRecord`` says whether
  an objective is better larger or smaller).  A default would be right half the time, and the
  wrong half exports the *worst* configuration under the label "best" -- indistinguishable
  from a success until somebody trains on it.  So the direction is a required argument.
* **Is this sample worth ranking at all?**  A search that lost a trial, or whose counts do not
  add up, still reads as a list of results.  :func:`analyze` therefore reports *which* checks
  ran (``SearchAnalysis.sample_trust``) instead of a boolean "complete", and
  :func:`export_best` refuses everything that was not count-checked.
* **Is the exported file the one the winning trial ran?**  The export is verified against the
  fingerprint the *trial record* has carried since the trial materialised, which is the only
  check that survives someone editing the source config and re-exporting.

Three things it deliberately is **not**:

* Not a sampler.  Task 7 owns the surrogate model and the acquisition function; nothing here
  builds a ``Sampler``, calls ``replay``, or proposes an assignment.  ``is_observation``
  (``hyperparameter_search.py:509``) is the ready-made criterion for that task, not a decision
  taken on its behalf.
* Not a second judge of pruning.  ``prune_summary`` is not called: it reads
  ``ledger.trials(run_ref=...)`` (``search_pruning.py:988``), which in a degraded read can be a
  *different, smaller* sample than the one ranked here, and a report that audits one sample
  while citing another is the failure this module exists to prevent.  Only the four pure
  reading helpers (``metric_value``, ``is_nonfinite``, ``reading_is_nulled``,
  ``canonical_metric``) are shared, so the two layers cannot disagree about what a reading *is*.
* Not a writer.  Not one line appends to the ledger, and no new record type is defined:
  writing any record before a run's ``search_run`` header makes the whole root permanently
  unreadable (measured by task 5: ``_validate_layout`` raises ``unknown_layout``,
  ``hyperparameter_search.py:574-622``).  An explanation of a search does not belong in the
  record of what the search did.

Two read-side rules that the rest of the module leans on:

**The sample comes from ``run()``, never from ``trials()``.**  ``trials()`` does run the hash
chain and ``_validate_layout`` -- it calls ``refresh()`` first (``:736`` -> ``:557-572``) --
despite its own docstring saying it "does not run ``run()``'s layout checks" (``:735``), which
means ``run()``'s *run-scoped* checks.  Those are the ones this module needs:
``_trials_of`` (``:655-683``: the trial's run, plan and space fingerprints agree with the
header, the indices are exactly ``1..N``, no two trials share an ``assignment_key``) and
``_check_counts`` (``:706-726``: ``trial_count == stats["proposals"] == len(trials)``).  Under
an index gap or a duplicate key ``trials()`` still returns records, so the failure mode of
reading through it is a *smaller, confident-looking* sample.  When ``run()`` fails, the
fallback below says so in ``error`` and downgrades ``sample_trust`` rather than going quiet.

**A ledger fact never raises; a shipped configuration always does.**  ``analyze()`` is the only
view an operator has of the run, so throwing would send them to parse ``trials.jsonl`` by hand
-- which is how a wrong conclusion gets drawn a week later.  The precedent is explicit:
``TrialLedgerStore.summary`` is documented "**Never raises**" and keeps a named
``integrity_error`` (``trial_ledger.py:723-750``).  ``export_best()`` is the other side of the
same line: its output gets copied into a training run, "this is the config the winning trial
ran" has exactly two truth values, and the caller's next line is naturally the export itself.
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any, Literal, Mapping, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config_injection import inject_into_file, load_config_file
from .hyperparameter_search import (
    TERMINAL_TRIAL_STATUSES,
    SearchError,
    SearchLedger,
    SearchRunRecord,
    TrialRecord,
    TrialStatus,
)
from .hyperparameter_space import SearchSpaceSchema
from .research_ledger import ArtifactRef, canonical_json, content_hash
from .search_pruning import canonical_metric, is_nonfinite, metric_value, reading_is_nulled

ANALYSIS_SCHEMA_VERSION = "rl-agent.search-analysis/v1"

#: The receipt carries its own version rather than reusing the report's: it is read back by
#: someone who has a config file and a manifest and nothing else, and a shared version number
#: would make two independently-evolving shapes claim to be the same shape.
EXPORT_SCHEMA_VERSION = "rl-agent.search-export/v1"

#: The suffix of a receipt.  ``<config>.manifest.json`` keeps the pair adjacent on disk, which
#: is what makes "the receipt next to a config explains that config" true without a registry.
MANIFEST_SUFFIX = ".manifest.json"

RankDirection = Literal["maximize", "minimize"]

#: Every way a trial can fail to yield a number.  ``absent`` and ``not_a_number`` are separated
#: here although the pruning layer folds them together (``metric_value`` returns ``None`` for
#: both), because "this key was never in the payload" and "this key holds a bool" are two
#: different sentences to an operator asking why a trial was not ranked.  See
#: :func:`_reading` for the classification order, which is fixed.
ReadingState = Literal["value", "no_objective", "absent", "nulled", "not_a_number", "nonfinite"]

#: How much of the read path actually ran.  Not a boolean "complete": a run that is still open
#: and a run whose counts disagree are both incomplete, and only one of them can be finished.
SampleTrust = Literal["counts_checked", "layout_checked", "unreadable"]

#: ``unreadable`` is a fourth run status the ledger cannot store (``SearchRunRecord.status`` is
#: ``open``/``closed``/``abandoned``): it is what the *reader* found, not what was recorded.
RunStatus = Literal["open", "closed", "abandoned", "unreadable"]

CoverageVerdict = Literal[
    "never_closed",
    "abandoned",
    "unreadable",
    "unstated",
    "walked_the_grid",
    "budget_truncated_the_grid",
    "drew_its_budget",
    "stream_ended_early",
]

#: The verdicts, as a runtime tuple so "the table has no hole" is an assertable fact and a new
#: entry cannot be added to the ``Literal`` alone.  Order is the table's order in the design.
COVERAGE_VERDICTS: tuple[CoverageVerdict, ...] = (
    "unreadable",
    "never_closed",
    "abandoned",
    "unstated",
    "walked_the_grid",
    "budget_truncated_the_grid",
    "drew_its_budget",
    "stream_ended_early",
)

#: The two halves of a grid verdict, and the one that is not about the grid at all.  Named so
#: the verdict sentences and the table cannot drift apart.
_GRID_KIND = "grid"

#: Read off the status vocabulary rather than retyped: a new ``TrialStatus`` that is not
#: terminal and not ``pending``/``running`` would otherwise be counted by no bucket at all.
_TRIAL_STATUSES: tuple[TrialStatus, ...] = get_args(TrialStatus)

_VERDICT_DETAIL: dict[str, str] = {
    "unreadable": "the ledger could not be read, so nothing about this run is known",
    "never_closed": (
        "the run has no closing record, so its counts were never checked against the sampler's"
    ),
    "abandoned": (
        "the run was abandoned: whatever counts it holds were never checked against the "
        "sampler's, so they were recorded but not confirmed"
    ),
    "unstated": (
        "the run closed without recording whether the sampler's stream was exhausted, so it is "
        "not known whether the plan was finished or cut short"
    ),
    "walked_the_grid": (
        "the grid plan was closed and the sampler reported its stream exhausted: every point of "
        "the grid was proposed and recorded"
    ),
    "budget_truncated_the_grid": (
        "the grid plan was closed with the sampler's stream not exhausted: something ended it "
        "early, so parts of the grid were never proposed"
    ),
    "drew_its_budget": (
        "the plan was closed with the sampler's stream exhausted: it drew the budget it was "
        "given, which says nothing about how much of the space that budget covered"
    ),
    "stream_ended_early": (
        "a plan that is not a grid was closed with the sampler's stream not exhausted: the run "
        "either stopped before the stream ran out or was closed without recording that it had "
        "run out (measured: ``SamplerStats.exhausted`` defaults to False, so "
        "``close_run(SamplerStats(proposals=N))`` writes exactly this)"
    ),
}


class SearchAnalysisError(SearchError):
    """Base class for everything this module raises.

    Under ``SearchError`` so a caller that already handles task 4's failures keeps working.
    """


class ExportRefusedError(SearchAnalysisError):
    """The analysis cannot be turned into a configuration, and here is every reason why.

    ``reasons`` holds all of them rather than the first: an operator fixing one at a time
    through successive exceptions is how a refusal turns into a guessing game.
    """

    def __init__(self, reasons: tuple[str, ...] | list[str]) -> None:
        self.reasons: tuple[str, ...] = tuple(reasons)
        super().__init__(f"refusing to export: {'; '.join(self.reasons)}")


# --- Reading a value out of a trial ------------------------------------------------------------


def _located(metrics: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    """Follow a dotted ``path`` and report whether it exists and what raw value is there.

    Display only: :func:`_reading` classifies with the pruning layer's functions first, and
    reaches this one **only** where they both said "there is no reading here".  That keeps this
    helper from becoming a second opinion on what a reading is.
    """
    current: Any = metrics
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _describe_unrepresentable(raw: Any) -> str:
    """The shape of a number no float can hold, without printing all of it.

    ``repr`` of ``10**400`` is 401 digits, and the digits are the one part an operator cannot
    use; "an integer of 401 digits" is the fact.  Anything else is shown as it is, because this
    branch is not supposed to be reachable for a non-integer.
    """
    if isinstance(raw, int) and not isinstance(raw, bool):
        return f"an integer of {len(str(raw))} digits"
    return repr(raw)


def _reading(rule: "RankingRule", record: TrialRecord) -> "Reading":
    """This trial's reading on the rule's key, by the fixed order: value, null, else raw.

    1. ``metric_value`` decides whether there is a number, and ``is_nonfinite`` separates a
       restored ``"nan"``/``"inf"`` text from a usable one.
    2. ``reading_is_nulled`` catches the other half of the JSON dump: the key survived and the
       value did not, which is how ``model_dump(mode="json")`` writes a non-finite float
       (``trial_ledger.py:565``).
    3. Only then does this module look at the raw payload, to say "absent" or "not a number".

    ``rule.key == "objective"`` reads the record's **field**, never a same-named metric: the
    field is the surrogate model's y, and a ``metrics`` bag that happens to carry an
    ``objective`` key is a different number that would silently take its place.
    """
    key = rule.key
    if key == "objective":
        if record.objective is None:
            return Reading(
                index=record.index,
                trial_id=record.id,
                key=key,
                state="no_objective",
                value=None,
                raw="",
            )
        return Reading(
            index=record.index,
            trial_id=record.id,
            key=key,
            state="value",
            value=record.objective,
            raw="",
        )

    try:
        value = metric_value(record.metrics, key)
    except OverflowError:
        #: ``metric_value`` ends in ``float(current)`` for any ``int`` (``search_pruning.py:108``),
        #: and a JSON integer has no upper bound: an exponent in a metrics bag -- a surrogate's
        #: log-likelihood, a loss that diverged -- can land ``10**400`` in the ledger, which no
        #: float can hold.  Measured: unwrapped, this escaped ``analyze`` as ``OverflowError: int
        #: too large to convert to float``, out of a function documented never to raise for a fact
        #: about the ledger.  The reading is not "no number"; it is a number this module cannot
        #: compare, which is what ``nonfinite`` names and what keeps it out of the ranking.
        _, raw = _located(record.metrics, key)
        return Reading(
            index=record.index,
            trial_id=record.id,
            key=key,
            state="nonfinite",
            value=None,
            raw=_describe_unrepresentable(raw),
        )
    if value is not None:
        if is_nonfinite(value):
            return Reading(
                index=record.index,
                trial_id=record.id,
                key=key,
                state="nonfinite",
                value=None,
                raw=str(canonical_metric(value)),
            )
        return Reading(
            index=record.index, trial_id=record.id, key=key, state="value", value=value, raw=""
        )
    if reading_is_nulled(record.metrics, key):
        return Reading(
            index=record.index, trial_id=record.id, key=key, state="nulled", value=None, raw="null"
        )
    found, raw = _located(record.metrics, key)
    if found:
        return Reading(
            index=record.index,
            trial_id=record.id,
            key=key,
            state="not_a_number",
            value=None,
            raw=repr(raw),
        )
    return Reading(
        index=record.index, trial_id=record.id, key=key, state="absent", value=None, raw=""
    )


def _why_unranked(reading: "Reading") -> str:
    """One sentence an operator can act on, not a state name repeated back at them."""
    key = reading.key
    if reading.state == "no_objective":
        return "the record carries no objective, so there is no score to rank"
    if reading.state == "absent":
        return f"the metrics payload has no key at {key!r}"
    if reading.state == "nulled":
        return (
            f"the metrics payload has a null at {key!r}: the value was dropped on the way to "
            f"disk, which is what a non-finite number becomes"
        )
    if reading.state == "not_a_number":
        return f"the metrics payload holds {reading.raw} at {key!r}, which is not a number"
    return f"{key!r} is {reading.raw}, which is not a finite number"


# --- Records -----------------------------------------------------------------------------------


class RankingRule(BaseModel):
    """What the caller means by "best".

    ``direction`` has **no default** on purpose.  A coin-flip default is an invisible
    configuration line whose failure mode (exporting the worst configuration as the winner) is
    indistinguishable from success until somebody trains on it, so the caller has to say it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = "objective"
    direction: RankDirection

    @model_validator(mode="after")
    def _validate_rule(self) -> "RankingRule":
        if not self.key.strip():
            raise ValueError("a ranking key must not be blank")
        return self

    def fingerprint(self) -> str:
        """Identity over the JSON form (law R3: never hash the model itself)."""
        return content_hash(self.model_dump(mode="json"))

    def describe(self) -> str:
        return f"{self.key} ({self.direction})"


class Reading(BaseModel):
    """One trial's reading on one key, with the state that says why it may not be a number."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int
    trial_id: str
    key: str
    state: ReadingState
    value: float | None = None
    raw: str = ""

    @model_validator(mode="after")
    def _validate_reading(self) -> "Reading":
        if (self.state == "value") != (self.value is not None):
            raise ValueError(
                f"a reading is a value exactly when it has a number: state is {self.state!r} "
                f"with value {self.value!r}"
            )
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError(f"a reading that is a value must be finite, got {self.value}")
        if self.index < 1:
            raise ValueError(f"a trial index starts at 1, got {self.index}")
        return self


class RankedTrial(BaseModel):
    """One row of the ranking.

    Carries both the number the ranking used and the record's own ``objective`` verbatim: when
    they are different numbers (the rule keys on a metric), the row has to show both or a
    reader cannot tell which one the order came from.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int
    index: int
    trial_id: str
    attempt: int
    status: TrialStatus
    value: float
    objective: float | None = None
    assignment: dict[str, Any]
    assignment_key: str
    config_path: str
    injected_fingerprint: str

    @model_validator(mode="after")
    def _validate_ranked(self) -> "RankedTrial":
        if self.rank < 1:
            raise ValueError(f"a rank starts at 1, got {self.rank}")
        if self.index < 1:
            raise ValueError(f"a trial index starts at 1, got {self.index}")
        if not math.isfinite(self.value):
            raise ValueError(f"a ranked value must be finite, got {self.value}")
        return self


class UnrankedTrial(BaseModel):
    """A terminal trial the rule could not score, named rather than dropped.

    This record is what stops "the best of 5" and "the best of 500" from printing the same
    sentence: the sample size is only the size of ``ranked``, and every terminal trial outside
    it is listed here with its state and a reason.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int
    trial_id: str
    status: TrialStatus
    state: ReadingState
    detail: str

    @model_validator(mode="after")
    def _validate_unranked(self) -> "UnrankedTrial":
        if self.index < 1:
            raise ValueError(f"a trial index starts at 1, got {self.index}")
        if self.status not in TERMINAL_TRIAL_STATUSES:
            raise ValueError(
                f"only an ended trial can be unrankable for want of a number: {self.status} has "
                f"not ended, so it is counted as pending or running instead"
            )
        return self


class TrialCensus(BaseModel):
    """Every recorded trial, in exactly one bucket.

    ``running`` is separate from ``failed`` because "the result was lost when the process died"
    and "the result is a low score" are different facts, and merging them corrupts both
    conclusions.  ``pending`` is separate for the same reason: materialised, never trained.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    recorded: int
    succeeded: int = 0
    failed: int = 0
    pruned: int = 0
    stopped: int = 0
    pending: int = 0
    running: int = 0
    pending_indices: tuple[int, ...] = ()
    running_indices: tuple[int, ...] = ()

    def terminal(self) -> int:
        """The trials that ended.  A method, not a computed field (law R1)."""
        return self.succeeded + self.failed + self.pruned + self.stopped

    @model_validator(mode="after")
    def _validate_census(self) -> "TrialCensus":
        parts = (
            self.succeeded
            + self.failed
            + self.pruned
            + self.stopped
            + self.pending
            + self.running
        )
        if parts != self.recorded:
            raise ValueError(
                f"the census parts add up to {parts} but {self.recorded} trials were recorded: "
                f"a trial in no bucket is a trial the report lost"
            )
        if len(self.pending_indices) != self.pending:
            raise ValueError(
                f"the census counts {self.pending} pending trials but names "
                f"{len(self.pending_indices)}"
            )
        if len(self.running_indices) != self.running:
            raise ValueError(
                f"the census counts {self.running} running trials but names "
                f"{len(self.running_indices)}"
            )
        for name in ("pending_indices", "running_indices"):
            indices = getattr(self, name)
            if list(indices) != sorted(indices) or len(set(indices)) != len(indices):
                raise ValueError(f"{name} must be strictly increasing, got {indices}")
            if indices and indices[0] < 1:
                raise ValueError(f"a trial index starts at 1, got {indices[0]}")
        return self


class Coverage(BaseModel):
    """How far the search got, as four separate facts rather than one boolean ``complete``.

    The verdict is recomputed from the other fields by this validator, so a record cannot claim
    ``walked_the_grid`` while its own ``kind`` says otherwise.  See :func:`_coverage_verdict`
    for the table and the design note for why each row is its own name.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_status: RunStatus
    kind: str = ""
    closing_present: bool = False
    exhausted: bool | None = None
    budget: int | None = None
    proposed: int | None = None
    declared_trial_count: int | None = None
    recorded: int = 0
    verdict: CoverageVerdict
    detail: str = ""

    @model_validator(mode="after")
    def _validate_coverage(self) -> "Coverage":
        if self.verdict not in COVERAGE_VERDICTS:
            raise ValueError(f"{self.verdict!r} is not one of {COVERAGE_VERDICTS}")
        if self.closing_present != (self.run_status == "closed"):
            raise ValueError(
                f"a closing record exists exactly when the run closed: status is "
                f"{self.run_status!r} with closing_present {self.closing_present}"
            )
        expected = _coverage_verdict(
            self.run_status, self.kind, self.exhausted, self.closing_present
        )
        if self.verdict != expected:
            raise ValueError(
                f"the coverage verdict {self.verdict!r} contradicts its own fields: they say "
                f"{expected!r}"
            )
        if self.recorded < 0:
            raise ValueError(f"a trial count cannot be negative, got {self.recorded}")
        return self


class SearchAnalysis(BaseModel):
    """One search, ranked, with the caveats a reader needs in order not to over-read it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = ANALYSIS_SCHEMA_VERSION
    run_ref: str
    plan_fingerprint: str = ""
    space_fingerprint: str = ""
    #: Blank here is a **recorded fact** ("this run recorded no source config"), not "not read":
    #: ``sample_trust == "unreadable"`` is the only signal that nothing was read at all.
    source_config: str = ""
    source_config_fingerprint: str = ""
    space_artifact: ArtifactRef | None = None
    rule: RankingRule
    rule_fingerprint: str
    sample_trust: SampleTrust
    error: str = ""
    coverage: Coverage
    census: TrialCensus
    ranked: tuple[RankedTrial, ...] = ()
    #: How many trials the rule *could* score, which is not ``len(ranked)``: ``top_n`` caps the
    #: rows stored, never the count, so a truncated report still reports its real sample size.
    ranked_count: int = 0
    unranked: tuple[UnrankedTrial, ...] = ()
    best_value: float | None = None
    #: Every **stored** row that shares ``best_value``, in rank order.  A tie is a fact about the
    #: sample, not something to break by picking the first one.
    ties: tuple[int, ...] = ()
    #: How many trials of the whole sample share ``best_value``, which is not ``len(ties)``:
    #: ``top_n`` caps the rows, so a two-row report of a five-way tie would otherwise show a
    #: champion.  Separate from ``ties`` rather than derived, because the rows that were truncated
    #: away are exactly what the field has to remember.
    tied_count: int = 0
    caveats: tuple[str, ...] = ()

    def best(self) -> RankedTrial | None:
        """The top row, or ``None`` when nothing could be scored."""
        return self.ranked[0] if self.ranked else None

    @model_validator(mode="after")
    def _validate_analysis(self) -> "SearchAnalysis":
        if self.rule_fingerprint != self.rule.fingerprint():
            raise ValueError(
                f"the report cites rule {self.rule_fingerprint} but the rule it carries "
                f"fingerprints to {self.rule.fingerprint()}"
            )
        #: The one coupling between the two: a count-checked sample cannot also carry an error.
        #: Nothing else ties them, which is what makes a degraded report constructible at all.
        if self.sample_trust == "counts_checked" and self.error:
            raise ValueError(
                f"a count-checked sample cannot also carry an error: {self.error!r}"
            )
        if self.sample_trust != "unreadable":
            for name in ("plan_fingerprint", "space_fingerprint"):
                if not getattr(self, name).strip():
                    raise ValueError(
                        f"a read that reached the run must carry its {name}: blank here would "
                        f"read as 'not read', which is what sample_trust is for"
                    )
        if self.coverage.recorded != self.census.recorded:
            raise ValueError(
                f"the coverage block counts {self.coverage.recorded} trials and the census "
                f"counts {self.census.recorded}"
            )
        self._validate_ranking()
        self._validate_no_trial_vanishes()
        return self

    def _validate_ranking(self) -> None:
        """Rank order, contiguity, and the tie bookkeeping."""
        if [row.rank for row in self.ranked] != list(range(1, len(self.ranked) + 1)):
            raise ValueError(
                f"ranks must be 1..{len(self.ranked)} with no gaps, got "
                f"{[row.rank for row in self.ranked]}"
            )
        if self.ranked_count < len(self.ranked):
            raise ValueError(
                f"ranked_count is the size of the sample, not of the rows stored: "
                f"{self.ranked_count} < {len(self.ranked)}"
            )
        if self.ranked_count > self.census.terminal():
            raise ValueError(
                f"{self.ranked_count} trials were ranked but only {self.census.terminal()} "
                f"trials ended: a trial that has not ended has no result to rank"
            )
        values = [row.value for row in self.ranked]
        ordered = (
            all(a >= b for a, b in zip(values, values[1:]))
            if self.rule.direction == "maximize"
            else all(a <= b for a, b in zip(values, values[1:]))
        )
        if not ordered:
            raise ValueError(
                f"the rows are not in {self.rule.direction} order: {values}"
            )
        if self.ranked:
            if self.best_value is None or self.ranked[0].value != self.best_value:
                raise ValueError(
                    f"best_value must be the first row's value: {self.best_value!r} against "
                    f"{self.ranked[0].value!r}"
                )
            expected = tuple(
                row.index for row in self.ranked if row.value == self.best_value
            )
            if self.ties != expected:
                raise ValueError(
                    f"ties must name every stored row sharing the best value: {self.ties} "
                    f"against {expected}"
                )
            #: The tied candidates all share the best value, so the ``(value, index)`` order puts
            #: them in the first rows: what is stored of them is ``min(tied_count, len(ranked))``.
            #: Checked as an identity rather than a bound, because a bound would let a report
            #: claim five tied trials while storing two rows of a three-way tie.
            if len(self.ties) != min(self.tied_count, len(self.ranked)):
                raise ValueError(
                    f"{self.tied_count} trials share the best value but {len(self.ties)} of the "
                    f"{len(self.ranked)} stored rows name it: the stored tied rows are the first "
                    f"min(tied_count, len(ranked))"
                )
        elif self.best_value is not None or self.ties:
            raise ValueError("an empty ranking has no best value and no ties")
        elif self.tied_count:
            raise ValueError(
                f"an empty ranking has no ties, got tied_count {self.tied_count}"
            )

    def _validate_no_trial_vanishes(self) -> None:
        """Every recorded trial is in exactly one of: ranked, unranked, pending or running.

        Checked as arithmetic (counts) *and* as set membership (indices).  The counts alone
        would let two swapped trials cancel out; the indices alone would not notice a duplicate
        row appended to ``ranked``.
        """
        total = (
            self.ranked_count
            + len(self.unranked)
            + self.census.pending
            + self.census.running
        )
        if total != self.census.recorded:
            raise ValueError(
                f"{self.ranked_count} ranked + {len(self.unranked)} unranked + "
                f"{self.census.pending} pending + {self.census.running} running = {total}, but "
                f"{self.census.recorded} trials were recorded: a trial is missing from the report"
            )
        named: list[int] = [row.index for row in self.unranked]
        named += list(self.census.pending_indices) + list(self.census.running_indices)
        if len(set(named)) != len(named):
            raise ValueError(f"an unranked, pending or running trial is named twice: {named}")
        for index in named:
            #: Bounded below only.  The upper bound cannot be ``census.recorded``: a *gapped*
            #: ledger records indices 2 and 3 and nothing else, and that is precisely the sample
            #: this report exists to describe.  Measured: with the upper bound in place,
            #: ``analyze`` raised ``ValidationError: a trial index must be within 1..1, got 2``
            #: out of a ledger whose only trial was index 2 -- a pydantic error escaping a
            #: function documented never to raise for a ledger fact.  What the bound would have
            #: caught (a hand-edited report naming a trial that does not exist) is caught at
            #: export time instead, where the row is compared against the ledger field by field.
            if index < 1:
                raise ValueError(f"a trial index starts at 1, got {index}")


class ExportArtifact(BaseModel):
    """The receipt for one exported configuration.

    Deliberately **no** ``verified: bool`` field: a stored flag that is true whenever the record
    exists is not evidence, it is the assertion this repository has been burned by.  What the
    receipt carries instead is the pair of numbers a reader can check for themselves
    (:meth:`verified`, or :func:`verify_export` when a reason is wanted).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = EXPORT_SCHEMA_VERSION
    run_ref: str
    rule: RankingRule
    rule_fingerprint: str
    trial_id: str
    index: int
    attempt: int
    status: TrialStatus
    assignment: dict[str, Any]
    assignment_key: str
    value: float
    #: How many trials of the analysed sample scored ``value``.  The receipt is read days later by
    #: someone who has this file and the config and nothing else, and "this config scored the best
    #: value" is a different claim when two others scored it too.  The report's tie caveat is not
    #: available at that point, so the number travels with the artifact that names the winner.
    tied_count: int = 1
    objective: float | None = None
    source_config: str
    source_config_fingerprint: str
    #: The fingerprint the winning trial recorded for its own config.  It has been in the ledger
    #: since the trial materialised (``hyperparameter_search.py:1090``), which is what makes the
    #: comparison evidence rather than a self-check.
    expected_fingerprint: str
    output_config: str
    output_fingerprint: str
    #: Whether the exported tree was also compared against the winning trial's own file.  A
    #: recorded boolean because the trial workspace is a plain directory that may be long gone:
    #: ``False`` means "that check could not run", never "it passed".
    compared_against_trial_file: bool = False
    trial_config_path: str = ""
    coverage_verdict: CoverageVerdict
    sample_trust: SampleTrust
    manifest_path: str = ""

    @model_validator(mode="after")
    def _validate_artifact(self) -> "ExportArtifact":
        if self.rule_fingerprint != self.rule.fingerprint():
            raise ValueError(
                f"the receipt cites rule {self.rule_fingerprint} but the rule it carries "
                f"fingerprints to {self.rule.fingerprint()}"
            )
        if self.output_fingerprint != self.expected_fingerprint:
            raise ValueError(
                f"the receipt describes a config that is not the one the trial ran: it cites "
                f"{self.expected_fingerprint} and wrote {self.output_fingerprint}"
            )
        if self.sample_trust != "counts_checked":
            raise ValueError(
                f"a receipt can only be written for a count-checked sample, got "
                f"{self.sample_trust!r}"
            )
        if self.index < 1:
            raise ValueError(f"a trial index starts at 1, got {self.index}")
        if not self.output_config.strip():
            raise ValueError("a receipt must name the config it wrote")
        if not self.expected_fingerprint.strip():
            raise ValueError("a receipt must carry the trial's own config fingerprint")
        if self.tied_count < 1:
            raise ValueError(
                f"a receipt names a trial that scored its value, so at least one trial scores it, "
                f"got tied_count {self.tied_count}"
            )
        return self

    def verified(self) -> bool:
        """Is the file still the config this receipt was written for?

        The cheap read-side question; :func:`verify_export` is the one that raises with a reason.
        """
        path = Path(self.output_config)
        if not path.is_file():
            return False
        try:
            return content_hash(load_config_file(path)) == self.expected_fingerprint
        except (OSError, ValueError, yaml.YAMLError):
            #: ``yaml.YAMLError`` is neither of the other two -- it is not an ``OSError`` and it
            #: does not derive from ``ValueError`` -- so without it a tampered file that is *not
            #: valid YAML* raised out of a predicate whose whole contract is "False means no".
            return False


# --- The coverage table ------------------------------------------------------------------------


def _coverage_verdict(
    run_status: RunStatus, kind: str, exhausted: bool | None, closing_present: bool
) -> CoverageVerdict:
    """The one function that decides a verdict, so the table and the record cannot disagree.

    Order is load-bearing twice:

    * An unread ledger is reported before anything about kinds, because nothing was read.
    * "closed without saying whether the stream was exhausted" is reported **before** the grid
      rows, so a grid that does not say cannot be filed under either half of the grid story.

    ``kind`` is compared to ``"grid"`` and everything else -- including a kind this version has
    never heard of -- takes the non-grid rows.  That tolerance is about not adding a *second*
    place that has to know the vocabulary, not about kinds being unknown in practice: the
    non-grid rows are named for what they describe, and a stream that reported itself exhausted
    drew the whole budget it was given whatever kind drew it.

    This paragraph used to end by naming a backstop that no longer exists: it said a third
    ``SamplerKind`` "would already be refused by ``SearchRunRecord``'s own validator when the
    ledger is read (it re-validates the plan, ``hyperparameter_search.py:315``)".  That was true
    while ``SamplerKind`` held two kinds and false the moment task 7 added ``bayesian`` --  the
    validator accepts it now.  The claim was removed rather than left standing, because a
    justification that has quietly expired reads exactly like one that still holds.  The
    behaviour it was defending is unchanged and is separately pinned by a test: a ``bayesian``
    run reads as ``drew_its_budget``, not as a ``ValidationError``.
    """
    if run_status == "unreadable":
        return "unreadable"
    if run_status == "open":
        return "never_closed"
    if run_status == "abandoned":
        return "abandoned"
    if not closing_present or exhausted is None:
        return "unstated"
    if kind == _GRID_KIND:
        return "walked_the_grid" if exhausted else "budget_truncated_the_grid"
    return "drew_its_budget" if exhausted else "stream_ended_early"


# --- analyze() ---------------------------------------------------------------------------------


class _RunRead(BaseModel):
    """What the read attempt produced.  Internal: never stored, never fingerprinted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    header: SearchRunRecord | None = None
    trials: tuple[TrialRecord, ...] = ()
    trust: SampleTrust = "unreadable"
    error: str = ""
    #: Whether the *ledger* could be read at all, which is not the same question as whether this
    #: run is in it: a mistyped ``run_ref`` leaves a perfectly readable ledger with no such run,
    #: and the two deserve different sentences.  Measured: the first version of this module
    #: raised out of ``analyze`` on exactly that case, because a missing header left the plan
    #: fingerprint blank while the trust said a read had happened.
    ledger_readable: bool = False


def _read_run(ledger: SearchLedger, run_ref: str) -> _RunRead:
    """Read one run, and on failure read as much as the ledger will still give.

    The fallback goes through ``runs()`` and ``trials()``, which skip the run-scoped checks --
    that is the point: index gaps, duplicate assignment keys and count mismatches are precisely
    the failures worth describing, and refusing to describe them helps nobody.  What the
    fallback cannot survive is a broken hash chain or a broken layout, because both of those
    reads run those checks; that case is reported as ``unreadable`` rather than faked.

    A run name that is not in the ledger is ``unreadable`` too, and for the same reason: it is
    the statement "nothing about this run is known", which is what an empty report means.  The
    two cases stay distinguishable in ``error`` and in ``coverage.detail``.
    """
    try:
        run = ledger.run(run_ref)
    except Exception as error:  # noqa: BLE001 -- "a ledger fact never raises" includes OSError
        first = f"{type(error).__name__}: {error}"
        try:
            header = next(
                (record for record in ledger.runs() if record.run_ref == run_ref), None
            )
            trials = ledger.trials(run_ref=run_ref)
        except Exception as fallback:  # noqa: BLE001 -- the same promise, one level down
            return _RunRead(
                error=(
                    f"{first}; the fallback read failed too "
                    f"({type(fallback).__name__}: {fallback})"
                ),
                trust="unreadable",
            )
        if header is None:
            return _RunRead(error=first, trust="unreadable", ledger_readable=True)
        return _RunRead(
            header=header,
            trials=tuple(trials),
            trust="layout_checked",
            error=first,
            ledger_readable=True,
        )
    return _RunRead(
        header=run.header,
        trials=run.trials,
        trust="counts_checked" if run.complete else "layout_checked",
        ledger_readable=True,
    )


def _declared_trial_count(header: SearchRunRecord) -> int | None:
    """The header's trial count, with the placeholder ``0`` of an unclosed run mapped to ``None``.

    ``SearchRunRecord.trial_count`` defaults to ``0`` (``hyperparameter_search.py:307``), and a
    run that was opened and never closed carries exactly that default.  Reporting it would turn
    "nothing was recorded" into "it recorded nothing", so for ``open`` the value is ``None``.
    An abandoned run is **not** treated this way: whatever counts it holds are on disk and were
    written deliberately, so they are read and labelled as unconfirmed by the verdict.
    """
    if header.status == "open":
        return None
    return header.trial_count


def _census(trials: tuple[TrialRecord, ...]) -> TrialCensus:
    counts: dict[str, int] = {status: 0 for status in _TRIAL_STATUSES}
    pending: list[int] = []
    running: list[int] = []
    for record in trials:
        counts[record.status] = counts.get(record.status, 0) + 1
        if record.status == "pending":
            pending.append(record.index)
        elif record.status == "running":
            running.append(record.index)
    return TrialCensus(
        recorded=len(trials),
        succeeded=counts["succeeded"],
        failed=counts["failed"],
        pruned=counts["pruned"],
        stopped=counts["stopped"],
        pending=counts["pending"],
        running=counts["running"],
        pending_indices=tuple(sorted(pending)),
        running_indices=tuple(sorted(running)),
    )


def _coverage(read: _RunRead, recorded: int, run_ref: str) -> Coverage:
    header = read.header
    if header is None:
        return Coverage(
            run_status="unreadable",
            recorded=recorded,
            verdict="unreadable",
            detail=(
                f"this ledger holds no search run named {run_ref!r}"
                if read.ledger_readable
                else _VERDICT_DETAIL["unreadable"]
            ),
        )
    plan = header.plan
    kind = plan.get("kind") if isinstance(plan.get("kind"), str) else ""
    closing_present = header.status == "closed"
    #: Read from ``stats`` whether or not the run closed.  An open run's stats are empty
    #: (measured: ``open_run`` writes ``{}``), so this yields ``None`` there -- and the verdict
    #: for an open run does not consult it anyway.
    raw_exhausted = header.stats.get("exhausted")
    exhausted = raw_exhausted if isinstance(raw_exhausted, bool) else None
    raw_proposed = header.stats.get("proposals")
    proposed = raw_proposed if isinstance(raw_proposed, int) and not isinstance(raw_proposed, bool) else None
    verdict = _coverage_verdict(header.status, kind, exhausted, closing_present)
    return Coverage(
        run_status=header.status,
        kind=kind,
        closing_present=closing_present,
        exhausted=exhausted,
        budget=header.budget,
        proposed=proposed,
        declared_trial_count=_declared_trial_count(header),
        recorded=recorded,
        verdict=verdict,
        detail=_VERDICT_DETAIL[verdict],
    )


def _rank(
    rule: RankingRule, trials: tuple[TrialRecord, ...], top_n: int
) -> tuple[
    tuple[RankedTrial, ...],
    int,
    tuple[UnrankedTrial, ...],
    float | None,
    tuple[int, ...],
    int,
]:
    """Score every ended trial, order them, and keep the first ``top_n`` rows.

    The order key is ``(value, index)`` with the value negated for ``maximize``: ``index`` is
    unique within a run and a retried index folds to one record, so the order is total and
    reproducible.  Comparison is exact rather than tolerant -- both numbers came out of the same
    JSON -- because a tolerance would call two different trials one winner.
    """
    candidates: list[tuple[float, TrialRecord]] = []
    unranked: list[UnrankedTrial] = []
    for record in trials:
        if record.status not in TERMINAL_TRIAL_STATUSES:
            continue
        reading = _reading(rule, record)
        if reading.state != "value" or reading.value is None:
            unranked.append(
                UnrankedTrial(
                    index=record.index,
                    trial_id=record.id,
                    status=record.status,
                    state=reading.state,
                    detail=_why_unranked(reading),
                )
            )
            continue
        candidates.append((reading.value, record))
    maximize = rule.direction == "maximize"
    candidates.sort(key=lambda pair: ((-pair[0]) if maximize else pair[0], pair[1].index))
    ranked = tuple(
        RankedTrial(
            rank=position,
            index=record.index,
            trial_id=record.id,
            attempt=record.attempt,
            status=record.status,
            value=value,
            objective=record.objective,
            assignment=dict(record.assignment),
            assignment_key=record.assignment_key,
            config_path=record.config_path,
            injected_fingerprint=record.injected_fingerprint,
        )
        for position, (value, record) in enumerate(candidates[:top_n], start=1)
    )
    best = candidates[0][0] if candidates else None
    #: Counted over ``candidates``, not over the stored rows: ``top_n`` truncates the rows and
    #: never the sample, so a tie counted after the slice would report a unique champion when
    #: ``top_n`` is 1.  Measured: with three trials tied at 0.7, ``top_n=1`` gave ``ties == (1,)``
    #: and the tie caveat did not fire at all, while ``top_n=2`` said "2 trials tie" -- the number
    #: the operator reads as the size of the tie was an artefact of the row limit.
    tied_count = sum(1 for value, _ in candidates if value == best) if best is not None else 0
    ties = tuple(row.index for row in ranked if row.value == best) if best is not None else ()
    return ranked, len(candidates), tuple(unranked), best, ties, tied_count


def _caveats(
    rule: RankingRule,
    read: _RunRead,
    coverage: Coverage,
    census: TrialCensus,
    ranked: tuple[RankedTrial, ...],
    ranked_count: int,
    unranked: tuple[UnrankedTrial, ...],
    ties: tuple[int, ...],
    tied_count: int,
) -> tuple[str, ...]:
    """The sentences a reader needs so the numbers above cannot be over-read.  Order is fixed."""
    notes: list[str] = []
    if read.trust == "unreadable":
        notes.append(
            "nothing about this run could be read, so every number below is absent rather "
            "than zero"
        )
    elif read.trust == "layout_checked":
        if read.error:
            notes.append(
                f"the run-scoped checks did not pass, so this sample was read without them "
                f"({read.error}); it describes the run, but it cannot be used to pick a winner"
            )
        else:
            notes.append(
                "the run has no closing record, so the sample was read without the count "
                "check: it describes the run, but it cannot be used to pick a winner"
            )
    if rule.key != "objective":
        notes.append(
            f"trials are ordered by {rule.key!r} ({rule.direction}) read out of each trial's "
            f"metrics, NOT by the objective the record carries"
        )
    if coverage.exhausted and coverage.kind != _GRID_KIND:
        notes.append(
            "the sampler reported its stream exhausted, which for a non-grid plan means it "
            "drew the budget it was given, not that it covered the space: see the coverage block"
        )
    if unranked:
        notes.append(
            f"{len(unranked)} ended trial(s) could not be scored and are listed in unranked "
            f"with the reason: they are not ranked last, they are absent from the sample of "
            f"{ranked_count}"
        )
    pruned = [row for row in ranked if row.status == "pruned"]
    if pruned:
        notes.append(
            f"{len(pruned)} pruned trial(s) carry a score and are ordered alongside the trials "
            f"that ran to an end: being stopped early is not the same as having no result"
        )
    if census.running:
        notes.append(
            f"{census.running} trial(s) were interrupted and have no record of how they ended: "
            f"their results are lost rather than low"
        )
    if tied_count > 1:
        #: ``tied_count``, not ``len(ties)``: the sample's tie is the fact, the stored rows are a
        #: consequence of ``top_n``.  Firing on ``len(ties)`` made ``top_n=1`` over a three-way tie
        #: silent, which reads as a single winner.
        if len(ties) == tied_count:
            shown = f"all of them are stored here (indices {', '.join(str(i) for i in ties)})"
        else:
            shown = (
                f"only {len(ties)} of them are among the {len(ranked)} rows stored here (indices "
                f"{', '.join(str(i) for i in ties)}), because top_n caps the rows and not the "
                f"sample"
            )
        notes.append(
            f"{tied_count} trials tie for first place at {ranked[0].value}: {shown}. The order "
            f"keeps the plan's sequence and claims no preference among them, so nothing in this "
            f"ledger makes one of them the winner"
        )
    if ranked_count > len(ranked):
        notes.append(
            f"{ranked_count} trials could be scored and only the best {len(ranked)} rows are "
            f"stored here (top_n); ranked_count is the sample size"
        )
    return tuple(notes)


def analyze(
    ledger: SearchLedger, run_ref: str, *, rule: RankingRule, top_n: int = 20
) -> SearchAnalysis:
    """Read one run, rank its ended trials by ``rule``, and explain what the ranking is worth.

    Never raises for a fact about the ledger: an unreadable root, a missing run, a lost trial and
    a count mismatch all come back as a report whose ``sample_trust``, ``error`` or ``coverage``
    says so.  ``ValueError`` is still raised for a caller error (``top_n`` below 1, a blank key),
    because that is a mistake in the question, not a fact about the answer.
    """
    if top_n < 1:
        raise ValueError(f"top_n must be at least 1, got {top_n}")
    read = _read_run(ledger, run_ref)
    header = read.header
    census = _census(read.trials)
    coverage = _coverage(read, census.recorded, run_ref)
    ranked, ranked_count, unranked, best_value, ties, tied_count = _rank(rule, read.trials, top_n)
    return SearchAnalysis(
        run_ref=run_ref,
        plan_fingerprint=header.plan_fingerprint if header else "",
        space_fingerprint=header.space_fingerprint if header else "",
        source_config=header.source_config if header else "",
        source_config_fingerprint=header.source_config_fingerprint if header else "",
        space_artifact=header.space_artifact if header else None,
        rule=rule,
        rule_fingerprint=rule.fingerprint(),
        sample_trust=read.trust,
        error=read.error,
        coverage=coverage,
        census=census,
        ranked=ranked,
        ranked_count=ranked_count,
        unranked=unranked,
        best_value=best_value,
        ties=ties,
        tied_count=tied_count,
        caveats=_caveats(
            rule, read, coverage, census, ranked, ranked_count, unranked, ties, tied_count
        ),
    )


# --- export_best() -----------------------------------------------------------------------------


def _discard(path: Path) -> bool:
    """Remove a file this call created, and say whether it is gone."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return False
    return not path.exists()


def _write_manifest(path: Path, payload: str) -> None:
    """Stage and rename, the same way the injector writes a config.

    A reader may be handed this path by someone else, and half a JSON document parses as a
    different document rather than as an error.
    """
    parent = path.parent
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f"{path.name}.",
        suffix=".tmp",
        dir=parent,
        delete=False,
    )
    staged = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, path)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def _agrees_with_ledger(winner: RankedTrial, record: TrialRecord) -> str:
    """``""`` when the ranked row is the record it names, else the mismatch to report."""
    for name in ("index", "attempt", "assignment_key", "assignment", "objective", "status"):
        stored = getattr(winner, name)
        found = getattr(record, name)
        if stored != found:
            return f"the report's {name!r} is {stored!r} but the ledger's is {found!r}"
    return ""


def export_best(
    analysis: SearchAnalysis,
    *,
    ledger: SearchLedger,
    space: SearchSpaceSchema,
    output_path: str | os.PathLike[str],
    rank: int = 1,
    manifest_path: str | os.PathLike[str] | None = None,
) -> ExportArtifact:
    """Write the winning assignment into a copy of the source config, and a receipt beside it.

    Three kinds of exception can leave this call, and the first is the one a caller should catch:

    * :class:`ExportRefusedError` for everything that makes the export *unjustified* -- including a
      ledger that can no longer be read, because "I cannot confirm this file is the one the trial
      ran" is a refusal and not an accident.
    * ``ValueError`` for a caller error or a fact about the space; the injector's own refusal to
      write a parameter the space does not declare is passed through unchanged, the same split
      ``begin_trial`` keeps.
    * ``OSError`` and, through it, ``yaml.YAMLError`` from the filesystem itself -- a directory
      that turns read-only between the checks and the write, a full disk.  These are *accidents*
      rather than judgements, which is why they are not refusals; what they are not is a reason to
      leave half an artifact behind.

    An exception raised **after** the config was written -- a refusal, an ``OSError``, a
    ``KeyboardInterrupt`` -- removes it again before propagating, so a caller that sees any of
    them never has to wonder whether a partial artifact is on disk.
    """
    if rank < 1:
        raise ValueError(f"rank must be at least 1, got {rank}")

    #: The refusals are collected **before** the rank is bounded, and the order matters: an
    #: unexportable report has no rows to rank, so checking the bound first reports "there is no
    #: row 1" -- true, but it hides the reason and it arrives as a ``ValueError`` that a caller
    #: catching the documented refusal does not catch.  Measured: with the two swapped, exporting
    #: from an unclosed run raised ``ValueError: rank 1 is outside the 0 rows``.
    reasons: list[str] = []
    if analysis.sample_trust != "counts_checked":
        reasons.append(
            f"the sample was not count-checked ({analysis.sample_trust}"
            f"{': ' + analysis.error if analysis.error else ''}): the counts that would show "
            f"this search recorded every trial it proposed never ran"
        )
    if not analysis.ranked:
        reasons.append(f"no trial of {analysis.run_ref!r} could be scored by {analysis.rule.describe()}")
    if space.fingerprint() != analysis.space_fingerprint:
        reasons.append(
            f"the space given is {space.fingerprint()} but the run was drawn from "
            f"{analysis.space_fingerprint}"
        )
    if reasons:
        raise ExportRefusedError(reasons)
    if rank > len(analysis.ranked):
        raise ValueError(
            f"rank {rank} is outside the {len(analysis.ranked)} rows this report stores "
            f"(top_n caps the rows, and ranked_count is {analysis.ranked_count}): re-run analyze "
            f"with a larger top_n to reach it"
        )

    winner = analysis.ranked[rank - 1]

    try:
        run = ledger.run(analysis.run_ref)
    except Exception as error:  # noqa: BLE001 -- a refusal, see the docstring
        raise ExportRefusedError(
            [
                f"the run can no longer be read back ({type(error).__name__}: {error}), so the "
                f"winning row cannot be confirmed against the ledger"
            ]
        ) from error
    record = next((item for item in run.trials if item.id == winner.trial_id), None)
    if record is None:
        raise ExportRefusedError(
            [f"the ledger no longer holds the trial this report ranks first ({winner.trial_id})"]
        )
    mismatch = _agrees_with_ledger(winner, record)
    if mismatch:
        raise ExportRefusedError(
            [f"the ranked row and the ledger disagree: {mismatch}"]
        )
    if not record.injected_fingerprint.strip():
        raise ExportRefusedError(
            [
                f"trial {record.id} records no injected config fingerprint, so there is nothing "
                f"to check the exported file against"
            ]
        )

    destination = Path(output_path)
    manifest = (
        Path(manifest_path) if manifest_path is not None else Path(f"{destination}{MANIFEST_SUFFIX}")
    )
    source = Path(analysis.source_config) if analysis.source_config.strip() else None
    if source is None:
        raise ExportRefusedError(
            [
                f"run {analysis.run_ref!r} records no source config, so there is no config to "
                f"copy and inject into"
            ]
        )
    #: A non-resolving ``source_config`` is a refusal; a missing ``config_path`` is not, because
    #: the trial workspace is the caller's to keep or delete.  Recorded paths are relative to the
    #: analyst's working directory unless absolute (the header stores what the caller passed).
    if destination.resolve() == source.resolve():
        raise ExportRefusedError(
            [f"the output path is the run's own source config: {destination}"]
        )
    if record.config_path.strip() and destination.resolve() == Path(record.config_path).resolve():
        raise ExportRefusedError(
            [
                f"the output path is the winning trial's own config ({record.config_path}), "
                f"which is the evidence this export would be checked against"
            ]
        )
    trial_config_path = Path(record.config_path) if record.config_path.strip() else None
    #: The receipt is written *after* the config and by ``os.replace``, so a receipt path that is
    #: one of the files this export reads or writes does not collide -- it overwrites.  Measured
    #: with ``manifest_path == output_path``: the call returned normally, the exported config was
    #: the receipt JSON (its first bytes were ``{"analysis":``), ``artifact.verified()`` was
    #: ``False`` from that moment, and only ``verify_export`` days later noticed.  These come
    #: *before* the occupancy checks for the same reason the output-path identities do: "the receipt
    #: path already exists" is true of a file someone is still using, and the caller who aimed the
    #: receipt at their own source config needs to be told which mistake they made.
    if manifest.resolve() == destination.resolve():
        raise ExportRefusedError(
            [
                f"the receipt path and the output path are the same file ({destination}): the "
                f"receipt would be written over the config it describes"
            ]
        )
    if manifest.resolve() == source.resolve():
        raise ExportRefusedError(
            [
                f"the receipt path is the run's own source config ({source}): writing the "
                f"receipt there would destroy the evidence the export is checked against"
            ]
        )
    if trial_config_path is not None and manifest.resolve() == trial_config_path.resolve():
        raise ExportRefusedError(
            [
                f"the receipt path is the winning trial's own config ({trial_config_path}): "
                f"writing the receipt there would destroy the file the export is checked against"
            ]
        )
    if destination.exists():
        raise ExportRefusedError(
            [f"the output path already exists: {destination}; refusing to overwrite it"]
        )
    if manifest.exists():
        raise ExportRefusedError(
            [
                f"the receipt path already exists: {manifest}; refusing to overwrite a receipt "
                f"that may point at a config someone is still using"
            ]
        )
    if not destination.parent.is_dir():
        raise ExportRefusedError(
            [f"the output directory does not exist: {destination.parent}"]
        )
    if not manifest.parent.is_dir():
        raise ExportRefusedError(
            [f"the receipt's directory does not exist: {manifest.parent}"]
        )

    try:
        source_fingerprint = content_hash(load_config_file(source))
    except FileNotFoundError as error:
        raise ExportRefusedError(
            [f"the run's source config is gone ({source}): which file this search started from "
             f"can no longer be established"]
        ) from error
    except (OSError, ValueError, yaml.YAMLError) as error:
        #: ``yaml.YAMLError`` is the third family and the one that matters most here: a source
        #: config someone edited into invalid YAML is precisely the state this check exists for,
        #: and it is neither an ``OSError`` nor a ``ValueError``.  Measured: unwrapped, it escaped
        #: as ``yaml.parser.ParserError`` out of a function documented to raise two things.
        raise ExportRefusedError(
            [f"the run's source config cannot be read ({source}): {type(error).__name__}: {error}"]
        ) from error
    if source_fingerprint != analysis.source_config_fingerprint:
        raise ExportRefusedError(
            [
                f"the source config has changed since this run started ({source} hashes to "
                f"{source_fingerprint}, the run recorded {analysis.source_config_fingerprint}): "
                f"the exported file would differ from the winning trial's in ways the search "
                f"never explored"
            ]
        )

    try:
        report = inject_into_file(
            source, destination, space, dict(winner.assignment)
        )
    except ValueError:
        # A caller error or a fact about the space (an undeclared parameter, a blank assignment)
        # -- **or** the injector's own read-back, which runs *after* the write
        # (``config_injection.py``: ``write_injection`` then ``load_config_file(written)`` then the
        # fingerprint comparison that raises).  That third case arrives with a config already on
        # disk, so "the injector failed before writing" -- what this comment used to say, and what
        # the bare ``raise`` assumed -- was false, and the partial file stayed behind for the next
        # reader to pick up as a training config.  Measured by forcing the read-back to disagree.
        _discard(destination)
        raise
    except BaseException:
        # Includes a keyboard interrupt: a half-written config left behind would be read later
        # as a configuration rather than as an accident.
        _discard(destination)
        raise
    if report.injected_fingerprint != record.injected_fingerprint:
        gone = _discard(destination)
        raise ExportRefusedError(
            [
                f"the config written to {destination} is not the one the winning trial ran: it "
                f"hashes to {report.injected_fingerprint} where the trial recorded "
                f"{record.injected_fingerprint}"
                + ("" if gone else f" (and the partial file could not be removed)")
            ]
        )

    compared = False
    if trial_config_path is not None and trial_config_path.is_file():
        try:
            if load_config_file(destination) != load_config_file(trial_config_path):
                gone = _discard(destination)
                raise ExportRefusedError(
                    [
                        f"the exported config differs from the one trial {record.id} ran "
                        f"({trial_config_path})"
                        + ("" if gone else f" (and the partial file could not be removed)")
                    ]
                )
        except ExportRefusedError:
            raise
        except (OSError, ValueError, yaml.YAMLError):
            # The trial's own file exists but cannot be compared -- including a trial workspace
            # holding YAML that no longer parses: that is a check that could not run, which the
            # receipt records as False rather than claiming it passed.
            compared = False
        else:
            compared = True

    artifact = ExportArtifact(
        run_ref=analysis.run_ref,
        rule=analysis.rule,
        rule_fingerprint=analysis.rule_fingerprint,
        trial_id=record.id,
        index=record.index,
        attempt=record.attempt,
        status=record.status,
        assignment=dict(record.assignment),
        assignment_key=record.assignment_key,
        value=winner.value,
        tied_count=analysis.tied_count,
        objective=record.objective,
        source_config=str(source),
        source_config_fingerprint=analysis.source_config_fingerprint,
        expected_fingerprint=record.injected_fingerprint,
        output_config=str(destination),
        output_fingerprint=report.injected_fingerprint,
        compared_against_trial_file=compared,
        trial_config_path=record.config_path,
        coverage_verdict=analysis.coverage.verdict,
        sample_trust=analysis.sample_trust,
        manifest_path=str(manifest),
    )
    payload = canonical_json(
        {"artifact": artifact.model_dump(mode="json"), "analysis": analysis.model_dump(mode="json")}
    )
    try:
        _write_manifest(manifest, payload + "\n")
    except BaseException:
        # The pair is the artifact: a config without its receipt is exactly what this module
        # exists to prevent, so the config goes too.
        _discard(destination)
        raise
    return artifact


def verify_export(artifact: ExportArtifact) -> str:
    """Re-read the exported config and confirm it is still the one the receipt was written for.

    This is the check a third party runs days later with nothing but the receipt: it needs a YAML
    parser and a hash, not this repository.  Raises :class:`ExportRefusedError` naming what
    failed; returns ``"verified"``.
    """
    path = Path(artifact.output_config)
    if not path.is_file():
        raise ExportRefusedError([f"the exported config is not there: {path}"])
    try:
        actual = content_hash(load_config_file(path))
    except (OSError, ValueError, yaml.YAMLError) as error:
        #: Three families, and ``yaml`` is the one that hides: ``yaml.YAMLError`` derives from
        #: ``Exception``, not from ``ValueError``, so the first version of this function let
        #: ``yaml.parser.ParserError`` escape -- out of the check a third party runs with nothing
        #: but the receipt -- and the caller caught an exception its documented refusal does not
        #: cover.  Measured before the fix, with the exported file overwritten by ``a: [unclosed``.
        raise ExportRefusedError(
            [f"the exported config cannot be read ({path}): {type(error).__name__}: {error}"]
        ) from error
    if actual != artifact.expected_fingerprint:
        raise ExportRefusedError(
            [
                f"the config at {path} hashes to {actual}, but the receipt was written for the "
                f"config the winning trial ran ({artifact.expected_fingerprint}): this file is "
                f"not that config"
            ]
        )
    return "verified"


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "COVERAGE_VERDICTS",
    "EXPORT_SCHEMA_VERSION",
    "MANIFEST_SUFFIX",
    "Coverage",
    "CoverageVerdict",
    "ExportArtifact",
    "ExportRefusedError",
    "RankDirection",
    "RankedTrial",
    "RankingRule",
    "Reading",
    "ReadingState",
    "RunStatus",
    "SampleTrust",
    "SearchAnalysis",
    "SearchAnalysisError",
    "TrialCensus",
    "UnrankedTrial",
    "analyze",
    "export_best",
    "verify_export",
]
