"""The adaptive sampler: the encoder, the posterior, the acquisition, the decisions, and their check.

Every test here pins one way the search can go **quietly** wrong -- wrong without raising, so the
ledger reads normally and the conclusion is simply false.  The failures this file is built around:

* A parameter that never applied and one that applied and took the bottom of its range encode to
  the same vector, so the surrogate learns a mapping the trainer never had.
* A value outside its declared domain is clamped instead of refused, so the model is trained on a
  point nobody ran.
* Observations arrive in whatever order the ledger happened to fold them, so two readings of the
  same evidence choose different next points.
* A pruned trial's truncated objective is learned as if it were a final one.
* The pool runs short and the search reports a full budget, or the budget runs past the space and
  the search reports a complete design.
* A decision is recorded after the point is proposed, so a proposal exists with no evidence behind
  it -- the one thing the record type exists to prevent.
* ``verify_step`` answers "fine" when it cannot recompute at all, which is the answer that makes a
  tampered sequence look verified.
* The policy's ``direction`` is guessed by a default, and every later step chases the wrong end.

What this file deliberately does **not** assert: that the adaptive sampler finds a better optimum
than a random one, or uses fewer trials.  There is no evidence for that here, and the one probe the
design records points the other way (an unvisited corner scores a higher expected improvement than
the true optimum).  A test asserting it would be a claim with nothing behind it.

The two halves of the run are pinned separately on purpose: the arithmetic (groups A-C) is a pure
function with no ledger, and the wiring (groups D-F) is checked against real records, because a
correct kernel feeding a wrong training set is a wrong search with a correct core.

Two guards in the module are **measured** as unreached by this file, and they are named here rather
than left to a reader's guess.  Coverage of the module under ``python -m coverage run --branch
--source=autotuner/research -m pytest tests/autotuner/research/test_bayesian_sampler.py`` at the time
of writing: 481 of 483 statements, 194 of 196 branches, with these two outstanding.

* ``_encode_discrete``'s ``identifies`` branch (``bayesian_sampler.py:316-317``): a value the domain
  check admits whose grid point still does not identify with it.  A sweep over discrete domains from
  ``1e5`` to ``1e15`` produced no such value -- ``contains`` and ``identifies`` both decide
  membership from the same rounded offset, which is why the branch is hard to reach -- but that is
  evidence and not a proof, so the branch stays and is recorded as unverified.  Its neighbour in the
  same function, the index-out-of-range guard, *is* reachable and is pinned by
  ``test_a_discrete_value_the_domain_check_admits_outside_the_grid_is_refused_not_clamped``.
* ``choose_next``'s non-finite-acquisition guard (``bayesian_sampler.py:1221-1225``): every policy
  and evidence combination measured so far produces finite scores.  Whether some extreme length
  scale against a near-singular kernel can produce an infinity is not known either way.

An earlier revision of this paragraph called the ``identifies`` branch "the one guard here with no
test behind it".  That was wrong on both the count and the identification: four lines were
unexecuted when it was measured (the two raises in ``_encode_discrete``, the inactive-parameter skip
in ``_draw_candidate`` and the guard above), tests were added for the two that turned out to be
reachable, and two unrelated ones remain.
"""
from __future__ import annotations

import ast
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, get_args

import pytest
import yaml
from pydantic import ValidationError

from autotuner.research import bayesian_sampler as bayes
from autotuner.research.bayesian_sampler import (
    DECISION_RECORD_TYPE,
    POLICY_RECORD_TYPE,
    SURROGATE_SCHEMA_VERSION,
    Acquisition,
    BayesianPolicy,
    BayesianSampler,
    DecisionMode,
    Direction,
    EncodingBlock,
    EncodingSpec,
    GaussianProcess,
    LedgerSurrogateSink,
    SurrogateAuditError,
    SurrogateDecisionRecord,
    SurrogateError,
    SurrogatePolicyRecord,
    SurrogateSink,
    _cholesky,
    _encode_discrete,
    _pool,
    _standardise,
    _step_rng,
    _training_set,
    bayesian_plan,
    choose_next,
    decision_id,
    expected_improvement,
    policy_for,
    store_policy_record,
    surrogate_policy_record,
    verify_step,
)
from autotuner.research.hyperparameter_sampler import (
    RandomSampler,
    Sampler,
    SamplerKind,
    SearchPlan,
    assignment_key,
    build_sampler,
    replay,
)
from autotuner.research.hyperparameter_search import (
    NON_REPLAYABLE_SAMPLER_KINDS,
    REPLAYABLE_SAMPLER_KINDS,
    SearchError,
    SearchLedger,
    SearchRunRecord,
    SearchTracker,
    TrialOutcome,
    TrialRecord,
    default_run_ref,
)
from autotuner.research.hyperparameter_space import (
    HyperparameterSpec,
    ParameterCondition,
    ParameterConstraint,
    SearchSpaceSchema,
    load_search_space,
)
from autotuner.research.research_ledger import content_hash
from autotuner.research.search_analysis import RankDirection
from autotuner.research.trial_ledger import TrialLedgerIntegrityError, TrialLedgerStore

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "autotuner" / "research" / "bayesian_sampler.py"
RESEARCH_DIR = REPO_ROOT / "autotuner" / "research"
_PRODUCT_DIR = REPO_ROOT / "products" / "taili" / "blind_locomotion"
SPACE_PATH = _PRODUCT_DIR / "hyperparameter_space.yaml"
CONFIG_PATH = _PRODUCT_DIR / "taili_blind_config.yaml"

LEARNING_RATE = "skrl.agent.learning_rate"
KL_THRESHOLD = "skrl.agent.learning_rate_scheduler_kwargs.kl_threshold"
MINI_BATCHES = "skrl.agent.mini_batches"
ROLLOUTS = "skrl.agent.rollouts"


# --- Real product files, small synthetic spaces, and the fixtures the records need --------------


@lru_cache(maxsize=None)
def _taili_space() -> SearchSpaceSchema:
    """The shipped space, read from the product files rather than invented here."""
    return load_search_space(yaml.safe_load(SPACE_PATH.read_text(encoding="utf-8")))


def _taili_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _continuous(
    name: str,
    low: float,
    high: float,
    default: float,
    *,
    log: bool = False,
    grid_points: int | None = None,
) -> HyperparameterSpec:
    return HyperparameterSpec(
        name=name,
        kind="continuous",
        low=low,
        high=high,
        default=default,
        log=log,
        grid_points=grid_points,
    )


def _discrete(name: str, low: float, high: float, step: float, default: float) -> HyperparameterSpec:
    return HyperparameterSpec(
        name=name, kind="discrete", low=low, high=high, step=step, default=default
    )


def _guard_space() -> SearchSpaceSchema:
    """A categorical guard, and a guarded discrete declared after it."""
    return SearchSpaceSchema(
        specs=(
            HyperparameterSpec(name="sched", kind="categorical", choices=("on", "off"), default="on"),
            HyperparameterSpec(
                name="tuned",
                kind="discrete",
                low=1,
                high=2,
                step=1,
                default=1,
                condition=ParameterCondition(parameter="sched", values=("on",)),
            ),
        )
    )


def _guard_continuous_space() -> SearchSpaceSchema:
    """The same shape with a continuous guarded parameter whose range does not start at zero."""
    return SearchSpaceSchema(
        specs=(
            HyperparameterSpec(name="sched", kind="categorical", choices=("on", "off"), default="on"),
            HyperparameterSpec(
                name="tuned",
                kind="continuous",
                low=2.0,
                high=8.0,
                default=2.0,
                condition=ParameterCondition(parameter="sched", values=("on",)),
            ),
        )
    )


def _small_space() -> SearchSpaceSchema:
    """Six combinations, four of them feasible: 6x{12,18,24} minus the non-multiples."""
    return SearchSpaceSchema(
        specs=(_discrete("a", 6, 9, 3, 6), _discrete("b", 12, 24, 6, 12)),
        constraints=(ParameterConstraint(kind="multiple", parameter="b", of="a"),),
    )


def _one_point_space() -> SearchSpaceSchema:
    """A single-point discrete domain, and a one-choice categorical: both degenerate, differently."""
    return SearchSpaceSchema(
        specs=(
            _discrete("a", 4, 4, 1, 4),
            HyperparameterSpec(name="only", kind="categorical", choices=("x",), default="x"),
        )
    )


def _impossible_space() -> SearchSpaceSchema:
    """No multiple of 20 or 28 lands on the b grid, so no draw can satisfy the constraint."""
    return SearchSpaceSchema(
        specs=(_discrete("a", 20, 28, 8, 20), _discrete("b", 24, 96, 24, 24)),
        constraints=(ParameterConstraint(kind="multiple", parameter="b", of="a"),),
    )


def _two_constraint_space() -> SearchSpaceSchema:
    """Two relations, no assignment satisfying both, and every draw breaking at least one.

    ``b`` must be a multiple of ``a`` (only ``a=2, b=4`` survives) and ``c`` a multiple of ``b``
    (no ``c`` in ``{2, 3}`` is a multiple of 4), so the pool always comes up empty.  The relations
    are declared in the reverse of alphabetical order on purpose: a message that lists them in
    declaration order is a different message from one that sorts them, and only a space where the
    two orders differ can tell those apart.
    """
    return SearchSpaceSchema(
        specs=(
            _discrete("a", 2, 3, 1, 2),
            _discrete("b", 4, 5, 1, 4),
            _discrete("c", 2, 3, 1, 2),
        ),
        constraints=(
            ParameterConstraint(kind="multiple", parameter="c", of="b"),
            ParameterConstraint(kind="multiple", parameter="b", of="a"),
        ),
    )


def _feasible_keys(space: SearchSpaceSchema) -> list[str]:
    """Brute force over the declared points, computed independently of the sampler."""
    order = space.sampling_order()
    found: list[dict[str, Any]] = []

    def walk(position: int, partial: dict[str, Any]) -> None:
        if position == len(order):
            if space.is_feasible(partial):
                found.append(dict(partial))
            return
        name = order[position]
        if not space.is_active(name, partial):
            walk(position + 1, partial)
            return
        for value in space.spec(name).grid_values() or ():
            walk(position + 1, {**partial, name: value})

    walk(0, {})
    return [assignment_key(item) for item in found]


def _policy(**overrides: Any) -> BayesianPolicy:
    """A policy that reaches the posterior quickly, so a test does not have to run a whole design."""
    settings: dict[str, Any] = {
        "direction": "maximize",
        "initial_design": 0,
        "min_train": 2,
        "pool_size": 8,
    }
    settings.update(overrides)
    return BayesianPolicy(**settings)


def _plan(
    space: SearchSpaceSchema,
    *,
    policy: BayesianPolicy | None = None,
    seed: int = 5,
    budget: int = 4,
    **overrides: Any,
) -> SearchPlan:
    chosen = policy if policy is not None else _policy(**overrides)
    return bayesian_plan(space, seed=seed, budget=budget, policy=chosen)


def _record(
    space: SearchSpaceSchema,
    plan: SearchPlan,
    index: int,
    assignment: dict[str, Any],
    objective: float | None,
    *,
    status: str = "succeeded",
    attempt: int = 1,
) -> TrialRecord:
    """A trial record as task 4 would write it, with every field the validators demand."""
    injection: dict[str, Any] = {"index": index, "assignment_key": assignment_key(assignment)}
    return TrialRecord(
        id=f"trial:{plan.fingerprint()[:8]}:{index:04d}",
        run_ref=f"search:{plan.fingerprint()[:8]}",
        index=index,
        attempt=attempt,
        assignment=assignment,
        assignment_key=assignment_key(assignment),
        plan=plan.model_dump(mode="json"),
        plan_fingerprint=plan.fingerprint(),
        space_fingerprint=space.fingerprint(),
        status=status,
        run_dir="/tmp/probe-trial" if status == "succeeded" else "",
        config_path="/tmp/probe-trial/trial_config.yaml",
        injection=injection,
        injection_fingerprint=content_hash(injection),
        source_fingerprint="source",
        injected_fingerprint="injected",
        objective=objective,
        stop_reason="the pruner cut it short" if status in {"pruned", "stopped"} else "",
        error="the backend died" if status == "failed" else "",
        started_at=1.0,
        finished_at=2.0,
    )


class _FakeSource:
    """An observation source that filters by fingerprint, like ``SearchLedger`` does.

    It also records the fingerprints it was asked with, so a test can tell "the sampler asked for
    the right evidence" from "it asked for evidence and happened to get some".
    """

    # No knob for the order it hands records back in: it returns them in the order it was given,
    # and the one test that cares about observation order (``_training_set`` folding by index and
    # attempt) pins the *module* sorting them rather than pinning the source scrambling them.  An
    # ``order`` argument that ``observations`` never read was here and has been removed.
    def __init__(self, records: tuple[TrialRecord, ...] = ()):
        self.records = records
        self.calls: list[tuple[str, str]] = []

    def observations(self, *, plan_fingerprint: str, space_fingerprint: str) -> tuple[TrialRecord, ...]:
        self.calls.append((plan_fingerprint, space_fingerprint))
        found = tuple(
            record
            for record in self.records
            if record.plan_fingerprint == plan_fingerprint
            and record.space_fingerprint == space_fingerprint
        )
        return found


class _RecordingSink:
    """A sink that keeps what it was handed, and can be made to fail at one step."""

    def __init__(self, *, fail_at: int | None = None) -> None:
        self.records: list[SurrogateDecisionRecord] = []
        self.fail_at = fail_at

    def record_decision(self, record: SurrogateDecisionRecord) -> None:
        if self.fail_at is not None and record.index == self.fail_at:
            raise SurrogateAuditError(f"the ledger refused step {record.index}")
        self.records.append(record)


class _ScoringRunner:
    """A runner whose objective is a formula over the assignment, so the search has real signal.

    A constant objective would send every later step down the ``fallback`` path, and a test that
    ran the whole loop against it would be pinning "the sampler gives up in the same way N times"
    rather than "the sampler adapts".
    """

    def __init__(self) -> None:
        self.requests: list[Any] = []

    @staticmethod
    def score(assignment: dict[str, Any]) -> float:
        """The mean of every number the assignment carries.

        Written this way rather than by naming three parameters on purpose: the shipped space turns
        parameters on and off with conditions, so a score that indexed a dotted name would fail on
        whichever assignment happened to leave it inactive -- and it would be a different assignment
        from run to run.  What the test needs is any objective that exists for every assignment and
        varies with it; this is that, and it is a pure function of the assignment.
        """
        numbers: list[float] = []

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                for value in node.values():
                    walk(value)
            elif isinstance(node, bool):
                numbers.append(float(node))
            elif isinstance(node, (int, float)):
                numbers.append(float(node))

        walk(assignment)
        return math.fsum(numbers) / len(numbers)

    def preflight(self, request: Any) -> None:
        return None

    def run(self, request: Any) -> TrialOutcome:
        self.requests.append(request)
        return TrialOutcome(status="succeeded", objective=self.score(request.assignment))


def _bayesian_tracker(
    tmp_path: Path,
    *,
    policy: BayesianPolicy | None = None,
    budget: int = 4,
    seed: int = 11,
    runner: Any = None,
    sink: SurrogateSink | None = None,
    max_in_flight: int = 1,
    observations: Any = None,
) -> tuple[SearchTracker, BayesianSampler, TrialLedgerStore]:
    """The sampler wired to the real space, config and store, with the sampler task 4 cannot build.

    ``observations`` defaults to the store's own search view, which is what a real search uses; a
    test that wants to advance the sampler without writing anything passes a source that holds no
    records and a sink that keeps them in memory.
    """
    space = _taili_space()
    plan = _plan(space, policy=policy, seed=seed, budget=budget)
    store = TrialLedgerStore(tmp_path / "ledger")
    run_ref = default_run_ref(plan, space)
    sampler = BayesianSampler(
        space,
        plan,
        observations=SearchLedger(store) if observations is None else observations,
        sink=sink if sink is not None else LedgerSurrogateSink(store, precondition=None),
        run_ref=run_ref,
    )
    tracker = SearchTracker(
        store=store,
        space=space,
        plan=plan,
        source_config=_taili_config(),
        source_config_path=CONFIG_PATH,
        work_root=tmp_path / "work",
        runner=runner if runner is not None else _ScoringRunner(),
        sampler=sampler,
        max_in_flight=max_in_flight,
    )
    return tracker, sampler, store


def _decisions(store: TrialLedgerStore) -> list[SurrogateDecisionRecord]:
    """Every decision on disk, in step order."""
    payloads = store.records(DECISION_RECORD_TYPE)
    return sorted(
        (SurrogateDecisionRecord.model_validate(payload) for payload in payloads.values()),
        key=lambda record: record.index,
    )


# --- A: the encoder ---------------------------------------------------------------------------


def test_the_encoding_covers_every_parameter_in_the_sampling_order():
    """The layout is ``(order, contiguous slices)``, and both are checkable from the record alone.

    A vector whose dimensions do not follow the sampling order cannot be decoded by a reader, and a
    gap or an overlap would put a parameter's value in another parameter's slot without changing
    anything visible.
    """
    space = _taili_space()
    spec = EncodingSpec.for_space(space)

    assert spec.order == space.sampling_order()
    assert tuple(block.name for block in spec.blocks) == spec.order

    cursor = 0
    for block in spec.blocks:
        assert block.start == cursor
        assert block.width >= 1
        cursor = block.start + block.width
    assert spec.width() == cursor
    assert spec.width() == sum(block.width for block in spec.blocks)

    assignment = RandomSampler(space, SearchPlan.for_random(space, seed=4, budget=1)).collect()[0]
    point = spec.encode(space, assignment)
    assert len(point) == spec.width()
    assert all(isinstance(value, float) for value in point)
    assert all(0.0 <= value <= 1.0 for value in point)


def test_the_layout_is_a_pure_function_of_the_space_and_a_recorded_one():
    """Two layouts of one space are equal, and the layout round-trips through its JSON form.

    ``space_fingerprint`` is stored with the layout and compared at encode time: encoding against
    another space's layout produces a vector that still looks like a point of the unit cube.
    """
    space = _taili_space()
    first = EncodingSpec.for_space(space)
    second = EncodingSpec.for_space(space)

    assert first == second
    assert first.schema_version == SURROGATE_SCHEMA_VERSION
    assert first.space_fingerprint == space.fingerprint()
    assert EncodingSpec.model_validate(first.model_dump(mode="json")) == first


def test_a_continuous_value_lands_between_zero_and_one_with_exact_endpoints():
    """Endpoints are pinned exactly, because the kernel measures distances from them."""
    space = SearchSpaceSchema(specs=(_continuous("lr", 1.0e-3, 1.0e-1, 1.0e-2),))
    spec = EncodingSpec.for_space(space)

    assert spec.encode(space, {"lr": 1.0e-3}) == (0.0,)
    assert spec.encode(space, {"lr": 1.0e-1}) == (1.0,)
    assert spec.encode(space, {"lr": 5.05e-2})[0] == pytest.approx(0.5, abs=1e-12)


def test_a_log_scaled_domain_is_encoded_geometrically_not_linearly():
    """A linear position across two decades would put almost every point in the upper decade.

    That is the same reason ``RandomSampler`` draws logarithmically, and the difference is asserted
    against a linear twin of the same domain rather than against a fixed number.
    """
    log_space = SearchSpaceSchema(specs=(_continuous("lr", 1.0e-3, 1.0e-1, 1.0e-2, log=True),))
    plain_space = SearchSpaceSchema(specs=(_continuous("lr", 1.0e-3, 1.0e-1, 1.0e-2),))
    log_spec = EncodingSpec.for_space(log_space)
    plain_spec = EncodingSpec.for_space(plain_space)

    # 1e-2 is the geometric midpoint of [1e-3, 1e-1]: two equal decades on either side.
    assert log_spec.encode(log_space, {"lr": 1.0e-2})[0] == pytest.approx(0.5, abs=1e-12)
    assert plain_spec.encode(plain_space, {"lr": 1.0e-2})[0] == pytest.approx(0.090909, abs=1e-6)
    assert log_spec.encode(log_space, {"lr": 1.0e-2}) != plain_spec.encode(plain_space, {"lr": 1.0e-2})


def test_a_discrete_value_is_its_index_over_the_count_minus_one():
    """Position by index, so the kernel measures distance in *grid steps*, not in raw units."""
    space = _small_space()
    spec = EncodingSpec.for_space(space)
    a_block = next(block for block in spec.blocks if block.name == "a")
    b_block = next(block for block in spec.blocks if block.name == "b")

    assert spec.encode(space, {"a": 6, "b": 12})[a_block.start] == 0.0
    assert spec.encode(space, {"a": 9, "b": 12})[a_block.start] == 1.0
    assert spec.encode(space, {"a": 6, "b": 18})[b_block.start] == pytest.approx(0.5)
    assert spec.encode(space, {"a": 6, "b": 24})[b_block.start] == 1.0


def test_a_discrete_value_the_domain_check_admits_outside_the_grid_is_refused_not_clamped():
    """The index guard, reached by a real hole in ``contains`` rather than argued about.

    ``contains`` decides "is this a point of the grid" with a tolerance scaled by the magnitude of
    the domain's own numbers (``_step_scale``), so a domain starting at a billion admits points a
    whole step outside itself: ``contains(1000000003.0)`` is True for ``[1000000000, 1000000002]``
    with step 1, and so is ``contains(999999999.0)``.  This is where that stops.  Clamping instead
    would train the posterior on a point nobody ran, and the clamp would be invisible: the encoded
    vector would be a perfect ``0.0`` or ``1.0``.

    Measured, not assumed: an earlier revision of this file's header called ``_encode_discrete``'s
    ``identifies`` branch the one guard in the module with no test behind it, and a coverage run put
    three guards in that position -- this one among them.  A sweep over
    magnitudes from ``1e5`` to ``1e15`` (one-step-past values at each) found this guard firing at
    ``1e9`` and at every magnitude from ``1e10`` up; below ``1e9`` the domain check itself refuses
    the value, and at ``1e16`` the spec refuses the domain.  So the window is narrow, real, and
    covered here in both directions.
    """
    spec = HyperparameterSpec(name="x", kind="discrete", low=1e9, high=1e9 + 2, step=1, default=1e9)
    assert spec.discrete_count() == 3

    for over, index in ((1e9 + 3, 3), (1e9 - 1, -1)):
        assert spec.contains(over) is True, "the premise: the domain check lets this one through"
        assert spec.validate_value(over) is None
        with pytest.raises(ValueError) as caught:
            _encode_discrete(spec, over)
        assert f"sits at index {index}, outside 0..2" in str(caught.value)


def test_a_one_point_domain_encodes_to_a_constant_and_carries_no_information():
    """``count == 1`` is 0.0 and a one-choice categorical is 1.0, and neither can be otherwise.

    Recorded because the two degenerate cases look like a bug and are not: a domain with one value
    is expressible, it simply contributes a constant dimension that no posterior can learn from.
    """
    space = _one_point_space()
    spec = EncodingSpec.for_space(space)
    only = next(block for block in spec.blocks if block.name == "only")

    assert spec.width() == 2
    assert spec.encode(space, {"a": 4, "only": "x"}) == (0.0, 1.0)
    assert only.width == 1
    assert only.activity_dim is None


def test_a_categorical_becomes_a_one_hot_that_sums_to_one():
    """One dimension per choice, exactly one of them set: a distance between two choices is the same
    for every pair, which is the honest encoding of "these are not ordered"."""
    space = _guard_space()
    spec = EncodingSpec.for_space(space)
    block = next(block for block in spec.blocks if block.name == "sched")

    on = spec.encode(space, {"sched": "on", "tuned": 1})
    off = spec.encode(space, {"sched": "off"})
    assert on[block.start : block.start + block.width] == (1.0, 0.0)
    assert off[block.start : block.start + block.width] == (0.0, 1.0)


def test_a_guarded_categorical_keeps_its_activity_dimension_off_the_one_hot():
    """The one block shape where "one dimension per choice plus an activity flag" can collide.

    Every other guarded block is one value wide, so the activity dimension sits at ``start + 1``
    whether the width was ``choices + 1`` or ``2`` -- a layout that added the activity dimension
    *before* the choices, or that forgot the ``+ 1`` when measuring the block, would still look
    right.  With three choices the two off-by-ones become visible: an activity flag written onto a
    one-hot dimension is a point the posterior reads as "this parameter was off" when it was on, and
    a one-hot written onto the flag is a value dimension that is never set.
    """
    space = SearchSpaceSchema(
        specs=(
            HyperparameterSpec(name="sched", kind="categorical", choices=("on", "off"), default="on"),
            HyperparameterSpec(
                name="mode",
                kind="categorical",
                choices=("fast", "balanced", "cheap"),
                default="fast",
                condition=ParameterCondition(parameter="sched", values=("on",)),
            ),
        )
    )
    spec = EncodingSpec.for_space(space)
    block = next(block for block in spec.blocks if block.name == "mode")

    assert block.width == 4
    assert block.activity_dim == block.start + 3

    active = spec.encode(space, {"sched": "on", "mode": "cheap"})
    values = active[block.start : block.start + 3]
    assert values == (0.0, 0.0, 1.0)
    assert sum(values) == 1.0
    assert active[block.activity_dim] == 1.0

    absent = spec.encode(space, {"sched": "off"})
    assert absent[block.start : block.start + block.width] == (0.0, 0.0, 0.0, 0.0)
    assert active != absent


def test_an_inactive_parameter_cannot_be_confused_with_one_at_the_bottom_of_its_range():
    """The failure the activity dimension exists to prevent, on both a discrete and a continuous guard.

    Both assignments below give the guarded parameter the *same value dimensions* -- "it took the
    bottom of its range" -- and the vectors still differ, in the activity dimension alone.  Without
    that dimension the two points would be identical and the posterior would be asked to explain two
    different objectives with one input.
    """
    space = _guard_space()
    spec = EncodingSpec.for_space(space)
    block = next(block for block in spec.blocks if block.name == "tuned")

    active = spec.encode(space, {"sched": "on", "tuned": 1})
    inactive = spec.encode(space, {"sched": "off"})
    active_values = active[block.start : block.start + block.width - 1]
    inactive_values = inactive[block.start : block.start + block.width - 1]

    assert block.activity_dim == block.start + block.width - 1
    assert active_values == inactive_values == (0.0,)
    assert active[block.activity_dim] == 1.0
    assert inactive[block.activity_dim] == 0.0
    assert active != inactive

    continuous = _guard_continuous_space()
    continuous_spec = EncodingSpec.for_space(continuous)
    c_block = next(item for item in continuous_spec.blocks if item.name == "tuned")
    at_low = continuous_spec.encode(continuous, {"sched": "on", "tuned": 2.0})
    absent = continuous_spec.encode(continuous, {"sched": "off"})
    assert at_low[c_block.start] == absent[c_block.start] == 0.0
    assert at_low[c_block.activity_dim] == 1.0
    assert absent[c_block.activity_dim] == 0.0


def test_the_activity_dimension_sits_inside_the_block_it_describes():
    """Two different rules, checked on the two things that can break them.

    The **constructor** accepts any dimension inside the block's own slice: "inside" is the
    invariant it can check on its own, since a block does not know whether its parameter is
    conditioned.  ``for_space`` places the dimension **last**, after the value dimensions, so the
    values' own slice stays contiguous -- that placement is a property of the layout and is pinned
    below on real blocks, because an assertion over a block this test constructed itself
    (``activity_dim == start + width - 1`` for a literal ``activity_dim=3``) is arithmetic on two
    numbers the test wrote down and cannot fail.
    """
    inside = EncodingBlock(name="tuned", kind="discrete", start=2, width=2, activity_dim=2)
    assert inside.activity_dim == 2

    with pytest.raises(ValueError) as caught:
        EncodingBlock(name="tuned", kind="discrete", start=2, width=2, activity_dim=4)
    assert "must fall inside the block's own slice" in str(caught.value)

    with pytest.raises(ValueError):
        EncodingBlock(name="", kind="discrete", start=0, width=1)
    with pytest.raises(ValueError):
        EncodingBlock(name="tuned", kind="ordinal", start=0, width=1)
    with pytest.raises(ValueError):
        EncodingBlock(name="tuned", kind="discrete", start=-1, width=1)
    with pytest.raises(ValueError):
        EncodingBlock(name="tuned", kind="discrete", start=0, width=0)

    space = _taili_space()
    spec = EncodingSpec.for_space(space)
    guarded = [block for block in spec.blocks if block.activity_dim is not None]
    assert guarded, "the shipped space has a conditional parameter; this test is about its layout"
    for block in guarded:
        assert block.start <= block.activity_dim < block.start + block.width
        assert block.activity_dim == block.start + block.width - 1


def test_the_layout_refuses_gaps_overlaps_and_a_missing_order():
    """A gap or an overlap places dimensions wrongly while every number still looks like a number."""
    space = _taili_space()
    good = EncodingSpec.for_space(space)

    with pytest.raises(ValueError) as caught:
        EncodingSpec(
            space_fingerprint=good.space_fingerprint,
            order=good.order,
            blocks=good.blocks[1:],
        )
    assert "are not the sampling order" in str(caught.value)

    shifted = good.blocks[:-1] + (
        EncodingBlock(
            name=good.blocks[-1].name,
            kind=good.blocks[-1].kind,
            start=good.blocks[-1].start + 1,
            width=good.blocks[-1].width,
        ),
    )
    with pytest.raises(ValueError) as caught:
        EncodingSpec(space_fingerprint=good.space_fingerprint, order=good.order, blocks=shifted)
    assert "a gap or an overlap" in str(caught.value)

    with pytest.raises(ValueError):
        EncodingSpec(space_fingerprint="", order=good.order, blocks=good.blocks)
    with pytest.raises(ValueError):
        EncodingSpec(space_fingerprint=good.space_fingerprint, order=(), blocks=())


def test_a_value_outside_its_declared_domain_is_refused_rather_than_clamped():
    """A clamp would turn "the ledger holds an illegal value" into "we sampled the nearest legal one".

    The posterior would then be trained on a point nobody ran, and every conclusion drawn from it
    would be about a trial that does not exist.
    """
    space = SearchSpaceSchema(specs=(_continuous("lr", 1.0e-3, 1.0e-1, 1.0e-2, log=True),))
    spec = EncodingSpec.for_space(space)

    with pytest.raises(ValueError) as caught:
        spec.encode(space, {"lr": 1.0})
    assert "lr" in str(caught.value)

    with pytest.raises(ValueError):
        spec.encode(space, {"lr": "not a number"})

    guard = _guard_space()
    guard_spec = EncodingSpec.for_space(guard)
    with pytest.raises(ValueError) as caught:
        guard_spec.encode(guard, {"sched": "maybe"})
    assert "is not one of" in str(caught.value)

    with pytest.raises(ValueError) as caught:
        guard_spec.encode(guard, {"sched": "on"})
    assert "carries no value" in str(caught.value)


def test_a_value_the_domain_check_tolerates_is_clamped_into_the_cube():
    """The clamp's real reason, measured: the domain check accepts a hair outside the bounds.

    ``contains`` allows ``1e-9 * max(1, |low|, |high|)`` -- ``1e-9`` for ``[0, 1]`` -- so
    ``validate_value(-1e-18)`` passes, and the position the encoder computes for it is ``-1e-18``.
    Handing that to the kernel makes a squared-distance term slightly negative.  Both branches of
    the clamp are exercised here, with the raw positions asserted alongside the encoded ones so
    that the test says what it is clamping *from*.
    """
    space = SearchSpaceSchema(specs=(_continuous("p", 0.0, 1.0, 0.5),))
    spec = EncodingSpec.for_space(space)
    low, high = 0.0, 1.0

    below, above = -1.0e-18, 1.0 + 1.0e-15
    for value, raw, expected in ((below, (below - low) / (high - low), 0.0), (above, (above - low) / (high - low), 1.0)):
        space.spec("p").validate_value(value)  # accepted, or this test proves nothing
        assert (raw < 0.0) if expected == 0.0 else (raw > 1.0)
        assert spec.encode(space, {"p": value}) == (expected,)

    # And the far side of the same check, so the clamp is not read as "anything is accepted".
    with pytest.raises(ValueError):
        spec.encode(space, {"p": -1.0e-6})
    with pytest.raises(ValueError):
        spec.encode(space, {"p": 1.0001})


def test_an_encoding_laid_out_for_another_space_or_another_order_is_refused():
    """Both checks are needed: the fingerprint ignores declaration order, so a reordered space
    still matches it while laying its dimensions out differently."""
    forward = SearchSpaceSchema(
        specs=(_continuous("lr", 1.0e-3, 1.0e-1, 1.0e-2), _discrete("a", 6, 9, 3, 6))
    )
    spec = EncodingSpec.for_space(forward)

    other = SearchSpaceSchema(specs=(_discrete("z", 1, 3, 1, 1),))
    with pytest.raises(ValueError) as caught:
        spec.encode(other, {"z": 1})
    assert "was laid out for space" in str(caught.value)

    reversed_space = SearchSpaceSchema(specs=tuple(reversed(forward.specs)))
    assert reversed_space.fingerprint() == forward.fingerprint()
    assert reversed_space.sampling_order() != forward.sampling_order()
    with pytest.raises(ValueError) as caught:
        spec.encode(reversed_space, {"lr": 1.0e-2, "a": 6})
    assert "was laid out for the order" in str(caught.value)


def test_a_continuous_domain_with_no_room_is_refused_by_the_space_itself():
    """``low == high`` cannot be expressed continuously, which is why the encoder has no branch for it.

    A branch for a state the space refuses is a branch no test can execute, and it would suggest the
    encoder handles a case the ledger can never hold.
    """
    with pytest.raises(ValueError) as caught:
        _continuous("lr", 1.0, 1.0, 1.0)
    assert "low must be strictly below high" in str(caught.value)


# --- B: the Gaussian process ------------------------------------------------------------------


def test_the_cholesky_factor_is_lower_triangular_with_a_positive_diagonal():
    """The factor is the whole numerical content of the posterior, and a non-positive pivot is how
    "this kernel matrix is not positive definite" announces itself."""
    lower = _cholesky([[1.0, 0.2], [0.2, 1.0]])
    assert lower[0][1] == 0.0
    assert lower[0][0] == pytest.approx(1.0)
    assert lower[1][1] == pytest.approx(math.sqrt(1.0 - 0.04))
    assert all(lower[i][i] > 0.0 for i in range(len(lower)))

    with pytest.raises(ValueError) as caught:
        _cholesky([[1.0, 1.0], [1.0, 1.0]])
    assert "not positive definite" in str(caught.value)

    with pytest.raises(ValueError):
        _cholesky([[0.0]])


def test_the_posterior_reproduces_the_target_at_a_training_point():
    """Mean in, mean out -- and the variance there is on the order of the noise, not the prior."""
    model = GaussianProcess(length_scale=0.5, noise=1.0e-6, train_x=[[0.0], [1.0]], train_y=[1.0, -1.0])
    mean, variance = model.predict([0.0])

    assert model.points == 2
    assert mean == pytest.approx(1.0, abs=1e-4)
    assert variance == pytest.approx(2.0e-6, abs=1e-6)


def test_the_variance_returns_to_the_prior_far_from_every_training_point():
    """A short length scale makes "far" reachable inside the unit cube, which is why this test uses
    one: with a long length scale every point of a 2-dimensional cube is a near neighbour."""
    model = GaussianProcess(
        length_scale=0.25, noise=1.0e-6, train_x=[[0.0]], train_y=[1.0]
    )
    _, near = model.predict([0.0])
    _, far = model.predict([1.0])

    assert near < 1.0e-5
    assert far == pytest.approx(1.0, abs=1e-3)
    assert far > near * 1000.0


def test_predict_is_bit_for_bit_repeatable_and_not_merely_close():
    """Equality, not ``approx``: a recorded decision is recomputed and compared exactly."""
    rows = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
    model = GaussianProcess(length_scale=0.5, noise=1.0e-6, train_x=rows, train_y=[1.0, 0.5, -1.0])
    other = GaussianProcess(length_scale=0.5, noise=1.0e-6, train_x=rows, train_y=[1.0, 0.5, -1.0])

    assert model.predict([0.25, 0.75]) == model.predict([0.25, 0.75])
    assert model.predict([0.25, 0.75]) == other.predict([0.25, 0.75])


def test_a_single_observation_is_a_posterior():
    """``min_train`` is 2 by default, but one observation must still build a model rather than
    divide by an empty sum: the boundary is 1, and a policy may set min_train to it."""
    model = GaussianProcess(length_scale=0.5, noise=1.0e-6, train_x=[[0.3]], train_y=[2.0])
    mean, variance = model.predict([0.3])
    assert model.points == 1
    assert mean == pytest.approx(2.0, abs=1e-4)
    assert variance >= 0.0


def test_a_duplicate_point_without_noise_has_no_posterior_and_says_so():
    """Two rows on one point and no jitter is a singular kernel matrix.  Nudging the pivot would
    hide it; the honest answer is that no posterior exists, which is what the caller records."""
    with pytest.raises(ValueError) as caught:
        GaussianProcess(length_scale=1.0, noise=0.0, train_x=[[0.0], [0.0]], train_y=[1.0, 2.0])
    assert "not positive definite" in str(caught.value)

    # The same rows are fine once the jitter the policy carries by default is present.
    model = GaussianProcess(length_scale=1.0, noise=1.0e-6, train_x=[[0.0], [0.0]], train_y=[1.0, 2.0])
    mean, _ = model.predict([0.0])
    assert 1.0 <= mean <= 2.0


def test_the_posterior_refuses_rows_that_are_not_one_space_and_targets_that_are_not_finite():
    """A ragged training set is a space mismatch nobody sees; a non-finite target is invisible in a
    mean and poisons every later covariance."""
    with pytest.raises(ValueError) as caught:
        GaussianProcess(length_scale=1.0, noise=1.0e-6, train_x=[[0.0]], train_y=[1.0, 2.0])
    assert "missing a row" in str(caught.value)

    with pytest.raises(ValueError):
        GaussianProcess(length_scale=1.0, noise=1.0e-6, train_x=[], train_y=[])

    with pytest.raises(ValueError) as caught:
        GaussianProcess(length_scale=1.0, noise=1.0e-6, train_x=[[0.0], [0.0, 1.0]], train_y=[1.0, 2.0])
    assert "not all points of one space" in str(caught.value)

    with pytest.raises(ValueError) as caught:
        GaussianProcess(length_scale=1.0, noise=1.0e-6, train_x=[[0.0]], train_y=[float("nan")])
    assert "must be finite" in str(caught.value)

    with pytest.raises(ValueError):
        GaussianProcess(length_scale=0.0, noise=1.0e-6, train_x=[[0.0]], train_y=[1.0])
    with pytest.raises(ValueError):
        GaussianProcess(length_scale=float("inf"), noise=1.0e-6, train_x=[[0.0]], train_y=[1.0])
    with pytest.raises(ValueError):
        GaussianProcess(length_scale=1.0, noise=-1.0, train_x=[[0.0]], train_y=[1.0])


def test_the_posterior_refuses_a_query_point_of_the_wrong_width():
    """A point of another space is a number in the cube like any other, so the width is checked."""
    model = GaussianProcess(length_scale=1.0, noise=1.0e-6, train_x=[[0.0, 1.0]], train_y=[1.0])
    with pytest.raises(ValueError) as caught:
        model.predict([0.0])
    assert "not 1" in str(caught.value)


def test_the_kernel_falls_with_distance_and_a_longer_length_scale_falls_more_slowly():
    """The one knob the module does not fit is the one the model's smoothness rests on, so its
    effect is pinned rather than assumed."""
    near = GaussianProcess(length_scale=0.5, noise=1.0e-6, train_x=[[0.0]], train_y=[1.0])
    wide = GaussianProcess(length_scale=2.0, noise=1.0e-6, train_x=[[0.0]], train_y=[1.0])

    assert near._kernel([0.0], [0.0]) == 1.0
    assert near._kernel([0.0], [0.25]) > near._kernel([0.0], [0.5]) > near._kernel([0.0], [1.0])
    assert wide._kernel([0.0], [0.5]) > near._kernel([0.0], [0.5])
    assert near.points == 1


def _raw_variance(
    rows: list[list[float]],
    targets: list[float],
    *,
    length_scale: float,
    noise: float,
    point: list[float],
) -> float:
    """The posterior variance recomputed from scratch, **without** the floor ``predict`` applies.

    Written here rather than assembled from the module's own helpers: a check built out of the code
    under test cannot see a mistake the two share, and the whole point of this number is to disagree
    with ``predict`` whenever the floor is not exactly ``max(raw, 0)``.
    """
    def kernel(left: list[float], right: list[float]) -> float:
        return math.exp(
            -0.5 * math.fsum(((a - b) / length_scale) ** 2 for a, b in zip(left, right))
        )

    count = len(rows)
    gram = [
        [kernel(rows[i], rows[j]) + (noise if i == j else 0.0) for j in range(count)]
        for i in range(count)
    ]
    low = [[0.0] * count for _ in range(count)]
    for i in range(count):
        for j in range(i + 1):
            total = gram[i][j] - math.fsum(low[i][k] * low[j][k] for k in range(j))
            low[i][j] = math.sqrt(total) if i == j else total / low[j][j]
    cross = [kernel(row, point) for row in rows]
    forward = [0.0] * count
    for i in range(count):
        forward[i] = (cross[i] - math.fsum(low[i][j] * forward[j] for j in range(i))) / low[i][i]
    backward = [0.0] * count
    for i in reversed(range(count)):
        backward[i] = (
            forward[i] - math.fsum(low[j][i] * backward[j] for j in range(i + 1, count))
        ) / low[i][i]
    return 1.0 + noise - math.fsum(cross[i] * backward[i] for i in range(count))


def test_the_variance_the_posterior_returns_is_the_raw_one_floored_at_zero():
    """The floor, checked against a variance recomputed here instead of against itself.

    An earlier version of this test swept the cube and asserted ``min(variances) >= 0.0`` -- which
    **cannot fail**, because ``predict`` floors whatever it returns, so the assertion was true by
    construction and said nothing about the floor.  What is worth pinning is that the returned
    number is exactly ``max(raw, 0)`` over a grid that does contain raw negatives: the collinear
    triple below leaves some of these probes at ``-2.22e-16``, which is what makes this a floor
    rather than a decoration.
    """
    rows = [[0.0, 0.0], [0.0, 0.25], [0.0, 0.5]]
    targets = [0.0, 1.0, 0.5]
    model = GaussianProcess(length_scale=0.05, noise=0.0, train_x=rows, train_y=targets)

    probes = [[x / 16.0, y / 16.0] for x in range(17) for y in range(17)] + rows
    raws = [
        _raw_variance(rows, targets, length_scale=0.05, noise=0.0, point=point)
        for point in probes
    ]
    returned = [model.predict(point)[1] for point in probes]

    assert min(raws) < 0.0, "this grid no longer reaches a negative raw variance; re-measure it"
    assert returned == [max(raw, 0.0) for raw in raws]
    assert all(math.isfinite(value) for value in returned)


def test_the_variance_floors_at_zero_on_the_case_that_would_otherwise_go_negative():
    """The floor's reachability, measured rather than argued: no input above is known to push the
    raw variance below zero, so the case that does is pinned here by hand.

    Three collinear points, ``noise=0.0`` and a short length scale leave the raw sum at
    ``-2.22e-16`` -- the ULP of the terms that cancel -- at the middle training point.  Without the
    floor that number reaches ``sqrt`` as a negative and the step dies; with it, the answer is
    ``0.0`` exactly, which is the honest statement that the model knows this point.
    """
    model = GaussianProcess(
        length_scale=0.05,
        noise=0.0,
        train_x=[[0.0, 0.0], [0.0, 0.25], [0.0, 0.5]],
        train_y=[0.0, 1.0, 0.5],
    )
    mean, variance = model.predict([0.0, 0.25])
    assert variance == 0.0
    assert mean == pytest.approx(1.0, abs=1e-9)
    # The three points are the same three the design's probe used; the query sits on one of them.
    assert model.predict([0.0, 0.0])[1] == 0.0
    assert model.predict([0.0, 0.5])[1] == 0.0


# --- C: the acquisition ------------------------------------------------------------------------


def test_expected_improvement_matches_the_hand_computed_value():
    """The numbers the design's probe produced, to six decimals.

    They are the reason the arithmetic is checked at all: the posterior mean and variance feeding
    this call came from the same formulas, so a shared mistake would cancel out in a
    self-consistency test and not here.
    """
    optimum = expected_improvement(-0.109735, math.sqrt(0.471054), -0.1725, xi=0.01)
    corner = expected_improvement(-0.276399, math.sqrt(0.896901), -0.1725, xi=0.01)

    assert optimum == pytest.approx(0.300999, abs=1.0e-6)
    assert corner == pytest.approx(0.323597, abs=1.0e-6)


def test_an_unvisited_corner_can_outrank_the_true_optimum_which_is_why_nothing_claims_efficacy():
    """The probe's conclusion, pinned so it cannot be quietly forgotten.

    ``corner`` is a point the search has never run and, in the probe, is not the best point either;
    it wins on uncertainty alone.  That is correct behaviour for expected improvement and it is also
    the reason no test in this file asserts that the adaptive sampler beats a random one.

    The means are named rather than inlined because the sentence that carries the point -- "the
    corner's mean is the worse one, and it still wins" -- is a claim about *these* arguments.  As
    ``assert -0.276399 < -0.109735`` it compared two literals, which no edit to the module could
    make false; as written here, changing either mean into one that does not show the effect fails
    on the premise line instead of passing silently.
    """
    optimum_mean, optimum_sigma = -0.109735, math.sqrt(0.471054)
    corner_mean, corner_sigma = -0.276399, math.sqrt(0.896901)
    best = -0.1725

    assert corner_mean < optimum_mean, "the corner's mean must be the worse one to say anything"
    assert corner_sigma > optimum_sigma, "and it must win on uncertainty, which is the whole claim"

    optimum = expected_improvement(optimum_mean, optimum_sigma, best, xi=0.01)
    corner = expected_improvement(corner_mean, corner_sigma, best, xi=0.01)

    assert corner > optimum


def test_a_point_the_model_knows_exactly_cannot_improve_on_anything():
    """``sigma <= 0`` returns 0.0 rather than dividing by zero, at both zero and a negative."""
    assert expected_improvement(1.0, 0.0, 0.0, xi=0.01) == 0.0
    assert expected_improvement(1.0, -0.5, 0.0, xi=0.01) == 0.0
    assert expected_improvement(-100.0, 0.0, 0.0, xi=0.0) == 0.0


def test_the_expected_improvement_rises_with_the_mean():
    values = [expected_improvement(mean, 0.4, 0.0, xi=0.01) for mean in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    assert values == sorted(values)
    assert values[0] >= 0.0


def test_the_expected_improvement_rises_with_uncertainty_below_the_best():
    """Below the incumbent, uncertainty is the only reason to look: a point we are sure is worse is
    worth nothing, and an uncertain one is worth the chance."""
    below = [expected_improvement(-0.5, sigma, -0.2, xi=0.0) for sigma in (0.05, 0.2, 0.5, 1.0)]
    assert below == sorted(below)
    assert below[0] < below[-1]


def test_the_margin_xi_discounts_the_incumbent():
    """``xi`` is the price of moving away from the best point seen, so it can only lower the value
    of a point that is exactly at the incumbent and of every point below it."""
    at_incumbent = [expected_improvement(0.0, 0.5, 0.0, xi=xi) for xi in (0.0, 0.05, 0.2)]
    assert at_incumbent == sorted(at_incumbent, reverse=True)
    assert at_incumbent[0] > 0.0

    below = [expected_improvement(-0.5, 0.4, 0.0, xi=xi) for xi in (0.0, 0.05, 0.2)]
    assert below == sorted(below, reverse=True)


def test_the_expected_improvement_is_never_meaningfully_negative_and_is_exactly_reproducible():
    """Not "never negative": the two terms cancel to a rounding error where the true value is
    below what the arithmetic can resolve, so the sum lands a fraction of an ULP below zero.

    Measured on the grid this test builds (``mean = k/10`` for ``k`` in ``-20..20``, ``sigma = j/50``
    for ``j`` in ``0..50``, ``best=0.0``, ``xi=0.01``; 2091 samples) the smallest value is
    ``-2.5131000553498602e-17``, at ``mean=-1.8, sigma=0.22``, and the largest is
    ``1.9987209215492046``.  A wider sweep with the same ``best`` and ``xi``
    (``mean = k/10`` for ``k`` in ``-400..400``, ``sigma = j/100`` for ``j`` in ``0..200``; 161001
    samples) goes to ``-4.1715443090332904e-16`` at ``mean=-15.9, sigma=1.93``.  Both are the ULP
    scale of the terms that cancel, and in both cases the true expectation is a positive number
    below the resolution of a double, so the sign of the result is not information.

    (An earlier revision of this docstring quoted ``-1.85e-16`` at ``mean=-7.9, sigma=0.96`` as
    the minimum of *this* grid.  That value is real -- it is what ``expected_improvement`` returns
    at that point -- but the point is not in this grid, whose ``mean`` only spans ``[-2, 2]``, so
    the sentence attributed a wider sweep's number to the grid under test.  Replaced with the two
    measurements above, both reproducible from the grids written here.)

    That is why the assertion is a bound at the rounding scale rather than ``>= 0.0``: claiming
    exact non-negativity would be a claim the formula does not make and cannot make.
    """
    grid = [mean / 10.0 for mean in range(-20, 21)]
    values = [expected_improvement(mean, sigma / 50.0, 0.0, xi=0.01) for mean in grid for sigma in range(0, 51)]
    assert min(values) > -1.0e-15
    assert min(values) < 0.0, "the cancellation this test bounds is not present; re-measure it"
    assert max(values) > 0.0
    assert all(math.isfinite(value) for value in values)
    assert values == [
        expected_improvement(mean, sigma / 50.0, 0.0, xi=0.01) for mean in grid for sigma in range(0, 51)
    ]


# --- D: the sampler's decisions ----------------------------------------------------------------


def test_the_sampler_fills_its_budget_and_reports_itself_exhausted():
    """``exhausted`` is the difference between "this was the whole space" and "this was the start
    of it", and a caller reads it to decide whether to keep going."""
    space = _small_space()
    plan = _plan(space, budget=4)
    sampler = BayesianSampler(space, plan, observations=_FakeSource())

    drawn = sampler.collect()
    keys = [assignment_key(item) for item in drawn]

    assert len(keys) == 4
    assert len(set(keys)) == 4
    assert set(keys) == set(_feasible_keys(space))
    assert sampler.stats.proposals == 4
    assert sampler.stats.exhausted is True


def test_a_budget_past_the_end_of_the_space_refuses_and_names_both_causes():
    """A short *search* reads as a complete one, so the budget is not quietly truncated.

    The message has to name both causes, because they call for different responses: a constraint
    that keeps rejecting draws means the space is wrong, while repeated draws mean the space is
    exhausted and the attempt budget is not the problem.
    """
    space = _small_space()
    policy = _policy(pool_size=1, max_attempts=200)
    plan = _plan(space, policy=policy, budget=5)
    sampler = BayesianSampler(space, plan, observations=_FakeSource())

    with pytest.raises(SurrogateError) as caught:
        sampler.collect()
    message = str(caught.value)

    assert sampler.stats.proposals == 4
    assert sampler.stats.exhausted is False
    assert "broke a constraint" in message
    assert "b must be a whole multiple of a" in message
    assert "repeated an assignment already proposed" in message
    assert "200 attempt(s)" in message


def test_a_pool_holds_only_feasible_unseen_candidates_and_counts_its_rejections():
    """The pool is the set of points a step may choose from, so a feasible point missing from it is
    a point the sampler cannot ever propose."""
    space = _small_space()
    policy = _policy(pool_size=8)
    plan = _plan(space, policy=policy)
    seen = {assignment_key({"a": 6, "b": 12})}

    pool, counts = _pool(space, plan, policy, index=1, seen=seen)
    keys = [assignment_key(candidate) for candidate in pool]

    assert set(keys) == set(_feasible_keys(space)) - seen
    assert len(keys) == len(set(keys))
    assert counts["attempts"] >= len(keys)
    assert counts["duplicate_draws"] >= 1
    assert counts["infeasible_draws"] >= 1
    assert counts["constraint:b must be a whole multiple of a"] == counts["infeasible_draws"]
    for candidate in pool:
        assert space.is_feasible(candidate)


def test_a_draw_leaves_out_the_parameters_whose_condition_is_off():
    """The drawing half of the conditional-parameter rule, which nothing else here reaches.

    Every pool in this file is drawn from a space with no condition, so the branch in
    ``_draw_candidate`` that skips an inactive parameter was never executed -- measured, not
    guessed: coverage put it among the module's unexecuted lines.  Without it a draw would carry a
    value for ``tuned`` even when ``sched`` is off, and every such candidate would be rejected as
    infeasible, so the pool would quietly shrink to the "sched is on" half of the space while the
    counts blamed a constraint that does not exist.
    """
    space = _guard_space()
    policy = _policy(pool_size=12, max_attempts=200)
    plan = _plan(space, policy=policy)

    pool, _ = _pool(space, plan, policy, index=1, seen=set())

    assert pool, "an empty pool would make this test vacuous"
    off = [candidate for candidate in pool if candidate["sched"] == "off"]
    on = [candidate for candidate in pool if candidate["sched"] == "on"]
    assert off and on, "both branches of the condition must appear, or the skip is untested"
    for candidate in off:
        assert "tuned" not in candidate
    for candidate in on:
        assert candidate["tuned"] in (1, 2)


def test_a_pool_emptied_by_duplicates_alone_names_duplicates_and_not_a_constraint():
    """The other cause, on its own, because the message is what a reader acts on.

    "Every draw broke a constraint" sends the reader to the space; "every draw repeated an
    assignment already proposed" sends them to the budget.  A space with no constraints at all, and
    every assignment in it already recorded, is the case that separates the two -- and it is the
    case that needs ``_describe_draws`` to say nothing about constraints rather than say "0 of 25".
    """
    space = _one_point_space()
    policy = _policy(pool_size=4, max_attempts=25)
    plan = _plan(space, policy=policy)
    only = {"a": 4, "only": "x"}

    assert _feasible_keys(space) == [assignment_key(only)], "the premise of the test, not an assumption"

    with pytest.raises(SurrogateError) as caught:
        choose_next(space, plan, policy, index=1, train=(), seen={assignment_key(only)})
    message = str(caught.value)

    assert "25 attempt(s)" in message
    assert "25 of 25 repeated an assignment already proposed" in message
    assert "broke a constraint" not in message


def test_a_short_pool_is_allowed_and_explained_unlike_a_short_search():
    """``RandomSampler`` refuses to return fewer trials than asked; a pool does not, because "only
    three unseen points are left" is a fact the caller needs rather than a truncation to hide."""
    space = _small_space()
    policy = _policy(pool_size=10)
    plan = _plan(space, policy=policy)

    pool, counts = _pool(space, plan, policy, index=1, seen=set())
    assert len(pool) == 4
    assert counts["duplicate_draws"] >= 1

    # The contrast, measured rather than described: the same space, asked for five trials by the
    # sampler whose docstring promises not to truncate, refuses instead of returning four.
    with pytest.raises(ValueError) as caught:
        RandomSampler(space, SearchPlan.for_random(space, seed=1, budget=5)).collect()
    assert "5" in str(caught.value)


def test_an_impossible_space_is_reported_as_a_constraint_and_not_as_a_tired_sampler():
    """The two causes are told apart by which counter moved, which is the only way a reader can
    decide whether raising ``max_attempts`` would help."""
    space = _impossible_space()
    policy = _policy(pool_size=1, max_attempts=50)
    plan = _plan(space, policy=policy)

    with pytest.raises(SurrogateError) as caught:
        choose_next(space, plan, policy, index=1, train=(), seen=set())
    message = str(caught.value)

    assert "must be a whole multiple of a" in message
    assert "repeated an assignment" not in message
    assert "50 attempt(s)" in message


def test_each_of_several_constraints_is_named_with_its_own_share_of_the_draws():
    """With two relations rejecting draws, the per-relation count is what separates "this space is
    wrong" from "this space is narrow" -- the same reason the single-relation case names its cause.

    The listing is asserted as one exact substring, which pins three things at once: the relations
    are listed in sorted order rather than the order they were declared in, each carries its own
    share of the infeasible draws, and the shares are separated.  The numbers (37 of 50 at the
    first relation, 50 of 50 at the second) come from the stream ``seed=5`` produces for this space
    and policy: they are as reproducible as the plan's fingerprint, which is what the stream is
    keyed on, so a change to ``SearchPlan``'s fields re-measures them.
    """
    space = _two_constraint_space()
    policy = _policy(pool_size=1, max_attempts=50)
    plan = _plan(space, policy=policy)

    with pytest.raises(SurrogateError) as caught:
        choose_next(space, plan, policy, index=1, train=(), seen=set())
    message = str(caught.value)

    assert (
        "50 of 50 broke a constraint: b must be a whole multiple of a (37 of 50 infeasible draws), "
        "c must be a whole multiple of b (50 of 50 infeasible draws)"
    ) in message


def test_the_initial_design_takes_the_first_candidate_without_building_a_posterior():
    """Before it has seen anything the surrogate has nothing to say, and it must say so in the
    record rather than by scoring every candidate zero and calling that a decision."""
    space = _small_space()
    policy = _policy(initial_design=2)
    plan = _plan(space, policy=policy)
    train = [
        (_record(space, plan, 1, {"a": 6, "b": 12}, 1.0), 1.0),
    ]

    choice = choose_next(space, plan, policy, index=2, train=train, seen=set())

    assert choice.mode == "initial_design"
    assert choice.chosen_key == choice.pool[0]
    # Not ``assignment_key(choice.chosen) == choice.chosen_key``, which stood here and proved
    # nothing: ``chosen_key`` is a property that *is* ``assignment_key(self.chosen)``, so comparing
    # the two compares a value with itself.  The line above is the load-bearing one -- ``pool`` holds
    # every candidate and ``chosen`` holds the pick, so it fails if the step ever takes anything but
    # the first.
    assert space.is_feasible(choice.chosen)
    assert choice.chosen_key not in {record.assignment_key for record, _ in train}
    assert choice.chosen_ei is None
    assert choice.runner_up_ei is None
    assert "within the initial design of 2" in choice.detail


def test_min_train_binds_when_the_design_is_over_but_the_evidence_is_thin():
    """The two knobs count different things -- steps and observations -- and a search whose early
    trials failed or were pruned hits this one with the design long past."""
    space = _small_space()
    policy = _policy(initial_design=0, min_train=2)
    plan = _plan(space, policy=policy)
    train = [(_record(space, plan, 1, {"a": 6, "b": 12}, 1.0), 1.0)]

    choice = choose_next(space, plan, policy, index=4, train=train, seen=set())

    assert choice.mode == "initial_design"
    assert "only 1 scored observation(s), below min_train=2" in choice.detail
    assert choice.train_keys == (assignment_key({"a": 6, "b": 12}),)


def test_a_step_with_enough_evidence_consults_the_posterior_and_records_what_it_scored():
    """The acquisition value is recorded with the decision, so "won outright" and "won by a hair"
    are distinguishable afterwards -- and ``None`` never stands in for a real zero."""
    space = _small_space()
    policy = _policy()
    plan = _plan(space, policy=policy)
    train = [
        (_record(space, plan, 1, {"a": 6, "b": 12}, 1.0), 1.0),
        (_record(space, plan, 2, {"a": 6, "b": 18}, 4.0), 4.0),
    ]

    choice = choose_next(space, plan, policy, index=3, train=train, seen=set())

    assert choice.mode == "surrogate"
    assert choice.chosen_ei is not None
    assert choice.runner_up_ei is not None
    # Strictly greater, where ``>=`` stood before.  ``>=`` is satisfied by a step that recorded the
    # runner-up's score as the winner's -- the exact confusion the field pair exists to expose -- so
    # it pinned nothing.  These four candidates are not tied: measured 0.3981479087225951 for the
    # winner against 0.10268566477970706 for the runner-up, and the two values are pinned to six
    # decimals in ``test_expected_improvement_matches_the_hand_computed_value``.
    assert choice.chosen_ei > choice.runner_up_ei
    assert len(choice.pool) == 4
    assert choice.chosen_key in choice.pool
    assert "expected_improvement, maximize" in choice.detail


def test_a_pool_of_one_records_a_real_zero_and_no_runner_up():
    """The two ways a value can be absent, on the step that produces both at once.

    A one-candidate pool still consults the posterior.  ``seen`` is empty here so that a point the
    model was trained on is a legal draw, and with ``pool_size=1`` that draw is the whole pool: the
    acquisition value of a point the model knows exactly is ``0.0`` -- a number the acquisition
    really returns, not a placeholder -- while a pool with no second candidate has no runner-up.
    Defaulting the first to ``None`` or the second to ``0.0`` would make "the best candidate scored
    nothing" and "nothing was scored" the same record, which is the distinction ``chosen_ei`` is
    optional for.

    The record is built rather than only inspected: ``chosen_ei=0.0`` is the value the validator's
    ``is None`` check has to let through, and a truthiness test there would reject it.
    """
    space = _small_space()
    policy = _policy(pool_size=1)
    plan = _plan(space, policy=policy)
    train = [
        (_record(space, plan, 1, {"a": 6, "b": 12}, 1.0), 1.0),
        (_record(space, plan, 2, {"a": 6, "b": 18}, 4.0), 4.0),
    ]

    choice = choose_next(space, plan, policy, index=3, train=train, seen=set())

    assert choice.mode == "surrogate"
    assert len(choice.pool) == 1
    assert choice.chosen_key == choice.pool[0]
    assert choice.chosen_key in {record.assignment_key for record, _ in train}, (
        "the single candidate has to be a training point; otherwise this is not the zero case"
    )
    assert choice.chosen_ei == 0.0
    assert choice.runner_up_ei is None

    record = SurrogateDecisionRecord.build(
        run_ref="search:test", plan=plan, space=space, policy=policy, index=3, choice=choice
    )
    assert record.pool_size == 1
    assert record.chosen_ei == 0.0 and record.chosen_ei is not None
    assert record.runner_up_ei is None


def test_a_pool_where_every_candidate_ties_takes_the_first_one_in_pool_order():
    """The tie-break rule, which is otherwise only visible by reading ``choose_next``.

    ``noise=0.0`` and a pool made entirely of the points the model was fitted on is a real
    configuration, not a contrived one: with no observation noise the posterior reproduces the
    target at a training point exactly, and the expected improvement of "the value we already have"
    is ``0.0``.  Every candidate here scores that same ``0.0``, so the step is decided entirely by
    the rule for equal scores -- the first one in pool order wins, and the runner-up records the
    same number rather than ``None``: a tie is not a pool of one.

    The assertion is on ``pool[0]`` rather than on any particular assignment, because the pool's
    order is not sorted and cannot be predicted from the space.  That is what makes this
    load-bearing: an implementation that scanned for the best score with ``max()`` over an
    ``(ei, key)`` pair, or that re-sorted the pool before scoring it, would still return *a*
    candidate and would still report a tie, but not this one.
    """
    space = _small_space()
    policy = _policy(min_train=1, pool_size=8, noise=0.0)
    plan = _plan(space, policy=policy)
    train = [
        (_record(space, plan, index, assignment, float(index)), float(index))
        for index, assignment in enumerate(
            ({"a": 6, "b": 12}, {"a": 6, "b": 18}, {"a": 6, "b": 24}, {"a": 9, "b": 18}), start=1
        )
    ]
    trained = {record.assignment_key for record, _ in train}

    choice = choose_next(space, plan, policy, index=5, train=train, seen=set())

    assert len(choice.pool) == len(_feasible_keys(space)), (
        "the tie is only the whole story if every candidate ties"
    )
    assert set(choice.pool) == trained, (
        "every candidate has to be a point the model reproduces exactly; otherwise the scores "
        "differ and the rule under test is never applied"
    )
    assert choice.chosen_ei == 0.0
    assert choice.runner_up_ei == 0.0, "a tie has a runner-up that scored the same, not no runner-up"
    assert choice.chosen_key == choice.pool[0]


def test_a_constant_objective_goes_to_the_fallback_and_says_the_surrogate_had_nothing_to_fit():
    """Every scored trial coming out the same is a real outcome, not an error: the model has no
    variation to fit, so the step takes the first candidate and records why.

    The record is built and audited here as well as inspected: a ``fallback`` is the one mode whose
    ``chosen_ei`` is legitimately ``None``, so it is the mode that exercises the validator's
    "an acquisition value with no posterior behind it" branch and the verifier's rebuild of a
    choice nothing was scored for.
    """
    space = _small_space()
    policy = _policy()
    plan = _plan(space, policy=policy)
    train = [
        (_record(space, plan, 1, {"a": 6, "b": 12}, 2.0), 2.0),
        (_record(space, plan, 2, {"a": 6, "b": 18}, 2.0), 2.0),
    ]
    seen = {record.assignment_key for record, _ in train}

    choice = choose_next(space, plan, policy, index=3, train=train, seen=seen)

    assert choice.mode == "fallback"
    assert choice.chosen_key == choice.pool[0]
    assert choice.chosen_ei is None
    assert "the same objective" in choice.detail
    assert "no variation to fit" in choice.detail

    record = SurrogateDecisionRecord.build(
        run_ref="search:test", plan=plan, space=space, policy=policy, index=3, choice=choice
    )
    assert record.mode == "fallback"
    assert record.chosen_ei is None and record.runner_up_ei is None
    assert record.train_count == 2
    assert (
        verify_step(
            record,
            plan=plan,
            prefix=tuple(item for item, _ in train),
            space=space,
            policy=policy,
        )
        == "verified"
    )


def test_a_posterior_that_cannot_be_built_falls_back_with_the_reason_attached():
    """A policy that cannot build a posterior at all would fail on every later step; raising would
    end the search where the caller would otherwise have got a random design."""
    space = _small_space()
    policy = _policy(noise=0.0)
    plan = _plan(space, policy=policy)
    duplicated = {"a": 6, "b": 12}
    train = [
        (_record(space, plan, 1, duplicated, 1.0), 1.0),
        (_record(space, plan, 2, duplicated, 2.0), 2.0),
    ]

    choice = choose_next(space, plan, policy, index=3, train=train, seen=set())

    assert choice.mode == "fallback"
    assert "no posterior could be built from 2 observation(s)" in choice.detail
    assert "not positive definite" in choice.detail


def test_the_pool_of_one_step_does_not_depend_on_how_many_draws_the_steps_before_it_made():
    """One stream per step, derived from the plan and the step number -- not one stream for the run.

    A shared stream would make step 5's pool a function of how many random numbers steps 1-4
    happened to consume, which is a function of the space and the evidence and nothing anybody
    recorded.  ``seen`` is held equal across the calls: it legitimately differs between steps, and
    holding it equal is the only way to isolate the stream.
    """
    space = _small_space()
    policy = _policy(pool_size=4)
    plan = _plan(space, policy=policy, budget=5)

    direct = _pool(space, plan, policy, index=5, seen=set())
    for index in (1, 2, 3, 4):
        _pool(space, plan, policy, index=index, seen=set())
    after = _pool(space, plan, policy, index=5, seen=set())

    assert direct == after
    assert direct[0], "the pool is not empty, or this test proves nothing"
    assert _step_rng(plan, 5).random() == _step_rng(plan, 5).random()

    # And the step number is part of the derivation: another step is another pool.
    other = _pool(space, plan, policy, index=4, seen=set())
    assert other[0] != direct[0] or other[1] != direct[1]


def test_the_same_seed_under_another_plan_is_another_stream():
    """The plan's fingerprint is part of the derivation, not decoration: two searches that share a
    seed but differ in policy or budget are different experiments, and sharing a pool stream would
    tie their draws together for no reason a reader could see.

    Measured on the streams themselves *and* on the pool they produce, so the claim is not pinned
    to a private function: a stream that differed while the pools did not would be a stream that
    does not matter.
    """
    space = _small_space()
    one = _plan(space, policy=_policy(direction="maximize", pool_size=4), seed=5, budget=4)
    other_policy = _plan(space, policy=_policy(direction="minimize", pool_size=4), seed=5, budget=4)
    other_budget = _plan(space, policy=_policy(direction="maximize", pool_size=4), seed=5, budget=9)
    policy = _policy(direction="maximize", pool_size=4)

    def pools(plan: SearchPlan) -> list[tuple[str, ...]]:
        return [
            tuple(
                assignment_key(item)
                for item in _pool(space, plan, policy, index=index, seen=set())[0]
            )
            for index in (1, 2, 3, 4)
        ]

    assert len({one.fingerprint(), other_policy.fingerprint(), other_budget.fingerprint()}) == 3
    assert [_step_rng(one, i).random() for i in (1, 2, 3)] != [
        _step_rng(other_policy, i).random() for i in (1, 2, 3)
    ]

    # Over four steps rather than one: the pool of a step holds every feasible point of this small
    # space in draw order, so two different streams can agree on one step's order by chance.  The
    # streams differ at every step; the pools differ over the four.
    ours, theirs, longer = pools(one), pools(other_policy), pools(other_budget)
    assert ours != theirs
    assert ours != longer
    for step in range(4):
        assert sorted(ours[step]) == sorted(theirs[step]) == sorted(longer[step])


def test_seen_covers_every_status_while_train_covers_only_the_statuses_the_policy_learns_from():
    """A pruned or failed trial is a point already spent -- proposing it again wastes the budget --
    while its objective may be exactly the thing that must not be regressed against."""
    space = _small_space()
    policy = _policy()
    plan = _plan(space, policy=policy)
    prefix = (
        _record(space, plan, 1, {"a": 6, "b": 12}, 1.0, status="succeeded"),
        _record(space, plan, 2, {"a": 6, "b": 18}, 9.0, status="pruned"),
        _record(space, plan, 3, {"a": 9, "b": 18}, None, status="failed"),
    )

    train = _training_set(prefix, policy=policy, before=4)
    trained = tuple(record.assignment_key for record, _ in train)
    seen = {record.assignment_key for record in prefix if record.index < 4}

    assert trained == (assignment_key({"a": 6, "b": 12}),)
    assert len(seen) == 3
    pool, _ = _pool(space, plan, policy, index=4, seen=seen)
    assert {assignment_key(candidate) for candidate in pool} == {assignment_key({"a": 6, "b": 24})}


def test_a_policy_that_learns_from_pruned_trials_takes_them_into_the_model():
    """The same evidence, read the other way, because it is the policy that decides and the record
    that has to say which policy decided."""
    space = _small_space()
    policy = _policy(learn_from=("succeeded", "pruned"))
    plan = _plan(space, policy=policy)
    prefix = (
        _record(space, plan, 1, {"a": 6, "b": 12}, 1.0, status="succeeded"),
        _record(space, plan, 2, {"a": 6, "b": 18}, 9.0, status="pruned"),
    )

    train = _training_set(prefix, policy=policy, before=3)
    assert tuple(record.assignment_key for record, _ in train) == (
        assignment_key({"a": 6, "b": 12}),
        assignment_key({"a": 6, "b": 18}),
    )

    choice = choose_next(space, plan, policy, index=3, train=train, seen=set())
    assert choice.mode == "surrogate"
    assert choice.train_keys == tuple(record.assignment_key for record, _ in train)
    assert len(choice.train_keys) == 2


def test_the_training_set_is_ordered_by_index_and_attempt_and_not_by_its_order_on_disk():
    """The model's output depends on the order its rows are summed in, so an order that depended on
    how the ledger happened to fold would let two readings of one evidence disagree."""
    space = _small_space()
    policy = _policy()
    plan = _plan(space, policy=policy)
    first = _record(space, plan, 1, {"a": 6, "b": 12}, 1.0)
    second = _record(space, plan, 2, {"a": 6, "b": 18}, 2.0)
    retried = _record(space, plan, 1, {"a": 6, "b": 12}, 3.0, attempt=2)

    forward = _training_set((first, second, retried), policy=policy, before=3)
    backward = _training_set((retried, second, first), policy=policy, before=3)

    assert tuple((record.index, record.attempt) for record, _ in forward) == ((1, 1), (1, 2), (2, 1))
    assert forward == backward

    shuffled = [first, second, retried]
    choices = [
        choose_next(
            space,
            plan,
            policy,
            index=3,
            train=_training_set(tuple(order), policy=policy, before=3),
            seen=set(),
        ).chosen_key
        for order in (shuffled, list(reversed(shuffled)))
    ]
    assert choices[0] == choices[1]


def test_a_minimisation_policy_is_the_same_search_with_the_targets_negated():
    """One formula, one place for the direction to be wrong: minimising y is maximising -y, and the
    two must pick the same point from the same evidence."""
    space = _small_space()
    maximizing = _policy(direction="maximize")
    minimizing = _policy(direction="minimize")
    plan = _plan(space, policy=maximizing)
    higher = [
        (_record(space, plan, 1, {"a": 6, "b": 12}, 1.0), 1.0),
        (_record(space, plan, 2, {"a": 9, "b": 18}, 3.0), 3.0),
    ]
    lower = [
        (_record(space, plan, 1, {"a": 6, "b": 12}, -1.0), -1.0),
        (_record(space, plan, 2, {"a": 9, "b": 18}, -3.0), -3.0),
    ]

    up = choose_next(space, plan, maximizing, index=3, train=higher, seen=set())
    down = choose_next(space, plan, minimizing, index=3, train=lower, seen=set())

    assert up.mode == down.mode == "surrogate"
    assert up.chosen_key == down.chosen_key
    assert up.chosen_ei == pytest.approx(down.chosen_ei)
    assert "minimize" in down.detail


def test_the_training_set_ignores_a_record_with_no_objective():
    """A succeeded trial without an objective is not an (x, y) pair, whatever its status says."""
    space = _small_space()
    policy = _policy()
    plan = _plan(space, policy=policy)
    prefix = (
        _record(space, plan, 1, {"a": 6, "b": 12}, None),
        _record(space, plan, 2, {"a": 6, "b": 18}, 2.0),
    )
    train = _training_set(prefix, policy=policy, before=3)
    assert tuple(record.index for record, _ in train) == (2,)


def test_the_sampler_refuses_a_plan_of_another_kind():
    """A bayesian sampler re-derives its sequence from evidence, so it cannot stand in for a kind
    whose sequence the plan already fixes."""
    space = _small_space()
    with pytest.raises(ValueError) as caught:
        BayesianSampler(space, SearchPlan.for_random(space, seed=1, budget=2), observations=_FakeSource())
    assert "needs a bayesian plan" in str(caught.value)
    assert "cannot stand in for another kind" in str(caught.value)


def test_an_observation_source_that_matches_nothing_leaves_every_step_in_the_initial_design():
    """The failure this prevents is the quiet one: observations from another plan teach the model a
    mapping nobody ran, and a wrong mapping does not announce itself.

    The source here holds a real, well-formed record from a *different* plan -- a legal record of a
    real trial, which is exactly why only the fingerprints can tell it apart from one that belongs.
    """
    space = _small_space()
    policy = _policy(initial_design=0, min_train=2)
    plan = _plan(space, policy=policy, budget=3)
    foreign = _record(space, _plan(space, policy=policy, budget=9), 1, {"a": 6, "b": 12}, 1.0)
    source = _FakeSource((foreign,))
    decisions = _RecordingSink()

    sampler = BayesianSampler(space, plan, observations=source, sink=decisions)
    sampler.collect()

    assert source.calls == [(plan.fingerprint(), space.fingerprint())] * 3
    assert foreign.plan_fingerprint != plan.fingerprint()
    assert [record.mode for record in decisions.records] == ["initial_design"] * 3
    assert [record.train_count for record in decisions.records] == [0, 0, 0]


def test_a_search_whose_every_trial_was_pruned_never_consults_a_posterior():
    """The other half of "the train set is not the seen set", stated as the outcome rather than the
    mechanism.

    ``pruned`` is not in the shipped policy's ``learn_from``, so a run in which every trial was cut
    short has nothing to fit -- and "nothing to fit" is not "a flat posterior": it is
    ``initial_design`` at every step with ``train_count == 0`` and no acquisition value recorded at
    all.  A sampler that trained on the truncated objectives anyway would produce a full-looking
    sequence of decisions with plausible numbers in them, which is why this is pinned on the records
    and not on the proposals.

    The same record pins the *other* use of the evidence.  The pruned assignment is still ``seen``,
    so the step after it cannot propose it again, while it stays out of ``train``: if the two were
    the same set, either the model would fit a truncated trial or the search would re-run one.  The
    pool size is what tells those apart, and the point is chosen to be one the design would not have
    proposed anyway -- otherwise "excluded from the pool" and "drawn by luck" look the same, which
    is what makes this an assertion about ``seen`` rather than an assertion about the seed.
    """
    space = _small_space()
    policy = _policy(min_train=2, pool_size=8)
    plan = _plan(space, policy=policy, budget=2)
    points = (
        {"a": 6, "b": 12},
        {"a": 6, "b": 18},
        {"a": 6, "b": 24},
        {"a": 9, "b": 18},
    )
    free = BayesianSampler(space, plan, observations=_FakeSource()).collect()[0]
    tried = next(item for item in points if assignment_key(item) != assignment_key(free))
    pruned = (_record(space, plan, 1, tried, 1.0, status="pruned"),)
    decisions = _RecordingSink()

    sampler = BayesianSampler(space, plan, observations=_FakeSource(pruned), sink=decisions)
    drawn = sampler.collect()

    assert len(drawn) == 2
    assert [record.mode for record in decisions.records] == ["initial_design"] * 2
    assert [record.train_count for record in decisions.records] == [0, 0]
    assert all(record.chosen_ei is None for record in decisions.records), (
        "a step that never built a posterior has no acquisition value to report"
    )
    assert assignment_key(drawn[0]) == assignment_key(free), (
        "step 1 precedes the trial's index, so it cannot know about it and draws what it would "
        "have drawn with no evidence at all"
    )
    assert assignment_key(drawn[1]) != assignment_key(tried)
    assert decisions.records[1].pool_size == 2, (
        "four feasible points minus the step's own proposal minus the pruned trial; a sampler that "
        "built seen out of train would leave the pruned trial in and report 3"
    )


def test_a_step_cannot_see_a_trial_that_ran_after_it():
    """The index bound on ``seen``, which nothing else pinned: this mutation survived a sweep.

    ``seen`` is not "every recorded trial" -- it is this run's own proposals plus the trials whose
    index is *below* the step.  Dropping that bound (``record.index < index`` becoming something
    true of every record) is invisible to every other test in this file, because the observation
    source in those tests either holds nothing or holds trials that carry no proposal of their own.
    It is not cosmetic: with the bound gone, a step's candidate pool depends on trials that had not
    run yet, so the sequence stops being a function of the plan and starts being a function of how
    the runner interleaved the trials -- the exact thing per-step streams exist to prevent.

    The trial is placed on the point step 1 would otherwise propose, at an index *after* step 1, so
    both halves of the rule are visible at once: the pool step 1 draws from is the whole feasible
    set, and what it draws is what it draws with no evidence at all.
    """
    space = _small_space()
    policy = _policy(min_train=2, pool_size=8)
    plan = _plan(space, policy=policy, budget=3)
    free = BayesianSampler(space, plan, observations=_FakeSource()).collect()
    later = _record(space, plan, 2, free[0], 1.0, status="pruned")
    decisions = _RecordingSink()

    drawn = BayesianSampler(space, plan, observations=_FakeSource((later,)), sink=decisions).collect()

    assert len(drawn) == 3
    assert decisions.records[0].pool_size == len(_feasible_keys(space)), (
        "trial 2 has not run when step 1 is decided, so it cannot shrink step 1's pool"
    )
    assert assignment_key(drawn[0]) == assignment_key(free[0])
    assert [record.train_count for record in decisions.records] == [0, 0, 0]
    assert decisions.records[1].pool_size == len(_feasible_keys(space)) - 1, (
        "step 2 still cannot see trial 2 -- what it excludes is its own earlier proposal"
    )
    assert all(record.mode == "initial_design" for record in decisions.records)


def test_the_sampler_asks_the_source_for_the_plan_and_the_space_it_is_working_on():
    """Both fingerprints are required: an observation from a different space would teach the wrong
    mapping, and one from a different plan would teach a sequence the caller never chose."""
    space = _small_space()
    plan = _plan(space, budget=2)
    source = _FakeSource()
    BayesianSampler(space, plan, observations=source).collect()

    assert source.calls == [(plan.fingerprint(), space.fingerprint())] * 2


def test_the_policy_has_no_default_direction_and_refuses_the_values_it_cannot_use():
    """A coin-flip default is an invisible decision, and getting it backwards makes every later step
    chase the wrong end of the objective without anything looking broken."""
    with pytest.raises(ValidationError) as caught:
        BayesianPolicy()
    assert "direction" in str(caught.value)

    with pytest.raises(ValidationError):
        BayesianPolicy(direction="sideways")

    for bad in (
        {"length_scale": 0.0},
        {"length_scale": float("inf")},
        {"noise": -1.0},
        {"xi": -1.0},
        {"initial_design": -1},
        {"min_train": 0},
        {"pool_size": 0},
        {"max_attempts": 0},
        {"learn_from": ()},
        {"learn_from": ("succeeded", "succeeded")},
        {"learn_from": ("pending",)},
        {"learn_from": ("running",)},
    ):
        with pytest.raises(ValidationError):
            BayesianPolicy(direction="maximize", **bad)

    policy = _policy()
    assert policy.acquisition == "expected_improvement"
    assert policy.learn_from == ("succeeded",)
    assert set(get_args(Acquisition)) == {"expected_improvement"}
    assert set(get_args(Direction)) == {"maximize", "minimize"}
    assert set(get_args(DecisionMode)) == {"initial_design", "surrogate", "fallback"}


def test_the_plan_takes_its_attempt_budget_from_the_policy_and_has_no_second_knob():
    """One number, or two that disagree -- and the third is the drawn attempt count, not the field.

    The plan carries ``max_attempts`` and the policy carries one too, and the drawing loop reads the
    *policy's*; the sampler rebuilds its policy from ``plan.policy``.  So the plan's field has to be
    a copy rather than an independent knob: a plan-level override was accepted, recorded in the run
    header and in every trial, and ignored by the loop that bounds the search.

    The behavioural half matters as much as the field: the last assertion counts the draws a step
    actually made against an unfillable pool, which is the only way to pin that the number the
    header shows is the number that ran.
    """
    space = _small_space()
    policy = _policy(max_attempts=7)

    plan = bayesian_plan(space, seed=1, budget=2, policy=policy)
    assert plan.max_attempts == 7
    assert plan.policy is not None and plan.policy["max_attempts"] == 7

    # Eight candidates asked for, seven draws allowed, and the space cannot produce eight anyway:
    # the loop stops at the policy's number, recorded and counted.
    impossible = _impossible_space()
    tight = _policy(pool_size=8, max_attempts=7)
    tight_plan = bayesian_plan(impossible, seed=1, budget=2, policy=tight)
    assert tight_plan.policy is not None
    assert tight_plan.max_attempts == tight_plan.policy["max_attempts"] == tight.max_attempts == 7
    _, counts = _pool(impossible, tight_plan, tight, index=1, seen=set())
    assert counts["attempts"] == 7


# --- E: the wiring to task 4 -------------------------------------------------------------------


def test_each_step_sees_the_trials_before_it_and_only_those(tmp_path):
    """The whole adaptive claim in one assertion: step N's training set is the scored trials with an
    index below N, and it grows by one each time."""
    tracker, sampler, store = _bayesian_tracker(
        tmp_path, policy=_policy(initial_design=1, min_train=2, pool_size=16), budget=4
    )
    run = tracker.run()

    assert run.complete is True
    assert len(run.trials) == 4
    assert sampler.stats.proposals == 4
    assert sampler.stats.exhausted is True

    decisions = _decisions(store)
    assert [record.index for record in decisions] == [1, 2, 3, 4]
    assert [record.train_count for record in decisions] == [0, 1, 2, 3]
    assert [record.mode for record in decisions] == [
        "initial_design",
        "initial_design",
        "surrogate",
        "surrogate",
    ]

    for decision in decisions:
        expected = [
            trial.assignment_key for trial in run.trials if trial.index < decision.index
        ]
        assert list(decision.train_keys) == expected
        assert decision.plan_fingerprint == tracker.plan.fingerprint()
        assert decision.space_fingerprint == tracker.space.fingerprint()


def test_every_proposal_has_a_decision_on_disk_with_the_same_key(tmp_path):
    """A proposal with no evidence behind it is the one thing the decision record exists to
    prevent, so the two are tied together by key and by index."""
    tracker, _, store = _bayesian_tracker(tmp_path, budget=3)
    run = tracker.run()

    decisions = {record.index: record for record in _decisions(store)}
    assert set(decisions) == {1, 2, 3}
    for trial in run.trials:
        decision = decisions[trial.index]
        assert decision.chosen_key == trial.assignment_key
        assert decision.id == decision_id(decision.run_ref, trial.index)
        assert decision.run_ref == tracker.run_ref
        assert decision.pool_size >= 1


def test_the_search_ledger_verifies_every_step_of_a_completed_run(tmp_path):
    """The end-to-end check: recomputing each recorded decision from the records before it picks the
    same assignment, which is what makes the sequence auditable after the fact."""
    tracker, _, store = _bayesian_tracker(tmp_path, budget=3)
    tracker.run()

    ledger = SearchLedger(store)
    decisions = _decisions(store)
    assert decisions
    for decision in decisions:
        prefix = tuple(trial for trial in ledger.trials() if trial.index < decision.index)
        assert (
            verify_step(
                decision,
                plan=tracker.plan,
                prefix=prefix,
                space=tracker.space,
                policy=BayesianPolicy.model_validate(tracker.plan.policy),
            )
            == "verified"
        )


def test_a_batch_of_an_adaptive_plan_is_refused_before_anything_is_written(tmp_path):
    """The header would declare a batch size the adaptive property cannot survive once anyone acts
    on it: within a batch, every point is drawn before any of them has a result."""
    tracker, _, store = _bayesian_tracker(tmp_path, budget=2, max_in_flight=2)

    with pytest.raises(SearchError) as caught:
        tracker.run()
    assert "max_in_flight=2" in str(caught.value)
    assert "a batch destroys" in str(caught.value)
    assert "max_in_flight=1" in str(caught.value)
    assert store.events() == ()


def test_the_plan_that_cannot_be_built_without_observations_is_refused_by_both_entry_points():
    """``build_sampler`` has nowhere to read evidence from, and it says what to construct instead
    rather than falling back to a random draw that would look adaptive in the record."""
    space = _taili_space()
    policy = _policy()
    plan = _plan(space, policy=policy)

    for refused in (build_sampler, replay):
        with pytest.raises(ValueError) as caught:
            refused(space, plan)
        assert "no observations" in str(caught.value)
        assert "BayesianSampler(space, plan, observations=...)" in str(caught.value)

    assert isinstance(
        BayesianSampler(space, plan, observations=_FakeSource()), Sampler
    )
    assert "bayesian" in NON_REPLAYABLE_SAMPLER_KINDS
    assert "bayesian" not in REPLAYABLE_SAMPLER_KINDS
    assert set(get_args(SamplerKind)) == {"grid", "random", "bayesian"}


def test_each_kind_of_plan_demands_exactly_the_fields_its_sequence_needs():
    """The shape check is the plan's, and it is what makes the duplicated ``Direction`` vocabulary
    and the missing sampler-side checks safe: a plan that reached a sampler is a legal plan."""
    space = _small_space()
    fingerprint = space.fingerprint()
    order = space.sampling_order()

    with pytest.raises(ValidationError) as caught:
        SearchPlan(kind="grid", space_fingerprint=fingerprint, sampling_order=order, budget=2, seed=1)
    assert "no randomness" in str(caught.value)

    with pytest.raises(ValidationError) as caught:
        SearchPlan(
            kind="grid",
            space_fingerprint=fingerprint,
            sampling_order=order,
            budget=2,
            policy={"direction": "maximize"},
        )
    assert "consults no observations" in str(caught.value)

    # A grid plan is the one kind that legitimately carries neither, and it is spelled out here
    # so that "grid refuses a seed" is not read as "a plan needs a seed".
    grid = SearchPlan(kind="grid", space_fingerprint=fingerprint, sampling_order=order, budget=2)
    assert (grid.seed, grid.policy) == (None, None)

    with pytest.raises(ValidationError):
        SearchPlan(kind="random", space_fingerprint=fingerprint, sampling_order=order, budget=2)
    with pytest.raises(ValidationError):
        SearchPlan(kind="random", space_fingerprint=fingerprint, sampling_order=order, seed=1)
    with pytest.raises(ValidationError):
        SearchPlan(kind="bayesian", space_fingerprint=fingerprint, sampling_order=order, seed=1)
    with pytest.raises(ValidationError):
        SearchPlan(
            kind="bayesian", space_fingerprint=fingerprint, sampling_order=order, budget=2
        )
    with pytest.raises(ValidationError) as caught:
        SearchPlan(
            kind="bayesian", space_fingerprint=fingerprint, sampling_order=order, seed=1, budget=2
        )
    assert "requires a surrogate policy" in str(caught.value)

    with pytest.raises(ValidationError):
        SearchPlan(
            kind="random",
            space_fingerprint=fingerprint,
            sampling_order=order,
            seed=1,
            budget=2,
            policy={"direction": "maximize"},
        )
    with pytest.raises(ValidationError):
        SearchPlan(
            kind="random",
            space_fingerprint=fingerprint,
            sampling_order=order,
            seed=1,
            budget=2,
            policy={"direction": object()},
        )

    # A policy whose ``max_attempts`` is not the default, so that "the plan took it from the
    # policy" is distinguishable from "both happened to be 1000".
    numbered = _policy(max_attempts=7)
    legal = _plan(space, policy=numbered, budget=2)
    assert legal.kind == "bayesian"
    assert legal.policy is not None
    assert legal.max_attempts == 7
    assert legal.max_attempts == legal.policy["max_attempts"]
    assert legal.policy["direction"] == "maximize"


def test_the_run_header_records_the_adaptive_plan_and_its_sampler(tmp_path):
    """A reader of the header alone must be able to tell that the sequence was adaptive and which
    policy decided it, without the sampler or the decision records."""
    tracker, _, store = _bayesian_tracker(tmp_path, budget=2)
    header = tracker.open_run()

    assert header.plan["kind"] == "bayesian"
    assert header.budget == 2
    assert header.max_in_flight == 1
    assert header.status == "open"

    from_disk = SearchRunRecord.model_validate(store.latest("search_run", header.id))
    assert from_disk.plan["policy"] == header.plan["policy"]
    assert from_disk.plan_fingerprint == tracker.plan.fingerprint()
    assert SearchPlan.model_validate(from_disk.plan).fingerprint() == from_disk.plan_fingerprint


def test_the_plan_fingerprint_changes_when_the_policy_changes_and_not_otherwise():
    """The policy is part of the plan's identity, so two searches with different policies can never
    be folded into one run -- and adding the field moved every grid and random fingerprint too,
    which is why the design records that cost."""
    space = _small_space()
    first = _plan(space, policy=_policy(direction="maximize"), budget=3)
    same = _plan(space, policy=_policy(direction="maximize"), budget=3)
    other = _plan(space, policy=_policy(direction="minimize"), budget=3)

    assert first.fingerprint() == same.fingerprint()
    assert first.fingerprint() != other.fingerprint()

    grid = SearchPlan.for_grid(space)
    assert grid.policy is None
    assert "policy" in grid.model_dump(mode="json")


def test_the_sampler_refuses_a_plan_drawn_from_another_space(tmp_path):
    """Reusing a plan against another space is refused by the base class, which is why the encoder's
    own space check is a second line and not the only one."""
    space = _small_space()
    plan = _plan(space, budget=2)
    other = SearchSpaceSchema(specs=(_discrete("a", 1, 3, 1, 1),))
    with pytest.raises(ValueError) as caught:
        BayesianSampler(other, plan, observations=_FakeSource())
    assert "different search space" in str(caught.value)


def test_an_injected_sampler_that_already_proposed_is_refused_by_the_tracker(tmp_path):
    """A stateful sampler handed in twice would bind the *next* stretch of its stream to indices
    1..N, and no read-side check can see it: every assignment in the record is individually legal.

    The sampler is advanced against an in-memory source and sink so that the only thing this test
    measures is the tracker's guard: a decision written into an empty root would fail on the layout
    check first, and the test would then be pinning the wrong refusal.
    """
    tracker, sampler, store = _bayesian_tracker(
        tmp_path, budget=2, observations=_FakeSource(), sink=_RecordingSink()
    )
    sampler.collect()
    assert sampler.stats.proposals == 2
    with pytest.raises(SearchError) as caught:
        tracker.run()
    message = str(caught.value)
    assert "already proposed 2 assignment(s)" in message
    assert "Pass a fresh sampler" in message
    assert store.events() != (), "the guard fires after the header, which is what run() opens first"


# --- F: the decisions, the policy record, and the audit -----------------------------------------


def test_a_decision_is_stored_under_its_derived_id_and_read_back_unchanged(tmp_path):
    """The id is derived from ``(run, index)``, so two spellings cannot both claim one step."""
    store = TrialLedgerStore(tmp_path / "ledger")
    sink = LedgerSurrogateSink(store, precondition=None)
    space = _small_space()
    policy = _policy()
    plan = _plan(space, policy=policy)
    train = [
        (_record(space, plan, 1, {"a": 6, "b": 12}, 1.0), 1.0),
        (_record(space, plan, 2, {"a": 6, "b": 18}, 4.0), 4.0),
    ]
    choice = choose_next(space, plan, policy, index=3, train=train, seen=set())
    record = SurrogateDecisionRecord.build(
        run_ref="search:test", plan=plan, space=space, policy=policy, index=3, choice=choice
    )
    sink.record_decision(record)

    payload = store.latest(DECISION_RECORD_TYPE, "surrogate-decision:search:test:3")
    assert payload is not None
    restored = SurrogateDecisionRecord.model_validate(payload)
    assert restored == record
    assert restored.id == decision_id("search:test", 3)
    assert restored.chosen_key == choice.chosen_key
    assert restored.policy_fingerprint == policy.fingerprint
    assert restored.mode == "surrogate"
    assert restored.pool_size == len(choice.pool)
    assert restored.train_count == len(choice.train_keys)


def test_a_decision_that_cannot_be_written_is_never_proposed():
    """Recorded before it is yielded, so a search that cannot explain a step does not take it."""
    space = _small_space()
    policy = _policy()
    plan = _plan(space, policy=policy, budget=3)
    sink = _RecordingSink(fail_at=2)
    sampler = BayesianSampler(space, plan, observations=_FakeSource(), sink=sink)

    drawn: list[dict[str, Any]] = []
    with pytest.raises(SurrogateAuditError):
        for assignment in sampler.proposals():
            drawn.append(assignment)

    assert len(drawn) == 1
    assert [record.index for record in sink.records] == [1]


def test_a_decision_already_recorded_identically_is_not_appended_a_second_time(tmp_path):
    """Resuming a search must not grow duplicates, and an identical recomputation is the normal
    case for a resumed step rather than a contradiction."""
    store = TrialLedgerStore(tmp_path / "ledger")
    sink = LedgerSurrogateSink(store, precondition=None)
    space = _small_space()
    policy = _policy()
    plan = _plan(space, policy=policy)
    choice = choose_next(space, plan, policy, index=1, train=(), seen=set())
    record = SurrogateDecisionRecord.build(
        run_ref="search:test", plan=plan, space=space, policy=policy, index=1, choice=choice
    )

    sink.record_decision(record)
    before = len(store.events())
    sink.record_decision(record)
    assert len(store.events()) == before


def test_a_decision_that_recomputes_to_something_else_is_refused_rather_than_overwritten(tmp_path):
    """Quietly keeping the newer one would leave the ledger holding a claim nothing supports."""
    store = TrialLedgerStore(tmp_path / "ledger")
    sink = LedgerSurrogateSink(store, precondition=None)
    space = _small_space()
    policy = _policy()
    plan = _plan(space, policy=policy)
    choice = choose_next(space, plan, policy, index=1, train=(), seen=set())
    record = SurrogateDecisionRecord.build(
        run_ref="search:test", plan=plan, space=space, policy=policy, index=1, choice=choice
    )
    sink.record_decision(record)

    # A different *feasible* key of the same space: still a legal assignment, and still not the one
    # the evidence picks, which is the whole point -- a tamper that changed nothing a reader could
    # act on would not be worth refusing.
    other = next(key for key in _feasible_keys(space) if key != record.chosen_key)
    tampered = SurrogateDecisionRecord.model_validate(
        {**record.model_dump(mode="json"), "chosen_key": other}
    )
    with pytest.raises(SurrogateAuditError) as caught:
        sink.record_decision(tampered)
    message = str(caught.value)
    assert "already" in message
    assert record.chosen_key in message
    assert other in message
    assert len(_decisions(store)) == 1


def test_verify_step_says_unverifiable_when_the_evidence_it_names_is_not_there():
    """A partial ledger is the normal state of an interrupted search, and calling it a
    contradiction would make every resume look like tampering."""
    space = _small_space()
    policy = _policy()
    plan = _plan(space, policy=policy)
    train = [
        (_record(space, plan, 1, {"a": 6, "b": 12}, 1.0), 1.0),
        (_record(space, plan, 2, {"a": 6, "b": 18}, 4.0), 4.0),
    ]
    # ``seen`` is the two assignments already recorded, not the empty set: a step taken with these
    # two trials behind it cannot propose either of them again, and a record built with ``seen``
    # empty describes a step that never happened -- which is a thing only the field-for-field
    # recomputation in ``verify_step`` can notice.
    seen = {record.assignment_key for record, _ in train}
    choice = choose_next(space, plan, policy, index=3, train=train, seen=seen)
    record = SurrogateDecisionRecord.build(
        run_ref="search:test", plan=plan, space=space, policy=policy, index=3, choice=choice
    )

    assert verify_step(record, plan=plan, prefix=tuple(item for item, _ in train), space=space, policy=policy) == "verified"
    assert verify_step(record, plan=plan, prefix=(), space=space, policy=policy) == "unverifiable"
    assert (
        verify_step(
            record, plan=plan, prefix=(train[0][0],), space=space, policy=policy
        )
        == "unverifiable"
    )
    # A record whose train_keys name an assignment the prefix does not hold: the count can match
    # while the members differ, which is why the keys are compared and not the length.
    stranger = _record(space, plan, 1, {"a": 9, "b": 18}, 1.0)
    assert (
        verify_step(record, plan=plan, prefix=(stranger, train[1][0]), space=space, policy=policy)
        == "unverifiable"
    )


def test_verify_step_says_unverifiable_rather_than_raising_for_another_plan_space_or_policy():
    """Three answers are kept apart on purpose: not recomputable, recomputed identically, and
    recomputed differently.  Only the third is a contradiction."""
    space = _small_space()
    policy = _policy()
    plan = _plan(space, policy=policy)
    train = [
        (_record(space, plan, 1, {"a": 6, "b": 12}, 1.0), 1.0),
        (_record(space, plan, 2, {"a": 6, "b": 18}, 4.0), 4.0),
    ]
    choice = choose_next(space, plan, policy, index=3, train=train, seen=set())
    prefix = tuple(item for item, _ in train)
    record = SurrogateDecisionRecord.build(
        run_ref="search:test", plan=plan, space=space, policy=policy, index=3, choice=choice
    )

    other_policy = _policy(xi=0.5)
    assert verify_step(record, plan=plan, prefix=prefix, space=space, policy=other_policy) == "unverifiable"

    other_plan = _plan(space, policy=policy, budget=9)
    assert verify_step(record, plan=other_plan, prefix=prefix, space=space, policy=policy) == "unverifiable"

    other_space = SearchSpaceSchema(specs=(_discrete("a", 6, 9, 3, 6), _discrete("b", 12, 24, 6, 12)))
    assert verify_step(record, plan=plan, prefix=prefix, space=other_space, policy=policy) == "unverifiable"


def test_verify_step_raises_when_the_same_records_choose_something_else():
    """The only answer that is a contradiction, and the one that has to be loud: the sequence is no
    longer re-derivable."""
    space = _small_space()
    policy = _policy()
    plan = _plan(space, policy=policy)
    train = [
        (_record(space, plan, 1, {"a": 6, "b": 12}, 1.0), 1.0),
        (_record(space, plan, 2, {"a": 6, "b": 18}, 4.0), 4.0),
    ]
    prefix = tuple(item for item, _ in train)
    seen = {record.assignment_key for record, _ in train}
    choice = choose_next(space, plan, policy, index=3, train=train, seen=seen)
    honest = SurrogateDecisionRecord.build(
        run_ref="search:test", plan=plan, space=space, policy=policy, index=3, choice=choice
    )
    assert verify_step(honest, plan=plan, prefix=prefix, space=space, policy=policy) == "verified"

    # A legal assignment the step could have taken and did not: the recomputation runs, picks the
    # recorded step's own winner, and disagrees with the claim on disk.
    other = next(key for key in _feasible_keys(space) if key != honest.chosen_key)
    tampered = SurrogateDecisionRecord.model_validate(
        {**honest.model_dump(mode="json"), "chosen_key": other}
    )

    with pytest.raises(SurrogateAuditError) as caught:
        verify_step(tampered, plan=plan, prefix=prefix, space=space, policy=policy)
    message = str(caught.value)
    assert "step 3 of run search:test" in message
    assert f"chosen_key: recorded {other!r} vs recomputed {honest.chosen_key!r}" in message
    assert "can no longer be re-derived" in message


def test_verify_step_refuses_a_record_that_disagrees_about_anything_not_only_the_winner():
    """Every field, because an audit narrower than the write path it audits is not an audit.

    ``verify_step`` compared only ``chosen_key``, and every test in this file tampered with only
    ``chosen_key`` -- so a record whose acquisition value, pool size, mode or detail had been
    changed was called "verified" here and refused by ``LedgerSurrogateSink``, which compares the
    whole record.  Both now compare the same JSON form, and this pins each field on its own: the
    message has to name the field and show both numbers, because "the record disagrees with the
    recomputation" without saying how is not something a reader can act on.
    """
    space = _small_space()
    policy = _policy()
    plan = _plan(space, policy=policy)
    train = [
        (_record(space, plan, 1, {"a": 6, "b": 12}, 1.0), 1.0),
        (_record(space, plan, 2, {"a": 6, "b": 18}, 4.0), 4.0),
    ]
    prefix = tuple(item for item, _ in train)
    seen = {record.assignment_key for record, _ in train}
    choice = choose_next(space, plan, policy, index=3, train=train, seen=seen)
    honest = SurrogateDecisionRecord.build(
        run_ref="search:test", plan=plan, space=space, policy=policy, index=3, choice=choice
    )
    assert honest.chosen_ei is not None and honest.runner_up_ei is not None

    payload = honest.model_dump(mode="json")
    cases: list[tuple[str, dict[str, Any]]] = [
        ("chosen_ei", {"chosen_ei": honest.chosen_ei + 1.0}),
        ("runner_up_ei", {"runner_up_ei": honest.runner_up_ei - 1.0}),
        ("pool_size", {"pool_size": honest.pool_size + 1}),
        ("detail", {"detail": "scored 999 candidate(s) against 2 observation(s)"}),
        # A mode change has to drop the acquisition values with it: the record's own validator
        # refuses a fallback that carries one, which is a different guard from this one.
        ("mode", {"mode": "fallback", "chosen_ei": None, "runner_up_ei": None}),
    ]
    for field, changes in cases:
        tampered = SurrogateDecisionRecord.model_validate({**payload, **changes})
        with pytest.raises(SurrogateAuditError) as caught:
            verify_step(tampered, plan=plan, prefix=prefix, space=space, policy=policy)
        message = str(caught.value)
        assert f"{field}: recorded" in message, field
        assert "vs recomputed" in message, field
        assert "can no longer be re-derived" in message, field

    # And the honest record still passes, so the loop above is not proving that everything raises.
    assert verify_step(honest, plan=plan, prefix=prefix, space=space, policy=policy) == "verified"


def test_the_policy_fingerprint_covers_every_field_that_changes_a_decision():
    """A policy stored content-addressed is only a claim if two policies that decide differently
    cannot share a fingerprint."""
    base = _policy()
    variants = [
        _policy(direction="minimize"),
        _policy(acquisition="expected_improvement", xi=0.5),
        _policy(length_scale=1.0),
        _policy(noise=1.0e-3),
        _policy(initial_design=3),
        _policy(min_train=3),
        _policy(pool_size=16),
        _policy(learn_from=("succeeded", "pruned")),
        _policy(max_attempts=9),
    ]
    fingerprints = {policy.fingerprint for policy in variants}
    assert len(fingerprints) == len(variants)
    assert base.fingerprint not in fingerprints
    assert _policy().fingerprint == base.fingerprint
    assert base.fingerprint == content_hash(base.model_dump(mode="json"))


def test_a_policy_record_is_stored_once_and_read_back(tmp_path):
    """Content-addressed, so storing the same policy twice appends nothing the second time."""
    store = TrialLedgerStore(tmp_path / "ledger")
    policy = _policy()
    record = surrogate_policy_record(policy)

    assert record.id == f"surrogate-policy:{policy.fingerprint}"
    assert record.schema_version == SURROGATE_SCHEMA_VERSION
    assert policy_for(store, policy.fingerprint) is None

    stored = store_policy_record(store, policy, precondition=None, actor="test")
    assert stored == record
    assert policy_for(store, policy.fingerprint) == policy

    count = len(store.events())
    store_policy_record(store, policy, precondition=None, actor="test")
    assert len(store.events()) == count

    payload = store.latest(POLICY_RECORD_TYPE, record.id)
    with pytest.raises(ValidationError) as caught:
        SurrogatePolicyRecord.model_validate({**payload, "id": "surrogate-policy:other"})
    assert "does not carry the fingerprint" in str(caught.value)

    with pytest.raises(ValidationError):
        SurrogatePolicyRecord.model_validate({**payload, "policy": {"direction": "maximize"}})


def test_a_policy_stored_before_the_run_header_costs_the_root_as_a_search_ledger(tmp_path):
    """The measured hazard, not an inferred one: the search reader requires the first event to be
    the run header, so a policy written into an empty root makes that root unreadable as a search.

    ``TrialLedgerStore`` itself still writes and reads -- it is the *search* reader that refuses the
    root from then on, which is why the precondition is a required argument of both writers.
    """
    root = tmp_path / "ledger"
    store = TrialLedgerStore(root)
    store_policy_record(store, _policy(), precondition=None)

    assert store.summary()["chain_verified"] is True
    assert len(store.records()) == 1

    with pytest.raises(TrialLedgerIntegrityError) as caught:
        SearchLedger(store).refresh()
    assert "unknown_layout" in str(caught.value)

    tracker, _, _ = _bayesian_tracker(tmp_path)
    with pytest.raises(TrialLedgerIntegrityError) as caught:
        SearchTracker(
            store=store,
            space=tracker.space,
            plan=tracker.plan,
            source_config=_taili_config(),
            source_config_path=CONFIG_PATH,
            work_root=tmp_path / "work",
            runner=_ScoringRunner(),
        ).open_run()
    assert "no search_run record to explain them" in str(caught.value)


def test_a_policy_stored_after_the_header_leaves_the_search_ledger_readable(tmp_path):
    """The other half of the hazard: the same append, in the right order, costs nothing."""
    tracker, _, store = _bayesian_tracker(tmp_path, budget=2)
    tracker.open_run()
    policy = BayesianPolicy.model_validate(tracker.plan.policy)
    store_policy_record(store, policy, precondition=None)

    assert policy_for(store, policy.fingerprint) == policy
    ledger = SearchLedger(store)
    ledger.refresh()
    assert ledger.run(tracker.run_ref).header.plan["kind"] == "bayesian"


def test_a_surrogate_decision_must_record_the_posterior_it_claims_to_have_consulted():
    """The combination that must never be written: ``surrogate`` with nothing to consult would read
    as "a posterior decided this" while listing no observation that could have."""
    with pytest.raises(ValidationError) as caught:
        SurrogateDecisionRecord(
            id=decision_id("search:test", 3),
            run_ref="search:test",
            policy_fingerprint="p",
            plan_fingerprint="p",
            space_fingerprint="s",
            index=3,
            mode="surrogate",
            train_count=0,
            train_keys=(),
            pool_size=2,
            chosen_key='{"a":6,"b":12}',
            chosen_ei=0.1,
        )
    assert "at least one training observation" in str(caught.value)

    with pytest.raises(ValidationError) as caught:
        SurrogateDecisionRecord(
            id=decision_id("search:test", 3),
            run_ref="search:test",
            policy_fingerprint="p",
            plan_fingerprint="p",
            space_fingerprint="s",
            index=3,
            mode="surrogate",
            train_count=1,
            train_keys=('{"a":6,"b":12}',),
            pool_size=2,
            chosen_key='{"a":6,"b":12}',
        )
    assert "acquisition value it chose on" in str(caught.value)

    for mode in ("initial_design", "fallback"):
        with pytest.raises(ValidationError) as caught:
            SurrogateDecisionRecord(
                id=decision_id("search:test", 3),
                run_ref="search:test",
                policy_fingerprint="p",
                plan_fingerprint="p",
                space_fingerprint="s",
                index=3,
                mode=mode,
                train_count=0,
                train_keys=(),
                pool_size=2,
                chosen_key='{"a":6,"b":12}',
                chosen_ei=0.0,
            )
        assert "scored no candidate" in str(caught.value)


def test_a_decision_refuses_the_shapes_that_would_read_as_something_they_are_not():
    """Every one of these is a record that a reader would draw a conclusion from, wrongly."""
    payload: dict[str, Any] = {
        "id": decision_id("search:test", 3),
        "run_ref": "search:test",
        "policy_fingerprint": "p",
        "plan_fingerprint": "p",
        "space_fingerprint": "s",
        "index": 3,
        "mode": "initial_design",
        "train_count": 0,
        "train_keys": (),
        "pool_size": 2,
        "chosen_key": '{"a":6,"b":12}',
    }
    assert SurrogateDecisionRecord(**payload).chosen_ei is None

    with pytest.raises(ValidationError):
        SurrogateDecisionRecord(**{**payload, "run_ref": "  "})
    with pytest.raises(ValidationError):
        SurrogateDecisionRecord(**{**payload, "index": 0})
    with pytest.raises(ValidationError):
        SurrogateDecisionRecord(**{**payload, "pool_size": 0})
    with pytest.raises(ValidationError):
        SurrogateDecisionRecord(**{**payload, "chosen_key": ""})
    # The id is the key the ledger stores under, so it cannot be a free-form string: another index
    # and another run are two ways of spelling a step that is not this one.
    for wrong_id in (decision_id("search:test", 4), decision_id("search:other", 3), "surrogate"):
        with pytest.raises(ValidationError) as caught:
            SurrogateDecisionRecord(**{**payload, "id": wrong_id})
        assert "is not the derived id of step 3 of run 'search:test'" in str(caught.value)

    with pytest.raises(ValidationError) as caught:
        SurrogateDecisionRecord(**{**payload, "train_count": 1, "train_keys": ()})
    assert "two claims about one thing" in str(caught.value)
    with pytest.raises(ValidationError):
        SurrogateDecisionRecord(**{**payload, "train_count": -1, "train_keys": ()})

    for field in ("chosen_ei", "runner_up_ei"):
        with pytest.raises(ValidationError) as caught:
            SurrogateDecisionRecord(
                **{
                    **payload,
                    "mode": "surrogate",
                    "train_count": 1,
                    "train_keys": ('{"a":6,"b":12}',),
                    "chosen_ei": 0.5,
                    field: float("inf"),
                }
            )
        assert "must be a finite number" in str(caught.value)


def test_the_records_round_trip_through_their_json_form():
    """Law R4: ``content_hash(model_dump(mode="json"))`` is the identity, so a record that cannot
    survive its own JSON form has no identity to compare."""
    policy = _policy()
    records: list[Any] = [
        policy,
        surrogate_policy_record(policy),
        EncodingSpec.for_space(_guard_space()),
        SurrogateDecisionRecord(
            id=decision_id("search:test", 1),
            run_ref="search:test",
            policy_fingerprint=policy.fingerprint,
            plan_fingerprint="p",
            space_fingerprint="s",
            index=1,
            mode="surrogate",
            train_count=1,
            train_keys=('{"a":6,"b":12}',),
            pool_size=3,
            chosen_key='{"a":6,"b":18}',
            chosen_ei=0.25,
            runner_up_ei=0.1,
            detail="scored 3 candidate(s)",
        ),
    ]
    for record in records:
        payload = record.model_dump(mode="json")
        # Through the *text*, not just the dict: ``json.dumps(allow_nan=False)`` refuses a non-finite
        # float anywhere in the tree, and ``loads`` is what a reader of the ledger actually does.
        # The hash is then taken over the record rebuilt from that text and compared with the one
        # taken over the original dump -- the identity a content-addressed ledger depends on.  An
        # earlier version wrote ``content_hash(payload) == content_hash(record.model_dump(mode="json"))``,
        # which hashes the same object twice and cannot fail.
        text = json.dumps(payload, allow_nan=False, sort_keys=True)
        restored = type(record).model_validate(json.loads(text))
        assert restored == record
        assert content_hash(restored.model_dump(mode="json")) == content_hash(payload)


def test_the_standardiser_reports_no_variation_rather_than_dividing_by_zero():
    """``None`` is a decision the caller records, not an error: a constant target has no scale, and
    the alternative is a model that would divide by zero and produce infinities."""
    scaled = _standardise([1.0, 2.0, 3.0])
    assert scaled is not None
    assert scaled == [scaled[0], 0.0, -scaled[0]]
    assert scaled[0] < 0.0 < scaled[2]
    assert _standardise([2.0, 2.0]) is None
    assert _standardise([1.0]) is None


# --- G: the shape of the module itself ----------------------------------------------------------


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

#: Exactly what the module imports today -- no more.  An allowlist, because a blocklist only refuses
#: the names someone thought of.  It is asserted in **both** directions below: an import that is not
#: here is refused, and an entry that is here but unused is refused too, because an allowlist that
#: quietly outlives the code it describes reads as a permission somebody checked.  (``functools``
#: and ``json`` were here and are not imported by the module at all; the plan's *own* allowlist is
#: deliberately wider than this one and is not this set -- it holds ``os``, ``pathlib``,
#: ``tempfile`` and ``yaml``, which this module must not reach for.)
ALLOWED_IMPORTS = {
    "__future__",
    "math",
    "random",
    "typing",
    "pydantic",
    ".hyperparameter_sampler",
    ".hyperparameter_search",
    ".hyperparameter_space",
    ".research_ledger",
    ".trial_ledger",
}


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * (node.level or 0) + (node.module or ""))
    return imported


def test_the_module_stays_inside_the_layers_it_may_import():
    """The AST, not the prose: a dependency on any of the seven would let the sampler reach the
    console, the product tree or the trainer."""
    imported = _imported_names(MODULE_PATH)
    assert imported <= ALLOWED_IMPORTS, f"unexpected imports: {sorted(imported - ALLOWED_IMPORTS)}"
    stale = ALLOWED_IMPORTS - imported
    assert not stale, (
        f"the allowlist names {sorted(stale)}, which the module no longer imports: a permission "
        f"nobody is using is a permission nobody re-checked"
    )
    for forbidden in FORBIDDEN_PREFIXES:
        assert not any(name.startswith(forbidden) for name in imported), forbidden


def test_no_module_in_the_research_layer_imports_a_third_party_math_stack():
    """Zero numpy, scipy, sklearn or pandas in the whole layer, not just in this module.

    The structure gate does not check third-party packages at all -- it only checks the direction of
    dependencies between packages -- so "the gate is green" is not evidence for this.  The sampler
    is handed to the tracker, which sits on the console's import path, and the console declares
    ``pydantic`` and YAML: an undeclared heavy dependency there is one nobody chose.
    """
    offenders: dict[str, list[str]] = {}
    modules = sorted(RESEARCH_DIR.glob("*.py"))
    assert len(modules) >= 18, f"the layer holds {len(modules)} modules; this scan would be vacuous"
    for path in modules:
        for name in _imported_names(path):
            root = name.split(".")[0]
            if root in {"numpy", "scipy", "sklearn", "pandas", "torch"}:
                offenders.setdefault(str(path.name), []).append(name)
    assert offenders == {}


def test_the_module_uses_no_model_copy_and_no_computed_field():
    """Laws R1 and R2 checked on the AST: both are invisible in review and fatal to the ledger's
    round trip."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"model_copy", "computed_field"}:
            offenders.append(node.attr)
        if isinstance(node, ast.Name) and node.id == "computed_field":
            offenders.append("computed_field")
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                if isinstance(target, ast.Name) and target.id == "computed_field":
                    offenders.append("computed_field decorator")
    assert offenders == []


def test_the_direction_vocabulary_is_the_ranking_rule_s_and_the_two_modules_do_not_import_each_other():
    """The duplication is forced by an import cycle, so the drift is prevented by this test.

    ``search_analysis`` -> ``search_pruning`` -> ``hyperparameter_search`` -> this module, so
    importing the direction back from there is a cycle; the words are therefore written twice and
    pinned here rather than kept in step by discipline.
    """
    assert get_args(Direction) == get_args(RankDirection) == ("maximize", "minimize")

    imported = _imported_names(MODULE_PATH)
    assert not any(name.startswith(".search_analysis") for name in imported)
    assert not any(name.startswith(".search_pruning") for name in imported)

    analysis_imports = _imported_names(RESEARCH_DIR / "search_analysis.py")
    assert not any(name.startswith(".bayesian_sampler") for name in analysis_imports)


def test_the_module_publishes_the_vocabulary_the_ledger_stores():
    """Two readers of one ledger must agree on the strings, so they are pinned rather than described."""
    assert SURROGATE_SCHEMA_VERSION == "rl-agent.bayesian-sampler/v1"
    assert POLICY_RECORD_TYPE == "surrogate_policy"
    assert DECISION_RECORD_TYPE == "surrogate_decision"
    assert decision_id("search:abc", 7) == "surrogate-decision:search:abc:7"
    assert isinstance(BayesianSampler(_small_space(), _plan(_small_space(), budget=1), observations=_FakeSource()), Sampler)
    assert issubclass(SurrogateAuditError, SurrogateError)
