"""Deciding which trial to run next, from the trials already run.

A grid sampler and a random sampler answer "what comes next" from the plan alone: replay the plan
and the same sequence comes back.  This module answers it from *evidence* -- the trials that already
finished -- so its sequence is a function of the plan **and the observations so far**.  Three things
it deliberately is **not**:

* Not a ranking and not an export.  Which trial won is task 6's question
  (``search_analysis.analyze``).  All this module produces is the next candidate.
* Not pruning.  Task 5 decides when a running trial is hopeless; this module only reads what
  finished and never ends anything.
* Not a claim that adapting finds a better optimum.  **No such claim is made anywhere in this
  module, and none has been measured.**  What is true is narrower and testable: given the same
  space, the same plan and the same observations in the same order, it proposes the same next
  assignment, and it writes down how it got there.  Whether the acquisition function helps is an
  open question; a probe run while designing this module found the exploration term preferring an
  unvisited far corner over the true optimum on a small synthetic case.

Four things shape the whole design:

* **The assignment is encoded here, not stored pre-encoded.**  (hyperparameters, objective) pairs
  arrive as the raw records ``TrialObservationSource`` returns, and every domain fact an encoding
  needs is read from ``SearchSpaceSchema`` at encode time.  A feature vector copied into the record
  layer would let the surrogate's input space drift away from the sampling space with nobody
  noticing.
* **An inactive parameter is not the same point as that parameter at its low bound.**  A guarded
  parameter that never activated carries no value; encoding its dimensions as ``0.0`` would make it
  indistinguishable from the same parameter active at the bottom of its range, which teaches the
  surrogate that two different trials are one point.  Each conditionable parameter therefore gets an
  extra *activity* dimension.
* **The training set is chosen, and the choice is recorded.**  A pruned trial's objective was
  measured over a truncated horizon; task 6 ranks such a trial alongside the others on purpose (a
  score is a score), but mixing truncated-horizon targets into a regression teaches a mapping that
  does not hold.  ``BayesianPolicy.learn_from`` defaults to ``("succeeded",)`` and is part of the
  policy fingerprint, so the choice cannot happen quietly.
* **One random stream per step, not one stream for the run.**  The pool a step scores is derived
  from ``(seed, plan, step)``.  With a single continuous stream, how many draws earlier steps
  consumed would decide what step N's pool looks like, so rebuilding state would mean re-running
  every earlier step exactly.  Per-step streams make a step's pool independent of how many steps
  came before it -- which is what makes "rebuild from the ledger and continue" work.

Everything written to the ledger is JSON-native (law R4): a computed number is checked with
``math.isfinite`` before it is recorded, because ``model_dump(mode="json")`` turns ``nan`` into
``null`` and a reader could no longer tell "the arithmetic produced NaN" from "this field was never
set".
"""
from __future__ import annotations

import math
import random
from typing import (
    Any,
    Callable,
    Iterator,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, model_validator

from .hyperparameter_sampler import (
    Sampler,
    SearchPlan,
    assignment_key,
    draw_value,
)
from .hyperparameter_search import (
    TERMINAL_TRIAL_STATUSES,
    TrialObservationSource,
    TrialRecord,
    TrialStatus,
    default_run_ref,
)
from .hyperparameter_space import (
    HyperparameterSpec,
    SearchSpaceSchema,
    get_parameter,
    json_form,
    set_parameter,
)
from .research_ledger import content_hash
from .trial_ledger import TrialLedgerStore


SURROGATE_SCHEMA_VERSION = "rl-agent.bayesian-sampler/v1"

POLICY_RECORD_TYPE = "surrogate_policy"
DECISION_RECORD_TYPE = "surrogate_decision"

#: Which way the objective improves.  ``search_analysis.RankDirection`` holds the same two words
#: and **cannot** be imported from here: ``search_analysis`` imports ``search_pruning``, which
#: imports ``hyperparameter_search``, which imports ``hyperparameter_sampler``, so importing it
#: from this side closes a cycle.  The two vocabularies are held together by a test instead of by
#: discipline.
Direction = Literal["maximize", "minimize"]

#: One acquisition function, on purpose.  A second would double the numeric surface while being
#: exactly as unevidenced about helping; both would need the same missing experiment.
Acquisition = Literal["expected_improvement"]

#: Why a step was chosen the way it was.  ``fallback`` is kept distinct from ``surrogate``: a
#: reader who sees ``surrogate`` is entitled to assume a posterior was actually consulted.
DecisionMode = Literal["initial_design", "surrogate", "fallback"]

#: The posterior's prior variance, in standardised-target units.  The kernel's diagonal is 1 and
#: the targets are scaled to unit variance, so this is the only value that makes the two agree.
_PRIOR_VARIANCE = 1.0

#: The terminal statuses, sorted, for the validator's message.  Read off the vocabulary rather
#: than retyped, so a status this version has never heard of cannot be named.
_TERMINAL_STATUSES = tuple(sorted(TERMINAL_TRIAL_STATUSES))


class SurrogateError(Exception):
    """The adaptive sampler cannot proceed: nothing left to propose, or the arithmetic failed."""


class SurrogateAuditError(SurrogateError):
    """A recorded decision the records contradict: recomputing it chooses a different point.

    Distinct from "unverifiable", which means the records needed to check were not there.  This one
    means they were there and they say something else.
    """


# --- Encoding --------------------------------------------------------------------------------


class EncodingBlock(BaseModel):
    """One parameter's slice of the feature vector.

    A block records the *layout* and not the domain: no ``low``/``high``/``choices`` are copied
    here, because the encoder reads those from the space at encode time.  A copy would be a second
    source of truth for the geometry, and the two would drift apart exactly when a space changes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    #: ``"continuous"`` / ``"discrete"`` / ``"categorical"``, as the spec declares it.
    kind: str
    start: int
    width: int
    #: The index of this parameter's "does it apply at all" dimension, or ``None`` when the
    #: parameter has no condition and therefore always applies.
    activity_dim: int | None = None

    @model_validator(mode="after")
    def _validate_block(self) -> "EncodingBlock":
        if not self.name.strip():
            raise ValueError("an encoding block must name a parameter")
        if self.kind not in ("continuous", "discrete", "categorical"):
            raise ValueError(f"{self.name}: unknown parameter kind {self.kind!r}")
        if self.start < 0:
            raise ValueError(f"{self.name}: a block cannot start before the vector, got {self.start}")
        if self.width < 1:
            raise ValueError(f"{self.name}: a block must be at least one dimension wide, got {self.width}")
        if self.activity_dim is not None and not self.start <= self.activity_dim < self.start + self.width:
            raise ValueError(
                f"{self.name}: the activity dimension {self.activity_dim} must fall inside the "
                f"block's own slice {self.start}..{self.start + self.width - 1}"
            )
        return self


class EncodingSpec(BaseModel):
    """How one space is laid out as a feature vector.  A pure function of the space.

    ``space_fingerprint`` is stored and re-checked at encode time.  Encoding an assignment drawn
    from one space against another space's layout would silently produce a vector in the wrong
    geometry -- every number in it would still look like a number in ``[0, 1]``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SURROGATE_SCHEMA_VERSION
    space_fingerprint: str
    order: tuple[str, ...]
    blocks: tuple[EncodingBlock, ...]

    @model_validator(mode="after")
    def _validate_spec(self) -> "EncodingSpec":
        if not self.space_fingerprint.strip():
            raise ValueError("space_fingerprint must not be blank")
        if not self.order:
            raise ValueError("an encoding spec must cover at least one parameter")
        if tuple(block.name for block in self.blocks) != tuple(self.order):
            raise ValueError(
                f"the blocks {tuple(block.name for block in self.blocks)} are not the sampling "
                f"order {tuple(self.order)}: a vector whose dimensions do not follow the order "
                f"cannot be decoded, and a reader cannot check it"
            )
        expected = 0
        for block in self.blocks:
            if block.start != expected:
                raise ValueError(
                    f"{block.name}: its slice starts at {block.start}, but the blocks before it "
                    f"end at {expected}: a gap or an overlap would place dimensions wrongly"
                )
            expected = block.start + block.width
        return self

    @classmethod
    def for_space(cls, space: SearchSpaceSchema) -> "EncodingSpec":
        """Lay ``space`` out.  One block per parameter, in the space's own sampling order."""
        blocks: list[EncodingBlock] = []
        cursor = 0
        for name in space.sampling_order():
            spec = space.spec(name)
            width = len(spec.choices) if spec.kind == "categorical" else 1
            activity_dim: int | None = None
            if spec.condition is not None:
                # Placed *after* the value dimensions so a value's own slice stays contiguous.
                # Where it sits inside the block is otherwise arbitrary, and this way the block
                # reads as "the values, then whether they apply".
                activity_dim = cursor + width
                width += 1
            blocks.append(
                EncodingBlock(
                    name=name, kind=spec.kind, start=cursor, width=width, activity_dim=activity_dim
                )
            )
            cursor += width
        return cls(
            space_fingerprint=space.fingerprint(), order=space.sampling_order(), blocks=tuple(blocks)
        )

    def width(self) -> int:
        """How many dimensions the vector has."""
        last = self.blocks[-1]
        return last.start + last.width

    def encode(self, space: SearchSpaceSchema, assignment: Mapping[str, Any]) -> tuple[float, ...]:
        """Encode one assignment as a point of the unit cube.

        Raises ``ValueError`` for a value outside its declared domain rather than clamping it: a
        clamp would turn "the ledger holds an illegal value" into "we sampled the nearest legal
        one", and the surrogate would be trained on a point that was never run.
        """
        fingerprint = space.fingerprint()
        if fingerprint != self.space_fingerprint:
            raise ValueError(
                f"this encoding was laid out for space {self.space_fingerprint}, not {fingerprint}"
            )
        order = space.sampling_order()
        if order != self.order:
            # The space's fingerprint ignores declaration order, so a reordered space still matches
            # it while laying its dimensions out differently.
            raise ValueError(f"this encoding was laid out for the order {self.order}, not {order}")
        point = [0.0] * self.width()
        for block in self.blocks:
            if block.activity_dim is not None and not space.is_active(block.name, assignment):
                # Left at zero, activity dimension included: "this parameter never applied" then
                # stays distinguishable from "it applied and took the bottom of its range".
                continue
            if block.activity_dim is not None:
                point[block.activity_dim] = 1.0
            found, value = get_parameter(assignment, block.name)
            if not found:
                raise ValueError(
                    f"{block.name} applies to this assignment but carries no value: there is "
                    f"nothing to encode, and defaulting it would invent a trial nobody ran"
                )
            spec = space.spec(block.name)
            if spec.kind == "categorical":
                forms = [json_form(choice) for choice in spec.choices]
                wanted = json_form(value)
                if wanted not in forms:
                    raise ValueError(f"{block.name}: {value!r} is not one of {list(spec.choices)!r}")
                point[block.start + forms.index(wanted)] = 1.0
            elif spec.kind == "discrete":
                point[block.start] = _encode_discrete(spec, value)
            else:
                point[block.start] = _encode_continuous(spec, value)
        return tuple(point)


def _encode_continuous(spec: HyperparameterSpec, value: Any) -> float:
    """A continuous value as a position in ``[0, 1]``, geometrically when the domain is log-scaled.

    No branch for ``low == high``: a continuous domain is refused outright when the two are equal
    ("low must be strictly below high"), so such a value cannot reach here and a branch for it would
    be code no test could execute.  A one-point domain is expressible only as a discrete or
    categorical parameter, and both of those are handled below.
    """
    spec.validate_value(value)
    low, high = float(spec.low), float(spec.high)
    number = float(value)
    if spec.log:
        # Logarithmically, for the reason the random sampler draws that way: a linear position
        # across two decades would put almost every point in the upper decade.
        position = (math.log(number) - math.log(low)) / (math.log(high) - math.log(low))
    else:
        position = (number - low) / (high - low)
    return _unit(position)


def _encode_discrete(spec: HyperparameterSpec, value: Any) -> float:
    """A discrete value as its index over ``count - 1``.

    The index is computed from the offset and then confirmed with ``identifies``, rather than
    searched for by walking every point: a domain may hold up to a million points, and the claim
    that matters is that the index names the value it came from.
    """
    spec.validate_value(value)
    count = spec.discrete_count()
    if count == 1:
        return 0.0
    offset = (float(value) - float(spec.low)) / float(spec.step)
    index = int(round(offset))
    if index < 0 or index >= count:
        raise ValueError(f"{spec.name}: {value!r} sits at index {index}, outside 0..{count - 1}")
    if not spec.identifies(value, spec.discrete_value(index)):
        raise ValueError(
            f"{spec.name}: {value!r} is inside the domain but is not one of its points "
            f"(index {index} holds {spec.discrete_value(index)!r})"
        )
    # Not through ``_unit``: the index is checked against ``0..count - 1`` above, so the quotient
    # is in ``[0, 1]`` exactly, and a clamp that cannot change its argument reads as though it
    # might.  ``_encode_continuous`` is the one caller that needs it, for the reason its docstring
    # gives.
    return index / (count - 1)


def _unit(position: float) -> float:
    """Clamp a computed position into ``[0, 1]``.

    What it protects is the encoder's **contract**, and the reason is the domain check's tolerance
    rather than float error at the endpoints: both endpoints were measured to encode to exactly
    ``0.0`` and ``1.0``, because the numerator and the denominator are literally the same
    subexpressions there.  What does reach past the ends is a value the domain *accepts*:
    ``contains`` allows anything within ``_TOLERANCE * max(1.0, |low|, |high|)`` (``1e-9`` for
    ``[0, 1]``), so ``validate_value(-1e-18)`` passes and ``(value - low) / (high - low)`` is
    ``-1e-18``.

    That tolerance is absolute in the domain's units, not relative to its width, so a narrow domain
    can be overshot by its whole width: ``[0, 1e-9]`` accepts ``-1e-9`` and ``2e-9``, whose
    positions are ``-1.0`` and ``2.0``.  Both branches are exercised by
    ``tests/autotuner/research/test_bayesian_sampler.py`` with those exact values.

    Honest scope, because an earlier revision of this docstring overstated it: no arithmetic here
    *breaks* on an out-of-cube position.  The RBF is smooth everywhere, so a coordinate of ``-1.0``
    costs nothing in ``predict`` -- it simply makes the point far from the training rows, which is
    correct for a point that far away.  An earlier revision said "a squared-distance kernel term
    would go slightly negative", which is false: the kernel sums *squares*.  The claim this clamp
    actually upholds is that ``encode``'s output is inside the cube its callers are told to expect.

    Only the continuous encoder calls it: a discrete index cannot leave the range, because
    ``0 <= index <= count - 1`` is checked before the division, so ``_encode_discrete`` divides
    directly rather than passing a value through a clamp that provably does nothing to it.
    """
    if position < 0.0:
        return 0.0
    if position > 1.0:
        return 1.0
    return position


# --- The surrogate ---------------------------------------------------------------------------


class GaussianProcess:
    """A zero-mean Gaussian process with an RBF kernel, in plain Python.

    ``n`` is "how many trials have been scored", a number in the dozens, so the cubic Cholesky is
    not the cost that matters; a BLAS-backed implementation would add a heavyweight, undeclared
    dependency to a layer that currently imports nothing beyond ``pydantic`` and its own modules.
    Cholesky rather than an explicit inverse: an inverse is a second chance to lose symmetry, and
    the solve is what is actually needed.

    The kernel is ``k(p, q) = exp(-0.5 * sum((p_i - q_i)^2) / length_scale^2)`` over points of the
    unit cube, and the observation model is ``y = f(x) + noise``.  Everything is deterministic: the
    same inputs give the same outputs bit for bit, which is what lets a recorded decision be
    recomputed and compared exactly.
    """

    def __init__(
        self,
        *,
        length_scale: float,
        noise: float,
        train_x: Sequence[Sequence[float]],
        train_y: Sequence[float],
    ) -> None:
        if not (math.isfinite(length_scale) and length_scale > 0.0):
            raise ValueError(f"length_scale must be positive and finite, got {length_scale!r}")
        if not (math.isfinite(noise) and noise >= 0.0):
            raise ValueError(f"noise must be finite and non-negative, got {noise!r}")
        if len(train_x) != len(train_y):
            raise ValueError(
                f"{len(train_x)} training points but {len(train_y)} targets: one of the two is "
                f"missing a row"
            )
        if not train_x:
            raise ValueError("a posterior needs at least one observation")
        width = len(train_x[0])
        for row, point in enumerate(train_x):
            if len(point) != width:
                raise ValueError(
                    f"training point {row} has {len(point)} dimensions, not {width}: the rows are "
                    f"not all points of one space"
                )
        for target in train_y:
            if not math.isfinite(target):
                raise ValueError(f"a training target must be finite, got {target!r}")
        self._length_scale = length_scale
        self._noise = noise
        self._x = tuple(tuple(float(value) for value in point) for point in train_x)
        self._y = tuple(float(target) for target in train_y)
        self._cholesky = _cholesky(
            [
                [
                    self._kernel(self._x[i], self._x[j]) + (noise if i == j else 0.0)
                    for j in range(len(self._x))
                ]
                for i in range(len(self._x))
            ]
        )

    @property
    def points(self) -> int:
        """How many observations this posterior was built from."""
        return len(self._x)

    def _kernel(self, left: Sequence[float], right: Sequence[float]) -> float:
        total = 0.0
        for a, b in zip(left, right):
            difference = (a - b) / self._length_scale
            total += difference * difference
        return math.exp(-0.5 * total)

    def predict(self, point: Sequence[float]) -> tuple[float, float]:
        """Posterior **mean and variance** at ``point``, in the units the targets were given in.

        The variance is floored at zero: cancelled terms can leave it a hair negative, and a
        negative variance would reach ``sqrt`` as a negative number.  A caller wanting a standard
        deviation takes the square root itself, because that is the one place the difference
        between the two matters and it should be visible there.
        """
        if len(point) != len(self._x[0]):
            raise ValueError(
                f"this posterior was built on {len(self._x[0])}-dimensional points, not {len(point)}"
            )
        kernel = [self._kernel(self._x[i], point) for i in range(len(self._x))]
        alpha = _cholesky_solve(self._cholesky, self._y)
        mean = sum(kernel[i] * alpha[i] for i in range(len(kernel)))
        solved = _cholesky_solve(self._cholesky, kernel)
        variance = _PRIOR_VARIANCE + self._noise - sum(kernel[i] * solved[i] for i in range(len(kernel)))
        return mean, max(variance, 0.0)


def _cholesky(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """Lower triangular ``L`` with ``L L^T = matrix``.

    Raises on a non-positive pivot: that is the honest answer for a matrix that is not positive
    definite, and silently nudging the pivot would hide a kernel or a jitter that is wrong.
    """
    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            total = sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                pivot = matrix[i][i] - total
                if not (pivot > 0.0) or not math.isfinite(pivot):
                    raise ValueError(
                        f"the kernel matrix is not positive definite at row {i} (pivot {pivot!r}): "
                        f"check the noise term and the training points"
                    )
                lower[i][j] = math.sqrt(pivot)
            else:
                lower[i][j] = (matrix[i][j] - total) / lower[j][j]
    return lower


def _cholesky_solve(lower: Sequence[Sequence[float]], target: Sequence[float]) -> list[float]:
    """Solve ``L L^T x = target`` by forward then back substitution."""
    size = len(target)
    forward = [0.0] * size
    for i in range(size):
        forward[i] = (target[i] - sum(lower[i][k] * forward[k] for k in range(i))) / lower[i][i]
    backward = [0.0] * size
    for i in range(size - 1, -1, -1):
        backward[i] = (
            forward[i] - sum(lower[k][i] * backward[k] for k in range(i + 1, size))
        ) / lower[i][i]
    return backward


def expected_improvement(mean: float, sigma: float, best: float, *, xi: float) -> float:
    """Expected improvement over ``best`` for a quantity we want to maximise.

    ``EI = (mu - best - xi) * Phi(z) + sigma * phi(z)`` with ``z = (mu - best - xi) / sigma``.
    A minimisation objective is handled by negating the training targets once, rather than by a
    second formula here -- one formula, one place to be wrong.

    ``sigma <= 0`` returns ``0.0``: a point the model already knows exactly cannot improve on
    anything, and dividing by zero would be the alternative.
    """
    if sigma <= 0.0:
        return 0.0
    z = (mean - best - xi) / sigma
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return (mean - best - xi) * cdf + sigma * pdf


# --- The policy ------------------------------------------------------------------------------


class BayesianPolicy(BaseModel):
    """Every knob of the adaptive sampler, in one recorded place.

    A record rather than a live object: stored content-addressed in the ledger (the way task 5
    stores a pruning policy) so a recorded decision stays checkable after the fact.

    Two of these numbers are measured in **standardised-target units**, not in the objective's own
    units: the targets are centred and scaled to unit variance before any model is built, so
    ``noise`` is a variance on that scale and ``xi`` is a fraction of a standard deviation of the
    objective.  ``length_scale`` is measured in the unit cube the encoder produces.  Writing that
    down matters because the two scales are easy to confuse and the confusion is silent.

    ``direction`` has **no default**, for the reason ``search_analysis.RankingRule`` gives for the
    same choice: a coin-flip default is an invisible decision, and getting it backwards makes every
    later step chase the wrong end of the objective without anything looking broken.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SURROGATE_SCHEMA_VERSION
    direction: Direction
    acquisition: Acquisition = "expected_improvement"
    #: RBF length scale over the unit cube.  A knob the caller chooses rather than one this module
    #: fits: fitting it would add an inner optimisation whose answers are least stable exactly where
    #: this is used -- a handful of observations.
    length_scale: float = 0.5
    #: Diagonal jitter, in standardised-target variance units.  Not zero, because a kernel matrix
    #: of near-duplicate points is otherwise only barely positive definite, and the variance read
    #: off it at that scale means nothing.
    noise: float = 1e-6
    #: The improvement margin in EI, in standardised-target units.  Larger means more willing to
    #: move away from the incumbent.
    xi: float = 0.01
    #: How many of the first steps are drawn as a plain random design regardless of what has been
    #: scored.  The surrogate has nothing to say before it has seen anything.
    initial_design: int = 4
    #: Below this many scored observations no posterior is built at all.  Distinct from
    #: ``initial_design``: that one counts steps, this one counts evidence, and it binds when early
    #: trials failed or were pruned.
    min_train: int = 2
    #: How many candidates a step draws to be scored.  The best of a finite pool rather than a
    #: continuous maximisation: a bounded pool keeps a step's cost and its result reproducible.
    pool_size: int = 64
    #: Which terminal statuses contribute a target.  See the module docstring: a pruned trial's
    #: objective is measured over a truncated horizon.
    learn_from: tuple[TrialStatus, ...] = ("succeeded",)
    #: How many draws one step may spend looking for an unseen, feasible candidate.
    max_attempts: int = 1_000

    @model_validator(mode="after")
    def _validate_policy(self) -> "BayesianPolicy":
        for name in ("length_scale", "noise", "xi"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.length_scale <= 0.0:
            raise ValueError(f"length_scale must be positive, got {self.length_scale}")
        if self.noise < 0.0:
            raise ValueError(f"noise must be non-negative, got {self.noise}")
        if self.xi < 0.0:
            raise ValueError(f"xi must be non-negative, got {self.xi}")
        if self.initial_design < 0:
            raise ValueError(f"initial_design must not be negative, got {self.initial_design}")
        if self.min_train < 1:
            raise ValueError(
                f"min_train must be at least 1, got {self.min_train}: a posterior needs an "
                f"observation to be built from"
            )
        if self.pool_size < 1:
            raise ValueError(f"pool_size must be at least 1, got {self.pool_size}")
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {self.max_attempts}")
        if not self.learn_from:
            raise ValueError(
                "learn_from must name at least one terminal status: a policy that learns from "
                "nothing would silently never adapt"
            )
        unknown = [status for status in self.learn_from if status not in TERMINAL_TRIAL_STATUSES]
        if unknown:
            raise ValueError(
                f"learn_from names {unknown}, which are not terminal trial statuses ({_TERMINAL_STATUSES})"
            )
        if len(set(self.learn_from)) != len(self.learn_from):
            raise ValueError(f"learn_from repeats a status: {self.learn_from}")
        return self

    @property
    def fingerprint(self) -> str:
        """Content address: ``content_hash`` over ``model_dump(mode="json")`` (law R3)."""
        return content_hash(self.model_dump(mode="json"))


def bayesian_plan(
    space: SearchSpaceSchema,
    *,
    seed: int,
    budget: int,
    policy: BayesianPolicy,
) -> SearchPlan:
    """The plan for an adaptive search, with the policy validated here.

    The policy's schema lives in this module, so this factory is the only construction path that
    can check it before the plan exists.  ``SearchPlan`` itself checks the *shape* -- a bayesian plan
    carries a policy and neither of the other kinds does -- but it cannot check the policy's fields
    without importing this module, which is the cycle this factory exists to avoid.

    ``max_attempts`` is **copied** from the policy rather than taken as an argument, and that copy is
    the whole point: the drawing loop reads ``policy.max_attempts`` (``_pool``), the sampler rebuilds
    its policy from ``plan.policy``, and ``SearchPlan`` would otherwise leave the field at its own
    default of 1000 -- so a plan-level override would be a second number that nothing consults,
    recorded in the run header and in every trial, disagreeing with the one that actually bounded
    the search.  There is exactly one such number and the policy holds it.

    An earlier revision took an override as an argument and said in this docstring that a plan and
    its policy therefore "do not silently disagree".  That was true of the default path only: the
    override was accepted, recorded, and ignored, which is the disagreement the sentence denied.
    """

    return SearchPlan(
        kind="bayesian",
        space_fingerprint=space.fingerprint(),
        sampling_order=space.sampling_order(),
        seed=seed,
        budget=budget,
        max_attempts=policy.max_attempts,
        policy=policy.model_dump(mode="json"),
    )


class SurrogatePolicyRecord(BaseModel):
    """The policy, stored content-addressed so a decision stays checkable without it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SURROGATE_SCHEMA_VERSION
    id: str
    policy_fingerprint: str
    policy: dict[str, Any]

    @model_validator(mode="after")
    def _validate_record(self) -> "SurrogatePolicyRecord":
        if self.id != f"surrogate-policy:{self.policy_fingerprint}":
            raise ValueError(
                f"the id {self.id!r} does not carry the fingerprint {self.policy_fingerprint}: "
                f"a derived id means two spellings cannot both claim one policy"
            )
        carried = BayesianPolicy.model_validate(self.policy)
        if carried.fingerprint != self.policy_fingerprint:
            raise ValueError(
                f"the stored policy fingerprints to {carried.fingerprint}, not {self.policy_fingerprint}"
            )
        return self


def surrogate_policy_record(policy: BayesianPolicy) -> SurrogatePolicyRecord:
    """Wrap a policy for storage.  The id is derived, so two spellings cannot both claim it."""
    fingerprint = policy.fingerprint
    return SurrogatePolicyRecord(
        id=f"surrogate-policy:{fingerprint}",
        policy_fingerprint=fingerprint,
        policy=policy.model_dump(mode="json"),
    )


def store_policy_record(
    store: TrialLedgerStore,
    policy: BayesianPolicy,
    *,
    precondition: Callable[[], None] | None,
    actor: str = "bayesian_sampler",
) -> SurrogatePolicyRecord:
    """Store a policy, but only through a ``precondition`` that keeps the ledger readable.

    The hazard is measured, not inferred, and it is the one ``search_pruning`` documents: writing
    any record into an empty search root makes that root unreadable **as a search ledger**, because
    the layout check requires the first event to be the ``search_run`` append.  Measured on this
    module's own record type: after storing one policy into a fresh root, both
    ``SearchLedger(store).refresh()`` and ``SearchTracker.open_run()`` raise
    ``TrialLedgerIntegrityError`` with ``unknown_layout: the ledger holds trial events but no
    search_run record to explain them`` -- so ``TrialLedgerStore`` itself still writes and reads, and
    it is the *search* reader that refuses the root from then on.  A policy stored before the run
    header therefore loses that root for every search done on it.

    ``precondition`` is required to be passed explicitly rather than defaulting to ``None``: a
    default would make the dangerous form the terse one.

    Idempotent, because the id is content-addressed: storing the same policy twice appends nothing
    the second time.
    """
    record = surrogate_policy_record(policy)
    if store.latest(POLICY_RECORD_TYPE, record.id) is not None:
        return record
    store.append(POLICY_RECORD_TYPE, record, actor=actor, precondition=precondition)
    return record


def policy_for(store: TrialLedgerStore, fingerprint: str) -> BayesianPolicy | None:
    """Read a stored policy back.  ``None`` when nothing was ever stored under that fingerprint."""
    payload = store.latest(POLICY_RECORD_TYPE, f"surrogate-policy:{fingerprint}")
    if payload is None:
        return None
    return BayesianPolicy.model_validate(SurrogatePolicyRecord.model_validate(payload).policy)


# --- The decision ----------------------------------------------------------------------------


class SurrogateDecisionRecord(BaseModel):
    """One step's evidence: what the sampler had read, what it computed, and what it chose.

    Written **before** the candidate is proposed, so a decision that cannot be recorded is a
    decision that cannot be acted on: the other order leaves a proposal with no evidence behind it,
    which is the one thing this record type exists to prevent.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SURROGATE_SCHEMA_VERSION
    id: str
    run_ref: str
    policy_fingerprint: str
    plan_fingerprint: str
    space_fingerprint: str
    #: Where in the sequence this step sits -- the same number ``TrialRecord.index`` carries, so a
    #: decision can be matched to the trial it produced without trusting their order on disk.
    index: int
    mode: DecisionMode
    #: How many observations had a usable objective and a status in ``learn_from``.
    train_count: int
    #: Which observations those were, by ``assignment_key``, in the order the model saw them.
    train_keys: tuple[str, ...]
    #: How many candidates the step drew and kept as choosable -- the set the choice was taken
    #: from, in every mode.  **Not** "how many were scored": in ``initial_design`` and ``fallback``
    #: nothing is scored at all, and the pool is still what the assignment was taken out of.  The
    #: two readings are what ``pool_size >= 1`` above is there to keep apart: a *scored* count of 0
    #: would be a legitimate number for those modes, so a field that must never be 0 cannot be it.
    pool_size: int
    chosen_key: str
    #: The acquisition value of the chosen candidate, or ``None`` when no posterior ran.  ``None``
    #: rather than ``0.0``: zero is a value EI really returns, and a reader must be able to tell
    #: "the best candidate scored nothing" from "nothing was scored".
    chosen_ei: float | None = None
    #: The pool's second-best value, so "won by a hair" and "won outright" are distinguishable.
    #: ``None`` when fewer than two candidates were scored.
    runner_up_ei: float | None = None
    detail: str = ""

    @model_validator(mode="after")
    def _validate_decision(self) -> "SurrogateDecisionRecord":
        if not self.run_ref.strip():
            raise ValueError("a decision must name the run it belongs to")
        if self.index < 1:
            raise ValueError(f"a decision's index starts at 1, got {self.index}")
        expected_id = decision_id(self.run_ref, self.index)
        if self.id != expected_id:
            # The id is the key the ledger stores under, so an id that is merely *a* string lets two
            # spellings of the same step sit side by side instead of colliding -- and the second one
            # would read as a second decision rather than as the same one restated.
            raise ValueError(
                f"the id {self.id!r} is not the derived id of step {self.index} of run "
                f"{self.run_ref!r}, which is {expected_id!r}"
            )
        if self.train_count < 0:
            raise ValueError(f"train_count cannot be negative, got {self.train_count}")
        if self.train_count != len(self.train_keys):
            raise ValueError(
                f"train_count is {self.train_count} but {len(self.train_keys)} keys were recorded: "
                f"the count and the list are two claims about one thing"
            )
        if self.pool_size < 1:
            raise ValueError(f"pool_size must be at least 1, got {self.pool_size}")
        if not self.chosen_key.strip():
            raise ValueError("a decision must name the assignment it chose")
        if self.mode == "surrogate":
            if self.train_count == 0:
                # The combination that must never be written: it would read as "a posterior was
                # consulted" while listing nothing that could have been consulted.
                raise ValueError("a surrogate decision must record at least one training observation")
            if self.chosen_ei is None:
                raise ValueError("a surrogate decision must record the acquisition value it chose on")
        elif self.chosen_ei is not None or self.runner_up_ei is not None:
            # An acquisition value with no posterior behind it is the worst kind of record: it
            # looks computed.
            raise ValueError(
                f"a {self.mode} decision scored no candidate, so it cannot carry an acquisition "
                f"value (got chosen_ei={self.chosen_ei!r}, runner_up_ei={self.runner_up_ei!r})"
            )
        for name in ("chosen_ei", "runner_up_ei"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(
                    f"{name} must be a finite number, got {value!r}: the ledger writes non-finite "
                    f"floats as null, and a reader could not tell that apart from an unset field"
                )
        return self

    @classmethod
    def build(
        cls,
        *,
        run_ref: str,
        plan: SearchPlan,
        space: SearchSpaceSchema,
        policy: BayesianPolicy,
        index: int,
        choice: "_Choice",
    ) -> "SurrogateDecisionRecord":
        """One step's record, with the id derived from ``(run, index)`` so it cannot be duplicated."""
        return cls(
            id=decision_id(run_ref, index),
            run_ref=run_ref,
            policy_fingerprint=policy.fingerprint,
            plan_fingerprint=plan.fingerprint(),
            space_fingerprint=space.fingerprint(),
            index=index,
            mode=choice.mode,
            train_count=len(choice.train_keys),
            train_keys=choice.train_keys,
            pool_size=len(choice.pool),
            chosen_key=choice.chosen_key,
            chosen_ei=choice.chosen_ei,
            runner_up_ei=choice.runner_up_ei,
            detail=choice.detail,
        )


def decision_id(run_ref: str, index: int) -> str:
    """The derived id of the decision for one step of one run."""
    return f"surrogate-decision:{run_ref}:{index}"


@runtime_checkable
class SurrogateSink(Protocol):
    """Where a decision goes.  Injected, so this module never touches a file."""

    def record_decision(self, record: SurrogateDecisionRecord) -> None:
        """Write (or re-write, idempotently) one decision."""
        ...


class LedgerSurrogateSink:
    """The reference sink: decisions land through task 4's generic append.

    A decision already recorded under the same id is compared rather than overwritten.  An identical
    one is a no-op, so resuming a search does not grow duplicates; a **different** one raises,
    because a second computation saying something else means the sequence is no longer re-derivable
    -- and quietly keeping the newer one would leave the ledger holding a claim nothing supports.
    """

    def __init__(
        self,
        store: TrialLedgerStore,
        *,
        precondition: Callable[[], None] | None,
        actor: str = "bayesian_sampler",
    ) -> None:
        self._store = store
        self._precondition = precondition
        self._actor = actor

    def record_decision(self, record: SurrogateDecisionRecord) -> None:
        existing = self._store.latest(DECISION_RECORD_TYPE, record.id)
        if existing is not None:
            previous = SurrogateDecisionRecord.model_validate(existing)
            if previous.model_dump(mode="json") != record.model_dump(mode="json"):
                # The whole record, not just the chosen assignment, and with the same field-level
                # wording ``verify_step`` uses: two audit paths describing one disagreement
                # differently is how a reader ends up thinking they are two different checks.
                raise SurrogateAuditError(
                    f"the decision for step {record.index} of run {record.run_ref} is already "
                    f"recorded, and recomputing it now produces something else: "
                    f"{_differences(previous, record)}.  The recorded sequence can no longer be "
                    f"re-derived"
                )
            return
        self._store.append(
            DECISION_RECORD_TYPE, record, actor=self._actor, precondition=self._precondition
        )


def verify_step(
    decision: SurrogateDecisionRecord,
    *,
    plan: SearchPlan,
    prefix: Sequence[TrialRecord],
    space: SearchSpaceSchema,
    policy: BayesianPolicy,
) -> Literal["verified", "unverifiable"]:
    """Recompute one recorded decision from the trials before it.

    Three answers, kept apart on purpose:

    * ``verified`` -- recomputing reproduces the record **field for field**, which is the same
      comparison ``LedgerSurrogateSink`` makes before it refuses a second write.  Checking only the
      chosen assignment would be weaker than the write path it audits: a record naming the right
      point but a wrong acquisition value, mode or pool size would read as verified, and those are
      exactly the fields a reader draws "how close was this" from.
    * ``unverifiable`` -- it cannot be recomputed, because the evidence the record names is not in
      ``prefix``, or the record was written against a different plan, space or policy.  Not a
      contradiction: a partial ledger is the normal state of an interrupted search.
    * ``SurrogateAuditError`` -- it can be recomputed, and the recomputation says something else.
    """
    if policy.fingerprint != decision.policy_fingerprint:
        return "unverifiable"
    if plan.fingerprint() != decision.plan_fingerprint:
        return "unverifiable"
    if space.fingerprint() != decision.space_fingerprint:
        return "unverifiable"
    train = _training_set(prefix, policy=policy, before=decision.index)
    if tuple(record.assignment_key for record, _ in train) != decision.train_keys:
        # The count could match while the members differ, which is why the keys are compared and
        # not the length.
        return "unverifiable"
    seen = {record.assignment_key for record in prefix if record.index < decision.index}
    choice = choose_next(space, plan, policy, index=decision.index, train=train, seen=seen)
    recomputed = SurrogateDecisionRecord.build(
        run_ref=decision.run_ref,
        plan=plan,
        space=space,
        policy=policy,
        index=decision.index,
        choice=choice,
    )
    if recomputed.model_dump(mode="json") != decision.model_dump(mode="json"):
        raise SurrogateAuditError(
            f"step {decision.index} of run {decision.run_ref} is not what recomputing it from the "
            f"same records produces: {_differences(decision, recomputed)}.  The recorded sequence "
            f"can no longer be re-derived"
        )
    return "verified"


def _differences(
    recorded: SurrogateDecisionRecord, recomputed: SurrogateDecisionRecord
) -> str:
    """The fields where a record and its recomputation disagree, each with both values.

    Compared in JSON form, because that is the form the ledger stores and the form
    ``LedgerSurrogateSink`` compares: a tuple that round-trips as a list is not a disagreement, and
    reporting it as one would make the audit refuse records the write path accepted.
    """
    left = recorded.model_dump(mode="json")
    right = recomputed.model_dump(mode="json")
    return ", ".join(
        f"{name}: recorded {left[name]!r} vs recomputed {right[name]!r}"
        for name in sorted(left)
        if left[name] != right[name]
    )


# --- Choosing --------------------------------------------------------------------------------


class _Choice(BaseModel):
    """What one step decided, and everything that decision rested on.

    Internal, and never stored: the assignment itself is carried here because the sampler has to
    yield the very object that was scored, and re-deriving it afterwards would be a second chance
    for the record and the drawing to disagree.  What reaches the ledger is its ``assignment_key``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: DecisionMode
    pool: tuple[str, ...]
    train_keys: tuple[str, ...]
    chosen: dict[str, Any]
    chosen_ei: float | None = None
    runner_up_ei: float | None = None
    detail: str = ""

    @property
    def chosen_key(self) -> str:
        """The canonical key of the chosen assignment."""
        return assignment_key(self.chosen)


def _training_set(
    prefix: Sequence[TrialRecord],
    *,
    policy: BayesianPolicy,
    before: int,
) -> tuple[tuple[TrialRecord, float], ...]:
    """The observations that teach the model, in a total order.

    Ordered by ``(index, attempt)`` rather than by their order on disk: the model's output depends
    on the order its rows are summed in, so an order that depended on how the ledger happened to be
    folded would let two readings of the same evidence disagree.
    """
    selected = [
        record
        for record in prefix
        if record.index < before
        and record.status in policy.learn_from
        and record.objective is not None
    ]
    selected.sort(key=lambda record: (record.index, record.attempt))
    return tuple((record, float(record.objective)) for record in selected)


def _step_rng(plan: SearchPlan, index: int) -> random.Random:
    """The random stream for one step, derived from the plan and the step number.

    Not one stream for the whole run: see the module docstring.  The plan's fingerprint is part of
    the derivation so that the same seed under a different plan -- a different space, a different
    budget, a different policy -- does not produce the same sequence of pools.
    """
    return random.Random(f"{plan.seed}:{plan.fingerprint()[:16]}:{index}")


def _draw_candidate(
    space: SearchSpaceSchema, order: Sequence[str], rng: random.Random
) -> dict[str, Any]:
    """One assignment drawn in dependency order, before any constraint is checked."""
    partial: dict[str, Any] = {}
    for name in order:
        if not space.is_active(name, partial):
            continue
        set_parameter(partial, name, draw_value(space.spec(name), rng), create_missing=True)
    return partial


def _pool(
    space: SearchSpaceSchema,
    plan: SearchPlan,
    policy: BayesianPolicy,
    *,
    index: int,
    seen: set[str],
) -> tuple[tuple[dict[str, Any], ...], dict[str, int]]:
    """Draw up to ``pool_size`` feasible, unseen candidates, with the rejections counted.

    A **short** pool is allowed here, unlike in ``RandomSampler``: that class refuses to return
    fewer trials than asked because a short *search* reads as a complete one.  A pool is not a
    search -- "only three unseen points are left" is a fact the caller needs, not a truncation to
    hide, and the step still has to choose one of them.
    """
    rng = _step_rng(plan, index)
    feasible: list[dict[str, Any]] = []
    keys: set[str] = set()
    constraint_hits: dict[str, int] = {}
    infeasible_draws = 0
    duplicate_draws = 0
    attempts = 0
    while len(feasible) < policy.pool_size and attempts < policy.max_attempts:
        attempts += 1
        candidate = _draw_candidate(space, plan.sampling_order, rng)
        broken = space.unsatisfied_constraints(candidate)
        if broken:
            # Counted per draw, not per constraint: a draw that breaks two constraints is one
            # draw, and adding the per-constraint hits could claim more failing draws than were
            # made -- the same accounting ``_describe_dead_end`` uses.
            infeasible_draws += 1
            for item in broken:
                text = item.describe()
                constraint_hits[text] = constraint_hits.get(text, 0) + 1
            continue
        key = assignment_key(candidate)
        if key in seen or key in keys:
            duplicate_draws += 1
            continue
        keys.add(key)
        feasible.append(candidate)
    counts = {
        "attempts": attempts,
        "infeasible_draws": infeasible_draws,
        "duplicate_draws": duplicate_draws,
    }
    for text, count in constraint_hits.items():
        counts[f"constraint:{text}"] = count
    return tuple(feasible), counts


def _describe_draws(counts: Mapping[str, int]) -> str:
    """Why a pool came up short, naming each cause that occurred.

    The counts are what make the claim checkable: "every draw broke a constraint" and "this
    constraint rejected 9 of the 10 infeasible draws" describe the same run, but only the second
    says whether raising the attempt budget would help or whether the space has nothing left.
    """
    draws = counts["infeasible_draws"] + counts["duplicate_draws"]
    causes: list[str] = []
    constraints = {
        text.split(":", 1)[1]: count
        for text, count in counts.items()
        if text.startswith("constraint:")
    }
    if constraints:
        several = len(constraints) > 1
        listed = ", ".join(
            f"{text} ({count} of {counts['infeasible_draws']} infeasible draws)" if several else text
            for text, count in sorted(constraints.items())
        )
        causes.append(f"{counts['infeasible_draws']} of {draws} broke a constraint: {listed}")
    if counts["duplicate_draws"]:
        causes.append(
            f"{counts['duplicate_draws']} of {draws} repeated an assignment already proposed, so "
            f"the space holds fewer distinct trials than the pool asks for"
        )
    # No fallback string for "no cause at all".  ``_pool`` increments ``attempts`` before it can
    # leave the loop, and a policy's ``pool_size`` and ``max_attempts`` are both at least 1
    # (``BayesianPolicy._validate_policy``, pinned by
    # ``test_the_policy_has_no_default_direction_and_refuses_the_values_it_cannot_use``), so at
    # least one draw happens and an empty pool therefore carries at least one counted cause.  A
    # default for the empty case would be a branch no test can execute.
    return "; ".join(causes)


def _standardise(targets: Sequence[float]) -> list[float] | None:
    """Centre and scale the targets, or ``None`` when they carry no variation.

    ``None`` is not an error: "every scored trial came out the same" is a real outcome, and the
    caller records it as a ``fallback`` rather than fitting a model to it.

    The guard is on **this function's own division**: the last line divides by ``deviation``, so a
    target set with no spread has no scale to divide by.  (An earlier revision of this docstring
    said the reason was that such a posterior "has zero variance everywhere", which is false: the
    predictive variance is ``_PRIOR_VARIANCE + noise - kᵀK⁻¹k`` and is only ever exactly zero at a
    training point with ``noise=0``.  A constant target is refused because of the scale, not because
    the model would come out flat.)
    """
    mean = sum(targets) / len(targets)
    variance = sum((target - mean) ** 2 for target in targets) / len(targets)
    deviation = math.sqrt(variance)
    if deviation <= 0.0:
        return None
    return [(target - mean) / deviation for target in targets]


def choose_next(
    space: SearchSpaceSchema,
    plan: SearchPlan,
    policy: BayesianPolicy,
    *,
    index: int,
    train: Sequence[tuple[TrialRecord, float]],
    seen: set[str],
) -> _Choice:
    """The one function that decides a step, so the sampler and the verifier cannot disagree.

    ``seen`` is every assignment this run already recorded, whatever its status; ``train`` holds
    only the ones allowed to teach the model.  The two sets are deliberately different: a pruned or
    failed trial is a point already spent -- proposing it again wastes the budget -- while its
    objective may be exactly the thing that must not be regressed against.

    Ties are broken by pool position: the winner is the first candidate in the pool attaining the
    best score, not the last and not a re-sorted one.  That matters because a tie is not an edge
    case here -- with ``noise=0`` every candidate the model was fitted on scores exactly ``0.0``,
    so a whole pool can be tied and the rule is what decides the step.  Pinned by
    ``test_a_pool_where_every_candidate_ties_takes_the_first_one_in_pool_order``.
    """
    pool, counts = _pool(space, plan, policy, index=index, seen=seen)
    if not pool:
        raise SurrogateError(
            f"step {index}: no unseen feasible candidate could be drawn in {counts['attempts']} "
            f"attempt(s) ({_describe_draws(counts)}): the space holds no more distinct trials than "
            f"have already been proposed, so the budget cannot be filled"
        )
    keys = tuple(assignment_key(candidate) for candidate in pool)
    train_keys = tuple(record.assignment_key for record, _ in train)
    if index <= policy.initial_design or len(train) < policy.min_train:
        why = (
            f"step {index} is within the initial design of {policy.initial_design}"
            if index <= policy.initial_design
            else f"only {len(train)} scored observation(s), below min_train={policy.min_train}"
        )
        return _Choice(
            mode="initial_design",
            pool=keys,
            train_keys=train_keys,
            chosen=pool[0],
            detail=f"{why}, so the first unseen candidate was taken and no posterior was built",
        )
    encoded = EncodingSpec.for_space(space)
    targets = [-target if policy.direction == "minimize" else target for _, target in train]
    scaled = _standardise(targets)
    if scaled is None:
        return _Choice(
            mode="fallback",
            pool=keys,
            train_keys=train_keys,
            chosen=pool[0],
            detail=(
                f"all {len(train)} scored observation(s) carry the same objective, so the "
                f"surrogate has no variation to fit; the first unseen candidate was taken"
            ),
        )
    points = [encoded.encode(space, record.assignment) for record, _ in train]
    try:
        model = GaussianProcess(
            length_scale=policy.length_scale,
            noise=policy.noise,
            train_x=points,
            train_y=scaled,
        )
    except ValueError as error:
        # Recorded rather than raised: a policy that cannot build a posterior at all would fail on
        # every later step too, so raising would end the search instead of producing the random
        # design the caller would have got with kind="random" -- and every step says so in writing.
        return _Choice(
            mode="fallback",
            pool=keys,
            train_keys=train_keys,
            chosen=pool[0],
            detail=(
                f"no posterior could be built from {len(train)} observation(s) ({error}); the "
                f"first unseen candidate was taken"
            ),
        )
    scores: list[float] = []
    for candidate in pool:
        mean, variance = model.predict(encoded.encode(space, candidate))
        scores.append(expected_improvement(mean, math.sqrt(variance), max(scaled), xi=policy.xi))
    if not all(math.isfinite(score) for score in scores):
        # Guarded rather than replaced: a non-finite acquisition value means the arithmetic went
        # wrong somewhere the model did build, and recording it would persist as a null that no
        # reader can interpret.
        raise SurrogateError(
            f"step {index}: the acquisition produced a non-finite value, so no candidate can be "
            f"ranked; the kernel or the training targets are at fault"
        )
    best = max(scores)
    winner = scores.index(best)
    ordered = sorted(scores, reverse=True)
    return _Choice(
        mode="surrogate",
        pool=keys,
        train_keys=train_keys,
        chosen=pool[winner],
        chosen_ei=best,
        runner_up_ei=ordered[1] if len(ordered) > 1 else None,
        detail=(
            f"scored {len(pool)} candidate(s) against {len(train)} observation(s) with "
            f"length_scale={policy.length_scale}, noise={policy.noise}, xi={policy.xi} "
            f"({policy.acquisition}, {policy.direction})"
        ),
    )


# --- The sampler -----------------------------------------------------------------------------


class BayesianSampler(Sampler):
    """Proposes each next trial from the trials already recorded.

    Constructed with the observation source, because the sequence is a function of evidence and not
    of the plan alone: ``build_sampler(space, plan)`` has nowhere to read it from, and it refuses
    this kind rather than falling back to a random draw.

    ``_candidates`` is a generator, and that is the whole reason this plugs into task 4's run loop
    without changing it.  ``SearchTracker.run`` pulls one assignment at a time and runs it to
    completion before pulling the next, so by the time step N+1 asks for its observations, trial N's
    terminal record is already on disk.  A caller that batched trials would break that: points in
    one batch are drawn before any of them has a result, and the sampler would no longer be adapting
    to anything.
    """

    def __init__(
        self,
        space: SearchSpaceSchema,
        plan: SearchPlan,
        *,
        observations: TrialObservationSource,
        sink: SurrogateSink | None = None,
        run_ref: str = "",
    ) -> None:
        super().__init__(space, plan)
        if plan.kind != "bayesian":
            raise ValueError(
                f"this sampler needs a bayesian plan, got {plan.kind!r}: a bayesian sampler "
                f"re-derives its sequence from evidence, so it cannot stand in for another kind"
            )
        # No check for ``plan.policy is None``: ``SearchPlan``'s own validator refuses a bayesian
        # plan without a policy ("a bayesian sampler requires a surrogate policy"), so a plan that
        # reached here with ``kind="bayesian"`` carries one.  A branch for it here would be a branch
        # no test could execute -- measured, not assumed:
        # ``SearchPlan(kind="bayesian", ..., policy=None)`` raises before this constructor runs, and
        # ``tests/autotuner/research/test_bayesian_sampler.py`` pins that refusal.
        self.policy = BayesianPolicy.model_validate(plan.policy)
        self._observations = observations
        self._sink = sink
        self._run_ref = run_ref or default_run_ref(plan, space)
        self._proposed: list[str] = []

    def _candidates(self) -> Iterator[dict[str, Any]]:
        budget = self.plan.budget
        # No branch for ``budget is None``: ``SearchPlan``'s own validator refuses a bayesian plan
        # without one ("a bayesian sampler requires a budget"), and that refusal is pinned by a
        # test, so a check here would be a second, unexecutable copy of it.  The same holds for
        # ``policy is None`` in ``__init__``.
        for index in range(1, budget + 1):
            records = self._observations.observations(
                plan_fingerprint=self.plan.fingerprint(),
                space_fingerprint=self.space.fingerprint(),
            )
            # ``seen`` is this run's own proposals plus the trials *before* this step -- not every
            # recording the source hands back.  The index bound is what makes the sequence a
            # sequence: a step may not consult a trial that ran after it, or a search would depend
            # on how the runner interleaved the trials instead of on the plan.  The corollary is
            # that step 1 may propose exactly the point trial 1 recorded, because at that moment it
            # does not exist yet; pinned by
            # ``test_a_search_whose_every_trial_was_pruned_never_consults_a_posterior``.
            # The status filter is deliberately absent here and present in ``_training_set``: a
            # pruned trial is spent (so it must not be re-proposed) but must not be fitted.  That
            # test tells the two apart by the size of the following step's pool.
            seen = set(self._proposed)
            seen.update(record.assignment_key for record in records if record.index < index)
            train = _training_set(records, policy=self.policy, before=index)
            choice = choose_next(
                self.space, self.plan, self.policy, index=index, train=train, seen=seen
            )
            if self._sink is not None:
                # Recorded *before* it is yielded.  A decision that cannot be recorded must not be
                # acted on, or the ledger ends up holding a proposal with no evidence behind it --
                # which is the one thing this record type exists to prevent.
                self._sink.record_decision(
                    SurrogateDecisionRecord.build(
                        run_ref=self._run_ref,
                        plan=self.plan,
                        space=self.space,
                        policy=self.policy,
                        index=index,
                        choice=choice,
                    )
                )
            self._proposed.append(choice.chosen_key)
            yield choice.chosen
        self.stats.exhausted = True
