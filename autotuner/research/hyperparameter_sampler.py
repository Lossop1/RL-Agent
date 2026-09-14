"""Deterministic, replayable ways to draw trials from a search space.

A search space says what may vary; a sampler says in what order and with which
values to try it.  Keeping the two apart is what makes a recorded search
replayable: the space is data with a fingerprint, and a sampler is a pure
function of that fingerprint, a seed, and a budget.

Two properties are load-bearing and easy to lose:

* **Determinism.**  Every sampler draws from its own ``random.Random`` instance.
  The module-level random functions share one global stream, so a draw would then
  depend on how much randomness unrelated code consumed first, and a replay of
  the same seed would produce a different sequence.
* **No silent truncation.**  A sampler that quietly returns fewer trials than
  asked reads as a completed search.  A grid stopped by its budget reports
  ``exhausted=False``; a random draw that cannot find a feasible, unseen
  assignment within its attempt budget raises instead of returning a short run.

Parameters are drawn in dependency order, so a conditional parameter's guard is
always placed first: whether ``kl_threshold`` is active is undecided until the
scheduler has been chosen.
"""
from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Mapping

from pydantic import BaseModel, ConfigDict, model_validator

from .hyperparameter_space import HyperparameterSpec, SearchSpaceSchema, set_parameter
from .research_ledger import content_hash


SAMPLER_SCHEMA_VERSION = "rl-agent.hyperparameter-sampler/v1"

SamplerKind = Literal["grid", "random"]

_MAX_ATTEMPTS_DEFAULT = 1_000

# A grid with no budget must be fully enumerable in memory.  A single parameter
# is capped at a million points by the space, which says nothing about the
# product of several such parameters: five of them is 10^30 trials.  Refusing
# early costs one multiplication; discovering it later costs a hang.
_MAX_GRID_TRIALS = 100_000


def assignment_key(assignment: Mapping[str, Any]) -> str:
    """Canonical identity of a drawn assignment.

    Two trials are the same trial when they set the same parameters to the same
    values, whatever the insertion order.  The key is built from the serialized
    form rather than from the values, so ``1`` and ``1.0`` stay distinct even
    though they compare equal.

    There is deliberately no fallback for a value JSON cannot express.  Such a
    fallback would have to be ``repr``, whose output for a set of strings changes
    with the process's hash seed -- the same assignment would then key differently
    in two processes, and a dedup check or a replay would silently disagree with
    itself.  A sampler cannot produce such a value, because the space rejects a
    choice without a JSON form; raising here is the honest way to find out
    otherwise.
    """
    return json.dumps(assignment, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class SearchPlan(BaseModel):
    """What was sampled, to the detail needed to sample it again.

    The plan is the record a search run cites, in the same way a schema
    fingerprint is the record a space cites.  ``seed`` is absent for a grid
    because a grid has no randomness to reproduce: recording one would suggest
    the sequence depended on it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SAMPLER_SCHEMA_VERSION
    kind: SamplerKind
    space_fingerprint: str
    sampling_order: tuple[str, ...]
    seed: int | None = None
    budget: int | None = None
    max_attempts: int = _MAX_ATTEMPTS_DEFAULT

    @model_validator(mode="after")
    def _validate_plan(self) -> "SearchPlan":
        if not self.space_fingerprint.strip():
            raise ValueError("space_fingerprint must not be blank")
        if not self.sampling_order:
            raise ValueError("sampling_order must not be empty")
        if len(set(self.sampling_order)) != len(self.sampling_order):
            raise ValueError(f"sampling_order repeats a parameter: {self.sampling_order}")
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {self.max_attempts}")
        if self.budget is not None and self.budget < 1:
            raise ValueError(f"budget must be at least 1, got {self.budget}")
        if self.kind == "grid":
            if self.seed is not None:
                raise ValueError("a grid sampler has no randomness, so it takes no seed")
        elif self.seed is None:
            raise ValueError("a random sampler requires a seed, or it cannot be replayed")
        elif self.budget is None:
            # Without a budget the stream never ends, and a caller that iterates
            # it would hang rather than finish a search.
            raise ValueError("a random sampler requires a budget")
        return self

    @classmethod
    def for_grid(cls, space: SearchSpaceSchema, *, budget: int | None = None) -> "SearchPlan":
        """A plan that enumerates ``space`` in dependency order."""
        return cls(
            kind="grid",
            space_fingerprint=space.fingerprint(),
            sampling_order=space.sampling_order(),
            budget=budget,
        )

    @classmethod
    def for_random(
        cls,
        space: SearchSpaceSchema,
        *,
        seed: int,
        budget: int,
        max_attempts: int = _MAX_ATTEMPTS_DEFAULT,
    ) -> "SearchPlan":
        """A plan that draws ``budget`` feasible, unseen assignments from ``space``."""
        return cls(
            kind="random",
            space_fingerprint=space.fingerprint(),
            sampling_order=space.sampling_order(),
            seed=seed,
            budget=budget,
            max_attempts=max_attempts,
        )

    def fingerprint(self) -> str:
        """Content-addressed identity of this plan.

        ``sampling_order`` is part of it even though it is absent from the
        space's own fingerprint.  A space's identity deliberately ignores
        declaration order; the *sequence* a plan replays does not, because a
        reordered declaration enumerates the same grid in a different order.
        The plan is what promises a sequence, so the promise includes the order.
        """
        return content_hash(self.model_dump(mode="json"))


@dataclass
class SamplerStats:
    """What happened while drawing, for a search report to quote.

    ``exhausted`` is False whenever a budget stopped the stream early, which is
    the difference between "this was the whole grid" and "this was the first N
    points of it".  A grid that is neither exhausted nor truncated walked every
    point it had.
    """

    proposals: int = 0
    attempts: int = 0
    rejected_infeasible: int = 0
    rejected_duplicate: int = 0
    exhausted: bool = False


class Sampler:
    """Base class: a plan, a space, and a stream of validated assignments."""

    def __init__(self, space: SearchSpaceSchema, plan: SearchPlan) -> None:
        if plan.space_fingerprint != space.fingerprint():
            raise ValueError(
                "the plan describes a different search space: "
                f"plan cites {plan.space_fingerprint}, space is {space.fingerprint()}"
            )
        order = space.sampling_order()
        if plan.sampling_order != order:
            # The fingerprint ignores declaration order, so a reordered space
            # still matches it while enumerating in a different sequence.  A
            # plan that promised a sequence must not be replayed against one.
            raise ValueError(
                f"the plan was recorded against a different parameter order: "
                f"plan has {plan.sampling_order}, space has {order}"
            )
        self.space = space
        self.plan = plan
        self.stats = SamplerStats()

    def proposals(self) -> Iterator[dict[str, Any]]:
        """Yield each assignment once, in the plan's order.

        Every assignment is validated against the space before it is yielded, so
        a sampler bug surfaces here rather than in a training run that fails to
        start an hour later.
        """
        for candidate in self._candidates():
            self.space.validate_assignment(candidate)
            self.stats.proposals += 1
            yield candidate

    def collect(self) -> tuple[dict[str, Any], ...]:
        """The whole proposal stream, as a tuple."""
        return tuple(self.proposals())

    def _candidates(self) -> Iterator[dict[str, Any]]:
        raise NotImplementedError


class GridSampler(Sampler):
    """Enumerates every feasible point of the grid, in dependency order.

    The enumeration is recursive rather than a flat cartesian product because
    the set of active parameters depends on the values drawn before them: a
    parameter guarded by a scheduler that was never selected is not a dimension
    of the grid at all, and iterating it would emit assignments carrying a value
    the trainer never reads.

    Every parameter must be enumerable.  One without ``grid_points`` has no
    agreed discretization, and inventing a density here would put a hidden
    decision in code that belongs in the product's data; such a space needs
    ``RandomSampler``, or the parameter needs ``grid_points`` declared.
    """

    def __init__(self, space: SearchSpaceSchema, plan: SearchPlan) -> None:
        super().__init__(space, plan)
        grids: dict[str, tuple[Any, ...]] = {}
        unenumerable: list[str] = []
        for spec in space.specs:
            values = spec.grid_values()
            if values is None:
                unenumerable.append(spec.name)
            else:
                grids[spec.name] = values
        if unenumerable:
            # Skipping them would emit assignments that leave the parameter at
            # whatever the baseline config says while the search report claims to
            # have covered a grid -- a whole dimension searched with one value,
            # and nothing in the record showing it.
            raise ValueError(
                f"a grid cannot enumerate {unenumerable}: no grid_points declared for them. "
                f"Declare grid_points in the search space, or draw with RandomSampler"
            )
        self._grids = grids
        if plan.budget is None:
            product = 1
            for values in grids.values():
                product *= len(values)
            if product > _MAX_GRID_TRIALS:
                raise ValueError(
                    f"the grid holds up to {product} points, over the {_MAX_GRID_TRIALS} a "
                    f"complete enumeration may hold; set a budget to enumerate only part of it"
                )

    def _candidates(self) -> Iterator[dict[str, Any]]:
        budget = self.plan.budget
        produced = 0
        seen: set[str] = set()
        for candidate in self._enumerate({}, 0):
            if budget is not None and produced >= budget:
                return
            key = assignment_key(candidate)
            if key in seen:
                # A grid walks each combination once, so this means the walk
                # itself is wrong -- not that the space is coarse.
                self.stats.rejected_duplicate += 1
                raise ValueError(f"the grid enumerated the same assignment twice: {key}")
            seen.add(key)
            produced += 1
            yield candidate
        self.stats.exhausted = True

    def _enumerate(self, partial: dict[str, Any], position: int) -> Iterator[dict[str, Any]]:
        if position == len(self.plan.sampling_order):
            if not self.space.is_feasible(partial):
                self.stats.rejected_infeasible += 1
                return
            yield partial
            return
        name = self.plan.sampling_order[position]
        if not self.space.is_active(name, partial):
            # The guard already placed rules this parameter out, so the grid has
            # no dimension here and the branch is simply skipped.
            yield from self._enumerate(partial, position + 1)
            return
        for value in self._grids[name]:
            # Copied rather than mutated-and-undone: a value placed deeper in the
            # recursion would otherwise survive into a sibling branch that no
            # longer activates it, and the emitted assignment would carry a value
            # the trainer ignores.
            extended = copy.deepcopy(partial)
            set_parameter(extended, name, value, create_missing=True)
            yield from self._enumerate(extended, position + 1)


class RandomSampler(Sampler):
    """Draws a fixed number of feasible, unseen assignments from the domains.

    Rejection is bounded.  A draw that breaks a constraint is retried, and a
    run that cannot fill its budget within ``max_attempts`` draws per trial
    raises rather than returning a shorter run: a search that silently produced
    30 of 50 trials would be reported as complete, and the missing trials would
    look like they were never part of the plan.
    """

    def __init__(self, space: SearchSpaceSchema, plan: SearchPlan) -> None:
        super().__init__(space, plan)
        self._rng = random.Random(plan.seed)

    def _candidates(self) -> Iterator[dict[str, Any]]:
        budget = self.plan.budget
        if budget is None:  # unreachable: SearchPlan requires it for a random draw
            raise ValueError("a random sampler requires a budget")
        seen: set[str] = set()
        for trial in range(budget):
            # Counted rather than overwritten.  One trial's draws can fail for
            # both reasons at once -- a constraint rejecting most of the space
            # and a budget asking for more distinct trials than the space holds
            # -- and remembering only the last draw's cause would name one of
            # them while reading as if the other never happened.
            constraint_hits: dict[str, int] = {}
            infeasible_draws = 0
            duplicate_hits = 0
            for _ in range(self.plan.max_attempts):
                self.stats.attempts += 1
                candidate = self._draw()
                broken = self.space.unsatisfied_constraints(candidate)
                if broken:
                    self.stats.rejected_infeasible += 1
                    # Counted per draw, not per constraint.  A draw that breaks
                    # two constraints is one draw, and a count that added the
                    # per-constraint hits together could claim more failing draws
                    # than the trial made.
                    infeasible_draws += 1
                    for item in broken:
                        text = item.describe()
                        constraint_hits[text] = constraint_hits.get(text, 0) + 1
                    continue
                key = assignment_key(candidate)
                if key in seen:
                    self.stats.rejected_duplicate += 1
                    duplicate_hits += 1
                    continue
                seen.add(key)
                yield candidate
                break
            else:
                # The same total the causes are counted against, so the two halves
                # of the message cannot disagree about how many draws were made.
                draws = infeasible_draws + duplicate_hits
                plural = "draw" if draws == 1 else "draws"
                raise ValueError(
                    f"gave up after {draws} {plural} for trial {trial + 1} of {budget}: "
                    f"{_describe_dead_end(constraint_hits, infeasible_draws, duplicate_hits)}"
                )
        self.stats.exhausted = True

    def _draw(self) -> dict[str, Any]:
        """One assignment, drawn in dependency order."""
        partial: dict[str, Any] = {}
        for name in self.plan.sampling_order:
            if not self.space.is_active(name, partial):
                continue
            set_parameter(
                partial,
                name,
                _draw_value(self.space.spec(name), self._rng),
                create_missing=True,
            )
        return partial


def _describe_dead_end(
    constraint_hits: Mapping[str, int],
    infeasible_draws: int,
    duplicate_draws: int,
) -> str:
    """Why every draw of one trial failed, naming each cause that occurred.

    The counts are what make the claim checkable.  "Every draw broke a
    constraint" and "this constraint rejected 9 of the 10 infeasible draws"
    describe the same run, but only the second tells a reader whether widening
    the budget would help or whether the space itself has nothing left to give.

    Every denominator is derived here from the causes rather than passed in, so
    the message cannot report more failing draws than the trial made.  Summing
    them is sound because a draw is counted towards exactly one cause: a broken
    constraint short-circuits before the duplicate test.
    """
    draws = infeasible_draws + duplicate_draws
    causes: list[str] = []
    if constraint_hits:
        # One draw can break several constraints at once, so the breakdown only
        # earns its space when there is more than one cause to break down.  Its
        # denominator is the infeasible draws, not all of them: a draw rejected
        # as a duplicate never reached the constraints, so counting it here would
        # understate this constraint's share.
        several = len(constraint_hits) > 1
        listed = ", ".join(
            f"{text} ({count} of {infeasible_draws} infeasible draws)" if several else text
            for text, count in sorted(constraint_hits.items())
        )
        causes.append(f"{infeasible_draws} of {draws} broke a constraint: {listed}")
    if duplicate_draws:
        causes.append(
            f"{duplicate_draws} of {draws} repeated an assignment already proposed, so the "
            f"space holds fewer distinct trials than the budget asks for"
        )
    # Unreachable while max_attempts is positive: every attempt either yields or
    # records a cause, and one of the two counts is therefore non-zero here.  An
    # error message with no cause in it would be worse than this one, so the
    # empty case still says something true.
    return "; ".join(causes) if causes else "no draw produced an assignment"


def _draw_value(spec: HyperparameterSpec, rng: random.Random) -> Any:
    """One value from ``spec``'s domain, uniformly over the domain's points."""
    if spec.kind == "categorical":
        return rng.choice(spec.choices)
    if spec.kind == "discrete":
        # Drawn by index rather than by snapping a continuous draw, so both
        # endpoints carry the same weight as every interior point.  Snapping
        # would give the endpoints half the weight of their neighbours, because
        # only one side of the interval maps to them.
        return spec.discrete_value(rng.randrange(spec.discrete_count()))
    low, high = float(spec.low), float(spec.high)
    if spec.log:
        # Uniform over the exponent: a linear draw across two decades would put
        # almost every sample in the upper decade.
        return math.exp(rng.uniform(math.log(low), math.log(high)))
    return rng.uniform(low, high)


def build_sampler(space: SearchSpaceSchema, plan: SearchPlan) -> Sampler:
    """The sampler a plan describes, ready to draw from ``space``."""
    if plan.kind == "grid":
        return GridSampler(space, plan)
    return RandomSampler(space, plan)


def replay(space: SearchSpaceSchema, plan: SearchPlan) -> tuple[dict[str, Any], ...]:
    """Rebuild the exact sequence of assignments a plan produced.

    The caller compares this against the recorded trials, keyed by
    :func:`assignment_key`; a mismatch means the space or the sampler changed
    under a record that claims to describe this search.
    """
    return build_sampler(space, plan).collect()
