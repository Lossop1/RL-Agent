"""Declarative, versioned descriptions of what a training run may vary.

A search space states which configuration keys are tunable, how their values may
be drawn, and which combinations are legal.  It is deliberately data-only: the
sampler, the config injector, and the experiment planner all consume the same
description instead of each maintaining a private list of parameter names.

Parameter names are dotted paths into the effective training config, for example
``skrl.agent.learning_rate``.  A spec never carries the value used by the
baseline run in addition to ``default`` -- the baseline is whatever the product
config already says, and ``default`` exists so a sampler can fall back to it when
a conditional parameter is inactive.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, model_validator

from .research_ledger import content_hash


SEARCH_SPACE_SCHEMA_VERSION = "rl-agent.hyperparameter-search/v1"
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)*$")
_MAX_NAME_LENGTH = 128

ParameterKind = Literal["continuous", "discrete", "categorical"]


def _is_number(value: Any) -> bool:
    """Reject booleans, which are ``int`` subclasses and never a numeric domain."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _choice_key(value: Any) -> str:
    """Stable identity for a categorical choice, including unhashable ones."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


_TOLERANCE = 1.0e-9
_MAX_INTEGRALITY_SLACK = 1.0e-6
_MAX_ENUMERABLE_POINTS = 1_000_000


def _nearly_integral(value: float, *, tolerance: float = _TOLERANCE) -> bool:
    """Whether ``value`` is a whole number, allowing for float error.

    The slack widens with magnitude so that a large step count still compares, but
    it is capped.  An uncapped relative slack admits the midpoint between two
    neighbouring grid points as "integral" once a domain holds a billion of them,
    which would let ``contains`` accept a value that is not on the grid at all.
    """
    return abs(value - round(value)) <= min(tolerance * max(1.0, abs(value)), _MAX_INTEGRALITY_SLACK)


def _within_bounds(number: float, low: float, high: float, *, tolerance: float = _TOLERANCE) -> bool:
    """Compare against a bound while absorbing float error at the endpoints."""
    margin = tolerance * max(1.0, abs(low), abs(high))
    return low - margin <= number <= high + margin


def _find_non_finite(value: Any) -> float | None:
    """Return the first non-finite number nested in ``value``, or ``None``.

    Searched recursively because a choice may be a list or a mapping, and a ``nan``
    buried inside one is just as unloadable as a bare one.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            found = _find_non_finite(item)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            found = _find_non_finite(item)
            if found is not None:
                return found
    return None


def _validate_parameter_name(name: str, *, label: str) -> None:
    """Reject a name that :func:`split_parameter` would reject.

    Both grammars must agree.  A name this layer accepts but the path helpers
    refuse produces a spec that validates, fingerprints, and then raises on every
    lookup -- the failure surfaces far from its cause.
    """
    if not name or len(name) > _MAX_NAME_LENGTH or not _IDENTIFIER.fullmatch(name):
        raise ValueError(f"invalid {label}: {name!r}")


def split_parameter(name: str) -> tuple[str, ...]:
    """Split a dotted parameter path into its segments.

    The grammar forbids empty segments, so a doubled or trailing dot is a name
    error rather than an empty-string key written into a training config.
    """
    _validate_parameter_name(name, label="parameter name")
    return tuple(name.split("."))


def get_parameter(assignment: Mapping[str, Any], name: str) -> tuple[bool, Any]:
    """Return ``(found, value)`` for a dotted path.

    The flag distinguishes an absent key from one whose value is ``None``, which
    matters because a legal categorical choice may be ``None``.
    """
    node: Any = assignment
    for segment in split_parameter(name):
        if not isinstance(node, Mapping) or segment not in node:
            return False, None
        node = node[segment]
    return True, node


def set_parameter(
    assignment: dict[str, Any],
    name: str,
    value: Any,
    *,
    create_missing: bool = False,
) -> None:
    """Assign a dotted path in place.

    Following ``ChangeSet._set_key``, an unknown path is an error by default: a
    misspelled parameter name must fail loudly rather than add an orphan key that
    leaves the real hyperparameter at its baseline while the trial record claims
    otherwise.  ``create_missing`` opts in to building the path.
    """
    segments = split_parameter(name)
    node: Any = assignment
    for segment in segments[:-1]:
        if not isinstance(node, dict):
            raise ValueError(f"config key does not exist: {name}")
        if segment not in node:
            if not create_missing:
                raise ValueError(f"config key does not exist: {name}")
            node[segment] = {}
        child = node[segment]
        if not isinstance(child, dict):
            raise ValueError(f"{segment!r} on the path to {name} is not a mapping")
        node = child
    if not isinstance(node, dict):
        raise ValueError(f"config key does not exist: {name}")
    if segments[-1] not in node and not create_missing:
        raise ValueError(f"config key does not exist: {name}")
    node[segments[-1]] = value


class ParameterCondition(BaseModel):
    """Restricts a parameter to the combinations where it is meaningful.

    ``kl_threshold`` only affects training while the KL-adaptive scheduler is
    selected; sampling it unconditionally would produce configurations whose
    recorded value never reached the optimizer.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parameter: str
    values: tuple[Any, ...] = ()

    @model_validator(mode="after")
    def _validate_condition(self) -> "ParameterCondition":
        _validate_parameter_name(self.parameter, label="condition parameter")
        if not self.values:
            raise ValueError("condition requires at least one value")
        for value in self.values:
            runaway = _find_non_finite(value)
            if runaway is not None:
                raise ValueError(f"condition values must be finite, got {runaway!r}")
        keys = [_choice_key(value) for value in self.values]
        if len(set(keys)) != len(keys):
            raise ValueError("condition values must be unique")
        return self

    def applies_to(self, assignment: Mapping[str, Any]) -> bool:
        """Return whether the guarded parameter is active for this assignment."""
        found, value = get_parameter(assignment, self.parameter)
        if not found:
            return False
        return _choice_key(value) in {_choice_key(item) for item in self.values}


class ParameterConstraint(BaseModel):
    """A relation between two parameters that must hold before a trial runs.

    Some combinations are individually inside their domain but jointly rejected by
    the trainer: the Taili config keeps ``rollouts`` a whole multiple of
    ``mini_batches``, so a search that drew the two ranges independently would emit
    configurations the training run refuses to start -- a wasted GPU slot and a
    trial record that says nothing about the hyperparameters.

    Only relations a real config needs are modelled.  There is no expression
    language here, and none is planned: a new relation is a new ``kind``, added
    when a config demonstrates the need.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["multiple"]
    parameter: str
    of: str

    @model_validator(mode="after")
    def _validate_constraint(self) -> "ParameterConstraint":
        for name in (self.parameter, self.of):
            _validate_parameter_name(name, label="constraint parameter")
        if self.parameter == self.of:
            raise ValueError(f"{self.parameter} cannot be constrained against itself")
        return self

    def holds(self, assignment: Mapping[str, Any]) -> bool:
        """Return whether the relation holds for the values this assignment carries.

        An assignment that omits either side counts as satisfying the relation: a
        sampler checks feasibility mid-draw, before every parameter is placed, and
        a missing value is a completeness problem for
        :meth:`SearchSpaceSchema.validate_assignment`, not a violated relation.
        """
        found_left, left = get_parameter(assignment, self.parameter)
        found_right, right = get_parameter(assignment, self.of)
        if not (found_left and found_right):
            return True
        if not (_is_number(left) and _is_number(right)):
            return False
        left, right = float(left), float(right)
        if right == 0.0 or not (math.isfinite(left) and math.isfinite(right)):
            return False
        return _nearly_integral(left / right)

    def describe(self) -> str:
        """A message naming what the relation requires, for an error or a log."""
        return f"{self.parameter} must be a whole multiple of {self.of}"


class HyperparameterSpec(BaseModel):
    """One tunable key and the domain its values are drawn from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: ParameterKind
    default: Any
    low: float | None = None
    high: float | None = None
    step: float | None = None
    log: bool = False
    choices: tuple[Any, ...] = ()
    grid_points: int | None = None
    condition: ParameterCondition | None = None

    @model_validator(mode="after")
    def _validate_domain(self) -> "HyperparameterSpec":
        _validate_parameter_name(self.name, label="parameter name")
        if self.condition is not None and self.condition.parameter == self.name:
            raise ValueError(f"{self.name} cannot be conditional on itself")
        if self.kind == "continuous":
            self._validate_continuous()
        elif self.kind == "discrete":
            self._validate_discrete()
        else:
            self._validate_categorical()
        if not self.contains(self.default):
            raise ValueError(f"{self.name}: default {self.default!r} is outside the declared domain")
        return self

    def _validate_bounds(self) -> None:
        """Reject bounds no sampler could draw from."""
        for label, bound in (("low", self.low), ("high", self.high)):
            if not _is_number(bound) or not math.isfinite(float(bound)):
                raise ValueError(f"{self.name}: {label} must be a finite number")
        if self.step is not None and (not _is_number(self.step) or not math.isfinite(float(self.step))):
            raise ValueError(f"{self.name}: step must be a finite number")

    def _validate_continuous(self) -> None:
        if self.low is None or self.high is None:
            raise ValueError(f"{self.name}: continuous parameters require numeric low and high")
        self._validate_bounds()
        if float(self.low) >= float(self.high):
            raise ValueError(f"{self.name}: low must be strictly below high")
        if self.log and float(self.low) <= 0.0:
            raise ValueError(f"{self.name}: log-scaled parameters require a positive low bound")
        if self.step is not None:
            raise ValueError(f"{self.name}: continuous parameters cannot declare step")
        if self.choices:
            raise ValueError(f"{self.name}: continuous parameters cannot declare choices")
        if self.grid_points is not None:
            if self.grid_points < 2:
                raise ValueError(f"{self.name}: grid_points must be at least 2")
            if self.grid_points > _MAX_ENUMERABLE_POINTS:
                raise ValueError(
                    f"{self.name}: grid_points {self.grid_points} exceeds the "
                    f"{_MAX_ENUMERABLE_POINTS} a grid may enumerate"
                )

    def _validate_discrete(self) -> None:
        if self.low is None or self.high is None:
            raise ValueError(f"{self.name}: discrete parameters require numeric low and high")
        if not _is_number(self.step) or float(self.step) <= 0.0:
            raise ValueError(f"{self.name}: discrete parameters require a positive step")
        self._validate_bounds()
        if float(self.low) > float(self.high):
            raise ValueError(f"{self.name}: low must not exceed high")
        if self.log:
            raise ValueError(f"{self.name}: discrete parameters cannot be log-scaled")
        if self.choices:
            raise ValueError(f"{self.name}: discrete parameters cannot declare choices")
        if self.grid_points is not None:
            raise ValueError(f"{self.name}: discrete parameters are already enumerable")
        span = (float(self.high) - float(self.low)) / float(self.step)
        if not _nearly_integral(span):
            raise ValueError(f"{self.name}: the range is not divisible by step")

    def _validate_categorical(self) -> None:
        if not self.choices:
            raise ValueError(f"{self.name}: categorical parameters require at least one choice")
        for choice in self.choices:
            # A non-finite choice survives in memory but not through JSON: pydantic
            # writes ``nan`` as ``null``, so a persisted space would reload with a
            # different domain than the one that was fingerprinted.
            runaway = _find_non_finite(choice)
            if runaway is not None:
                raise ValueError(f"{self.name}: a choice must be finite, got {runaway!r}")
        keys = [_choice_key(choice) for choice in self.choices]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{self.name}: choices must be unique")
        for bound in (self.low, self.high, self.step):
            if bound is not None:
                raise ValueError(f"{self.name}: categorical parameters cannot declare numeric bounds")
        if self.log:
            raise ValueError(f"{self.name}: categorical parameters cannot be log-scaled")
        if self.grid_points is not None:
            raise ValueError(f"{self.name}: categorical parameters are already enumerable")

    def _has_whole_valued_points(self) -> bool:
        """Whether every grid point is a whole number.

        Decided by value rather than by the declared Python type: the fields are
        annotated ``float``, so an integer literal arrives already coerced.
        """
        return all(float(value).is_integer() for value in (self.low, self.high, self.step))

    def contains(self, value: Any) -> bool:
        """Return whether ``value`` lies inside the declared domain."""
        if self.kind == "categorical":
            return _choice_key(value) in {_choice_key(choice) for choice in self.choices}
        if not _is_number(value):
            return False
        number = float(value)
        if not math.isfinite(number):
            return False
        if not _within_bounds(number, float(self.low), float(self.high)):
            return False
        if self.kind == "continuous":
            return True
        offset = (number - float(self.low)) / float(self.step)
        return _nearly_integral(offset)

    def validate_value(self, value: Any) -> None:
        """Raise ``ValueError`` unless ``value`` lies inside the declared domain."""
        if self.contains(value):
            return
        raise ValueError(f"{self.name}={value!r} is outside {self.describe_domain()}")

    def describe_domain(self) -> str:
        if self.kind == "categorical":
            return f"choices {list(self.choices)!r}"
        if self.kind == "discrete":
            return f"discrete [{self.low}, {self.high}] step {self.step}"
        scale = "log-uniform" if self.log else "uniform"
        return f"{scale} [{self.low}, {self.high}]"

    def grid_values(self) -> tuple[Any, ...] | None:
        """Enumerate the domain, or return ``None`` when it is not finite.

        A continuous parameter is enumerable only when it declares
        ``grid_points``; otherwise a grid sampler would have nothing to iterate
        and must fall back to random draws.  Log-scaled continuous domains are
        spaced geometrically so that a coarse grid still covers several orders of
        magnitude instead of crowding every point near the lower bound.
        """
        if self.kind == "categorical":
            return self.choices
        if self.kind == "continuous":
            if self.grid_points is None:
                return None
            return self._continuous_grid()
        low, high, step = float(self.low), float(self.high), float(self.step)
        count = int(round((high - low) / step)) + 1
        if count > _MAX_ENUMERABLE_POINTS:
            # Materializing the list first would allocate gigabytes and look like a
            # hang; the caller wants a clear refusal it can fall back from.
            raise ValueError(
                f"{self.name}: enumerating {count} points exceeds the "
                f"{_MAX_ENUMERABLE_POINTS} a grid may materialize; widen step or sample randomly"
            )
        values = [low + index * step for index in range(count)]
        # Validation already proved the range is divisible by step, so the last
        # grid point is high by construction.  Pinning it to the declared bound
        # rather than the float coercion removes the accumulated error that would
        # otherwise put it just outside the domain, and keeps an integer domain
        # yielding ints.
        values[-1] = self.high
        if self._has_whole_valued_points():
            return tuple(int(round(value)) for value in values)
        return tuple(values)

    def _continuous_grid(self) -> tuple[float, ...]:
        """Place ``grid_points`` values across the domain, endpoints included."""
        low, high = float(self.low), float(self.high)
        last = self.grid_points - 1
        if self.log:
            ratio = (high / low) ** (1.0 / last)
            values = [low * (ratio**index) for index in range(self.grid_points)]
        else:
            values = [low + (high - low) * index / last for index in range(self.grid_points)]
        # Same pinning as the discrete branch: the arithmetic endpoints absorb
        # float error, and a grid point outside the domain would trip
        # ``validate_assignment`` on the sampler's own output.
        values[0], values[-1] = low, high
        return tuple(values)

    def condition_holds(self, assignment: Mapping[str, Any]) -> bool:
        """Return whether this parameter's own condition holds for the assignment.

        This is only the local check.  A parameter whose guard is itself inactive
        must not count as active, which is why callers go through
        :meth:`SearchSpaceSchema.is_active` rather than asking a lone spec.
        """
        return self.condition is None or self.condition.applies_to(assignment)


class SearchSpaceSchema(BaseModel):
    """A validated collection of tunable parameters with a stable fingerprint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SEARCH_SPACE_SCHEMA_VERSION
    specs: tuple[HyperparameterSpec, ...]
    constraints: tuple[ParameterConstraint, ...] = ()

    @model_validator(mode="after")
    def _validate_graph(self) -> "SearchSpaceSchema":
        if not self.schema_version.strip():
            raise ValueError("schema_version must not be blank")
        if not self.specs:
            raise ValueError("a search space requires at least one parameter")
        names = [spec.name for spec in self.specs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate parameter names: {duplicates}")
        self._validate_no_prefix_collision(names)
        known = set(names)
        by_name = {spec.name: spec for spec in self.specs}
        edges: dict[str, str] = {}
        for spec in self.specs:
            if spec.condition is None:
                continue
            guard = spec.condition.parameter
            if guard not in known:
                raise ValueError(f"{spec.name} is conditional on unknown parameter {guard!r}")
            self._validate_condition_reaches_its_guard(spec, by_name[guard])
            edges[spec.name] = guard
        cycle = _find_dependency_cycle(edges)
        if cycle is not None:
            raise ValueError(f"conditional dependency cycle involving {cycle!r}")
        self._validate_constraints(known)
        return self

    def _validate_no_prefix_collision(self, names: list[str]) -> None:
        """Reject one parameter name being a dotted prefix of another.

        ``skrl.agent`` and ``skrl.agent.mini_batches`` cannot both be satisfied:
        the first demands that path hold a value while the second demands it hold a
        mapping.  Such a space has no legal assignment at all, so it must not load.
        """
        shortest_first = sorted(names, key=len)
        for position, name in enumerate(shortest_first):
            for longer in shortest_first[position + 1:]:
                if longer.startswith(f"{name}."):
                    raise ValueError(
                        f"{name} is a prefix of {longer}: one would have to hold a value "
                        f"and the other a mapping"
                    )

    def _validate_condition_reaches_its_guard(
        self,
        spec: HyperparameterSpec,
        guard: HyperparameterSpec,
    ) -> None:
        """Reject a condition no assignment can ever satisfy.

        A misspelled value -- a dropped character, the wrong type -- leaves the
        guarded parameter permanently inactive, and nothing downstream notices:
        it is dropped from active sets, omitted from defaults, and never reported
        missing.  The declared knob would be silently absent from every search.
        """
        for value in spec.condition.values:
            if not guard.contains(value):
                raise ValueError(
                    f"{spec.name} is conditioned on {guard.name}={value!r}, which is "
                    f"outside {guard.describe_domain()}; it could never become active"
                )

    def _validate_constraints(self, known: set[str]) -> None:
        seen: set[tuple[str, str, str]] = set()
        for constraint in self.constraints:
            for name in (constraint.parameter, constraint.of):
                if name not in known:
                    raise ValueError(f"constraint names unknown parameter {name!r}")
            key = (constraint.kind, constraint.parameter, constraint.of)
            if key in seen:
                raise ValueError(f"duplicate constraint: {constraint.describe()}")
            seen.add(key)

    def spec(self, name: str) -> HyperparameterSpec:
        """Look up a parameter by name."""
        for candidate in self.specs:
            if candidate.name == name:
                return candidate
        raise ValueError(f"unknown parameter: {name!r}")

    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def _active_flags(self, assignment: Mapping[str, Any]) -> dict[str, bool]:
        """Resolve activation transitively along each parameter's condition chain.

        A parameter is active only when its own condition holds *and* the guard it
        depends on is itself active.  Without that, switching a scheduler off would
        leave its dependent tuning knob "active" and a sampler would draw a value
        that can never reach the optimizer.
        """
        by_name = {spec.name: spec for spec in self.specs}

        def resolve(name: str) -> bool:
            spec = by_name[name]
            if spec.condition is None:
                return True
            if not spec.condition.applies_to(assignment):
                return False
            return resolve(spec.condition.parameter)

        return {spec.name: resolve(spec.name) for spec in self.specs}

    def is_active(self, name: str, assignment: Mapping[str, Any]) -> bool:
        """Return whether ``name`` applies to this assignment."""
        if name not in {spec.name for spec in self.specs}:
            raise ValueError(f"unknown parameter: {name!r}")
        return self._active_flags(assignment)[name]

    def active_specs(self, assignment: Mapping[str, Any]) -> tuple[HyperparameterSpec, ...]:
        """Return the parameters that apply to this assignment, in declaration order."""
        flags = self._active_flags(assignment)
        return tuple(spec for spec in self.specs if flags[spec.name])

    def inactive_names(self, assignment: Mapping[str, Any]) -> tuple[str, ...]:
        flags = self._active_flags(assignment)
        return tuple(spec.name for spec in self.specs if not flags[spec.name])

    def active_names(self, assignment: Mapping[str, Any]) -> tuple[str, ...]:
        """The parameters that apply to this assignment, in declaration order."""
        return tuple(spec.name for spec in self.active_specs(assignment))

    def active_assignment(self, assignment: Mapping[str, Any]) -> dict[str, Any]:
        """The subset of ``assignment`` the active parameters actually fill.

        Nested like its input, so the result is itself a config fragment.  A
        flat ``name -> value`` view would carry dotted names that a config
        loader ignores, which is the trap :meth:`default_values` documents.
        """
        active: dict[str, Any] = {}
        for spec in self.active_specs(assignment):
            found, value = get_parameter(assignment, spec.name)
            if found:
                set_parameter(active, spec.name, value, create_missing=True)
        return active

    def sampling_order(self) -> tuple[str, ...]:
        """Order the parameters so every guard is drawn before what it guards.

        A conditional parameter's domain is only known once its guard holds a
        value, so a sampler that walked declaration order could draw a value for a
        parameter whose activation is still undecided.

        Among parameters that are equally ready, declaration order decides.  The
        result therefore depends on the declaration and is *not* shared between two
        schemas that list the same parameters differently -- which is why the order
        stays out of :meth:`fingerprint`: it is an execution detail, and two spaces
        with one identity must not be assumed to draw in the same sequence.
        """
        guard = {spec.name: (spec.condition.parameter if spec.condition else None) for spec in self.specs}
        ordered: list[str] = []
        placed: set[str] = set()
        pending = list(self.names())
        while pending:
            ready = next(
                (name for name in pending if guard[name] is None or guard[name] in placed),
                None,
            )
            if ready is None:
                # ``_validate_graph`` already rejected cycles, so reaching this
                # means the graph validator and this walk disagree.
                raise ValueError(f"cannot linearize conditional dependencies among {pending}")
            ordered.append(ready)
            placed.add(ready)
            pending.remove(ready)
        return tuple(ordered)

    def default_values(self) -> dict[str, Any]:
        """Flat ``name -> default`` view, for logging and for human inspection.

        Not an assignment: the names stay dotted here, so
        :meth:`validate_assignment` would not find them.  Use
        :meth:`default_assignment` for a mapping that can be injected.
        """
        return {spec.name: spec.default for spec in self.specs}

    def default_assignment(self) -> dict[str, Any]:
        """Nested defaults of the active parameters, ready to overlay on a config.

        Built with :func:`set_parameter` so a dotted name becomes real nesting
        rather than a flat key that a config loader would ignore.  Parameters
        whose condition does not hold are left out, matching what a sampler would
        actually have drawn.
        """
        assignment: dict[str, Any] = {}
        for spec in self.specs:
            set_parameter(assignment, spec.name, spec.default, create_missing=True)
        flags = self._active_flags(assignment)
        active: dict[str, Any] = {}
        for spec in self.specs:
            if flags[spec.name]:
                set_parameter(active, spec.name, spec.default, create_missing=True)
        self.validate_assignment(active)
        return active

    def unsatisfied_constraints(self, assignment: Mapping[str, Any]) -> tuple[ParameterConstraint, ...]:
        """The relations this assignment breaks, in declaration order.

        A relation is only checked while both of its parameters apply: the baseline
        config keeps a stale value for an inactive parameter, and judging a trial on
        a value the trainer never reads would reject legal combinations.
        """
        flags = self._active_flags(assignment)
        return tuple(
            constraint
            for constraint in self.constraints
            if flags[constraint.parameter]
            and flags[constraint.of]
            and not constraint.holds(assignment)
        )

    def is_feasible(self, assignment: Mapping[str, Any]) -> bool:
        """Whether every relation holds, for a sampler deciding to keep a draw."""
        return not self.unsatisfied_constraints(assignment)

    def validate_assignment(
        self,
        assignment: Mapping[str, Any],
        *,
        require_complete: bool = True,
        forbid_inactive: bool = False,
    ) -> None:
        """Raise ``ValueError`` listing every problem found in ``assignment``.

        Inactive parameters are ignored by default: the baseline config carries a
        value for them, and a sampler is not required to clear it.
        """
        issues: list[str] = []
        flags = self._active_flags(assignment)
        for spec in self.specs:
            active = flags[spec.name]
            found, value = get_parameter(assignment, spec.name)
            if not active:
                if forbid_inactive and found:
                    issues.append(f"{spec.name} does not apply to this assignment")
                continue
            if not found:
                if require_complete:
                    issues.append(f"{spec.name} is missing")
                continue
            if not spec.contains(value):
                issues.append(f"{spec.name}={value!r} is outside {spec.describe_domain()}")
        issues.extend(constraint.describe() for constraint in self.unsatisfied_constraints(assignment))
        if issues:
            raise ValueError("; ".join(issues))

    def fingerprint(self) -> str:
        """Content-addressed identity, so a search run cites an exact space.

        Declaration order is not part of the identity: two schemas listing the
        same parameters differ only in the order a grid is enumerated, which no
        downstream record should depend on.  ``schema_version`` *is* part of it --
        a v2 space that reuses a v1 parameter name describes a different thing,
        and a record citing the old identity must not silently match it.
        """
        ordered = sorted(self.specs, key=lambda spec: spec.name)
        return content_hash(
            {
                "schema_version": self.schema_version,
                "specs": [spec.model_dump(mode="json", exclude_none=True) for spec in ordered],
                "constraints": [
                    constraint.model_dump(mode="json")
                    for constraint in sorted(self.constraints, key=lambda item: (item.kind, item.parameter))
                ],
            }
        )


_FILE_KEYS = ("parameters", "constraints")


def load_search_space(data: Mapping[str, Any]) -> SearchSpaceSchema:
    """Build a schema from a product's declared search space, already parsed.

    The file spells the list ``parameters``, the word the domain uses; the model
    calls the field ``specs``.  Translating in one place keeps that spelling out of
    every call site.  Unknown keys are rejected here rather than by pydantic,
    because this function builds the model's input itself: a misspelled
    ``constraint:`` would be dropped silently and the space would load with no
    relations at all.

    ``schema_version`` is deliberately not a file key: it is part of the schema's
    identity and lives in code, so a file cannot drift from the reader that
    interprets it.
    """
    if not isinstance(data, Mapping):
        raise ValueError(f"search space must be a mapping, got {type(data).__name__}")
    unknown = sorted(set(data) - set(_FILE_KEYS))
    if unknown:
        raise ValueError(f"unknown search space keys: {unknown}")
    if "parameters" not in data:
        raise ValueError("search space requires a 'parameters' list")
    return SearchSpaceSchema.model_validate({
        "specs": data["parameters"],
        "constraints": data.get("constraints", ()),
    })


def _find_dependency_cycle(edges: Mapping[str, str]) -> str | None:
    """Return one parameter name that closes a dependency cycle, if any exists."""
    for start in sorted(edges):
        seen: dict[str, None] = {}
        current: str | None = start
        while current is not None:
            if current in seen:
                return current
            seen[current] = None
            current = edges.get(current)
    return None
