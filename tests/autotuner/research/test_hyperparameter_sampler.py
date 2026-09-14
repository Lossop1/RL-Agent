"""Grid and random sampling: coverage, feasibility, determinism, and replay.

The product space is read from the shipped files for the same reason the space
tests read it: a sampler that stops working against the real Taili space must
fail here, not at the start of a GPU run.
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from autotuner.research.hyperparameter_sampler import (
    GridSampler,
    RandomSampler,
    Sampler,
    SamplerStats,
    SearchPlan,
    assignment_key,
    build_sampler,
    replay,
)
from autotuner.research.hyperparameter_space import (
    HyperparameterSpec,
    ParameterCondition,
    ParameterConstraint,
    SearchSpaceSchema,
    get_parameter,
    load_search_space,
)

_PRODUCT_DIR = Path(__file__).resolve().parents[3] / "products" / "taili" / "blind_locomotion"
SPACE_PATH = _PRODUCT_DIR / "hyperparameter_space.yaml"

ROLLOUTS = "skrl.agent.rollouts"
MINI_BATCHES = "skrl.agent.mini_batches"
LEARNING_RATE = "skrl.agent.learning_rate"
ENTROPY = "skrl.agent.entropy_loss_scale"
KL_THRESHOLD = "skrl.agent.learning_rate_scheduler_kwargs.kl_threshold"


def _taili_space() -> SearchSpaceSchema:
    return load_search_space(yaml.safe_load(SPACE_PATH.read_text(encoding="utf-8")))


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
        name=name, kind="continuous", low=low, high=high, default=default,
        log=log, grid_points=grid_points,
    )


def _discrete(name: str, low: float, high: float, step: float, default: float) -> HyperparameterSpec:
    return HyperparameterSpec(name=name, kind="discrete", low=low, high=high, step=step, default=default)


def _small_space() -> SearchSpaceSchema:
    """Six combinations, four of them feasible: 6x{12,18,24} minus the non-multiples."""
    return SearchSpaceSchema(
        specs=(
            _discrete("a", 6, 9, 3, 6),
            _discrete("b", 12, 24, 6, 12),
        ),
        constraints=(ParameterConstraint(kind="multiple", parameter="b", of="a"),),
    )


def _grid_space() -> SearchSpaceSchema:
    """Every parameter enumerable, so a complete grid enumeration is possible."""
    return SearchSpaceSchema(
        specs=(
            _continuous("lr", 1.0e-3, 1.0e-1, 1.0e-2, log=True, grid_points=3),
            _discrete("a", 6, 9, 3, 6),
            _discrete("b", 12, 24, 6, 12),
        ),
        constraints=(ParameterConstraint(kind="multiple", parameter="b", of="a"),),
    )


def _conditional_space() -> SearchSpaceSchema:
    """A guarded parameter declared *after* its guard."""
    return SearchSpaceSchema(specs=(
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
    ))


def _conditional_space_reversed() -> SearchSpaceSchema:
    """The same space, with the guarded parameter declared first."""
    return SearchSpaceSchema(specs=tuple(reversed(_conditional_space().specs)))


def _grid_space_reversed() -> SearchSpaceSchema:
    """The same space, with the equally-ready parameters declared differently.

    A condition pins its guard's position whatever the declaration order says,
    so only unconstrained parameters can actually change the sequence.
    """
    forward = _grid_space()
    return SearchSpaceSchema(specs=tuple(reversed(forward.specs)), constraints=forward.constraints)


def _impossible_space() -> SearchSpaceSchema:
    """No multiple of 20 or 28 lands on the b grid, so no draw can satisfy it."""
    return SearchSpaceSchema(
        specs=(
            _discrete("a", 20, 28, 8, 20),
            _discrete("b", 24, 96, 24, 24),
        ),
        constraints=(ParameterConstraint(kind="multiple", parameter="b", of="a"),),
    )


def _single_point_space() -> SearchSpaceSchema:
    return SearchSpaceSchema(specs=(_discrete("a", 4, 4, 1, 4),))


def _feasible_points(space: SearchSpaceSchema) -> set[str]:
    """Brute-force reference: every assignment a grid must find, computed independently."""
    return {assignment_key(item) for item in _brute_force(space, 0, {})}


def _brute_force(space: SearchSpaceSchema, position: int, partial: dict[str, Any]) -> list[dict[str, Any]]:
    order = space.sampling_order()
    if position == len(order):
        return [partial] if space.is_feasible(partial) else []
    name = order[position]
    if not space.is_active(name, partial):
        return _brute_force(space, position + 1, partial)
    found: list[dict[str, Any]] = []
    for value in space.spec(name).grid_values() or ():
        extended = {**partial, name: value}
        found.extend(_brute_force(space, position + 1, extended))
    return found


# --- grid sampling ------------------------------------------------------------


def test_a_grid_walks_every_feasible_point_exactly_once():
    space = _grid_space()
    sampler = build_sampler(space, SearchPlan.for_grid(space))
    drawn = sampler.collect()
    keys = [assignment_key(item) for item in drawn]
    assert len(keys) == len(set(keys))
    assert set(keys) == _feasible_points(space)
    assert sampler.stats.exhausted
    assert sampler.stats.proposals == len(drawn)
    assert sampler.stats.rejected_duplicate == 0


def test_a_grid_skips_the_combinations_a_constraint_rejects():
    space = _grid_space()
    # 3 learning rates x 2 values of a x 3 values of b, minus the two b values
    # that are not multiples of a.
    assert len(build_sampler(space, SearchPlan.for_grid(space)).collect()) == 3 * 4


def test_a_grid_leaves_out_a_parameter_whose_condition_does_not_hold():
    space = _conditional_space()
    drawn = build_sampler(space, SearchPlan.for_grid(space)).collect()
    assert len(drawn) == 3
    for item in drawn:
        if item["sched"] == "off":
            assert "tuned" not in item
        else:
            assert item["tuned"] in (1, 2)


def test_a_grid_does_not_leak_a_value_from_a_sibling_branch():
    """The inactive branch must not inherit the value the active branch placed."""
    space = _conditional_space()
    for item in build_sampler(space, SearchPlan.for_grid(space)).collect():
        space.validate_assignment(item, forbid_inactive=True)


def test_a_grid_enumerates_the_same_points_whatever_the_declaration_order():
    forward = _conditional_space()
    reversed_space = _conditional_space_reversed()
    first = build_sampler(forward, SearchPlan.for_grid(forward)).collect()
    second = build_sampler(reversed_space, SearchPlan.for_grid(reversed_space)).collect()
    assert {assignment_key(item) for item in first} == {assignment_key(item) for item in second}


def test_a_grid_stops_at_its_budget_and_says_so():
    space = _grid_space()
    sampler = build_sampler(space, SearchPlan.for_grid(space, budget=5))
    drawn = sampler.collect()
    assert len(drawn) == 5
    assert not sampler.stats.exhausted


def test_a_grid_budget_larger_than_the_grid_still_exhausts_it():
    space = _conditional_space()
    sampler = build_sampler(space, SearchPlan.for_grid(space, budget=99))
    assert len(sampler.collect()) == 3
    assert sampler.stats.exhausted


def test_a_grid_refuses_a_parameter_it_cannot_enumerate():
    """The shipped space leaves two continuous parameters without grid_points."""
    space = _taili_space()
    with pytest.raises(ValueError, match="no grid_points") as raised:
        build_sampler(space, SearchPlan.for_grid(space))
    message = str(raised.value)
    assert ENTROPY in message
    assert KL_THRESHOLD in message


def test_a_partial_grid_still_refuses_an_unenumerable_parameter():
    """A budget does not make a missing dimension enumerable, it just hides it."""
    space = _taili_space()
    with pytest.raises(ValueError, match="no grid_points"):
        build_sampler(space, SearchPlan.for_grid(space, budget=3))


def test_a_grid_refuses_a_product_too_large_to_enumerate_in_full():
    space = SearchSpaceSchema(specs=(
        _discrete("x", 0, 399, 1, 0),
        _discrete("y", 0, 399, 1, 0),
    ))
    with pytest.raises(ValueError, match="over the 100000"):
        build_sampler(space, SearchPlan.for_grid(space))
    assert len(build_sampler(space, SearchPlan.for_grid(space, budget=3)).collect()) == 3


def test_a_grid_refuses_a_plan_recorded_against_another_space():
    space = _conditional_space()
    other = _single_point_space()
    with pytest.raises(ValueError, match="different search space"):
        build_sampler(space, SearchPlan.for_grid(other))


def test_a_grid_refuses_a_plan_recorded_against_another_declaration_order():
    """The fingerprints match, so only the recorded order can catch this."""
    forward = _grid_space()
    reordered = _grid_space_reversed()
    assert forward.fingerprint() == reordered.fingerprint()
    assert forward.sampling_order() != reordered.sampling_order()
    with pytest.raises(ValueError, match="different parameter order"):
        build_sampler(reordered, SearchPlan.for_grid(forward))


def test_a_condition_pins_the_order_regardless_of_declaration():
    """Both declaration orders must walk the guard before what it guards."""
    forward = _conditional_space()
    reversed_space = _conditional_space_reversed()
    assert forward.sampling_order() == reversed_space.sampling_order() == ("sched", "tuned")


# --- random sampling ----------------------------------------------------------


def test_a_random_search_fills_its_budget_with_distinct_feasible_trials():
    space = _taili_space()
    plan = SearchPlan.for_random(space, seed=20260914, budget=25)
    sampler = build_sampler(space, plan)
    drawn = sampler.collect()
    assert len(drawn) == 25
    assert len({assignment_key(item) for item in drawn}) == 25
    for item in drawn:
        space.validate_assignment(item)
        assert get_parameter(item, ROLLOUTS)[1] % get_parameter(item, MINI_BATCHES)[1] == 0
    assert sampler.stats.rejected_infeasible > 0
    assert sampler.stats.exhausted


def test_a_random_search_draws_the_parameters_a_grid_cannot_enumerate():
    """The two parameters without grid_points are still tunable, just not on a grid."""
    space = _taili_space()
    drawn = build_sampler(
        space, SearchPlan.for_random(space, seed=1, budget=12)
    ).collect()
    entropies = {get_parameter(item, ENTROPY)[1] for item in drawn}
    thresholds = {get_parameter(item, KL_THRESHOLD)[1] for item in drawn}
    assert len(entropies) > 1
    assert len(thresholds) > 1
    for value in entropies:
        assert 0.0 <= value <= 0.05
    for value in thresholds:
        assert 0.008 <= value <= 0.016


def test_a_random_search_stays_inside_every_domain():
    space = _taili_space()
    for item in build_sampler(space, SearchPlan.for_random(space, seed=3, budget=40)).collect():
        for name in space.names():
            found, value = get_parameter(item, name)
            if found:
                assert space.spec(name).contains(value)


def test_a_random_search_draws_a_log_domain_across_its_decades():
    """A linear draw would put almost every sample in the top decade."""
    space = SearchSpaceSchema(specs=(_continuous("lr", 1.0e-4, 1.0e-2, 1.0e-3, log=True),))
    drawn = build_sampler(space, SearchPlan.for_random(space, seed=5, budget=200)).collect()
    below_midpoint = [item for item in drawn if item["lr"] < 1.0e-3]
    assert 0.35 < len(below_midpoint) / len(drawn) < 0.65


def test_a_random_search_keeps_a_discrete_domain_integers():
    """A config declaring mini_batches: 16 must not be overwritten with 16.0."""
    space = _taili_space()
    drawn = build_sampler(
        space, SearchPlan.for_random(space, seed=8, budget=20)
    ).collect()
    for item in drawn:
        for name in (MINI_BATCHES, ROLLOUTS):
            assert isinstance(get_parameter(item, name)[1], int)


def test_a_random_search_omits_a_parameter_whose_condition_does_not_hold():
    space = SearchSpaceSchema(specs=(
        HyperparameterSpec(name="sched", kind="categorical", choices=("on", "off"), default="on"),
        HyperparameterSpec(
            name="tuned",
            kind="continuous",
            low=0.0,
            high=1.0,
            default=0.5,
            condition=ParameterCondition(parameter="sched", values=("on",)),
        ),
    ))
    drawn = build_sampler(space, SearchPlan.for_random(space, seed=11, budget=40)).collect()
    assert any(item["sched"] == "off" and "tuned" not in item for item in drawn)
    assert any(item["sched"] == "on" and "tuned" in item for item in drawn)


def test_a_random_search_covers_a_small_feasible_set_when_the_budget_allows():
    """Exactly four assignments are feasible, so four distinct draws are all of them."""
    space = _small_space()
    drawn = build_sampler(space, SearchPlan.for_random(space, seed=17, budget=4)).collect()
    assert {assignment_key(item) for item in drawn} == _feasible_points(space)


def test_a_random_search_refuses_to_return_a_short_run():
    """An unsatisfiable constraint must fail loudly, not yield fewer trials."""
    space = _impossible_space()
    plan = SearchPlan.for_random(space, seed=2, budget=3, max_attempts=50)
    with pytest.raises(ValueError, match="gave up after 50 draws for trial 1 of 3"):
        build_sampler(space, plan).collect()


def test_a_random_search_says_when_the_set_of_unseen_trials_ran_out():
    space = _single_point_space()
    plan = SearchPlan.for_random(space, seed=2, budget=2, max_attempts=10)
    with pytest.raises(ValueError, match="repeated an assignment already proposed"):
        build_sampler(space, plan).collect()


def test_a_trial_that_fails_for_both_reasons_names_both():
    """A constraint and an exhausted space can both be why the draws stopped.

    Naming only the last draw's cause would report the constraint as
    unsatisfiable when the truth is that the one feasible point had already been
    tried, and would send a reader to widen a domain that was never the problem.
    """
    space = SearchSpaceSchema(
        specs=(_discrete("a", 1, 2, 1, 1), _discrete("b", 1, 1, 1, 1)),
        constraints=(ParameterConstraint(kind="multiple", parameter="b", of="a"),),
    )
    mixed: list[int] = []
    for seed in range(64):
        plan = SearchPlan.for_random(space, seed=seed, budget=2, max_attempts=10)
        try:
            build_sampler(space, plan).collect()
        except ValueError as error:
            if "broke a constraint" in str(error) and "repeated an assignment" in str(error):
                mixed.append(seed)
    # Half the draws land on the infeasible point, so ten draws missing one of
    # the two causes is a one-in-five-hundred event, not a typical seed.
    assert mixed, "no seed produced a sequence of draws that failed for both reasons"


def test_a_failed_trial_blames_only_what_actually_happened():
    """Each pure dead end is reported alone, without the other verdict."""
    impossible = SearchPlan.for_random(_impossible_space(), seed=2, budget=3, max_attempts=50)
    with pytest.raises(ValueError) as unsatisfiable:
        build_sampler(_impossible_space(), impossible).collect()
    assert "broke a constraint" in str(unsatisfiable.value)
    assert "repeated an assignment" not in str(unsatisfiable.value)

    exhausted = SearchPlan.for_random(_single_point_space(), seed=2, budget=2, max_attempts=10)
    with pytest.raises(ValueError) as reused:
        build_sampler(_single_point_space(), exhausted).collect()
    assert "repeated an assignment" in str(reused.value)
    assert "broke a constraint" not in str(reused.value)


def test_a_draw_that_breaks_two_constraints_is_counted_once():
    """The failing-draw count must be draws, not the sum of per-constraint hits.

    Every draw here breaks both constraints, so a message that added the
    per-constraint hits together would report twice as many failing draws as the
    trial made.  "4 of 2" is self-contradictory on its face, and a reader sizing
    the space from it would be using a number that cannot be true.
    """
    space = SearchSpaceSchema(
        specs=(
            _discrete("a", 2, 3, 1, 2),
            _discrete("b", 1, 1, 1, 1),
            _discrete("c", 1, 1, 1, 1),
        ),
        constraints=(
            ParameterConstraint(kind="multiple", parameter="b", of="a"),
            ParameterConstraint(kind="multiple", parameter="c", of="a"),
        ),
    )
    plan = SearchPlan.for_random(space, seed=0, budget=1, max_attempts=2)
    sampler = build_sampler(space, plan)
    with pytest.raises(ValueError) as failure:
        sampler.collect()
    message = str(failure.value)
    assert "gave up after 2 draws" in message
    assert "2 of 2 broke a constraint" in message
    assert "4 of 2" not in message
    # Both constraints still get their own line, each blamed for both draws.
    assert message.count("(2 of 2 infeasible draws)") == 2
    assert sampler.stats.attempts == 2
    assert sampler.stats.rejected_infeasible == 2


def test_a_dead_end_reports_one_draw_in_the_singular():
    """A one-draw trial is described as one draw, not "1 draws"."""
    plan = SearchPlan.for_random(_impossible_space(), seed=2, budget=1, max_attempts=1)
    with pytest.raises(ValueError, match="gave up after 1 draw for trial 1 of 1"):
        build_sampler(_impossible_space(), plan).collect()


def test_the_two_causes_account_for_every_draw_the_trial_made():
    """Every number in the message is counted against the same total.

    A message whose parts describe different runs cannot be used to decide what
    to change: the two headlines have to sum to the draws actually made, and the
    breakdown under the first one has to quote the same total the headline does.
    A per-constraint share reading "(2 of 2 infeasible draws)" beneath a headline
    claiming "4 of N broke a constraint" leaves the reader no way to tell which
    number is about their run.
    """
    space = SearchSpaceSchema(
        specs=(
            _discrete("a", 1, 2, 1, 1),
            _discrete("b", 1, 2, 1, 1),
            _discrete("c", 1, 2, 1, 1),
        ),
        constraints=(
            ParameterConstraint(kind="multiple", parameter="b", of="a"),
            ParameterConstraint(kind="multiple", parameter="c", of="a"),
        ),
    )
    mixed = 0
    for seed in range(64):
        plan = SearchPlan.for_random(space, seed=seed, budget=5, max_attempts=10)
        try:
            build_sampler(space, plan).collect()
        except ValueError as error:
            message = str(error)
            found = re.findall(
                r"(\d+) of 10 (?:broke a constraint|repeated an assignment)", message
            )
            if len(found) < 2:
                continue
            mixed += 1
            assert sum(int(count) for count in found) == 10, message
            headline = re.search(r"(\d+) of 10 broke a constraint", message)
            for broken, counted in re.findall(r"\((\d+) of (\d+) infeasible draws\)", message):
                # The share a constraint is blamed for is quoted over the
                # infeasible draws, so that denominator and the headline's
                # numerator are the same count written twice.
                assert counted == headline.group(1), message
                assert int(broken) <= int(counted), message
    assert mixed, "no seed produced a sequence of draws that failed for both reasons"


def test_a_random_search_counts_its_attempts():
    space = _small_space()
    sampler = build_sampler(space, SearchPlan.for_random(space, seed=4, budget=4))
    sampler.collect()
    assert sampler.stats.attempts >= sampler.stats.proposals == 4


# --- determinism and replay --------------------------------------------------


def test_the_same_seed_replays_the_same_sequence():
    space = _taili_space()
    plan = SearchPlan.for_random(space, seed=4242, budget=15)
    assert build_sampler(space, plan).collect() == build_sampler(space, plan).collect()


def test_a_different_seed_draws_a_different_sequence():
    space = SearchSpaceSchema(specs=(_continuous("lr", 1.0e-4, 1.0e-2, 1.0e-3, log=True),))
    first = build_sampler(space, SearchPlan.for_random(space, seed=1, budget=3)).collect()
    second = build_sampler(space, SearchPlan.for_random(space, seed=2, budget=3)).collect()
    assert first != second


def test_a_draw_does_not_depend_on_the_global_random_stream():
    """Two processes that consumed different amounts of randomness must agree."""
    space = _taili_space()
    plan = SearchPlan.for_random(space, seed=7, budget=10)
    first = build_sampler(space, plan).collect()
    random.seed(12345)
    for _ in range(1000):
        random.random()
    second = build_sampler(space, plan).collect()
    assert first == second


def test_replaying_a_grid_reproduces_the_recorded_sequence():
    space = _grid_space()
    plan = SearchPlan.for_grid(space, budget=7)
    recorded = build_sampler(space, plan).collect()
    assert replay(space, plan) == recorded


def test_replaying_a_random_search_reproduces_the_recorded_sequence():
    space = _taili_space()
    plan = SearchPlan.for_random(space, seed=99, budget=10)
    recorded = build_sampler(space, plan).collect()
    assert replay(space, plan) == recorded


def test_a_replay_rejects_a_space_that_changed_under_the_record():
    space = _grid_space()
    plan = SearchPlan.for_grid(space)
    changed = SearchSpaceSchema(specs=space.specs, constraints=())
    with pytest.raises(ValueError, match="different search space"):
        replay(changed, plan)


# --- plan validation ---------------------------------------------------------


def test_a_plan_identity_covers_the_seed_the_budget_and_the_order():
    space = _grid_space()
    base = SearchPlan.for_random(space, seed=1, budget=5)
    assert SearchPlan.for_random(space, seed=2, budget=5).fingerprint() != base.fingerprint()
    assert SearchPlan.for_random(space, seed=1, budget=6).fingerprint() != base.fingerprint()
    assert SearchPlan.for_random(space, seed=1, budget=5).fingerprint() == base.fingerprint()


def test_a_grid_plan_refuses_a_seed():
    space = _grid_space()
    with pytest.raises(ValidationError, match="takes no seed"):
        SearchPlan(
            kind="grid",
            space_fingerprint=space.fingerprint(),
            sampling_order=space.sampling_order(),
            seed=3,
        )


def test_a_random_plan_requires_a_seed_and_a_budget():
    space = _grid_space()
    with pytest.raises(ValidationError, match="requires a seed"):
        SearchPlan(kind="random", space_fingerprint=space.fingerprint(),
                   sampling_order=space.sampling_order(), budget=5)
    with pytest.raises(ValidationError, match="requires a budget"):
        SearchPlan(kind="random", space_fingerprint=space.fingerprint(),
                   sampling_order=space.sampling_order(), seed=1)


def test_a_plan_rejects_a_degenerate_budget_or_attempt_count():
    space = _grid_space()
    with pytest.raises(ValidationError, match="budget must be at least 1"):
        SearchPlan.for_grid(space, budget=0)
    with pytest.raises(ValidationError, match="max_attempts must be at least 1"):
        SearchPlan.for_random(space, seed=1, budget=1, max_attempts=0)


def test_a_plan_rejects_a_blank_fingerprint_or_a_broken_order():
    space = _grid_space()
    with pytest.raises(ValidationError, match="space_fingerprint must not be blank"):
        SearchPlan(kind="grid", space_fingerprint=" ", sampling_order=("a",))
    with pytest.raises(ValidationError, match="must not be empty"):
        SearchPlan(kind="grid", space_fingerprint="x", sampling_order=())
    with pytest.raises(ValidationError, match="repeats a parameter"):
        SearchPlan(kind="grid", space_fingerprint="x", sampling_order=("a", "a"))


# --- assignment identity and the sampler contract ----------------------------


def test_assignment_identity_ignores_key_order_and_separates_values():
    assert assignment_key({"a": 1, "b": {"c": 2}}) == assignment_key({"b": {"c": 2}, "a": 1})
    assert assignment_key({"a": 1}) != assignment_key({"a": 2})
    assert assignment_key({"a": 1}) != assignment_key({"a": 1, "b": 1})


class _BrokenSampler(Sampler):
    """A sampler that ignores its domain, to exercise the contract check."""

    def _candidates(self):
        yield {"a": 999}


def test_the_sampler_contract_catches_an_out_of_domain_assignment():
    space = _small_space()
    with pytest.raises(ValueError, match="outside"):
        _BrokenSampler(space, SearchPlan.for_grid(space)).collect()


def test_stats_start_at_zero():
    stats = SamplerStats()
    assert (stats.proposals, stats.attempts, stats.exhausted) == (0, 0, False)


def test_the_sampler_kind_selects_the_implementation():
    space = _single_point_space()
    assert isinstance(build_sampler(space, SearchPlan.for_grid(space)), GridSampler)
    assert isinstance(
        build_sampler(space, SearchPlan.for_random(space, seed=1, budget=1)), RandomSampler
    )
