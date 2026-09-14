"""Domain, condition, constraint, and fingerprint checks for search spaces.

The Taili space and the baseline it is supposed to describe are read from the
product files rather than restated here: a parameter renamed in the training
config, or a default that drifts from the shipped value, has to fail these tests
instead of passing against a hand-copied fixture that drifted with it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pytest
import yaml
from pydantic import ValidationError

from autotuner.research.hyperparameter_space import (
    HyperparameterSpec,
    ParameterCondition,
    ParameterConstraint,
    SearchSpaceSchema,
    get_parameter,
    load_search_space,
    set_parameter,
    split_parameter,
)

_PRODUCT_DIR = Path(__file__).resolve().parents[3] / "products" / "taili" / "blind_locomotion"
SPACE_PATH = _PRODUCT_DIR / "hyperparameter_space.yaml"
CONFIG_PATH = _PRODUCT_DIR / "taili_blind_config.yaml"

SCHEDULER = "skrl.agent.learning_rate_scheduler"
KL_THRESHOLD = "skrl.agent.learning_rate_scheduler_kwargs.kl_threshold"


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


def _categorical(name: str, choices: tuple, default) -> HyperparameterSpec:
    return HyperparameterSpec(name=name, kind="categorical", choices=choices, default=default)


def _discrete(name: str, low: int, high: int, step: int, default: int) -> HyperparameterSpec:
    return HyperparameterSpec(name=name, kind="discrete", low=low, high=high, step=step, default=default)


def _taili_space() -> SearchSpaceSchema:
    """The space the Taili product ships, loaded the way a caller loads it."""
    return load_search_space(yaml.safe_load(SPACE_PATH.read_text(encoding="utf-8")))


def _baseline() -> dict:
    """The shipped training config, which is what every trial starts from."""
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _agent() -> dict:
    """A private copy of the config's agent section, safe to mutate."""
    return _baseline()["skrl"]["agent"]


def _switchable_space() -> SearchSpaceSchema:
    """A two-choice scheduler, so deactivation can be exercised at all.

    The product space pins the scheduler to the one name this repository can
    verify, so the mechanism is tested on a synthetic alternative rather than by
    asserting that a second skrl scheduler exists.
    """
    return SearchSpaceSchema(specs=(
        _categorical(SCHEDULER, ("KLAdaptiveLR", "synthetic-other"), "KLAdaptiveLR"),
        HyperparameterSpec(
            name=KL_THRESHOLD,
            kind="continuous",
            low=0.008,
            high=0.016,
            default=0.016,
            condition=ParameterCondition(parameter=SCHEDULER, values=("KLAdaptiveLR",)),
        ),
    ))


def _switchable_baseline(scheduler: str = "KLAdaptiveLR") -> dict:
    return {
        "skrl": {
            "agent": {
                "learning_rate_scheduler": scheduler,
                "learning_rate_scheduler_kwargs": {"kl_threshold": 0.016},
            }
        }
    }


def test_continuous_spec_accepts_values_inside_its_bounds():
    spec = _continuous("skrl.agent.learning_rate", 1.0e-5, 1.0e-3, 1.0e-4)
    for value in (1.0e-5, 5.0e-5, 1.0e-3):
        assert spec.contains(value)
        spec.validate_value(value)


def test_continuous_spec_rejects_values_outside_its_bounds():
    spec = _continuous("skrl.agent.learning_rate", 1.0e-5, 1.0e-3, 1.0e-4)
    assert not spec.contains(1.1e-3)
    assert not spec.contains(0.0)
    with pytest.raises(ValueError, match="outside"):
        spec.validate_value(1.1e-3)


def test_continuous_spec_rejects_a_non_finite_value():
    spec = _continuous("skrl.agent.learning_rate", 1.0e-5, 1.0e-3, 1.0e-4)
    assert not spec.contains(float("nan"))
    assert not spec.contains(float("inf"))


def test_continuous_spec_rejects_a_boolean_value():
    spec = _continuous("skrl.agent.learning_rate", 0.0, 1.0, 0.5)
    assert not spec.contains(True)


def test_log_scaled_spec_rejects_a_non_positive_low_bound():
    with pytest.raises(ValueError, match="positive low bound"):
        _continuous("skrl.agent.learning_rate", 0.0, 1.0e-3, 1.0e-4, log=True)


def test_continuous_spec_rejects_step_and_choices():
    with pytest.raises(ValueError, match="cannot declare step"):
        HyperparameterSpec(name="a", kind="continuous", low=0.0, high=1.0, default=0.5, step=0.1)
    with pytest.raises(ValueError, match="cannot declare choices"):
        HyperparameterSpec(name="a", kind="continuous", low=0.0, high=1.0, default=0.5, choices=(1,))


def test_continuous_spec_rejects_an_inverted_range():
    with pytest.raises(ValueError, match="strictly below high"):
        _continuous("a", 1.0, 0.0, 0.5)


def test_discrete_spec_rejects_a_range_not_divisible_by_step():
    with pytest.raises(ValueError, match="not divisible by step"):
        _discrete("skrl.agent.mini_batches", 4, 30, 4, 16)


def test_discrete_spec_rejects_a_non_integral_offset():
    spec = _discrete("skrl.agent.mini_batches", 4, 32, 4, 16)
    assert spec.contains(20)
    assert not spec.contains(19)


def test_discrete_spec_rejects_a_missing_or_negative_step():
    with pytest.raises(ValueError, match="positive step"):
        HyperparameterSpec(name="a", kind="discrete", low=1, high=5, default=1)
    with pytest.raises(ValueError, match="positive step"):
        HyperparameterSpec(name="a", kind="discrete", low=1, high=5, step=-1, default=1)


def test_categorical_spec_rejects_duplicate_choices():
    with pytest.raises(ValueError, match="choices must be unique"):
        _categorical("a", ("x", "x"), "x")


def test_categorical_spec_rejects_numeric_bounds():
    with pytest.raises(ValueError, match="cannot declare numeric bounds"):
        HyperparameterSpec(name="a", kind="categorical", choices=("x",), default="x", low=0.0)


def test_categorical_spec_rejects_an_empty_choice_set():
    with pytest.raises(ValueError, match="at least one choice"):
        HyperparameterSpec(name="a", kind="categorical", choices=(), default="x")


def test_categorical_spec_distinguishes_none_from_a_missing_key():
    spec = _categorical("skrl.agent.state_preprocessor", (None, "RunningStandardScaler"), None)
    assert spec.contains(None)
    assert spec.contains("RunningStandardScaler")
    assert not spec.contains("Missing")


def test_a_default_outside_the_declared_domain_is_rejected():
    with pytest.raises(ValueError, match="outside the declared domain"):
        _continuous("skrl.agent.learning_rate", 1.0e-5, 1.0e-3, 2.0e-3)


def test_a_parameter_cannot_be_conditional_on_itself():
    with pytest.raises(ValueError, match="conditional on itself"):
        HyperparameterSpec(
            name="a",
            kind="continuous",
            low=0.0,
            high=1.0,
            default=0.5,
            condition=ParameterCondition(parameter="a", values=(1,)),
        )


def test_a_condition_requires_at_least_one_value():
    with pytest.raises(ValueError, match="at least one value"):
        ParameterCondition(parameter="b", values=())


def test_a_condition_on_an_unknown_parameter_is_rejected():
    with pytest.raises(ValueError, match="unknown parameter"):
        SearchSpaceSchema(specs=(
            _continuous("a", 0.0, 1.0, 0.5, log=False),
            HyperparameterSpec(
                name="b",
                kind="continuous",
                low=0.0,
                high=1.0,
                default=0.5,
                condition=ParameterCondition(parameter="missing", values=(1,)),
            ),
        ))


def test_a_conditional_dependency_cycle_is_rejected():
    with pytest.raises(ValueError, match="cycle"):
        SearchSpaceSchema(specs=(
            HyperparameterSpec(
                name="a",
                kind="categorical",
                choices=(0, 1),
                default=0,
                condition=ParameterCondition(parameter="b", values=(0,)),
            ),
            HyperparameterSpec(
                name="b",
                kind="categorical",
                choices=(0, 1),
                default=0,
                condition=ParameterCondition(parameter="a", values=(0,)),
            ),
        ))


def test_duplicate_parameter_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate parameter names"):
        SearchSpaceSchema(specs=(
            _continuous("a", 0.0, 1.0, 0.5),
            _continuous("a", 0.0, 2.0, 1.0),
        ))


def test_an_empty_search_space_is_rejected():
    with pytest.raises(ValueError, match="at least one parameter"):
        SearchSpaceSchema(specs=())


def test_the_schema_keeps_declaration_order_for_enumeration():
    schema = _taili_space()
    assert schema.names()[0] == "skrl.agent.learning_rate"
    assert schema.spec("skrl.agent.mini_batches").kind == "discrete"
    with pytest.raises(ValueError, match="unknown parameter"):
        schema.spec("skrl.agent.not_a_parameter")


def test_a_conditional_parameter_is_active_only_for_matching_guards():
    schema = _switchable_space()
    assert schema.is_active(KL_THRESHOLD, _switchable_baseline())
    switched = _switchable_baseline("synthetic-other")
    assert not schema.is_active(KL_THRESHOLD, switched)
    assert schema.inactive_names(switched) == (KL_THRESHOLD,)
    with pytest.raises(ValueError, match="unknown parameter"):
        schema.is_active("skrl.agent.not_a_parameter", _switchable_baseline())


def test_a_spec_and_a_schema_reject_mutation():
    spec = _continuous("a", 0.0, 1.0, 0.5)
    with pytest.raises(ValidationError):
        spec.low = 2.0
    schema = _taili_space()
    with pytest.raises(ValidationError):
        schema.specs = ()


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        HyperparameterSpec(name="a", kind="continuous", low=0.0, high=1.0, default=0.5, bogus=1)
    with pytest.raises(ValidationError):
        SearchSpaceSchema(specs=(_continuous("a", 0.0, 1.0, 0.5),), bogus=1)


def test_a_chain_of_conditions_is_accepted():
    schema = SearchSpaceSchema(specs=(
        HyperparameterSpec(name="a", kind="categorical", choices=("on", "off"), default="on"),
        HyperparameterSpec(
            name="b",
            kind="categorical",
            choices=(1, 2),
            default=1,
            condition=ParameterCondition(parameter="a", values=("on",)),
        ),
        HyperparameterSpec(
            name="c",
            kind="continuous",
            low=0.0,
            high=1.0,
            default=0.5,
            condition=ParameterCondition(parameter="b", values=(1,)),
        ),
    ))
    assert schema.active_names({"a": "on", "b": 1, "c": 0.5}) == ("a", "b", "c")
    assert schema.active_names({"a": "on", "b": 2, "c": 0.5}) == ("a", "b")
    assert schema.active_names({"a": "off", "b": 1, "c": 0.5}) == ("a",)


def test_a_three_node_dependency_cycle_is_rejected():
    with pytest.raises(ValueError, match="cycle"):
        SearchSpaceSchema(specs=(
            HyperparameterSpec(
                name="a",
                kind="categorical",
                choices=(0, 1),
                default=0,
                condition=ParameterCondition(parameter="c", values=(0,)),
            ),
            HyperparameterSpec(
                name="b",
                kind="categorical",
                choices=(0, 1),
                default=0,
                condition=ParameterCondition(parameter="a", values=(0,)),
            ),
            HyperparameterSpec(
                name="c",
                kind="categorical",
                choices=(0, 1),
                default=0,
                condition=ParameterCondition(parameter="b", values=(0,)),
            ),
        ))


def test_active_names_report_every_parameter_of_a_fully_active_baseline():
    schema = _taili_space()
    assert schema.active_names(_baseline()) == schema.names()


def test_active_names_drop_a_parameter_whose_condition_no_longer_holds():
    schema = _switchable_space()
    switched = _switchable_baseline("synthetic-other")
    assert schema.active_names(switched) == (SCHEDULER,)


def test_the_active_assignment_is_a_nested_config_fragment():
    schema = _taili_space()
    active = schema.active_assignment(_baseline())
    assert active["skrl"]["agent"]["rollouts"] == 48
    assert active["skrl"]["agent"]["learning_rate_scheduler_kwargs"]["kl_threshold"] == 0.016


def test_the_active_assignment_drops_an_inactive_parameter_from_the_nested_shape():
    schema = _switchable_space()
    active = schema.active_assignment(_switchable_baseline("synthetic-other"))
    assert "learning_rate_scheduler_kwargs" not in active["skrl"]["agent"]
    schema.validate_assignment(active, forbid_inactive=True)


def test_the_active_assignment_omits_an_active_parameter_the_config_never_set():
    schema = _taili_space()
    incomplete = _baseline()
    del incomplete["skrl"]["agent"]["rollouts"]
    active = schema.active_assignment(incomplete)
    assert "rollouts" not in active["skrl"]["agent"]
    assert "mini_batches" in active["skrl"]["agent"]


def test_the_shipped_config_validates_against_the_product_space():
    schema = _taili_space()
    schema.validate_assignment(_baseline(), forbid_inactive=True)


def test_a_missing_active_parameter_is_reported():
    schema = _taili_space()
    incomplete = _baseline()
    del incomplete["skrl"]["agent"]["discount_factor"]
    with pytest.raises(ValueError, match="skrl.agent.discount_factor is missing"):
        schema.validate_assignment(incomplete)


def test_an_incomplete_assignment_is_tolerated_when_not_required():
    schema = _taili_space()
    incomplete = _baseline()
    del incomplete["skrl"]["agent"]["discount_factor"]
    schema.validate_assignment(incomplete, require_complete=False)


def test_an_assignment_lists_every_problem_it_found():
    schema = _taili_space()
    broken = _baseline()
    broken["skrl"]["agent"]["learning_rate"] = 5.0
    broken["skrl"]["agent"]["mini_batches"] = 18
    with pytest.raises(ValueError) as excinfo:
        schema.validate_assignment(broken)
    message = str(excinfo.value)
    assert "skrl.agent.learning_rate" in message
    assert "skrl.agent.mini_batches" in message


def test_an_inactive_parameter_is_ignored_unless_forbidden():
    schema = _switchable_space()
    switched = _switchable_baseline("synthetic-other")
    switched["skrl"]["agent"]["learning_rate_scheduler_kwargs"]["kl_threshold"] = 99.0
    schema.validate_assignment(switched)
    with pytest.raises(ValueError, match="does not apply"):
        schema.validate_assignment(switched, forbid_inactive=True)


def test_an_inactive_parameter_is_forbidden_even_without_completeness():
    schema = _switchable_space()
    switched = _switchable_baseline("synthetic-other")
    switched["skrl"]["agent"]["learning_rate_scheduler_kwargs"]["kl_threshold"] = 99.0
    del switched["skrl"]["agent"]["learning_rate_scheduler"]
    with pytest.raises(ValueError, match="does not apply"):
        schema.validate_assignment(switched, require_complete=False, forbid_inactive=True)


def test_a_missing_parameter_is_not_reported_when_completeness_is_not_required():
    schema = _taili_space()
    incomplete = _baseline()
    del incomplete["skrl"]["agent"]["discount_factor"]
    schema.validate_assignment(incomplete, require_complete=False, forbid_inactive=True)


def test_grid_values_enumerate_discrete_and_categorical_domains():
    assert _discrete("a", 4, 16, 4, 8).grid_values() == (4, 8, 12, 16)
    assert _categorical("b", ("x", "y"), "x").grid_values() == ("x", "y")


def test_a_log_scaled_domain_describes_its_own_scale():
    logged = _continuous("skrl.agent.learning_rate", 1.0e-5, 1.0e-3, 1.0e-4, log=True)
    assert "log-uniform" in logged.describe_domain()
    linear = _continuous("skrl.agent.entropy_loss_scale", 0.0, 0.05, 0.02)
    assert "log-uniform" not in linear.describe_domain()
    assert "uniform" in linear.describe_domain()


def test_grid_values_refuse_continuous_domains_without_grid_points():
    assert _continuous("a", 0.0, 1.0, 0.5).grid_values() is None
    assert _continuous("a", 1.0e-5, 1.0e-3, 1.0e-4, log=True).grid_values() is None


def test_a_continuous_grid_places_the_endpoints_on_the_declared_bounds():
    spec = HyperparameterSpec(name="a", kind="continuous", low=0.0, high=1.0, default=0.25, grid_points=5)
    values = spec.grid_values()
    assert values == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert all(spec.contains(value) for value in values)


def test_a_two_point_continuous_grid_is_the_two_bounds():
    spec = HyperparameterSpec(name="a", kind="continuous", low=0.1, high=0.3, default=0.1, grid_points=2)
    assert spec.grid_values() == (0.1, 0.3)


def test_a_log_scaled_grid_spaces_its_points_geometrically():
    """A coarse linear grid would cluster near the low bound and miss the decades."""
    spec = HyperparameterSpec(
        name="skrl.agent.learning_rate",
        kind="continuous",
        low=1.0e-5,
        high=1.0e-3,
        default=1.0e-4,
        log=True,
        grid_points=3,
    )
    values = spec.grid_values()
    assert values == pytest.approx((1.0e-5, 1.0e-4, 1.0e-3))
    assert all(spec.contains(value) for value in values)


def test_the_product_space_enumerates_the_grid_of_its_continuous_parameters():
    schema = _taili_space()
    assert schema.spec("skrl.agent.learning_rate").grid_values() == pytest.approx((1.0e-5, 1.0e-4, 1.0e-3))
    discount = schema.spec("skrl.agent.discount_factor").grid_values()
    assert len(discount) == 4
    assert discount[0] == pytest.approx(0.95)
    assert discount[-1] == pytest.approx(0.999)


def test_a_continuous_product_parameter_without_grid_points_cannot_be_enumerated():
    """The sampler has to fall back to random draws for these two."""
    schema = _taili_space()
    assert schema.spec("skrl.agent.entropy_loss_scale").grid_values() is None
    assert schema.spec(KL_THRESHOLD).grid_values() is None


def test_grid_points_must_leave_room_for_two_distinct_values():
    with pytest.raises(ValueError, match="at least 2"):
        HyperparameterSpec(name="a", kind="continuous", low=0.0, high=1.0, default=0.5, grid_points=1)


def test_an_already_enumerable_domain_cannot_declare_grid_points():
    with pytest.raises(ValueError, match="already enumerable"):
        HyperparameterSpec(name="a", kind="discrete", low=1, high=5, step=1, default=1, grid_points=3)
    with pytest.raises(ValueError, match="already enumerable"):
        HyperparameterSpec(
            name="a",
            kind="categorical",
            choices=("x", "y"),
            default="x",
            grid_points=3,
        )


def test_a_non_finite_bound_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        _continuous("a", 0.0, float("inf"), 0.5)
    with pytest.raises(ValueError, match="finite"):
        HyperparameterSpec(
            name="a",
            kind="discrete",
            low=0.0,
            high=1.0,
            step=float("nan"),
            default=0.5,
        )


def test_grid_values_stay_inside_the_domain_they_were_built_from():
    spec = HyperparameterSpec(name="a", kind="discrete", low=0.1, high=0.3, step=0.1, default=0.1)
    values = spec.grid_values()
    assert values == (0.1, 0.2, 0.3)
    assert all(spec.contains(value) for value in values)


def test_a_large_discrete_grid_keeps_every_point_inside_the_domain():
    spec = HyperparameterSpec(name="a", kind="discrete", low=0.0, high=1.0, step=0.00001, default=0.0)
    values = spec.grid_values()
    assert len(values) == 100001
    assert values[-1] == 1.0
    assert all(spec.contains(value) for value in values[::9973])


def test_a_bound_reached_through_float_error_is_still_inside():
    spec = _continuous("skrl.agent.learning_rate", 1.0e-5, 1.0e-3, 1.0e-4)
    assert spec.contains(1.0e-3 * (1.0 + 1.0e-12))
    assert not spec.contains(1.0e-3 * 1.01)


def test_an_integer_domain_grid_enumerates_ints():
    values = _discrete("a", 4, 16, 4, 8).grid_values()
    assert all(isinstance(value, int) and not isinstance(value, bool) for value in values)


def test_a_fractional_domain_grid_enumerates_floats():
    spec = HyperparameterSpec(name="a", kind="discrete", low=0.0, high=1.0, step=0.25, default=0.5)
    values = spec.grid_values()
    assert values == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert all(isinstance(value, float) for value in values)
    assert all(spec.contains(value) for value in values)


def test_the_product_space_declares_the_rollouts_multiple_constraint():
    constraints = _taili_space().constraints
    assert len(constraints) == 1
    assert constraints[0].kind == "multiple"
    assert constraints[0].parameter == "skrl.agent.rollouts"
    assert constraints[0].of == "skrl.agent.mini_batches"


def test_the_shipped_config_satisfies_the_declared_constraint():
    assert _taili_space().is_feasible(_baseline())


def test_a_constraint_rejects_values_that_are_each_inside_their_domain():
    """48 and 32 are both legal on their own; only the relation between them is not."""
    schema = _taili_space()
    broken = _baseline()
    broken["skrl"]["agent"]["rollouts"] = 48
    broken["skrl"]["agent"]["mini_batches"] = 32
    assert schema.spec("skrl.agent.rollouts").contains(48)
    assert schema.spec("skrl.agent.mini_batches").contains(32)
    assert not schema.is_feasible(broken)
    with pytest.raises(ValueError, match="whole multiple"):
        schema.validate_assignment(broken)


def test_a_constraint_is_skipped_while_either_side_does_not_apply():
    """A stale baseline value for an inactive parameter must not sink a legal trial."""
    schema = SearchSpaceSchema(
        specs=(
            _categorical("mode", ("on", "off"), "on"),
            HyperparameterSpec(
                name="base",
                kind="discrete",
                low=4,
                high=32,
                step=4,
                default=16,
                condition=ParameterCondition(parameter="mode", values=("on",)),
            ),
            _discrete("total", 24, 96, 24, 48),
        ),
        constraints=(ParameterConstraint(kind="multiple", parameter="total", of="base"),),
    )
    assert schema.is_feasible({"mode": "on", "base": 16, "total": 48})
    assert not schema.is_feasible({"mode": "on", "base": 32, "total": 48})
    assert schema.is_feasible({"mode": "off", "base": 32, "total": 48})


def test_a_constraint_cannot_be_defined_against_itself():
    with pytest.raises(ValueError, match="against itself"):
        ParameterConstraint(kind="multiple", parameter="a", of="a")


def test_a_constraint_on_an_unknown_parameter_is_rejected():
    with pytest.raises(ValueError, match="unknown parameter"):
        SearchSpaceSchema(
            specs=(_discrete("a", 4, 8, 4, 4),),
            constraints=(ParameterConstraint(kind="multiple", parameter="a", of="b"),),
        )


def test_a_repeated_constraint_is_rejected():
    constraint = ParameterConstraint(kind="multiple", parameter="total", of="base")
    with pytest.raises(ValueError, match="duplicate constraint"):
        SearchSpaceSchema(
            specs=(_discrete("base", 4, 8, 4, 4), _discrete("total", 4, 16, 4, 8)),
            constraints=(constraint, constraint),
        )


def test_a_constraint_holds_while_a_side_is_still_undrawn():
    constraint = ParameterConstraint(kind="multiple", parameter="total", of="base")
    assert constraint.holds({"total": 7})
    assert constraint.holds({})


def test_a_constraint_with_a_zero_divisor_is_not_satisfied():
    constraint = ParameterConstraint(kind="multiple", parameter="total", of="base")
    assert not constraint.holds({"total": 8, "base": 0})


def test_a_float_quotient_that_is_whole_counts_as_a_multiple():
    constraint = ParameterConstraint(kind="multiple", parameter="total", of="base")
    assert constraint.holds({"total": 0.3, "base": 0.1})
    assert not constraint.holds({"total": 0.31, "base": 0.1})


def test_default_values_expose_a_flat_view_of_every_parameter():
    schema = _taili_space()
    values = schema.default_values()
    assert set(values) == set(schema.names())
    assert values["skrl.agent.learning_rate"] == 1.0e-4


def test_the_default_assignment_is_nested_under_the_real_config_shape():
    schema = _taili_space()
    agent = schema.default_assignment()["skrl"]["agent"]
    assert agent["learning_rate"] == 1.0e-4
    assert agent["learning_rate_scheduler_kwargs"]["kl_threshold"] == 0.016
    assert agent["mini_batches"] == 16


def test_the_default_assignment_validates_against_its_own_schema():
    schema = _taili_space()
    schema.validate_assignment(schema.default_assignment(), forbid_inactive=True)


def test_the_default_assignment_omits_a_parameter_whose_condition_does_not_hold():
    schema = SearchSpaceSchema(specs=(
        HyperparameterSpec(name="a", kind="categorical", choices=("on", "off"), default="off"),
        HyperparameterSpec(
            name="b",
            kind="continuous",
            low=0.0,
            high=1.0,
            default=0.5,
            condition=ParameterCondition(parameter="a", values=("on",)),
        ),
    ))
    defaults = schema.default_assignment()
    assert set(defaults) == {"a"}
    schema.validate_assignment(defaults, forbid_inactive=True)


def test_sampling_order_places_each_guard_before_what_it_guards():
    schema = _taili_space()
    order = schema.sampling_order()
    assert set(order) == set(schema.names())
    assert order.index(SCHEDULER) < order.index(KL_THRESHOLD)


def test_sampling_order_leaves_an_already_ordered_declaration_alone():
    schema = _taili_space()
    assert schema.sampling_order() == schema.names()


def test_sampling_order_is_declaration_biased_among_equally_ready_parameters():
    """Two spaces with one fingerprint may draw in different sequences."""
    first = SearchSpaceSchema(specs=(
        _continuous("x", 0.0, 1.0, 0.5),
        _categorical("a", ("on", "off"), "on"),
    ))
    second = SearchSpaceSchema(specs=tuple(reversed(first.specs)))
    assert first.fingerprint() == second.fingerprint()
    assert first.sampling_order() == ("x", "a")
    assert second.sampling_order() == ("a", "x")


def test_a_condition_value_outside_the_guards_domain_is_rejected():
    """A dropped character would otherwise leave the parameter permanently inactive."""
    with pytest.raises(ValueError, match="could never become active"):
        SearchSpaceSchema(specs=(
            _categorical("sched", ("KLAdaptiveLR", "AdaptiveLR"), "KLAdaptiveLR"),
            HyperparameterSpec(
                name="kw.kl_threshold",
                kind="continuous",
                low=0.008,
                high=0.016,
                default=0.016,
                condition=ParameterCondition(parameter="sched", values=("KLAdaptive",)),
            ),
        ))


def test_a_condition_value_of_the_wrong_type_is_rejected():
    with pytest.raises(ValueError, match="could never become active"):
        SearchSpaceSchema(specs=(
            _discrete("mini_batches", 4, 32, 4, 16),
            HyperparameterSpec(
                name="kl_threshold",
                kind="continuous",
                low=0.008,
                high=0.016,
                default=0.016,
                condition=ParameterCondition(parameter="mini_batches", values=(8.5,)),
            ),
        ))


def test_a_parameter_name_that_prefixes_another_is_rejected():
    """One would have to hold a value and the other a mapping; nothing can satisfy both."""
    with pytest.raises(ValueError, match="is a prefix of"):
        SearchSpaceSchema(specs=(
            _continuous("skrl.agent", 0.0, 1.0, 0.5),
            _discrete("skrl.agent.mini_batches", 4, 32, 4, 16),
        ))


def test_a_name_may_share_a_prefix_without_being_a_path_prefix_of_it():
    """``learning_rate`` and ``learning_rate_scheduler`` are distinct keys, not a nesting."""
    schema = SearchSpaceSchema(specs=(
        _continuous("skrl.agent.learning_rate", 0.0, 1.0, 0.5),
        _categorical("skrl.agent.learning_rate_scheduler", ("KLAdaptiveLR",), "KLAdaptiveLR"),
    ))
    assert schema.sampling_order() == schema.names()


def test_an_overlong_name_is_rejected_when_the_spec_is_built():
    """Rejected at construction, not later at the first name lookup."""
    with pytest.raises(ValueError, match="invalid parameter name"):
        _continuous("a." + "b" * 200, 0.0, 1.0, 0.5)


def test_a_non_finite_choice_is_rejected():
    """It survives in memory but pydantic writes it as null, changing the domain."""
    with pytest.raises(ValueError, match="must be finite"):
        _categorical("a", (float("nan"), "x"), "x")
    with pytest.raises(ValueError, match="must be finite"):
        _categorical("a", ([float("inf")], "x"), "x")


def test_a_non_finite_condition_value_is_rejected():
    with pytest.raises(ValueError, match="must be finite"):
        ParameterCondition(parameter="a", values=(float("inf"),))


def test_a_grid_too_large_to_materialize_is_refused_rather_than_hung():
    spec = HyperparameterSpec(name="a", kind="discrete", low=0, high=10**8, step=1, default=0)
    with pytest.raises(ValueError, match="exceeds"):
        spec.grid_values()


def test_a_grid_point_count_beyond_the_ceiling_is_refused_at_construction():
    with pytest.raises(ValueError, match="exceeds"):
        HyperparameterSpec(
            name="a", kind="continuous", low=0.0, high=1.0, default=0.5, grid_points=10**9,
        )


def test_an_off_grid_value_in_a_huge_domain_is_still_rejected():
    """The integrality slack is capped, so a midpoint never counts as a grid value."""
    spec = HyperparameterSpec(name="a", kind="discrete", low=0, high=10**9, step=1, default=0)
    assert spec.contains(5 * 10**8)
    assert not spec.contains(5 * 10**8 + 0.5)
    assert not spec.contains(10**9 + 0.5)


def test_sampling_order_walks_a_backwards_declaration_into_dependency_order():
    schema = SearchSpaceSchema(specs=(
        HyperparameterSpec(
            name="c",
            kind="continuous",
            low=0.0,
            high=1.0,
            default=0.5,
            condition=ParameterCondition(parameter="b", values=(1,)),
        ),
        HyperparameterSpec(
            name="b",
            kind="categorical",
            choices=(1, 2),
            default=1,
            condition=ParameterCondition(parameter="a", values=("on",)),
        ),
        HyperparameterSpec(name="a", kind="categorical", choices=("on", "off"), default="on"),
    ))
    assert schema.sampling_order() == ("a", "b", "c")


def _with_specs_replaced(schema: SearchSpaceSchema, updates: Mapping[str, dict]) -> SearchSpaceSchema:
    """Rebuild a schema with specs changed, so the edit is validated.

    ``model_copy(update=...)`` would skip validation and let a test assert on a
    value the schema never agreed to hold.  Constraints are carried across, so a
    fingerprint that changes here changed because of the spec.

    Several specs move at once because an edit to one may only be legal together
    with an edit to another -- widening a conditional parameter's domain is only
    expressible by widening its guard as well.
    """
    return SearchSpaceSchema.model_validate({
        "schema_version": schema.schema_version,
        "specs": [
            {**spec.model_dump(mode="json"), **updates.get(spec.name, {})} for spec in schema.specs
        ],
        "constraints": [item.model_dump(mode="json") for item in schema.constraints],
    })


def _with_spec_replaced(schema: SearchSpaceSchema, name: str, **updates) -> SearchSpaceSchema:
    """Rebuild a schema with a single spec changed."""
    return _with_specs_replaced(schema, {name: updates})


def test_a_fingerprint_changes_when_choices_change():
    base = _taili_space()
    widened = _with_spec_replaced(base, SCHEDULER, choices=("KLAdaptiveLR", "synthetic-other"))
    assert widened.fingerprint() != base.fingerprint()


def test_a_fingerprint_changes_when_a_condition_changes():
    """Widening the guard is what makes the widened condition expressible at all."""
    base = _taili_space()
    loosened = _with_specs_replaced(base, {
        SCHEDULER: {"choices": ("KLAdaptiveLR", "synthetic-other")},
        KL_THRESHOLD: {
            "condition": ParameterCondition(
                parameter=SCHEDULER, values=("KLAdaptiveLR", "synthetic-other")
            )
        },
    })
    elsewhere = _baseline()
    set_parameter(elsewhere, SCHEDULER, "synthetic-other")
    assert not base.is_active(KL_THRESHOLD, elsewhere)
    assert loosened.is_active(KL_THRESHOLD, elsewhere)
    assert loosened.fingerprint() != base.fingerprint()


def test_a_fingerprint_ignores_declaration_order():
    forward = _taili_space()
    reordered = SearchSpaceSchema(specs=tuple(reversed(forward.specs)), constraints=forward.constraints)
    assert reordered.fingerprint() == forward.fingerprint()


def test_a_fingerprint_changes_when_a_grid_density_changes():
    """Two spaces that differ only in how finely a domain is cut are not the same space."""
    base = _taili_space()
    denser = _with_spec_replaced(base, "skrl.agent.discount_factor", grid_points=7)
    assert denser.fingerprint() != base.fingerprint()


def test_a_fingerprint_changes_when_a_constraint_disappears():
    base = _taili_space()
    relaxed = SearchSpaceSchema(specs=base.specs)
    assert relaxed.fingerprint() != base.fingerprint()


def test_a_fingerprint_distinguishes_schema_versions():
    base = _taili_space()
    bumped = SearchSpaceSchema(schema_version="rl-agent.hyperparameter-search/v2", specs=base.specs)
    assert bumped.fingerprint() != base.fingerprint()


def test_a_blank_schema_version_is_rejected():
    with pytest.raises(ValueError, match="schema_version"):
        SearchSpaceSchema(schema_version="  ", specs=(_continuous("a", 0.0, 1.0, 0.5),))


def test_a_fingerprint_changes_when_a_domain_moves():
    moved = _with_spec_replaced(_taili_space(), "skrl.agent.learning_rate", low=1.0e-6)
    assert moved.fingerprint() != _taili_space().fingerprint()


def test_a_search_space_survives_a_json_round_trip():
    schema = _taili_space()
    restored = SearchSpaceSchema.model_validate_json(schema.model_dump_json())
    assert restored == schema
    assert restored.fingerprint() == schema.fingerprint()
    assert restored.constraints == schema.constraints


def test_the_product_space_loads_every_parameter_from_the_shipped_config():
    """A renamed config key has to fail here, not on the first trial."""
    schema = _taili_space()
    config = _baseline()
    for spec in schema.specs:
        found, value = get_parameter(config, spec.name)
        assert found, f"{spec.name} is not a key of {CONFIG_PATH.name}"
        assert value == spec.default, f"{spec.name} ships as {value!r} but the space says {spec.default!r}"


def test_the_product_space_covers_the_acceptance_criteria_of_p4():
    assert set(_taili_space().names()) >= {
        "skrl.agent.learning_rate",
        "skrl.agent.entropy_loss_scale",
        "skrl.agent.discount_factor",
    }


def test_the_product_space_pins_the_scheduler_to_the_one_verified_name():
    """A second choice claims a training configuration nobody has run."""
    assert _taili_space().spec(SCHEDULER).choices == ("KLAdaptiveLR",)


def test_a_search_space_file_must_list_parameters():
    with pytest.raises(ValueError, match="requires a 'parameters' list"):
        load_search_space({"constraints": []})


def test_a_search_space_file_must_be_a_mapping():
    with pytest.raises(ValueError, match="must be a mapping"):
        load_search_space(["parameters"])


def test_an_unknown_file_key_is_rejected_rather_than_dropped():
    """A misspelled ``constraint:`` would otherwise load a space with no relations."""
    data = yaml.safe_load(SPACE_PATH.read_text(encoding="utf-8"))
    data["constraint"] = data.pop("constraints")
    with pytest.raises(ValueError, match="unknown search space keys"):
        load_search_space(data)


def test_the_schema_version_is_not_a_file_key():
    """It lives in code, so a file cannot drift from the reader that interprets it."""
    with pytest.raises(ValueError, match="unknown search space keys"):
        load_search_space({"schema_version": "rl-agent.hyperparameter-search/v1", "parameters": []})


def test_get_and_set_parameter_traverse_dotted_paths():
    assignment: dict = {}
    set_parameter(assignment, "skrl.agent.mini_batches", 16, create_missing=True)
    assert assignment == {"skrl": {"agent": {"mini_batches": 16}}}
    assert get_parameter(assignment, "skrl.agent.mini_batches") == (True, 16)


def test_set_parameter_refuses_an_unknown_path_by_default():
    """A misspelled name must fail rather than leave the real key at its baseline."""
    assignment = {"skrl": {"agent": {"rollouts": 48}}}
    with pytest.raises(ValueError, match="config key does not exist"):
        set_parameter(assignment, "skrl.agent.rollout", 24)
    assert assignment == {"skrl": {"agent": {"rollouts": 48}}}


def test_set_parameter_rejects_assignment_through_a_scalar():
    with pytest.raises(ValueError, match="not a mapping"):
        set_parameter({"skrl": {"agent": 4}}, "skrl.agent.learning_rate", 1.0e-4)


def test_a_dotted_path_must_start_with_a_letter():
    with pytest.raises(ValueError, match="invalid parameter name"):
        split_parameter("4rollouts")
