"""Injection, round trip, and the refusals that keep a trial honest.

The synthetic configs here are three keys deep on purpose: a path helper that walks
one level, or a diff that compares only the top of a section, passes against a flat
fixture and fails against a nested one.

Assignments are built through :func:`assign`, which is the same construction the
sampler performs, so the form these tests exercise is the form the two modules
actually exchange.  A test that hand-wrote ``{"skrl.agent.mini_batches": 24}`` would
pass every lookup here and fail in the search that composes the two.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from autotuner.research.config_injection import (
    InjectedValue,
    Injection,
    InjectionReport,
    inject_assignment,
    inject_into_file,
    load_config_file,
    verify_injection,
    write_injection,
)
from autotuner.research.hyperparameter_space import (
    HyperparameterSpec,
    ParameterCondition,
    ParameterConstraint,
    SearchSpaceSchema,
    load_search_space,
    set_parameter,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
_PRODUCT_DIR = REPO_ROOT / "products" / "taili" / "blind_locomotion"
SPACE_PATH = _PRODUCT_DIR / "hyperparameter_space.yaml"
CONFIG_PATH = _PRODUCT_DIR / "taili_blind_config.yaml"

MINI_BATCHES = "skrl.agent.mini_batches"
ROLLOUTS = "skrl.agent.rollouts"
LEARNING_RATE = "skrl.agent.learning_rate"
DISCOUNT_FACTOR = "skrl.agent.discount_factor"
GUARDED = "skrl.agent.learning_rate_scheduler_kwargs.kl_threshold"


def assign(values: Mapping[str, Any]) -> dict[str, Any]:
    """A nested assignment, built the way a sampler draws one.

    The dotted names stay the *keys of ``values``*; what comes back is the tree a
    config holds.  Nothing downstream reads a dotted key out of a config, so a
    fixture that kept them flat would be exercising a shape no caller produces.
    """
    tree: dict[str, Any] = {}
    for name, value in values.items():
        set_parameter(tree, name, value, create_missing=True)
    return tree


def _taili_space() -> SearchSpaceSchema:
    """The shipped search space, read from the product file."""
    return load_search_space(yaml.safe_load(SPACE_PATH.read_text(encoding="utf-8")))


def _taili_config() -> dict:
    """The shipped baseline config, read from the product file."""
    return load_config_file(CONFIG_PATH)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nested_config() -> dict:
    """A config with a guard and a guarded key, both two levels down."""
    return {
        "outer": {
            "inner": {
                "schedule": "adaptive",
                "threshold": 0.5,
                "scale": 1.0,
            },
            "note": "keep me",
        },
        "list": [1, 2, 3],
    }


def _nested_space() -> SearchSpaceSchema:
    """A space whose guarded parameter is active only under one schedule."""
    return SearchSpaceSchema(
        specs=(
            HyperparameterSpec(
                name="outer.inner.schedule",
                kind="categorical",
                choices=("adaptive", "constant"),
                default="adaptive",
            ),
            HyperparameterSpec(
                name="outer.inner.threshold",
                kind="continuous",
                low=0.1,
                high=1.0,
                default=0.5,
                condition=ParameterCondition(parameter="outer.inner.schedule", values=("adaptive",)),
            ),
        ),
        constraints=(),
    )


def _target(tmp_path: Path, name: str = "trial.yaml") -> Path:
    return tmp_path / name


# --- the real product pair ----------------------------------------------------


def test_an_injected_config_carries_every_value_it_was_given():
    space = _taili_space()
    source = _taili_config()
    values = {MINI_BATCHES: 24, ROLLOUTS: 72, LEARNING_RATE: 3.5e-05}
    injection = inject_assignment(source, space, assign(values))
    for name, wanted in values.items():
        found, actual = _dig(injection.config, name)
        assert found, f"{name} is missing from the injected config: {injection.config}"
        assert actual == wanted
    assert injection.report.moved_paths == tuple(sorted(values))
    assert injection.config["skrl"]["agent"]["discount_factor"] == source["skrl"]["agent"]["discount_factor"]


def _dig(tree: dict, dotted: str) -> tuple[bool, object]:
    node: object = tree
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def test_a_flat_assignment_is_refused_by_its_shape():
    """A dotted key is a config key nothing reads, so it must not read as "empty".

    ``validate_assignment`` walks the tree with the dotted name, so a flat mapping
    fills no parameter and every domain check on it passes vacuously.  Accepting it
    would inject nothing while reporting a trial that ran.
    """
    with pytest.raises(ValueError, match="nested the way a config holds it"):
        inject_assignment(_taili_config(), _taili_space(), {MINI_BATCHES: 24})
    with pytest.raises(ValueError, match="nested the way a config holds it"):
        verify_injection(_taili_config(), _taili_config(), _taili_space(), {MINI_BATCHES: 24})


def test_a_flat_assignment_is_not_silently_validated():
    """The same flat form, on a value that is out of domain, still says so."""
    space = SearchSpaceSchema(
        specs=(HyperparameterSpec(name="a.b", kind="discrete", low=1, high=4, step=1, default=1),),
        constraints=(),
    )
    with pytest.raises(ValueError, match="fills no parameter"):
        inject_assignment({"a": {"b": 1}}, space, {"a.b": 99})


def test_moving_a_parameter_nothing_reads_is_refused_before_the_write(tmp_path):
    """``value_loss_scale`` is a real key in the config and not a search parameter."""
    with pytest.raises(ValueError, match="not declared in the search space"):
        inject_assignment(
            _taili_config(), _taili_space(), assign({"skrl.agent.value_loss_scale": 0.5})
        )


def test_a_value_outside_its_declared_domain_is_refused():
    with pytest.raises(ValueError, match="outside"):
        inject_assignment(_taili_config(), _taili_space(), assign({MINI_BATCHES: 22}))


def test_a_combination_that_breaks_a_relation_is_refused():
    """mini_batches=16 with rollouts=72: 72 is not a whole multiple of 16."""
    with pytest.raises(ValueError, match="whole multiple"):
        inject_assignment(_taili_config(), _taili_space(), assign({MINI_BATCHES: 16, ROLLOUTS: 72}))


def test_a_relation_broken_against_an_untouched_parameter_is_refused():
    """Setting one parameter can leave its relation to a sibling unsatisfiable.

    ``rollouts`` stays at 48 while ``mini_batches`` is asked for 28, and 48 is not a
    multiple of 28.  The config would still train, so nothing downstream would notice
    that the trial ran outside the space it was recorded against.
    """
    with pytest.raises(ValueError, match="whole multiple"):
        inject_assignment(_taili_config(), _taili_space(), assign({MINI_BATCHES: 28}))
    report = inject_assignment(_taili_config(), _taili_space(), assign({MINI_BATCHES: 24})).report
    assert report.moved_paths == (MINI_BATCHES,)


def test_a_guarded_parameter_stays_injectable_while_its_guard_holds():
    """The shipped scheduler is the guarded parameter's guard value."""
    report = inject_assignment(_taili_config(), _taili_space(), assign({GUARDED: 0.012})).report
    assert GUARDED in report.moved_paths


def test_the_source_config_is_never_modified_in_place():
    source = _taili_config()
    before = yaml.safe_dump(source, sort_keys=False)
    inject_assignment(source, _taili_space(), assign({MINI_BATCHES: 24}))
    assert yaml.safe_dump(source, sort_keys=False) == before


def test_the_source_file_is_left_untouched_by_a_full_injection(tmp_path):
    before = _file_digest(CONFIG_PATH)
    inject_into_file(CONFIG_PATH, _target(tmp_path), _taili_space(), assign({MINI_BATCHES: 24}))
    assert _file_digest(CONFIG_PATH) == before


def test_a_full_injection_round_trips_through_the_file(tmp_path):
    target = _target(tmp_path)
    values = {MINI_BATCHES: 24, ROLLOUTS: 72}
    report = inject_into_file(CONFIG_PATH, target, _taili_space(), assign(values))
    reread = load_config_file(target)
    for name, wanted in values.items():
        assert _dig(reread, name) == (True, wanted)
    assert report.moved_paths == tuple(sorted(values))
    assert reread["skrl"]["agent"]["entropy_loss_scale"] == 0.02


def test_writing_over_the_source_is_refused(tmp_path):
    source = _target(tmp_path, "same.yaml")
    source.write_text(yaml.safe_dump(_nested_config()), encoding="utf-8")
    with pytest.raises(ValueError, match="over its own source"):
        inject_into_file(source, source, _nested_space(), assign({"outer.inner.threshold": 0.4}))


def test_writing_into_a_directory_that_does_not_exist_is_refused(tmp_path):
    with pytest.raises(ValueError, match="target directory does not exist"):
        write_injection(
            inject_assignment(_taili_config(), _taili_space(), assign({MINI_BATCHES: 24})),
            tmp_path / "missing" / "trial.yaml",
        )


def test_writing_a_config_its_report_does_not_describe_is_refused(tmp_path):
    """An ``Injection`` is a plain pair, so the pair has to be checked before use."""
    injection = inject_assignment(_taili_config(), _taili_space(), assign({MINI_BATCHES: 24}))
    impostor = Injection(
        config=injection.config,
        report=injection.report.model_copy(update={"injected_fingerprint": "0" * 64}),
    )
    with pytest.raises(ValueError, match="describes a different config"):
        write_injection(impostor, _target(tmp_path))


# --- the checks that need a config the product does not ship -------------------


def test_a_value_already_held_is_recorded_as_unchanged(tmp_path):
    """The baseline trial is a legal trial, and saying so is not the same as changing."""
    space = _taili_space()
    source = _taili_config()
    injection = inject_assignment(source, space, assign({MINI_BATCHES: 16}))
    assert injection.report.moved_paths == ()
    assert injection.report.changed() == ()
    assert "none of them new values" in injection.report.describe()
    assert injection.report.injected_fingerprint == injection.report.source_fingerprint
    assert inject_into_file(CONFIG_PATH, _target(tmp_path), space, assign({MINI_BATCHES: 16})).changed() == ()


def test_a_conditional_parameter_is_judged_by_the_config_not_by_the_assignment():
    """The guard's value lives in the config, and a partial injection must not hide it.

    The assignment names only the guarded parameter.  Judged against the assignment
    alone there is no guard to read, so the parameter would look dead; judged against
    the config -- which is what the trainer will hold -- it is live.
    """
    space = _nested_space()
    injection = inject_assignment(_nested_config(), space, assign({"outer.inner.threshold": 0.4}))
    assert injection.report.moved_paths == ("outer.inner.threshold",)
    assert injection.config["outer"]["inner"]["threshold"] == 0.4

    switched = _nested_config()
    switched["outer"]["inner"]["schedule"] = "constant"
    with pytest.raises(ValueError, match="inactive"):
        inject_assignment(switched, space, assign({"outer.inner.threshold": 0.4}))


def test_turning_a_guard_off_while_setting_the_guarded_parameter_is_refused():
    """The result would carry a value the trainer never reads."""
    with pytest.raises(ValueError, match="inactive"):
        inject_assignment(
            _nested_config(),
            _nested_space(),
            assign({"outer.inner.schedule": "constant", "outer.inner.threshold": 0.4}),
        )


def test_a_source_value_outside_the_space_is_refused_as_a_whole():
    """Every trial from such a config would be unreproducible, so fail once, now."""
    broken = _nested_config()
    broken["outer"]["inner"]["scale"] = 1.0  # untouched by the injection
    space = SearchSpaceSchema(
        specs=(
            HyperparameterSpec(
                name="outer.inner.schedule",
                kind="categorical",
                choices=("adaptive",),
                default="adaptive",
            ),
            HyperparameterSpec(name="outer.inner.scale", kind="discrete", low=1, high=4, step=1, default=1),
        ),
        constraints=(),
    )
    injected = inject_assignment(broken, space, assign({"outer.inner.scale": 2}))
    assert injected.config["outer"]["inner"]["scale"] == 2
    outside = _nested_config()
    outside["outer"]["inner"]["scale"] = 9
    with pytest.raises(ValueError, match="outside"):
        inject_assignment(outside, space, assign({"outer.inner.schedule": "adaptive"}))


def test_a_config_missing_a_declared_parameter_is_refused():
    """Creating the key would hide the drift; the space and the config must agree."""
    trimmed = _nested_config()
    del trimmed["outer"]["inner"]["threshold"]
    with pytest.raises(ValueError, match="does not have"):
        inject_assignment(trimmed, _nested_space(), assign({"outer.inner.schedule": "constant"}))


def test_an_empty_injection_is_refused():
    with pytest.raises(ValueError, match="at least one parameter"):
        inject_assignment(_nested_config(), _nested_space(), {})


def test_a_source_holding_a_value_with_no_json_form_is_refused():
    """A set has no JSON form, so its hash would not survive another process."""
    source = yaml.safe_load("a: !!set {x: null}\nb: 1\n")
    space = SearchSpaceSchema(
        specs=(HyperparameterSpec(name="b", kind="discrete", low=1, high=1, step=1, default=1),),
        constraints=(),
    )
    with pytest.raises(ValueError, match="no JSON form"):
        inject_assignment(source, space, assign({"b": 1}))


def test_a_value_the_config_cannot_express_is_refused_by_its_domain():
    space = SearchSpaceSchema(
        specs=(
            HyperparameterSpec(
                name="outer.inner.schedule",
                kind="categorical",
                choices=("adaptive", "constant"),
                default="adaptive",
            ),
        ),
        constraints=(),
    )
    with pytest.raises(ValueError, match="outside"):
        inject_assignment(_nested_config(), space, assign({"outer.inner.schedule": {"adaptive"}}))


def test_a_created_list_position_is_reported_as_moved():
    """A list written where a scalar was is a change a run would notice."""
    space = SearchSpaceSchema(
        specs=(HyperparameterSpec(name="list", kind="categorical", choices=([1], [2]), default=[1]),),
        constraints=(),
    )
    report = inject_assignment({"list": [1]}, space, assign({"list": [2]})).report
    assert report.moved_paths == ("list[0]",)
    assert report.changed()[0].previous_present


# --- the reading half ---------------------------------------------------------


def test_verification_catches_a_file_that_lost_a_value(tmp_path):
    target = _target(tmp_path)
    space = _taili_space()
    values = {MINI_BATCHES: 24, ROLLOUTS: 72}
    inject_into_file(CONFIG_PATH, target, space, assign(values))
    tampered = load_config_file(target)
    tampered["skrl"]["agent"]["mini_batches"] = 16
    with pytest.raises(ValueError, match="holds 16 where 24 was injected"):
        verify_injection(_taili_config(), tampered, space, assign(values))


def test_verification_catches_a_file_that_moved_something_else():
    space = _taili_space()
    source = _taili_config()
    tampered = _taili_config()
    tampered["skrl"]["agent"]["mini_batches"] = 24
    tampered["skrl"]["agent"]["entropy_loss_scale"] = 0.5  # not in the assignment
    with pytest.raises(ValueError, match="outside its parameters"):
        verify_injection(source, tampered, space, assign({MINI_BATCHES: 24}))


def test_verification_catches_a_missing_key():
    space = _taili_space()
    trimmed = _taili_config()
    del trimmed["skrl"]["agent"]["mini_batches"]
    with pytest.raises(ValueError, match="does not have this key"):
        verify_injection(_taili_config(), trimmed, space, assign({MINI_BATCHES: 24}))


def test_a_value_is_compared_in_the_form_the_file_will_hold():
    """``1`` and ``1.0`` compare equal in Python and are different scalars in YAML."""
    space = SearchSpaceSchema(
        specs=(HyperparameterSpec(name="x", kind="continuous", low=0.0, high=2.0, default=0.5),),
        constraints=(),
    )
    assert verify_injection({"x": 0.5}, {"x": 1.0}, space, {"x": 1.0}).moved_paths == ("x",)
    with pytest.raises(ValueError, match="holds 1.0 where 1 was injected"):
        verify_injection({"x": 0.5}, {"x": 1.0}, space, {"x": 1})
    with pytest.raises(ValueError, match="holds 1 where 1.0 was injected"):
        verify_injection({"x": 0.5}, {"x": 1}, space, {"x": 1.0})


def test_a_report_rejects_a_parameter_it_did_not_inject():
    with pytest.raises(ValueError, match="outside its parameters"):
        InjectionReport(
            space_fingerprint="a",
            source_fingerprint="b",
            injected_fingerprint="c",
            values=(InjectedValue(name=MINI_BATCHES, value=1),),
            moved_paths=(ROLLOUTS,),
        )


def test_a_value_is_marked_changed_by_its_own_values_not_by_a_flag():
    """The claim is derived, so a record cannot carry it and contradict it."""
    assert InjectedValue(name="x", value=1, previous=1, previous_present=True).changed is False
    assert InjectedValue(name="x", value=2, previous=1, previous_present=True).changed is True
    assert InjectedValue(name="x", value=1, previous=0.5, previous_present=True).changed is True
    assert InjectedValue(name="x", value=None).changed is True


def test_a_report_needs_a_parameter_and_sorted_paths():
    with pytest.raises(ValueError, match="at least one parameter"):
        InjectionReport(
            space_fingerprint="a",
            source_fingerprint="b",
            injected_fingerprint="c",
            values=(),
            moved_paths=(),
        )
    with pytest.raises(ValueError, match="sorted and unique"):
        InjectionReport(
            space_fingerprint="a",
            source_fingerprint="b",
            injected_fingerprint="c",
            values=(InjectedValue(name="x", value=1),),
            moved_paths=("b", "a"),
        )


def test_the_report_identity_changes_with_the_values():
    space = _taili_space()
    source = _taili_config()
    first = inject_assignment(source, space, assign({MINI_BATCHES: 24})).report
    second = inject_assignment(source, space, assign({MINI_BATCHES: 8})).report
    assert first.fingerprint() != second.fingerprint()
    same = inject_assignment(source, space, assign({MINI_BATCHES: 24})).report
    assert first.fingerprint() == same.fingerprint()


# --- the two modules together, and the launch they feed -----------------------


def test_a_drawn_trial_reaches_the_injector(tmp_path):
    """The composition the two modules exist for: a sampler draw, injected.

    The space's parameters are dotted while a config is nested, so a sampler and an
    injector that disagreed about the assignment's shape would each be green on its
    own and refuse every draw the other produced.
    """
    from autotuner.research.hyperparameter_sampler import SearchPlan, build_sampler

    space = _taili_space()
    plan = SearchPlan.for_random(space, seed=3, budget=2)
    drawn = build_sampler(space, plan).collect()
    assert len(drawn) == 2
    for assignment in drawn:
        report = inject_into_file(
            CONFIG_PATH, _target(tmp_path, f"{assignment['skrl']['agent']['mini_batches']}.yaml"),
            space, assignment,
        )
        assert report.moved_paths, "a drawn trial changed nothing, so it proves nothing"
        assert set(report.moved_paths) <= {item.name for item in report.values}


def test_a_launched_run_carries_the_drawn_values(tmp_path):
    """End to end through the launcher's own ``--config`` seam, without training.

    ``--dry-run`` performs every setup step except the training process, including
    deriving ``agent.skrl.yaml`` from the copied config -- which is the artifact the
    trainer actually reads.  Asserting on the injected file alone would prove the file
    was written, not that the value survived the launcher's own transformations.
    """
    from autotuner.research.hyperparameter_sampler import SearchPlan, build_sampler

    space = _taili_space()
    drawn = build_sampler(space, SearchPlan.for_random(space, seed=3, budget=1)).collect()[0]
    injected = tmp_path / "trial.yaml"
    report = inject_into_file(CONFIG_PATH, injected, space, drawn)

    run_root = tmp_path / "runs"
    before = _file_digest(CONFIG_PATH)
    completed = subprocess.run(
        [
            sys.executable, "-m", "taili.blind_locomotion.launch_taili_train",
            "--config", str(injected),
            "--run-root", str(run_root),
            "--run-id", "p1",
            "--data-root", str(tmp_path / "data"),
            "--dry-run",
        ],
        cwd=str(REPO_ROOT / "products"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "products"), str(REPO_ROOT)]),
        },
    )
    run_dir = run_root / "p1"
    assert completed.returncode == 0, f"launcher exited {completed.returncode}; see {run_dir}"

    # The repository's own config is the source of every trial, so a launch that
    # rewrote it would silently retarget every trial drawn afterwards.
    assert _file_digest(CONFIG_PATH) == before

    agent = yaml.safe_load((run_dir / "agent.skrl.yaml").read_text(encoding="utf-8"))
    copied = load_config_file(run_dir / "taili_blind_config.yaml")
    assert report.moved_paths
    for name in report.moved_paths:
        wanted = drawn
        for segment in name.split("."):
            wanted = wanted[segment]
        assert _dig(copied, name) == (True, wanted)
        if name.startswith("skrl."):
            # ``agent.skrl.yaml`` *is* the config's ``skrl`` section, so the prefix
            # comes off on the way in.  Checking the un-stripped path would pass on
            # the copied config and never reach the file the trainer reads.
            found, actual = _dig(agent, name[len("skrl.") :])
            assert found, f"{name} never reached agent.skrl.yaml: {sorted(agent)}"
            assert actual == wanted, f"agent.skrl.yaml holds {actual!r} where {wanted!r} was drawn"
