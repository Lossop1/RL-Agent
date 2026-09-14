"""Write a sampled assignment into a training config, and prove it landed.

A trial is only evidence if the trainer actually read the values the trial claims to
have tried.  Three things can quietly break that, and this module is arranged around
refusing each of them:

* **The value never reached the file.**  The assignment is written into a copy of the
  source config, the copy is read back, and every injected value is compared in its
  canonical JSON form.  ``8`` and ``8.0`` are different values here because they are
  different scalars in the file, whatever Python's ``==`` says.
* **Something else moved with it.**  Injecting a hyperparameter must not disturb any
  other part of a config a training run depends on, so the leaf-by-leaf diff between
  the source and the injected tree has to lie entirely under the injected parameter
  names.
* **The source got overwritten.**  A trial writes its own copy, never the repository's
  config: two trials must not race on one file, and a finished trial's evidence has to
  still match the values it ran with.

Why not ``config_overlay``.  That door is deliberately narrow -- its allowlist covers
task-contract sections such as ``reward`` and ``env`` and leaves ``skrl`` out -- because
it is shared with the LLM mechanism loop.  A hyperparameter search uses the launcher's
own ``--config`` seam instead: it already accepts an arbitrary source config, copies it
into the run directory, and derives ``agent.skrl.yaml`` from that copy.  Widening the
overlay allowlist so the search could squeeze through would loosen a guard that exists
for a different caller.

Two limits worth stating rather than implying.  The round trip proves that the
*launcher's* config carries the value, not that the trainer reads it: a declared key
nothing in the runtime consults would pass every check here and still cost a trial.
That is what the launcher's derived ``agent.skrl.yaml`` answers, and it needs a run.
And the space is checked against the config, not against the trainer's defaults, so a
parameter missing from the config is reported as such instead of being created.

The layer rule holds: this module takes already-parsed mappings, so nothing here
imports the product package.

An assignment is *nested*, keyed the way the config holds it -- ``{"skrl":
{"agent": {"mini_batches": 24}}}`` -- which is the form
:meth:`~hyperparameter_space.SearchSpaceSchema.default_assignment` returns and the
form a sampler draws.  A flat ``{"skrl.agent.mini_batches": 24}`` is refused, and
not merely by choice: the space walks the tree with the dotted name, so a flat
mapping is a set of keys its validation never looks at.  A layer that took the
flat form would validate nothing while appearing to validate.
"""
from __future__ import annotations

import copy
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, computed_field, model_validator

from .hyperparameter_space import SearchSpaceSchema, get_parameter, json_form, set_parameter
from .research_ledger import content_hash


INJECTION_SCHEMA_VERSION = "rl-agent.config-injection/v1"


def load_config_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a training config as a mapping.

    YAML, because that is what the launcher reads and therefore what this module
    writes; a JSON file would parse too, since JSON is a subset of YAML.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def _leaves(tree: Any, prefix: str = "") -> dict[str, Any]:
    """Every scalar in a config tree, keyed by dotted path.

    List positions are written ``[i]``, so each path names exactly one leaf.  An
    empty mapping or list is itself a leaf: dropping it would make "the section was
    emptied" and "the section was removed" look the same.
    """
    if isinstance(tree, Mapping):
        if not tree:
            return {prefix: tree}
        found: dict[str, Any] = {}
        for key, value in tree.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            found.update(_leaves(value, child))
        return found
    if isinstance(tree, list):
        if not tree:
            return {prefix: tree}
        found = {}
        for index, item in enumerate(tree):
            found.update(_leaves(item, f"{prefix}[{index}]"))
        return found
    return {prefix: tree}


def _moved_paths(source: Mapping[str, Any], injected: Mapping[str, Any]) -> tuple[str, ...]:
    """Paths whose value differs between two config trees, sorted.

    A path present on one side only counts as moved: a key that appeared or
    disappeared is a change a training run would notice just as much as a new value.

    Both trees have already been through :func:`content_hash`, which refuses a value
    with no JSON form, so comparing forms here cannot silently equate two values
    that only look alike through ``repr``.
    """
    before, after = _leaves(source), _leaves(injected)
    moved: list[str] = []
    for path in sorted(set(before) | set(after)):
        if path not in before or path not in after:
            moved.append(path)
        elif json_form(before[path]) != json_form(after[path]):
            moved.append(path)
    return tuple(moved)


def _under(path: str, name: str) -> bool:
    """Whether a leaf path lies at or inside a parameter's dotted path."""
    return path == name or path.startswith(name + ".") or path.startswith(name + "[")


def _assignment_names(
    space: SearchSpaceSchema, assignment: Mapping[str, Any]
) -> tuple[str, ...]:
    """The declared parameters an assignment fills, in the space's own order.

    Derived from the space rather than read off the assignment's keys, so the
    names are the space's vocabulary and a report cannot cite a name nothing
    declared.  The same walk also refuses the flat form, which fills nothing: it
    is caught here rather than left to produce an empty assignment, because a
    caller who wrote a flat mapping needs to hear about the shape, not the count.
    """
    names = tuple(name for name in space.names() if get_parameter(assignment, name)[0])
    if names:
        return names
    if not assignment:
        raise ValueError("an injection must set at least one parameter")
    raise ValueError(
        "the assignment fills no parameter the space declares, so nothing would be "
        "validated or injected. An assignment is nested the way a config holds it, "
        "so 'skrl.agent.mini_batches' is written as skrl: {agent: {mini_batches: ...}}; "
        "a flat name is a key no config loader reads"
    )


def _stray_leaves(
    space: SearchSpaceSchema, assignment: Mapping[str, Any]
) -> tuple[str, ...]:
    """Assignment leaves that lie under no declared parameter.

    ``validate_assignment`` walks the specs, so a key the space never declared
    would pass it unnoticed and be written into a config no search recorded and no
    domain constrained.
    """
    declared = space.names()
    return tuple(
        path
        for path in sorted(_leaves(assignment))
        if not any(_under(path, name) for name in declared)
    )


class InjectedValue(BaseModel):
    """One parameter's value, before and after the injection.

    ``previous_present`` separates an absent key from one that held ``None``: a
    legal categorical choice may be ``None``, and a record that cannot tell those
    apart cannot say whether it created a key.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    value: Any
    previous: Any = None
    previous_present: bool = False

    @computed_field
    @property
    def changed(self) -> bool:
        """Whether the injection gave this parameter a value it did not already hold.

        Derived rather than stored.  A record that carries both the two values and a
        claim about whether they differ can contradict itself, and a caller reading a
        report has no way to tell which half to believe.
        """
        if not self.previous_present:
            return True
        return json_form(self.previous) != json_form(self.value)


class InjectionReport(BaseModel):
    """What an injection did, in enough detail to audit or replay it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = INJECTION_SCHEMA_VERSION
    space_fingerprint: str
    source_fingerprint: str
    injected_fingerprint: str
    values: tuple[InjectedValue, ...]
    moved_paths: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_report(self) -> "InjectionReport":
        for label, digest in (
            ("space_fingerprint", self.space_fingerprint),
            ("source_fingerprint", self.source_fingerprint),
            ("injected_fingerprint", self.injected_fingerprint),
        ):
            if not digest.strip():
                raise ValueError(f"{label} must not be blank")
        if not self.values:
            raise ValueError("an injection report must name at least one parameter")
        names = [item.name for item in self.values]
        if len(set(names)) != len(names):
            raise ValueError(f"an injection report repeats a parameter: {names}")
        if list(self.moved_paths) != sorted(set(self.moved_paths)):
            raise ValueError("moved_paths must be sorted and unique")
        injected = tuple(names)
        stray = [path for path in self.moved_paths if not any(_under(path, name) for name in injected)]
        if stray:
            # The whole point of the diff: a value that moved outside the injected
            # parameters is a config this trial did not intend to run.
            raise ValueError(f"the injection moved config outside its parameters: {stray}")
        return self

    def changed(self) -> tuple[InjectedValue, ...]:
        """Only the parameters whose value the injection actually changed."""
        return tuple(item for item in self.values if item.changed)

    def describe(self) -> str:
        """One line naming what was injected, for a trial log."""
        changed = self.changed()
        unchanged = len(self.values) - len(changed)
        summary = ", ".join(f"{item.name}={item.value!r}" for item in changed)
        if not changed:
            return f"injected {len(self.values)} parameters, none of them new values"
        tail = f"; {unchanged} already held their injected value" if unchanged else ""
        return f"injected {len(changed)} of {len(self.values)} parameters: {summary}{tail}"

    def fingerprint(self) -> str:
        """Content-addressed identity of this injection, for a trial record."""
        return content_hash(self.model_dump(mode="json"))


@dataclass(frozen=True)
class Injection:
    """An injected config tree together with the report describing it."""

    config: dict[str, Any]
    report: InjectionReport


def inject_assignment(
    source: Mapping[str, Any],
    space: SearchSpaceSchema,
    assignment: Mapping[str, Any],
) -> Injection:
    """Build a config tree that carries ``assignment``, or raise saying why not.

    The injected tree is checked in full against the space, not only at the paths
    that were injected: a source config holding a value outside its own declared
    domain would make every trial from it unreproducible, and the moment to find
    that out is once, before the first trial, rather than per trial afterwards.

    ``assignment`` is nested, in the shape :meth:`_assignment_names` documents.  A
    parameter the assignment leaves out is left at the source config's value, which
    is what makes a partial injection meaningful -- and why the activation check
    below reads the config rather than the assignment.
    """
    if not assignment:
        raise ValueError("an injection must set at least one parameter")
    # Before the names are derived: an assignment naming a real config key that is
    # not a search parameter fills no declared parameter either, and "not declared"
    # names the offending key while "fills no parameter" only counts them.
    stray = _stray_leaves(space, assignment)
    if stray:
        raise ValueError(f"the assignment sets parameters not declared in the search space: {stray}")
    names = _assignment_names(space, assignment)
    missing = [name for name in space.names() if not get_parameter(source, name)[0]]
    if missing:
        raise ValueError(f"the search space declares parameters this config does not have: {missing}")
    try:
        source_fingerprint = content_hash(source)
    except TypeError as error:
        raise ValueError(f"the source config holds a value with no JSON form: {error}") from error

    # Checked before the write as well as after, because this is the error a caller
    # can act on: it names the parameter and the domain, not the file position.
    space.validate_assignment(assignment, require_complete=False)

    config = copy.deepcopy(dict(source))
    values: list[InjectedValue] = []
    for name in names:
        _, value = get_parameter(assignment, name)
        found, previous = get_parameter(config, name)
        set_parameter(config, name, value)
        values.append(
            InjectedValue(
                name=name,
                value=value,
                previous=previous if found else None,
                previous_present=found,
            )
        )

    inactive = sorted(name for name in names if not space.is_active(name, config))
    if inactive:
        # The guard is read from the config, not from the bare assignment: a partial
        # injection leaves the guard at whatever the source config says, and that
        # value decides whether the trainer will read the injected key at all.
        raise ValueError(
            f"these parameters are inactive in the injected config, so the trainer would "
            f"never read them: {inactive}"
        )
    space.validate_assignment(config, require_complete=False)

    return Injection(
        config=config,
        report=InjectionReport(
            space_fingerprint=space.fingerprint(),
            source_fingerprint=source_fingerprint,
            injected_fingerprint=content_hash(config),
            values=tuple(values),
            moved_paths=_moved_paths(source, config),
        ),
    )


def verify_injection(
    source: Mapping[str, Any],
    injected: Mapping[str, Any],
    space: SearchSpaceSchema,
    assignment: Mapping[str, Any],
) -> InjectionReport:
    """Check ``injected`` against ``source`` and ``assignment``, and report it.

    This is the reading half of :func:`inject_assignment`, applied to a config that
    may have come back from a file rather than from memory.  It re-derives the same
    report, so a caller compares the two reports to learn whether the file still
    says what was written.
    """
    if not assignment:
        raise ValueError("a verification must name at least one parameter")
    stray = _stray_leaves(space, assignment)
    if stray:
        raise ValueError(f"the assignment sets parameters not declared in the search space: {stray}")
    names = _assignment_names(space, assignment)
    try:
        source_fingerprint = content_hash(source)
        injected_fingerprint = content_hash(injected)
    except TypeError as error:
        raise ValueError(f"a config holds a value with no JSON form: {error}") from error

    values: list[InjectedValue] = []
    for name in names:
        _, wanted = get_parameter(assignment, name)
        found, actual = get_parameter(injected, name)
        if not found:
            raise ValueError(f"{name}: the injected config does not have this key")
        if json_form(actual) != json_form(wanted):
            raise ValueError(
                f"{name}: the config holds {actual!r} where {wanted!r} was injected"
            )
        before_found, previous = get_parameter(source, name)
        values.append(
            InjectedValue(
                name=name,
                value=wanted,
                previous=previous if before_found else None,
                previous_present=before_found,
            )
        )

    return InjectionReport(
        space_fingerprint=space.fingerprint(),
        source_fingerprint=source_fingerprint,
        injected_fingerprint=injected_fingerprint,
        values=tuple(values),
        moved_paths=_moved_paths(source, injected),
    )


def write_injection(
    injection: Injection,
    target: str | os.PathLike[str],
    *,
    source_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Write the injected config to ``target``, atomically, and return the path.

    The parent directory must exist.  Creating it would turn a mistyped trial
    directory into a stray tree somewhere else, and the caller is the one that knows
    where a run's files belong.
    """
    destination = Path(target)
    if source_path is not None and destination.resolve() == Path(source_path).resolve():
        raise ValueError(
            f"refusing to write the injected config over its own source: {destination}"
        )
    parent = destination.parent
    if not parent.is_dir():
        raise ValueError(f"the target directory does not exist: {parent}")
    try:
        actual = content_hash(injection.config)
    except TypeError as error:
        raise ValueError(f"the config to write holds a value with no JSON form: {error}") from error
    if actual != injection.report.injected_fingerprint:
        # An ``Injection`` is a plain pair, so a caller can assemble one whose report
        # describes a different tree.  Writing that would put a config on disk that no
        # record of it can explain.
        raise ValueError(
            f"the injection report describes a different config: report cites "
            f"{injection.report.injected_fingerprint}, config is {actual}"
        )
    text = yaml.safe_dump(dict(injection.config), sort_keys=False, allow_unicode=True)
    # Staged then renamed, so a reader never sees half a config: the launcher may be
    # handed this path by another thread, and a truncated YAML file parses as a
    # different training configuration rather than as an error.
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f"{destination.name}.",
        suffix=".tmp",
        dir=parent,
        delete=False,
    )
    staged = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, destination)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return destination


def inject_into_file(
    source_path: str | os.PathLike[str],
    target: str | os.PathLike[str],
    space: SearchSpaceSchema,
    assignment: Mapping[str, Any],
) -> InjectionReport:
    """Inject, write, read back, and verify -- the whole path a trial takes.

    Verification is part of this call rather than left to the caller, because the
    failure it catches (a config file that does not carry the value the trial claims)
    is invisible until a result is compared against a run that never happened.
    """
    source = load_config_file(source_path)
    injection = inject_assignment(source, space, assignment)
    written = write_injection(injection, target, source_path=source_path)
    report = verify_injection(source, load_config_file(written), space, assignment)
    if report.fingerprint() != injection.report.fingerprint():
        raise ValueError(
            f"the written config does not carry the injection: {written} "
            f"({report.describe()} against {injection.report.describe()})"
        )
    return report


__all__ = [
    "INJECTION_SCHEMA_VERSION",
    "InjectedValue",
    "Injection",
    "InjectionReport",
    "inject_assignment",
    "inject_into_file",
    "load_config_file",
    "verify_injection",
    "write_injection",
]
