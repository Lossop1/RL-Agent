"""Framework Library ② — the value core (SYSTEM_ARCHITECTURE §4②, §7).

Holds the composable/selectable/growable MECHANISM components (M1-M9 + the blind-TP redesign)
as a typed catalog: each component carries its adapt_kind (invariant / derive / regenerate /
scale, straight from §6/§7), the robot morphology it applies to, a provenance pointer to the
implementing code, and a version. Compositions name a set of components = one framework instance.

This is descriptive + validating, NOT a re-implementation: the mechanisms live in the current
Taili runtime/strategy modules (`blind_locomotion`, `taili_core`, adapter placeholders). The
catalog lets ① ConfigSet compose a framework and lets ③ Adapter know, per component, what to
derive/regenerate/scale vs leave invariant.
"""
from autotuner.framework_library.catalog import (  # noqa: F401
    FrameworkComponent,
    FrameworkComposition,
    CATALOG,
    COMPOSITIONS,
    get_component,
    get_composition,
    validate_composition,
    adapt_plan,
)
