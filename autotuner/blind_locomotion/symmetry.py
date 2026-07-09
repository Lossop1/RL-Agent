"""Shim: `taili_amp_env` imports `from . import symmetry`, but the implementation
lives in `_symmetry_local`. Re-export its public API so sym_augment wires up.
(0708) Without this module the `from . import symmetry` in taili_amp_env crashes
env construction whenever sym_augment is enabled — the reason mirror-augmentation
had been left off. Deploy alongside _symmetry_local.py."""
from ._symmetry_local import (  # noqa: F401
    set_active,
    set_mirrors,
    patch_memory_sample_all,
    build_obs_mirror,
    build_action_mirror,
    mirror_obs,
    mirror_action,
)
