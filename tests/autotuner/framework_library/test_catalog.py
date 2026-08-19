from __future__ import annotations

from collections.abc import Mapping

import pytest

from autotuner.framework_library import COMPOSITIONS, get_composition, get_compositions


def test_product_compositions_are_loaded_from_manifest() -> None:
    compositions = get_compositions("taili")

    assert set(compositions) == {"taili_amp_proven", "taili_blind_tp"}
    assert get_composition("taili_blind_tp", "taili").component_ids[:4] == (
        "B1",
        "B2",
        "B3",
        "B4",
    )


def test_legacy_compositions_view_is_dynamic_and_read_only() -> None:
    assert isinstance(COMPOSITIONS, Mapping)
    assert COMPOSITIONS["taili_amp_proven"].status == "validated"

    with pytest.raises(TypeError):
        COMPOSITIONS["new"] = object()  # type: ignore[index]
