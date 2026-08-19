from __future__ import annotations

import pytest

from autotuner.product.plugins import (
    ProductPluginError,
    call_product_plugin,
    load_product_plugin,
    plugin_reference,
)


def test_product_plugin_is_resolved_by_role() -> None:
    contract = {"plugins": {"utility": {"length": "builtins:len"}}}

    assert plugin_reference(contract, "utility", "length") == "builtins:len"
    assert load_product_plugin(contract, "utility", "length") is len
    assert call_product_plugin(contract, "utility", "length", [1, 2, 3]) == 3


def test_missing_product_plugin_never_falls_back() -> None:
    with pytest.raises(ProductPluginError, match=r"acceptance\.aggregate"):
        plugin_reference({"plugins": {}}, "acceptance", "aggregate")
