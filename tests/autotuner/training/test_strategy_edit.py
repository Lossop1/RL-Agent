"""The autonomous-tuning 'tune' primitive: allowlisted, bounded, comment-preserving strategy edits
with a reversible rollback stack. Runs on a temp copy — never touches the real config."""
import shutil
from pathlib import Path

import pytest

from autotuner.training import strategy_edit as SE


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    p = tmp_path / "cfg.yaml"
    shutil.copy(SE._STRATEGY_YAML, p)
    monkeypatch.setattr(SE, "_ROLLBACK_STACK", tmp_path / "rb.jsonl")
    return p


def test_rejects_non_allowlisted_and_out_of_bounds():
    assert SE.validate_changes({"os_system": 1})            # not allowlisted
    assert SE.validate_changes({"w_stance_slip": 999})      # out of bounds
    assert SE.validate_changes({"w_stance_slip": "x"})      # non-numeric
    assert SE.validate_changes({}) == []                    # empty is valid


def test_apply_preserves_comment_and_pushes_rollback(cfg):
    before_line = next(l for l in cfg.read_text().splitlines() if l.strip().startswith("w_stance_slip:"))
    r = SE.apply_weight_changes({"w_stance_slip": 0.42, "reward.w_clearance_over": 2.0},
                                yaml_path=cfg, stamp="t1")
    assert r["ok"] and len(r["applied"]) == 2
    line = next(l for l in cfg.read_text().splitlines() if l.strip().startswith("w_stance_slip:"))
    assert "0.42" in line
    if "#" in before_line:
        assert "#" in line                                  # value changed, existing comment kept
    assert len(SE.rollback_stack()) == 1


def test_rollback_restores_field_for_field(cfg):
    before_line = next(l for l in cfg.read_text().splitlines() if l.strip().startswith("w_stance_slip:"))
    before_value = before_line.split(":")[1].split("#")[0].strip()
    SE.apply_weight_changes({"w_stance_slip": 0.42}, yaml_path=cfg, stamp="t1")
    rb = SE.rollback_last(yaml_path=cfg)
    assert rb["ok"]
    line = next(l for l in cfg.read_text().splitlines() if l.strip().startswith("w_stance_slip:"))
    assert line.split(":")[1].split("#")[0].strip() == before_value   # original restored
    assert SE.rollback_stack() == []                            # stack popped


def test_missing_key_is_atomic_no_write(cfg):
    before = cfg.read_text()
    r = SE.apply_weight_changes({"w_stance_slip": 0.4, "w_not_a_real_key": 0.4}, yaml_path=cfg)
    # w_not_a_real_key isn't allowlisted → validation fails → nothing written
    assert not r["ok"] and cfg.read_text() == before
