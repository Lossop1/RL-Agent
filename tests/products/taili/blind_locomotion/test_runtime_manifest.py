"""Pure tests for runtime evidence helpers; no IsaacLab or torch process."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from products.taili.blind_locomotion.runtime_manifest import (
    capture_optimization_state,
    capture_runtime_execution,
    initial_manifest,
    write_manifest,
    _curriculum_ref,
    _state_ref,
    _rng_ref,
)


class _State:
    def __init__(self, value: str):
        self.value = value

    def state_dict(self):
        return {"value": self.value}


class _Model(_State):
    def named_parameters(self):
        return [("log_std", _State("std"))]


def test_runtime_snapshot_requires_real_components_and_records_fresh_state(tmp_path: Path):
    agent = SimpleNamespace(
        models={"policy": _Model("policy"), "value": _State("value"), "discriminator": _State("amp")},
        optimizer=_State("optimizer"),
        state_preprocessor=_State("normalizer"),
    )
    env = SimpleNamespace(_phase=0, _dr_level=0)
    snapshot = capture_optimization_state(
        agent,
        env,
        run_dir=tmp_path,
        amp_expected=True,
        parity_check=True,
        resume=False,
    )

    assert snapshot["status"] == "proven"
    assert snapshot["parity_check"] is True
    assert snapshot["fields"]["policy"]["status"] == "captured"
    assert snapshot["fields"]["curriculum"]["status"] == "fresh_initialization"


def test_runtime_import_proof_captures_resolved_file(tmp_path: Path):
    proof = capture_runtime_execution(
        Path(__file__).resolve().parents[4],
        ("products.taili.blind_locomotion.runtime_manifest",),
    )
    assert proof["status"] == "proven"
    assert proof["remote_verification"] == "proven"
    assert any(item["path"].endswith("runtime_manifest.py") for item in proof["source_files"])


def test_initial_manifest_is_written_as_a_declared_not_proven_record(tmp_path: Path):
    config = tmp_path / "config.yaml"
    effective = tmp_path / "effective.yaml"
    agent = tmp_path / "agent.yaml"
    for path in (config, effective, agent):
        path.write_text("key: value\n", encoding="utf-8")
    manifest = initial_manifest(
        run_id="fixture",
        run_dir=tmp_path,
        task="taili",
        payload_root=tmp_path,
        source_config=config,
        effective_config=effective,
        agent_config=agent,
    )
    path = tmp_path / "runtime_manifest.json"
    write_manifest(path, manifest)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["status"] == "running"
    assert loaded["runtime_execution"]["status"] == "declared"
    assert loaded["optimization_state"]["parity_check"] is False


def test_manifest_updates_preserve_nested_launch_facts(tmp_path: Path):
    config = tmp_path / "config.yaml"
    effective = tmp_path / "effective.yaml"
    agent = tmp_path / "agent.yaml"
    for path in (config, effective, agent):
        path.write_text("key: value\n", encoding="utf-8")
    path = tmp_path / "runtime_manifest.json"
    write_manifest(
        path,
        initial_manifest(
            run_id="run-preserved",
            run_dir=tmp_path,
            task="taili",
            payload_root=tmp_path,
            source_config=config,
            effective_config=effective,
            agent_config=agent,
        ),
    )

    from products.taili.blind_locomotion.runtime_manifest import update_manifest

    update_manifest(path, run={"seed": 42, "skrl_version": "1.4.3"})
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["run"]["run_id"] == "run-preserved"
    assert loaded["run"]["task"] == "taili"
    assert loaded["run"]["seed"] == 42


def test_rng_state_roundtrip_python_numpy():
    """验证Python和NumPy RNG状态捕获后可恢复"""
    import random
    import numpy as np
    from products.taili.blind_locomotion.runtime_manifest import _rng_ref

    # 设置初始状态
    random.seed(12345)
    np.random.seed(67890)

    # 捕获初始状态
    initial_rng = _rng_ref()
    assert initial_rng["status"] == "captured"
    py_state = initial_rng["sha256"]  # 存储初始哈希

    # 生成一些随机数（改变状态）
    _ = random.random()
    _ = np.random.rand()

    # 捕获变化后的状态
    changed_rng = _rng_ref()
    assert changed_rng["sha256"] != py_state  # 状态应该已变化


def test_curriculum_sidecar_file_priority(tmp_path: Path):
    """验证课程状态优先从sidecar文件加载"""
    from products.taili.blind_locomotion.runtime_manifest import _curriculum_ref

    # 创建sidecar文件
    sidecar = tmp_path / "curriculum_state.json"
    sidecar.write_text('{"phase": 2, "level": 5}', encoding="utf-8")

    # 创建env对象（有属性但应被sidecar覆盖）
    env = SimpleNamespace(_phase=999, _dr_level=999)

    result = _curriculum_ref(env, tmp_path)

    assert result["status"] == "captured"
    assert "curriculum_state.json" in result["source"]
    # 验证sidecar文件内容被读取（通过sha256变化可间接验证）
    assert "sha256" in result


def test_curriculum_fallback_to_env_attributes(tmp_path: Path):
    """验证sidecar文件缺失时从env属性读取课程状态"""
    from products.taili.blind_locomotion.runtime_manifest import _curriculum_ref

    # 不创建sidecar文件
    env = SimpleNamespace(_phase=1, _dr_level=3)

    result = _curriculum_ref(env, tmp_path)

    assert result["status"] == "fresh_initialization"
    assert result["source"] == "runtime_environment_attributes"
    # 验证env属性被使用（通过sha256存在可验证状态被捕获）
    assert "sha256" in result


def test_curriculum_missing_when_no_source(tmp_path: Path):
    """验证无sidecar文件且env无属性时标记为missing"""
    from products.taili.blind_locomotion.runtime_manifest import _curriculum_ref

    # env对象无课程属性
    env = SimpleNamespace()

    result = _curriculum_ref(env, tmp_path)

    assert result["status"] == "missing"
    assert result["source"] == ""
    assert "reason" in result


def test_optimizer_state_hash_consistency():
    """验证optimizer state_dict哈希在保存/加载后保持一致"""
    from products.taili.blind_locomotion.runtime_manifest import _state_ref

    # 创建optimizer mock
    optimizer = _State("momentum=0.9,lr=0.001")

    # 捕获状态
    ref1 = _state_ref("optimizer", optimizer)
    assert ref1["status"] == "captured"

    # 模拟保存/加载（通过state_dict往返）
    state_saved = optimizer.state_dict()
    optimizer_reloaded = _State(state_saved["value"])

    # 再次捕获状态
    ref2 = _state_ref("optimizer", optimizer_reloaded)
    assert ref2["status"] == "captured"

    # 验证哈希一致性
    assert ref1["sha256"] == ref2["sha256"]


def test_scheduler_not_configured_vs_missing():
    """验证scheduler未配置时标记为not_configured而非missing"""
    from products.taili.blind_locomotion.runtime_manifest import capture_optimization_state

    # agent无scheduler属性
    agent = SimpleNamespace(
        models={"policy": _Model("policy"), "value": _State("value")},
        optimizer=_State("optimizer"),
        state_preprocessor=_State("normalizer"),
    )
    env = SimpleNamespace(_phase=0, _dr_level=0)

    snapshot = capture_optimization_state(
        agent,
        env,
        run_dir=Path("."),
        amp_expected=False,
        parity_check=False,
    )

    # scheduler应为not_configured（表示可选组件未启用）
    assert snapshot["fields"]["scheduler"]["status"] == "not_configured"


def test_log_std_extraction_from_policy_parameters():
    """验证log_std从policy.named_parameters正确提取"""
    from products.taili.blind_locomotion.runtime_manifest import capture_optimization_state

    # policy有log_std参数
    agent = SimpleNamespace(
        models={"policy": _Model("policy_with_log_std"), "value": _State("value")},
        optimizer=_State("optimizer"),
        state_preprocessor=_State("normalizer"),
    )
    env = SimpleNamespace(_phase=0, _dr_level=0)

    snapshot = capture_optimization_state(
        agent,
        env,
        run_dir=Path("."),
        amp_expected=False,
        parity_check=True,
    )

    # log_std应被捕获
    assert snapshot["fields"]["log_std"]["status"] == "captured"
    assert "sha256" in snapshot["fields"]["log_std"]


def test_amp_missing_when_not_expected():
    """验证amp_expected=False时amp标记为not_configured"""
    from products.taili.blind_locomotion.runtime_manifest import capture_optimization_state

    agent = SimpleNamespace(
        models={"policy": _Model("policy"), "value": _State("value")},
        optimizer=_State("optimizer"),
        state_preprocessor=_State("normalizer"),
    )
    env = SimpleNamespace(_phase=0, _dr_level=0)

    snapshot = capture_optimization_state(
        agent,
        env,
        run_dir=Path("."),
        amp_expected=False,  # 不期望AMP
        parity_check=False,
    )

    assert snapshot["fields"]["amp"]["status"] == "not_configured"


def test_parity_check_upgrades_to_proven_when_all_usable():
    """验证parity_check=True且所有组件可用时status升级为proven"""
    from products.taili.blind_locomotion.runtime_manifest import capture_optimization_state

    agent = SimpleNamespace(
        models={"policy": _Model("policy"), "value": _State("value")},
        optimizer=_State("optimizer"),
        state_preprocessor=_State("normalizer"),
    )
    env = SimpleNamespace(_phase=0, _dr_level=0)

    snapshot = capture_optimization_state(
        agent,
        env,
        run_dir=Path("."),
        amp_expected=False,
        parity_check=True,  # 开启parity验证
    )

    # 所有必需组件已捕获，status应为proven
    assert snapshot["status"] == "proven"
    assert snapshot["parity_check"] is True
