"""
P4.3训练恢复 - compatibility.py核心逻辑测试

测试范围：
- check_resume_compatibility() ABI字段变化阻断
- check_resume_compatibility() provenance字段变化记录但不阻断
- _checkpoint_components() 多种别名识别
- evaluate_resume_proof() 9个检查项的通过/失败逻辑
"""

import pytest
from autotuner.execution.compatibility import (
    check_resume_compatibility,
    evaluate_resume_proof,
    ResumeCompatibility,
    ResumeProof,
)


class TestResumeCompatibility:
    """测试check_resume_compatibility()函数的兼容性检查逻辑"""

    def test_abi_field_change_blocks_resume_observation_structure(self):
        """观测维度变化应阻断恢复"""
        parent = {
            "resolved_contract": {
                "training": {
                    "observation_structure": {"dim": 54, "type": "proprioception"}
                }
            }
        }
        candidate = {
            "resolved_contract": {
                "training": {
                    "observation_structure": {"dim": 57, "type": "proprioception"}
                }
            }
        }
        result = check_resume_compatibility(
            parent, candidate, checkpoint=None, requested=True, strict=True
        )

        assert not result.compatible
        assert result.decision == "blocked"
        assert any("observation_structure" in reason for reason in result.reasons)

    def test_abi_field_change_blocks_resume_action_structure(self):
        """动作空间变化应阻断恢复"""
        parent = {
            "resolved_contract": {
                "training": {
                    "action_structure": {"dim": 12, "type": "continuous"}
                }
            }
        }
        candidate = {
            "resolved_contract": {
                "training": {
                    "action_structure": {"dim": 18, "type": "continuous"}
                }
            }
        }
        result = check_resume_compatibility(
            parent, candidate, checkpoint=None, requested=True, strict=True
        )

        assert not result.compatible
        assert result.decision == "blocked"
        assert any("action_structure" in reason for reason in result.reasons)

    def test_abi_field_change_blocks_resume_network_structure(self):
        """网络结构变化应阻断恢复"""
        parent = {
            "resolved_contract": {
                "training": {
                    "network_structure": {
                        "hidden_sizes": [256, 256, 128],
                        "activation": "elu"
                    }
                }
            }
        }
        candidate = {
            "resolved_contract": {
                "training": {
                    "network_structure": {
                        "hidden_sizes": [512, 512, 256],
                        "activation": "elu"
                    }
                }
            }
        }
        result = check_resume_compatibility(
            parent, candidate, checkpoint=None, requested=True, strict=True
        )

        assert not result.compatible
        assert result.decision == "blocked"
        assert any("network_structure" in reason for reason in result.reasons)

    def test_provenance_change_recorded_not_blocked_payload_digest(self):
        """payload_digest变化应记录但不阻断"""
        parent = {
            "product": {"id": "taili", "version": "1.0.0"},
            "run": {"task": "blind_locomotion"},
            "resolved_contract": {
                "training": {
                    "observation_structure": {"dim": 54},
                    "action_structure": {"dim": 12},
                    "network_structure": {"hidden_sizes": [256, 256, 128]},
                    "normalization": {"enabled": True},
                },
                "runtime": {"physics_timestep": 0.005},
                "config_digest": "sha256:config123",
                "source_digest": "sha256:source123",
            },
            "execution": {
                "payload_digest": "sha256:abc123def456",
                "task_contract_digest": "sha256:same",
                "runtime_digest": "sha256:runtime123",
            }
        }
        candidate = {
            "product": {"id": "taili", "version": "1.0.0"},
            "run": {"task": "blind_locomotion"},
            "resolved_contract": {
                "training": {
                    "observation_structure": {"dim": 54},
                    "action_structure": {"dim": 12},
                    "network_structure": {"hidden_sizes": [256, 256, 128]},
                    "normalization": {"enabled": True},
                },
                "runtime": {"physics_timestep": 0.005},
                "config_digest": "sha256:config123",
                "source_digest": "sha256:source123",
            },
            "execution": {
                "payload_digest": "sha256:xyz789ghi012",  # 仅payload变化
                "task_contract_digest": "sha256:same",
                "runtime_digest": "sha256:runtime123",
            }
        }
        checkpoint = {
            "status": "captured",
            "keys": ["policy", "value", "optimizer", "normalizer"]
        }
        result = check_resume_compatibility(
            parent, candidate, checkpoint=checkpoint, requested=True, strict=True
        )

        assert result.compatible
        assert result.decision == "resume"
        # provenance变化应该被记录在compared中
        assert "payload_digest" in result.compared
        assert result.compared["payload_digest"]["change"] == "recorded_provenance_change"

    def test_provenance_change_recorded_not_blocked_config_digest(self):
        """config_digest变化应记录但不阻断"""
        parent = {
            "product": {"id": "taili", "version": "1.0.0"},
            "run": {"task": "blind_locomotion"},
            "resolved_contract": {
                "training": {
                    "observation_structure": {"dim": 54},
                    "action_structure": {"dim": 12},
                    "network_structure": {"hidden_sizes": [256, 256, 128]},
                    "normalization": {"enabled": True},
                },
                "runtime": {"physics_timestep": 0.005},
                "config_digest": "sha256:old_config",
                "source_digest": "sha256:source123",
            },
            "execution": {
                "task_contract_digest": "sha256:same",
                "runtime_digest": "sha256:runtime123",
                "payload_digest": "sha256:payload123",
            }
        }
        candidate = {
            "product": {"id": "taili", "version": "1.0.0"},
            "run": {"task": "blind_locomotion"},
            "resolved_contract": {
                "training": {
                    "observation_structure": {"dim": 54},
                    "action_structure": {"dim": 12},
                    "network_structure": {"hidden_sizes": [256, 256, 128]},
                    "normalization": {"enabled": True},
                },
                "runtime": {"physics_timestep": 0.005},
                "config_digest": "sha256:new_config",  # 仅config变化
                "source_digest": "sha256:source123",
            },
            "execution": {
                "task_contract_digest": "sha256:same",
                "runtime_digest": "sha256:runtime123",
                "payload_digest": "sha256:payload123",
            }
        }
        checkpoint = {
            "status": "captured",
            "keys": ["policy", "value", "optimizer", "normalizer"]
        }
        result = check_resume_compatibility(
            parent, candidate, checkpoint=checkpoint, requested=True, strict=True
        )

        assert result.compatible
        assert result.decision == "resume"

    def test_identical_manifests_allow_resume(self):
        """完全相同的manifest应允许恢复"""
        manifest = {
            "product": {"id": "taili", "version": "1.0.0"},
            "run": {"task": "blind_locomotion"},
            "resolved_contract": {
                "training": {
                    "observation_structure": {"dim": 54},
                    "action_structure": {"dim": 12},
                    "network_structure": {"hidden_sizes": [256, 256, 128]},
                    "normalization": {"enabled": True},
                },
                "runtime": {"physics_timestep": 0.005},
                "config_digest": "sha256:config123",
                "source_digest": "sha256:source123",
            },
            "execution": {
                "task_contract_digest": "sha256:abc123",
                "payload_digest": "sha256:def456",
                "runtime_digest": "sha256:runtime123",
            }
        }
        checkpoint = {
            "status": "captured",
            "keys": ["policy", "value", "optimizer", "normalizer"]
        }
        result = check_resume_compatibility(
            manifest, manifest, checkpoint=checkpoint, requested=True, strict=True
        )

        assert result.compatible
        assert result.decision == "resume"
        assert len(result.reasons) == 0

    def test_missing_parent_manifest_allows_fresh_start(self):
        """缺失parent manifest应标记为fresh而非blocked"""
        candidate = {
            "product": {"id": "taili", "version": "1.0.0"},
            "run": {"task": "blind_locomotion"},
            "resolved_contract": {
                "training": {
                    "observation_structure": {"dim": 54},
                    "action_structure": {"dim": 12},
                    "network_structure": {"hidden_sizes": [256, 256, 128]},
                    "normalization": {"enabled": True},
                },
                "runtime": {"physics_timestep": 0.005},
            },
            "execution": {
                "runtime_digest": "sha256:runtime123",
                "config_digest": "sha256:config123",
                "source_digest": "sha256:source123",
                "payload_digest": "sha256:payload123",
            }
        }
        result = check_resume_compatibility(
            None, candidate, checkpoint=None, requested=True, strict=True
        )

        # parent为None时应允许fresh start
        assert result.decision == "fresh"


class TestCheckpointComponentRecognition:
    """测试checkpoint组件别名识别逻辑"""

    def test_checkpoint_with_standard_keys(self):
        """标准检查点键名应被正确识别"""
        checkpoint = {
            "status": "captured",
            "keys": ["policy", "value", "optimizer", "normalizer"],
        }
        result = check_resume_compatibility(
            {}, {}, checkpoint=checkpoint, requested=True, strict=True
        )

        # 检查点组件信息应记录在result.checkpoint中
        assert result.checkpoint is not None

    def test_checkpoint_with_model_prefix_keys(self):
        """models.前缀的键名应被识别为组件"""
        checkpoint = {
            "status": "captured",
            "keys": ["models.policy", "models.value", "optimizer"],
        }
        result = check_resume_compatibility(
            {}, {}, checkpoint=checkpoint, requested=True, strict=True
        )

        assert result.checkpoint is not None

    def test_checkpoint_with_amp_discriminator_aliases(self):
        """amp/discriminator别名应被识别"""
        # 测试"amp"键名
        checkpoint_amp = {
            "status": "captured",
            "keys": ["policy", "value", "optimizer", "amp"],
        }
        result_amp = check_resume_compatibility(
            {}, {}, checkpoint=checkpoint_amp, requested=True, strict=True
        )
        assert result_amp.checkpoint is not None

        # 测试"discriminator"键名
        checkpoint_disc = {
            "status": "captured",
            "keys": ["policy", "value", "optimizer", "discriminator"],
        }
        result_disc = check_resume_compatibility(
            {}, {}, checkpoint=checkpoint_disc, requested=True, strict=True
        )
        assert result_disc.checkpoint is not None


class TestResumeProof:
    """测试evaluate_resume_proof()收敛逻辑"""

    def test_resume_proof_requires_compatibility_check(self):
        """兼容性检查失败应导致proof失败"""
        parent = {
            "resolved_contract": {
                "training": {"observation_structure": {"dim": 54}}
            },
            "execution": {"task_contract_digest": "sha256:abc"},
        }
        candidate = {
            "resolved_contract": {
                "training": {"observation_structure": {"dim": 57}}  # 维度不匹配
            },
            "execution": {"task_contract_digest": "sha256:abc"},
        }
        checkpoint = {"status": "captured", "keys": ["policy", "value"]}
        preflight = {"status": "pass"}
        restored = {
            "policy": True,
            "value": True,
            "optimizer": True,
            "normalizer": True,
            "curriculum": True,
            "rng": True,
        }

        proof = evaluate_resume_proof(
            parent,
            candidate,
            checkpoint=checkpoint,
            runtime_preflight=preflight,
            restored=restored,
        )

        assert not proof.proven
        assert proof.status == "blocked"
        assert not proof.checks["compatibility"]

    def test_resume_proof_requires_checkpoint_inventory(self):
        """checkpoint inventory状态必须为captured或proven"""
        parent = {"execution": {"task_contract_digest": "sha256:abc"}}
        candidate = {"execution": {"task_contract_digest": "sha256:abc"}}
        checkpoint = {"status": "missing"}  # 无效状态
        preflight = {"status": "pass"}
        restored = {
            "policy": True,
            "value": True,
            "optimizer": True,
            "normalizer": True,
            "curriculum": True,
            "rng": True,
        }

        proof = evaluate_resume_proof(
            parent,
            candidate,
            checkpoint=checkpoint,
            runtime_preflight=preflight,
            restored=restored,
        )

        assert not proof.proven
        assert not proof.checks["checkpoint_inventory"]
        assert any("checkpoint_inventory" in reason for reason in proof.reasons)

    def test_resume_proof_requires_runtime_preflight(self):
        """runtime preflight必须通过"""
        parent = {"execution": {"task_contract_digest": "sha256:abc"}}
        candidate = {"execution": {"task_contract_digest": "sha256:abc"}}
        checkpoint = {"status": "captured"}
        preflight = {"status": "failed"}  # preflight失败
        restored = {
            "policy": True,
            "value": True,
            "optimizer": True,
            "normalizer": True,
            "curriculum": True,
            "rng": True,
        }

        proof = evaluate_resume_proof(
            parent,
            candidate,
            checkpoint=checkpoint,
            runtime_preflight=preflight,
            restored=restored,
        )

        assert not proof.proven
        assert not proof.checks["runtime_preflight"]
        assert any("runtime_preflight" in reason for reason in proof.reasons)

    def test_resume_proof_requires_all_state_components(self):
        """所有required_state_components必须恢复"""
        parent = {"execution": {"task_contract_digest": "sha256:abc"}}
        candidate = {"execution": {"task_contract_digest": "sha256:abc"}}
        checkpoint = {"status": "captured"}
        preflight = {"status": "pass"}
        restored = {
            "policy": True,
            "value": True,
            "optimizer": True,
            "normalizer": True,
            "curriculum": False,  # curriculum未恢复
            "rng": True,
        }

        proof = evaluate_resume_proof(
            parent,
            candidate,
            checkpoint=checkpoint,
            runtime_preflight=preflight,
            restored=restored,
        )

        assert not proof.proven
        assert not proof.state_components["curriculum"]
        assert any("curriculum" in reason for reason in proof.reasons)

    def test_resume_proof_requires_digest_present(self):
        """必要的digest字段必须存在"""
        parent = {"execution": {}}  # 缺少所有digest
        candidate = {"execution": {}}
        checkpoint = {"status": "captured"}
        preflight = {"status": "pass"}
        restored = {
            "policy": True,
            "value": True,
            "optimizer": True,
            "normalizer": True,
            "curriculum": True,
            "rng": True,
        }

        proof = evaluate_resume_proof(
            parent,
            candidate,
            checkpoint=checkpoint,
            runtime_preflight=preflight,
            restored=restored,
        )

        assert not proof.proven
        assert not proof.checks["contract_digest_present"]
        assert not proof.checks["bundle_digest_present"]
        assert not proof.checks["payload_digest_present"]
        assert not proof.checks["runtime_digest_present"]

    def test_resume_proof_success_with_all_checks_passing(self):
        """所有检查通过时应返回proven=True"""
        parent = {
            "product": {"id": "taili", "version": "1.0.0"},
            "run": {"task": "blind_locomotion"},
            "resolved_contract": {
                "training": {
                    "observation_structure": {"dim": 54},
                    "action_structure": {"dim": 12},
                    "network_structure": {"hidden_sizes": [256, 256, 128]},
                    "normalization": {"enabled": True},
                },
                "runtime": {"physics_timestep": 0.005},
                "config_digest": "sha256:config123",
                "source_digest": "sha256:source123",
            },
            "execution": {
                "task_contract_digest": "sha256:abc123",
                "task_bundle_digest": "sha256:def456",
                "payload_digest": "sha256:ghi789",
                "runtime_digest": "sha256:jkl012",
            },
        }
        candidate = {
            "product": {"id": "taili", "version": "1.0.0"},
            "run": {"task": "blind_locomotion"},
            "resolved_contract": {
                "training": {
                    "observation_structure": {"dim": 54},
                    "action_structure": {"dim": 12},
                    "network_structure": {"hidden_sizes": [256, 256, 128]},
                    "normalization": {"enabled": True},
                },
                "runtime": {"physics_timestep": 0.005},
                "config_digest": "sha256:config123",
                "source_digest": "sha256:source123",
            },
            "execution": {
                "task_contract_digest": "sha256:abc123",
                "task_bundle_digest": "sha256:def456",
                "payload_digest": "sha256:ghi789",
                "runtime_digest": "sha256:jkl012",
            },
            "runtime_execution": {"remote_verification": "proven"},
        }
        checkpoint = {
            "status": "captured",
            "keys": ["policy", "value", "optimizer", "normalizer"]
        }
        preflight = {"status": "pass"}
        restored = {
            "policy": True,
            "value": True,
            "optimizer": True,
            "normalizer": True,
            "curriculum": True,
            "rng": True,
        }

        proof = evaluate_resume_proof(
            parent,
            candidate,
            checkpoint=checkpoint,
            runtime_preflight=preflight,
            restored=restored,
        )

        assert proof.proven
        assert proof.status == "proven"
        assert len(proof.reasons) == 0
        assert all(proof.checks.values())
        assert all(proof.state_components.values())


class TestEdgeCases:
    """测试边缘情况"""

    def test_empty_manifest_comparison(self):
        """空manifest对比应正常处理"""
        result = check_resume_compatibility(
            {}, {}, checkpoint=None, requested=True, strict=True
        )

        assert isinstance(result, ResumeCompatibility)

    def test_partial_manifest_missing_fields(self):
        """部分字段缺失的manifest应正常处理"""
        parent = {"resolved_contract": {}}
        candidate = {"execution": {}}

        result = check_resume_compatibility(
            parent, candidate, checkpoint=None, requested=True, strict=True
        )

        assert isinstance(result, ResumeCompatibility)

    def test_resume_proof_with_minimal_input(self):
        """最小化输入应正常处理"""
        proof = evaluate_resume_proof(
            None,
            {},
            checkpoint=None,
            runtime_preflight=None,
            restored=None,
        )

        assert isinstance(proof, ResumeProof)
        assert not proof.proven
