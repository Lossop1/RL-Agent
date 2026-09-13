"""测试观测契约定义的完整性和一致性。

验证P5.1要求：
- 所有观测项有明确的物理单位
- 所有观测项有明确的数值范围
- 维度定义一致
- 组装顺序与实际实现匹配
"""
import pytest

from products.taili.core.taili_obs_spec import (
    ACTOR_OBS_SPACE,
    AMP_FRAME51_SPACE,
    BODY57_SPACE,
    TICK54_SPACE,
    validate_all_spaces,
)


class TestObservationSpecCompleteness:
    """测试观测契约定义的完整性。"""

    def test_all_spaces_have_valid_dimensions(self):
        """测试所有观测空间维度定义一致。"""
        results = validate_all_spaces()
        assert all(results.values()), f"维度不一致: {results}"

    def test_tick54_dimension(self):
        """测试tick54维度：12+12+12+12+3+3=54。"""
        assert TICK54_SPACE.validate_dimension()
        assert TICK54_SPACE.total_dimension == 54

    def test_body57_dimension(self):
        """测试body57维度：3+3+3+3+1+12+12+12+8=57。"""
        assert BODY57_SPACE.validate_dimension()
        assert BODY57_SPACE.total_dimension == 57

    def test_amp_frame51_dimension(self):
        """测试amp_frame51维度：1+3+3+4+12+12+12+12=51。"""
        assert AMP_FRAME51_SPACE.validate_dimension()
        assert AMP_FRAME51_SPACE.total_dimension == 51

    def test_actor_obs_dimension(self):
        """测试actor_obs维度：57+25*54=1407。"""
        assert ACTOR_OBS_SPACE.validate_dimension()
        assert ACTOR_OBS_SPACE.total_dimension == 1407

    def test_all_components_have_physical_units(self):
        """测试所有观测项有物理单位定义。"""
        spaces = [TICK54_SPACE, BODY57_SPACE, AMP_FRAME51_SPACE]
        for space in spaces:
            for comp in space.components:
                assert comp.physical_unit, f"{space.name}.{comp.name} 缺失物理单位"
                assert comp.physical_unit != "", f"{space.name}.{comp.name} 物理单位为空"

    def test_all_components_have_value_ranges(self):
        """测试所有观测项有数值范围定义。"""
        spaces = [TICK54_SPACE, BODY57_SPACE, AMP_FRAME51_SPACE]
        for space in spaces:
            for comp in space.components:
                assert comp.typical_range, f"{space.name}.{comp.name} 缺失数值范围"
                assert len(comp.typical_range) == 2, f"{space.name}.{comp.name} 范围格式错误"
                min_val, max_val = comp.typical_range
                assert min_val < max_val, f"{space.name}.{comp.name} 范围不合法: {comp.typical_range}"

    def test_all_components_have_descriptions(self):
        """测试所有观测项有描述。"""
        spaces = [TICK54_SPACE, BODY57_SPACE, AMP_FRAME51_SPACE]
        for space in spaces:
            for comp in space.components:
                assert comp.description, f"{space.name}.{comp.name} 缺失描述"
                assert len(comp.description) > 0, f"{space.name}.{comp.name} 描述为空"

    def test_normalization_flags_are_explicit(self):
        """测试所有观测项显式声明是否需要归一化。"""
        spaces = [TICK54_SPACE, BODY57_SPACE, AMP_FRAME51_SPACE]
        for space in spaces:
            for comp in space.components:
                assert isinstance(comp.normalization_required, bool), \
                    f"{space.name}.{comp.name} 归一化标志不是布尔值"


class TestObservationSpecSemantics:
    """测试观测契约语义一致性。"""

    def test_tick54_components_require_normalization(self):
        """测试tick54所有分量都需要归一化（除了已归一化的）。"""
        # tick54所有分量都是原始物理量，需要归一化
        assert all(c.normalization_required for c in TICK54_SPACE.components)

    def test_body57_normalized_components(self):
        """测试body57中已归一化的分量标记正确。"""
        normalized = [c.name for c in BODY57_SPACE.components if not c.normalization_required]
        expected = ["command_age", "last_action", "gait_clock"]
        assert normalized == expected, f"已归一化分量不匹配: {normalized} vs {expected}"

    def test_gyro_range_consistency(self):
        """测试陀螺仪范围在tick54和body57中一致。"""
        tick54_gyro = next(c for c in TICK54_SPACE.components if c.name == "gyro")
        body57_gyro = next(c for c in BODY57_SPACE.components if c.name == "gyro")
        assert tick54_gyro.typical_range == body57_gyro.typical_range
        assert tick54_gyro.physical_unit == body57_gyro.physical_unit

    def test_joint_velocity_range_consistency(self):
        """测试关节速度范围在tick54和body57中一致。"""
        tick54_dq = next(c for c in TICK54_SPACE.components if c.name == "dq")
        body57_jvel = next(c for c in BODY57_SPACE.components if c.name == "jvel")
        assert tick54_dq.typical_range == body57_jvel.typical_range
        assert tick54_dq.physical_unit == body57_jvel.physical_unit

    def test_privileged_flag_correct(self):
        """测试特权信息标记正确。"""
        assert not TICK54_SPACE.privileged
        assert not BODY57_SPACE.privileged
        assert AMP_FRAME51_SPACE.privileged  # AMP参考运动是训练时特权信息
        assert not ACTOR_OBS_SPACE.privileged

    def test_components_requiring_normalization_method(self):
        """测试components_requiring_normalization()方法。"""
        tick54_need_norm = TICK54_SPACE.components_requiring_normalization()
        assert len(tick54_need_norm) == 6  # tick54所有6个分量组都需要归一化

        body57_need_norm = BODY57_SPACE.components_requiring_normalization()
        assert len(body57_need_norm) == 6  # body57有6个需要归一化（9-3已归一化）


class TestObservationSpecPhysicalUnits:
    """测试物理单位定义的合理性。"""

    def test_angular_units(self):
        """测试角度相关单位统一使用rad。"""
        all_components = (
            TICK54_SPACE.components
            + BODY57_SPACE.components
            + AMP_FRAME51_SPACE.components
        )
        angular_comps = [
            c for c in all_components
            if any(keyword in c.name.lower() for keyword in ["q_", "joint", "jpos", "gyro", "angvel", "wz"])
        ]
        for comp in angular_comps:
            if "vel" in comp.name or "gyro" in comp.name or "wz" in comp.description:
                assert comp.physical_unit in ["rad/s", "mixed"], \
                    f"{comp.name} 应使用 rad/s: {comp.physical_unit}"
            elif "pos" in comp.name or comp.name.startswith("q_"):
                assert comp.physical_unit in ["rad", "quat"], \
                    f"{comp.name} 应使用 rad: {comp.physical_unit}"

    def test_linear_velocity_units(self):
        """测试线速度单位统一使用m/s。"""
        all_components = (
            TICK54_SPACE.components
            + BODY57_SPACE.components
            + AMP_FRAME51_SPACE.components
        )
        linear_vel_comps = [
            c for c in all_components
            if "linvel" in c.name or "foot_vel" in c.name
        ]
        for comp in linear_vel_comps:
            if "command" not in comp.name:  # command是混合单位
                assert comp.physical_unit == "m/s", \
                    f"{comp.name} 应使用 m/s: {comp.physical_unit}"

    def test_gravity_unit(self):
        """测试重力单位为m/s^2。"""
        gravity_comps = [
            c for c in TICK54_SPACE.components + BODY57_SPACE.components
            if "gravity" in c.name
        ]
        for comp in gravity_comps:
            assert comp.physical_unit == "m/s^2", \
                f"{comp.name} 应使用 m/s^2: {comp.physical_unit}"


class TestObservationSpecValueRanges:
    """测试数值范围定义的合理性。"""

    def test_joint_position_ranges(self):
        """测试关节位置范围合理（action_scale=0.35）。"""
        joint_pos_comps = [
            c for c in TICK54_SPACE.components + BODY57_SPACE.components + AMP_FRAME51_SPACE.components
            if c.name in ["q_rel", "q_des_rel", "q_error", "jpos", "joint_pos"]
        ]
        for comp in joint_pos_comps:
            min_val, max_val = comp.typical_range
            assert min_val == -0.35, f"{comp.name} 最小值应为-0.35: {min_val}"
            assert max_val == 0.35, f"{comp.name} 最大值应为0.35: {max_val}"

    def test_normalized_ranges(self):
        """测试已归一化分量范围在[-1,1]或[0,1]。"""
        normalized_comps = [
            c for c in BODY57_SPACE.components
            if not c.normalization_required
        ]
        for comp in normalized_comps:
            min_val, max_val = comp.typical_range
            if comp.name == "command_age":
                assert min_val == 0.0 and max_val == 1.0, f"{comp.name} 应在[0,1]: {comp.typical_range}"
            else:
                assert min_val == -1.0 and max_val == 1.0, f"{comp.name} 应在[-1,1]: {comp.typical_range}"

    def test_gravity_range(self):
        """测试重力范围合理（地球重力加速度~9.81）。"""
        gravity_comps = [
            c for c in TICK54_SPACE.components + BODY57_SPACE.components
            if "gravity" in c.name
        ]
        for comp in gravity_comps:
            min_val, max_val = comp.typical_range
            assert abs(min_val) == pytest.approx(9.81, rel=0.01), \
                f"{comp.name} 最小值应接近-9.81: {min_val}"
            assert abs(max_val) == pytest.approx(9.81, rel=0.01), \
                f"{comp.name} 最大值应接近9.81: {max_val}"
