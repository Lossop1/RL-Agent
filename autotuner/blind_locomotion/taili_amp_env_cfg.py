# Taili 四足 AMP 运动环境配置：速度跟踪由奖励负责，步态风格由 AMP/参考轨迹约束。
from __future__ import annotations
import os
from pathlib import Path
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import PhysxCfg, RigidBodyMaterialCfg, SimulationCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from .assets.taili import TAILI_DOG_CFG

MOTIONS_DIR = os.environ.get("TAILI_MOTIONS_DIR", str(Path(__file__).resolve().parent / "motions"))

TAILI_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=20,
    horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
    use_cache=False, curriculum=True,
    sub_terrains={
        "flat":      terrain_gen.MeshPlaneTerrainCfg(proportion=0.28),
        "slope":     terrain_gen.HfPyramidSlopedTerrainCfg(
                         proportion=0.18, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25),
        "slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                         proportion=0.14, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25),
        "rough":     terrain_gen.HfRandomUniformTerrainCfg(
                         proportion=0.13, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25),
        "boxes":     terrain_gen.MeshRandomGridTerrainCfg(
                         proportion=0.10, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0),
        "stairs":    terrain_gen.MeshPyramidStairsTerrainCfg(   # 从中心平台生成，主要训练下楼梯。
                         proportion=0.05, step_height_range=(0.05, 0.15), step_width=0.3,
                         platform_width=3.0, border_width=1.0, holes=False),
        # 上楼梯：倒金字塔楼梯让机器人从低处起步，课程等级会缩放台阶高度。
        # 这一路径用于训练真实跨越台阶，而不是只从平台向下走。
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                         proportion=0.12, step_height_range=(0.08, 0.30), step_width=0.32,
                         platform_width=3.0, border_width=1.0, holes=False),
    },
)


@configclass
class TailiAmpEnvCfg(DirectRLEnvCfg):
    # 速度跟踪：速度奖励是命令跟踪的主驱动，AMP 只负责风格与姿态，不承担速度语义。
    # 线速度和 yaw 都使用双侧误差核：欠速、超速都会降低奖励。
    rew_track_lin    = 3.5
    rew_track_ang    = 3.5
    track_sigma_lin  = 0.25      # 线速度误差尺度，单位 m/s。
    track_sigma_ang  = 0.25      # 角速度误差尺度，单位 rad/s。
    stand_sigma      = 0.10      # 站立姿态项使用的速度误差尺度。
    # 足端滞空时间奖励默认关闭；接触节律由 gait_match 和接触质量项负责。
    rew_feet_air_time = 0.0
    air_time_min     = 0.08
    air_time_target  = 0.30
    rew_alive        = 0.0
    rew_action_rate  = -0.025  # 限制动作突变，使速度切换和停步恢复更平滑。
    rew_joint_acc    = -2.5e-7
    rew_lin_vel_z    = -1.5
    rew_ang_vel_xy   = -0.4
    # 电机保护：只惩罚接近力矩上限的区间，正常步态力矩不应被过度压制。
    rew_torque       = -2.0e-4
    torque_limit_frac = 0.85
    # 无命令时抑制关节抖动，目标是自然回到稳定站立而不是原地碎步。
    rew_stand_still  = -2.0e-2

    # 步态节律：接触传感器不进入 AMP 判别器，因此对角小跑节律需要显式奖励。
    rew_gait_phase   = 3.0
    # 目标步态周期偏慢，用来抑制高频碎步和重落脚；速度越高周期越短。
    gait_period      = 0.55
    gait_period_min  = 0.40
    gait_period_slope = 0.10
    gait_duty        = 0.5
    rew_swing_drag   = 0.0     # 摆动相触地拖脚惩罚，默认关闭。
    rew_gait_enforce = 0.0     # 硬步态约束默认关闭；地形上允许策略打破固定节律。
    dr_unlock_terrain = 0.0    # 域随机化解锁所需的平均地形等级，0 表示不额外延迟。

    # 地形越粗糙，固定小跑节律约束越弱；高难度地形允许出现非平地步态。
    gait_rough_scale = 0.08

    # 步态质量：滑移、落脚速度和抬脚高度共同约束“轻、稳、不拖地”的足端轨迹。
    rew_stance_slip  = -0.8
    rew_swing_dir    = 0.0      # 摆动方向奖励默认关闭，避免形成无上限的速度激励。
    rew_land_decel   = -1.5     # 落脚前水平速度越大惩罚越强。
    rew_clearance    = -1.5     # 把摆动脚高度拉向目标区间，避免平地过度高抬腿。
    base_clearance   = 0.11
    # 粗糙地形提升抬脚目标和平滑加重 clearance 约束；平地保持低而高效的摆动高度。
    rew_clearance_heavy   = -8.0
    clr_rough_flat        = 0.008
    clr_rough_span        = 0.012
    clr_rough_bonus_max   = 0.06
    clearance_ramp_intervals = 25
    # AMP 参考轨迹也随粗糙度提高摆动脚高度，使风格判别器与地形需求一致。
    ref_clearance_rough_gain = 0.30

    # 关节轨迹模仿：父类中的旧 imitation 参数保留兼容；TP 环境实际使用 live imitation。
    rew_imitate      = 1.5
    imitate_sigma    = 0.5
    imitate_fwd_floor = 0.5
    # live imitation 是 TP 环境中的正向轨迹吸引项，用来约束足端回收、轻落脚和左右对称。
    w_imitate_live     = 1.6
    imitate_sigma_live = 0.09
    w_swing_dir = 0.0        # 摆动脚方向吸引项，默认关闭。
    swing_dir_margin = 0.15
    swing_dir_sigma  = 0.08

    # 地形事件奖励：严格盲狗语义，只在接触/支撑高度变化后给奖励或惩罚，不使用前视地形标签。
    rew_climb = 0.0
    climb_slip_gate = 0.18
    climb_slip_soft_span = 0.18
    climb_vz_cap = 0.8
    rew_terrain_up = 0.0
    rew_terrain_down = 0.0
    rew_terrain_support_transfer = 0.0
    rew_terrain_contact_quality = 0.0
    rew_terrain_event_collapse = 0.0
    terrain_transition_eps = 0.05
    terrain_transition_span = 0.08
    terrain_event_latch_s = 0.0
    terrain_up_vz_cap = 0.6
    terrain_down_vz_cap = 0.5
    terrain_down_vz_target = 0.18
    terrain_event_quality_floor = 0.35
    terrain_front_duty_margin = 0.12
    terrain_rear_duty_floor = 0.48
    terrain_support_scale = 0.22
    terrain_torque_soft_frac = 0.85
    terrain_event_collapse_height = 0.42
    terrain_event_collapse_wxy = 1.50
    terrain_event_collapse_speed_ratio = 1.60
    terrain_event_collapse_speed_min = 0.75
    terrain_event_collapse_speed_scale = 0.60
    terrain_curriculum_height_gain = 0.08
    terrain_curriculum_height_loss = 0.08
    terrain_curriculum_forward_min = 0.25
    terrain_curriculum_stable_h = 0.42
    terrain_curriculum_stable_upright = 0.85
    terrain_curriculum_stable_contact_min = 2.0
    terrain_curriculum_stable_wxy_max = 1.50
    terrain_curriculum_speed_cap_ratio = 1.60
    terrain_curriculum_speed_cap_min = 0.75
    terrain_curriculum_failure_h = 0.42
    terrain_curriculum_failure_upright = 0.75
    terrain_curriculum_failure_wxy = 2.00
    terrain_curriculum_floor_after_peak = 3
    terrain_curriculum_peak_drop = 0
    terrain_curriculum_move_down_patience = 1
    terrain_curriculum_ignore_flat = True
    w_settle_brake = 2.0    # 无命令但仍在运动时，奖励主动减速，帮助干净停步。
    # 不使用固定“楼梯专属高抬腿”。抬脚高度由粗糙度、接触事件和 clearance 目标共同决定。
    discrete_clearance = False
    # 参考步态中的前后站距偏置；YAML 会覆盖为实际策略值。
    stance_dx        = 0.0

    # 站立与姿态
    rew_stand_pose   = 5.0
    stand_height     = 0.52
    rew_base_height  = -9.0
    rew_hip_neutral  = -8.0     # 抑制髋关节外翻/内扣，保持直线方向下足端横向位置干净。
    hip_neutral_lat_scale = 0.25
    rew_offaxis_vel  = -3.5     # 抑制无关方向速度，减少前进/后退时的横摆和横移。
    rew_overshoot    = -2.0     # 超速刹车项。
    rew_wrong_dir    = 0.0      # 反向运动惩罚默认关闭，由 YAML 决定是否启用。
    rew_underspeed   = 0.0      # 通用欠速惩罚默认关闭。
    rew_backward_underspeed = 0.0  # 后退命令专用欠速惩罚。
    rew_backward_wrong_dir  = 0.0  # 后退命令下向前运动的惩罚。
    rew_lateral_underspeed = 0.0   # 纯横移命令专用欠速惩罚。
    directional_aux_backward_only = True
    speed_tol_abs    = 0.10     # 速度跟踪绝对容差。
    speed_tol_rel    = 0.15     # 速度跟踪相对容差。

    # 命令分布：早期训练保持前进、后退、横移和 yaw 四类命令均衡。
    cmd_lin_x        = (-1.0,  1.0)
    cmd_lin_y        = (-0.6,  0.6)
    cmd_ang_z        = (-1.0,  1.0)
    cmd_resample_s   = 5.0
    # 命令缓冲：self.commands 按一阶低通靠近采样目标，控制频率约 50Hz。
    # alpha=0.93 约等价于 0.3s 过渡，用于停步、反向和转向时的自然减速。
    # 稳态下命令仍会收敛到目标值，因此不改变最终速度跟踪语义。
    cmd_smooth_alpha = 0.93
    cmd_transition_enable = True
    cmd_transition_cycles = 0.75
    cmd_transition_min_s = 0.25
    cmd_transition_max_s = 0.80
    cmd_transition_fast_s = 0.16
    cmd_transition_sign_flip_v = 0.08
    cmd_transition_sign_flip_w = 0.10
    cmd_transition_low_speed_v = 0.12
    cmd_transition_low_speed_w = 0.16
    cmd_transition_contact_feet = 3.0
    cmd_transition_zero_frac_min = 0.48
    cmd_transition_zero_frac_max = 0.70
    cmd_transition_stop_zero_frac = 0.80
    stand_prob       = 0.25    # 提高站立/停步样本比例，增加从运动命令切到零命令的减速练习。
    cmd_prob_fwd     = 0.25    # bootstrap 阶段四个运动方向等概率。
    cmd_prob_back    = 0.25
    cmd_prob_lat     = 0.25
    cmd_prob_yaw     = 0.25
    # 各方向速度范围：(下限, 课程起始上限)。
    cmd_fwd_range    = (0.30, 0.70)
    cmd_back_range   = (0.20, 0.45)
    cmd_lat_range    = (0.15, 0.35)
    cmd_yaw_range    = (0.25, 0.80)

    # 分方向速度课程。
    # 每个方向都有独立进展和速度上限，互不代偿。
    cmd_fwd_max      = 2.5     # 前进速度课程目标上限。
    cmd_back_max     = 0.8     # 后退速度课程目标上限。
    cmd_lat_max      = 0.7     # 横移速度课程目标上限。
    cmd_yaw_max      = 1.2     # yaw 速度课程目标上限，单位 rad/s。
    vel_cur_up       = 0.85    # 方向进展高于该值时提高上限。
    vel_cur_down     = 0.55    # 方向进展低于该值时降低上限。
    vel_cur_step     = 0.04    # 每个日志间隔的速度上限调整步长。
    vel_terrain_decouple = True

    # 域随机化：根据 _dr_level 条件启用，先学会干净步态，再逐步扩大真实部署扰动。
    # DR 从 0 级开始，只有在持续满足能力门控后才升级。
    # 左右镜像增强用于约束左右对称，减少单侧偏置步态。
    sym_augment      = True

    # 统一课程阶段：phi0 建立基本运动，phi1 引入步态质量约束，phi2 并行推进速度、地形和 DR。
    # DR 与地形等级解耦；真实盲狗需要 sim2real 鲁棒性，不能等到高地形后才开始。
    # 每个阶段门控都要求连续 phase_intervals 个日志间隔满足条件。
    phase_intervals       = 5      # 进入下一阶段前需要连续满足门控的日志间隔数。
    penalty_ramp_intervals = 25    # phi1 中质量惩罚从 0 渐入到 1，避免 critic 冲击。
    phase_gate_prog_0     = 0.60   # phi0 到 phi1：四个方向速度进展的最小值。
    # 阶段门控使用命令进展与去时钟化步态质量；gait_match 只作为读数，不直接门控。
    phase_gate_prog_1     = 0.70   # phi1 到 phi2：保证能进入地形训练，同时不过早放松平地基础。
    phase_gate_slip_1     = 0.20   # 历史兼容字段，当前不再作为门控。
    phase_gate_terrain_2  = 6.0    # phi2 到 phi3：平均地形等级门槛。
    phase_gate_discrete_terrain_2 = 0.0  # 可选：离散障碍平均等级门槛，0 表示不单独门控。
    phase_gate_fall_2     = 0.05   # phi2 到 phi3：跌倒率门槛。
    phase_gate_prog_2     = 0.50   # phi2 到 phi3：地形上仍需保留基本方向跟踪。
    regress_fall          = 0.10   # phi2 回退保护：跌倒率过高时暂停推进地形/速度。
    regress_prog          = 0.40   # phi2 回退保护：方向进展过低时暂停推进。

    dr_enable        = True
    dr_start_level   = 0    # DR 从该等级起步；设为大于 0 可跳过低等级。
    # 0 级：不加 DR，用于建立干净基础步态。
    dr_push_interval_s_0 = 0.0
    dr_push_vel_0        = 0.0
    # 1 级：轻量扰动。
    dr_push_interval_s_1 = 30.0
    dr_push_vel_1        = 0.2
    dr_mass_range_1      = (-1.0, 2.0)
    dr_stiffness_scale_1 = (0.9,  1.1)
    dr_damping_scale_1   = (0.85, 1.15)
    # 2 级：中等 sim2real 扰动。
    dr_push_interval_s_2 = 20.0
    dr_push_vel_2        = 0.5
    dr_mass_range_2      = (-3.0, 6.0)
    dr_stiffness_scale_2 = (0.8,  1.2)
    dr_damping_scale_2   = (0.7,  1.3)
    # 3 级：部署级扰动包络。
    dr_push_interval_s_3 = 15.0
    dr_push_vel_3        = 0.8
    dr_push_ang_scale    = 0.8     # yaw 角速度扰动 = 该系数乘线速度扰动。
    dr_mass_range_3      = (-5.0, 20.0)    # 机体质量扰动范围。
    dr_stiffness_scale_3 = (0.6,  1.4)
    dr_damping_scale_3   = (0.6,  1.4)
    # 额外 DR 通道：摩擦、质心偏置和 IMU 偏置按等级逐步打开。
    dr_full_start_level  = 1               # 从该等级开始启用摩擦/质心/IMU 扰动。
    dr_friction_range_1  = (0.7, 1.3)      # 摩擦扰动逐级扩大，到 3 级覆盖完整范围。
    dr_friction_range_2  = (0.5, 1.4)
    dr_com_offset_1      = 0.02            # 质心偏置阶梯，单位 m。
    dr_com_offset_2      = 0.035
    dr_imu_gyro_bias_1   = 0.02            # 陀螺仪偏置阶梯，单位 rad/s。
    dr_imu_gyro_bias_2   = 0.035
    dr_imu_grav_bias_1   = 0.015           # 重力方向偏置阶梯。
    dr_imu_grav_bias_2   = 0.022
    dr_friction_range_3  = (0.4, 1.4)      # 足端/地面摩擦范围。
    dr_com_offset_3      = 0.05            # x/y 方向质心偏置幅度，模拟负载和安装不对称。
    dr_imu_gyro_bias_3   = 0.05            # 单 episode 固定陀螺仪偏置。
    dr_imu_grav_bias_3   = 0.03            # 单 episode 固定重力方向偏置。
    # DR 升级门槛：等级越高，要求的基础步态越强。
    dr_gate_progress     = 0.65   # 0 级到 1 级。
    dr_gate_progress_l2  = 0.75   # 1 级到 2 级。
    dr_gate_progress_l3  = 0.80   # 2 级到 3 级，要求更强。
    dr_gate_intervals    = 5      # 连续满足门槛后升级。

    # 观测噪声：只加在盲策略观测上，不加在 critic 特权信息上。
    obs_noise_jpos    = 0.01
    obs_noise_jvel    = 0.25
    obs_noise_angvel  = 0.05
    obs_noise_gravity = 0.05

    # 控制延迟：1 个控制步，约 20ms，用于模拟真实计算和通信延迟。
    action_delay_steps = 1

    episode_length_s = 20.0
    decimation       = 4
    dt               = 1 / 200

    # 观测空间
    # Actor 是盲策略：只使用电机、IMU、命令和本体历史，也就是实际部署可用的信息。
    #   current_blind(53) = ang_vel(3)+gravity(3)+cmd(3)+jpos(12)+jvel(12)+last_act(12)+gait(8)
    #   history(420)      = 42 x 10 steps of proprio
    #   BLIND_OBS_DIM     = 53 + 420 = 473，见 blind_policy.py。
    #
    # 环境 observation_space = 670：skrl 侧能拿到完整特权观测。
    #   blind(473) + lin_vel(3) + height_scan(187) + terrain_ctx(3) + foot_contact(4) = 670
    #
    # BlindGaussianPolicy 会显式切片 [:, :473]；critic 可以使用完整 670 维。
    obs_history_len  = 10         # H=10，约 200ms，用于感知接触冲击并估计速度。
    obs_history_dim  = 42         # 单步历史：jpos(12)+jvel(12)+angvel(3)+gravity(3)+lastact(12)。
    n_height_scan    = 17 * 11    # GridPatternCfg(resolution=0.1, size=(1.6,1.0)) 对应 187 条射线。
    observation_space = (53 + obs_history_dim * obs_history_len   # 盲策略部分：473
                         + 3 + n_height_scan + 3 + 4)             # 特权部分：197，总计 670。
    action_space     = 12
    state_space      = 0          # 不使用 skrl state_space 机制，模型内部自行切片。

    # AMP 观测只表达风格：基础运动学风格 43 维 + terrain_ctx 3 维 = 46。
    # 速度和命令不进入 AMP 判别器；速度跟踪由双侧速度奖励负责。
    # 这样可以避免 AMP 把滑移或欠速误判成“风格正确”。
    terrain_ctx_dim  = 3          # [前后坡度, 横向坡度, 粗糙度]，仅供 critic/AMP 使用。
    command_ctx_dim  = 3          # 保留作兼容说明，不再进入 amp_observation_space。
    num_amp_observations = 2
    amp_observation_space = 43 + terrain_ctx_dim  # 46，纯风格加地形上下文。
    # AMP stride 窗口：启用 TAILI_AMP_STRIDE=1 时，让风格窗口覆盖约一个步态周期。
    # 策略侧和参考侧都会按 amp_frame_stride 子采样，判别器输入维度自动随帧数扩展。
    amp_frame_stride  = 4          # 环境步子采样步长，50Hz 下约 80ms。
    amp_stride_frames = 6          # stride 窗口帧数，总跨度约 420ms。

    action_scale     = 0.35
    early_termination = True
    termination_height = 0.35
    contact_force_threshold = 10.0
    actuator_stiffness_by_joint = [120.0] * 12
    actuator_damping_by_joint = [10.0] * 12

    # 参考动作片段
    motion_files = [os.path.join(MOTIONS_DIR, "clips", f"taili_{g}.npz") for g in
                    ["fwd_030", "fwd_060", "fwd_090", "fwd_120", "fwd_160", "fwd_200",
                     "back_030", "back_060",
                     "left_025", "left_045", "right_025", "right_045",
                     "yawl_040", "yawl_080", "yawr_040", "yawr_080",
                     "uphill_fwd", "downhill_fwd", "uphill_yawl", "uphill_yawr",
                     "cross_fwd", "cross_back"]]
    cond_sigma       = 0.30
    slope_sigma      = 0.10
    log_every        = 500
    reference_body   = "base_link"
    foot_body_names  = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    reset_strategy   = "random-start"

    # 仿真配置
    sim: SimulationCfg = SimulationCfg(dt=dt, render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**24,
            gpu_total_aggregate_pairs_capacity=2**25,
            gpu_found_lost_aggregate_pairs_capacity=2**25,
            gpu_max_rigid_contact_count=2**24,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**27,
        ))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    robot: ArticulationCfg = TAILI_DOG_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="generator", terrain_generator=TAILI_TERRAINS_CFG,
        max_init_terrain_level=0, collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0),
        debug_vis=False,
    )
    # 高度扫描器始终在场景中，用于 terrain_ctx、critic 特权观测和 clearance 奖励。
    # Actor 盲策略不直接读取 height_scan；它只供 critic、AMP 和奖励计算使用。
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=False, mesh_prim_paths=["/World/ground"],
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, track_air_time=True)
    terrain_move_up_dist = TAILI_TERRAINS_CFG.size[0] / 2.0
