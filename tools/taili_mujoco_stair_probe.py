"""Measure Taili staircase completion and contact quality in MuJoCo."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from taili_mujoco_long_horizon import (
    Controller,
    DeploymentContract,
    quaternion_to_roll_pitch,
    rotate_inverse,
)


LEGS = ("FR", "FL", "RL", "RR")


@dataclass(frozen=True)
class StairCourse:
    direction: str
    start_x_m: float
    landing_start_x_m: float
    completion_x_m: float
    landing_height_m: float
    expected_base_height_m: float


def object_name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, object_type, index) or f"unnamed_{index}"


def infer_stair_course(model: mujoco.MjModel) -> StairCourse:
    geom_names = {
        object_name(model, mujoco.mjtObj.mjOBJ_GEOM, index): index
        for index in range(model.ngeom)
    }
    up_ids = [index for name, index in geom_names.items() if name.startswith("stair_up_")]
    down_ids = [index for name, index in geom_names.items() if name.startswith("stair_down_")]
    if up_ids and not down_ids:
        direction = "up"
        stair_ids = up_ids
        landing_id = geom_names.get("upper_landing", -1)
    elif down_ids and not up_ids:
        direction = "down"
        stair_ids = down_ids
        landing_id = geom_names.get("lower_landing", -1)
    else:
        raise ValueError("model must contain exactly one generated up or down staircase")
    if landing_id < 0:
        raise ValueError(f"{direction} staircase has no landing geom")

    start_x = min(float(model.geom_pos[index, 0] - model.geom_size[index, 0]) for index in stair_ids)
    landing_start = float(
        model.geom_pos[landing_id, 0] - model.geom_size[landing_id, 0]
    )
    landing_height = float(
        model.geom_pos[landing_id, 2] + model.geom_size[landing_id, 2]
    )
    initial_surface_height = 0.0
    base_clearance = float(model.key_qpos[0, 2] - initial_surface_height)
    return StairCourse(
        direction=direction,
        start_x_m=start_x,
        landing_start_x_m=landing_start,
        completion_x_m=landing_start + 0.15,
        landing_height_m=landing_height,
        expected_base_height_m=landing_height + base_clearance,
    )


def classify_robot_geoms(
    model: mujoco.MjModel,
) -> tuple[dict[int, str], dict[int, str]]:
    feet: dict[int, str] = {}
    segments: dict[int, str] = {}
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        if body_id == 0:
            continue
        body_name = object_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        leg = body_name[:2] if body_name[:2] in LEGS else "body"
        is_foot = (
            leg in LEGS
            and body_name.endswith("_calf")
            and int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
            and int(model.geom_contype[geom_id]) != 0
        )
        if is_foot:
            feet[geom_id] = leg
        else:
            segments[geom_id] = body_name
    if set(feet.values()) != set(LEGS):
        raise ValueError(f"could not identify one collision foot per leg: {feet}")
    return feet, segments


def quaternion_to_yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def percentile(values: list[float], quantile: float) -> float | None:
    return float(np.percentile(values, quantile)) if values else None


def rms(square_sum: float, count: int) -> float | None:
    return math.sqrt(square_sum / count) if count else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=None)
    parser.add_argument("--command", type=float, nargs=3, default=(0.4, 0.0, 0.0))
    parser.add_argument("--stand-seconds", type=float, default=2.0)
    parser.add_argument("--move-seconds", type=float, default=18.0)
    parser.add_argument("--post-completion-seconds", type=float, default=1.5)
    parser.add_argument("--terminate-upright", type=float, default=0.5)
    parser.add_argument("--log-interval", type=float, default=0.5)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    target = np.asarray(args.command, dtype=np.float64)
    if target[0] <= 0.0:
        raise ValueError("generated staircases are evaluated along positive body x")
    if args.stand_seconds < 0.0 or args.move_seconds <= 0.0:
        raise ValueError("stand-seconds must be nonnegative and move-seconds positive")

    contract_path = args.contract or args.policy.with_suffix(args.policy.suffix + ".json")
    contract = DeploymentContract(contract_path)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    course = infer_stair_course(model)
    feet, segments = classify_robot_geoms(model)
    policy_substeps = round(contract.policy_dt / model.opt.timestep)
    if policy_substeps < 1 or not math.isclose(
        policy_substeps * model.opt.timestep,
        contract.policy_dt,
        abs_tol=1.0e-9,
    ):
        raise ValueError("MuJoCo physics timestep must divide the policy period")

    data = mujoco.MjData(model)
    controller = Controller(model, data, args.policy, contract, None, 0)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    zero = np.zeros(3)
    end_time = args.stand_seconds + args.move_seconds
    completion_time: float | None = None
    completion_deadline: float | None = None
    failure_time: float | None = None
    start_position = data.qpos[:3].copy()
    next_log = args.stand_seconds
    step = 0
    motion_steps = 0
    max_x = float(data.qpos[0])
    min_upright = 1.0
    x_path_length = 0.0
    previous_xy = data.qpos[:2].copy()
    body_vx_sum = 0.0
    tracking_error_sq_sum = 0.0
    lateral_sq_sum = 0.0
    lateral_abs_max = 0.0
    yaw_sq_sum = 0.0
    yaw_abs_max = 0.0
    roll_sq_sum = 0.0
    pitch_sq_sum = 0.0
    wxy_sq_sum = 0.0
    wxy_peak = 0.0
    vertical_velocity_sq_sum = 0.0
    z_min = math.inf
    z_max = -math.inf
    contact_steps = Counter({leg: 0 for leg in LEGS})
    no_foot_contact_steps = 0
    feet_per_step_sum = 0
    nonfoot_contact_steps = 0
    nonfoot_contacts = Counter()
    nonfoot_forces: list[float] = []
    foot_normal_forces: list[float] = []
    foot_tangent_forces: list[float] = []
    foot_slip_speeds: list[float] = []
    touchdown_forces: list[float] = []
    previous_contact_legs: set[str] = set()
    saturation_samples = 0
    saturation_total = 0

    while data.time < end_time:
        command = zero if data.time < args.stand_seconds else target
        if step % policy_substeps == 0:
            controller.policy_step(command)
        controller.physics_step()
        step += 1

        if data.time < args.stand_seconds:
            previous_xy = data.qpos[:2].copy()
            continue

        motion_steps += 1
        quat = data.qpos[3:7].copy()
        body_velocity = rotate_inverse(quat, data.qvel[:3])
        gyro = data.sensordata[
            controller.gyro_adr:controller.gyro_adr + 3
        ].copy()
        roll, pitch = quaternion_to_roll_pitch(quat)
        yaw = quaternion_to_yaw(quat)
        upright = float(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2))
        min_upright = min(min_upright, upright)
        max_x = max(max_x, float(data.qpos[0]))
        x_path_length += float(np.linalg.norm(data.qpos[:2] - previous_xy))
        previous_xy = data.qpos[:2].copy()
        body_vx_sum += float(body_velocity[0])
        tracking_error_sq_sum += float((body_velocity[0] - target[0]) ** 2)
        lateral = float(data.qpos[1] - start_position[1])
        lateral_sq_sum += lateral * lateral
        lateral_abs_max = max(lateral_abs_max, abs(lateral))
        yaw_sq_sum += yaw * yaw
        yaw_abs_max = max(yaw_abs_max, abs(yaw))
        roll_sq_sum += roll * roll
        pitch_sq_sum += pitch * pitch
        wxy = float(np.linalg.norm(gyro[:2]))
        wxy_sq_sum += wxy * wxy
        wxy_peak = max(wxy_peak, wxy)
        vertical_velocity_sq_sum += float(body_velocity[2] ** 2)
        z_min = min(z_min, float(data.qpos[2]))
        z_max = max(z_max, float(data.qpos[2]))
        saturation_samples += int(controller.last_saturated.sum())
        saturation_total += controller.last_saturated.size

        current_legs: set[str] = set()
        step_nonfoot = False
        force_by_leg: dict[str, float] = {}
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(model.geom_bodyid[geom1])
            body2 = int(model.geom_bodyid[geom2])
            if body1 == 0 and body2 > 0:
                robot_geom = geom2
            elif body2 == 0 and body1 > 0:
                robot_geom = geom1
            else:
                continue

            contact_force = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact_index, contact_force)
            normal_force = abs(float(contact_force[0]))
            tangent_force = float(np.linalg.norm(contact_force[1:3]))
            if robot_geom in feet:
                leg = feet[robot_geom]
                current_legs.add(leg)
                force_by_leg[leg] = max(force_by_leg.get(leg, 0.0), normal_force)
                foot_normal_forces.append(normal_force)
                foot_tangent_forces.append(tangent_force)

                body_id = int(model.geom_bodyid[robot_geom])
                jacobian = np.zeros((3, model.nv))
                rotational_jacobian = np.zeros((3, model.nv))
                mujoco.mj_jac(
                    model,
                    data,
                    jacobian,
                    rotational_jacobian,
                    contact.pos,
                    body_id,
                )
                point_velocity = jacobian @ data.qvel
                normal = np.asarray(contact.frame[:3])
                tangent_velocity = point_velocity - np.dot(point_velocity, normal) * normal
                foot_slip_speeds.append(float(np.linalg.norm(tangent_velocity)))
            else:
                body_name = segments.get(robot_geom, f"geom_{robot_geom}")
                nonfoot_contacts[body_name] += 1
                nonfoot_forces.append(normal_force)
                step_nonfoot = True

        for leg in current_legs:
            contact_steps[leg] += 1
        feet_per_step_sum += len(current_legs)
        if not current_legs:
            no_foot_contact_steps += 1
        if step_nonfoot:
            nonfoot_contact_steps += 1
        for leg in current_legs - previous_contact_legs:
            touchdown_forces.append(force_by_leg.get(leg, 0.0))
        previous_contact_legs = current_legs

        if completion_time is None and data.qpos[0] >= course.completion_x_m:
            completion_time = float(data.time - args.stand_seconds)
            completion_deadline = float(data.time + args.post_completion_seconds)
            end_time = min(end_time, completion_deadline)

        if upright < args.terminate_upright:
            failure_time = float(data.time - args.stand_seconds)
            end_time = float(data.time)

        if data.time + 1.0e-9 >= next_log:
            print(
                f"t={data.time:.2f} x={data.qpos[0]:.3f} y={data.qpos[1]:.3f} "
                f"z={data.qpos[2]:.3f} roll={math.degrees(roll):.2f} "
                f"pitch={math.degrees(pitch):.2f} yaw={math.degrees(yaw):.2f} "
                f"vx={body_velocity[0]:.3f} contacts={sorted(current_legs)} "
                f"nonfoot={step_nonfoot}",
                flush=True,
            )
            next_log += args.log_interval

        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            break

    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    completed = completion_time is not None
    final_height_error = float(data.qpos[2] - course.expected_base_height_m)
    stable_on_landing = bool(
        completed
        and finite
        and min_upright > 0.4
        and abs(final_height_error) < 0.20
        and data.qpos[0] >= course.completion_x_m
    )
    course_distance = course.completion_x_m - course.start_x_m
    progress = max_x - course.start_x_m
    summary = {
        "model": str(args.model),
        "policy": str(args.policy),
        "course": asdict(course),
        "command": target.tolist(),
        "finite": finite,
        "completed": completed,
        "stable_on_landing": stable_on_landing,
        "completion_time_s": completion_time,
        "failure_time_s": failure_time,
        "progress_fraction": float(progress / course_distance),
        "start_position_m": start_position.tolist(),
        "final_position_m": data.qpos[:3].tolist(),
        "max_x_m": max_x,
        "final_landing_height_error_m": final_height_error,
        "mean_body_forward_velocity_m_s": body_vx_sum / max(motion_steps, 1),
        "forward_tracking_rmse_m_s": rms(tracking_error_sq_sum, motion_steps),
        "path_efficiency": float(progress / max(x_path_length, 1.0e-9)),
        "lateral_offset_rms_m": rms(lateral_sq_sum, motion_steps),
        "lateral_offset_abs_max_m": lateral_abs_max,
        "yaw_rms_deg": math.degrees(rms(yaw_sq_sum, motion_steps) or 0.0),
        "yaw_abs_max_deg": math.degrees(yaw_abs_max),
        "roll_rms_deg": math.degrees(rms(roll_sq_sum, motion_steps) or 0.0),
        "pitch_rms_deg": math.degrees(rms(pitch_sq_sum, motion_steps) or 0.0),
        "body_wxy_rms_rad_s": rms(wxy_sq_sum, motion_steps),
        "body_wxy_peak_rad_s": wxy_peak,
        "vertical_velocity_rms_m_s": rms(vertical_velocity_sq_sum, motion_steps),
        "base_height_range_m": [z_min, z_max],
        "minimum_upright": min_upright,
        "foot_contact_fraction": {
            leg: contact_steps[leg] / max(motion_steps, 1) for leg in LEGS
        },
        "mean_contacting_feet": feet_per_step_sum / max(motion_steps, 1),
        "no_foot_contact_fraction": no_foot_contact_steps / max(motion_steps, 1),
        "foot_slip_rms_m_s": rms(
            sum(value * value for value in foot_slip_speeds), len(foot_slip_speeds)
        ),
        "foot_slip_p95_m_s": percentile(foot_slip_speeds, 95.0),
        "foot_slip_max_m_s": max(foot_slip_speeds, default=None),
        "foot_normal_force_p95_n": percentile(foot_normal_forces, 95.0),
        "foot_normal_force_max_n": max(foot_normal_forces, default=None),
        "foot_tangent_force_p95_n": percentile(foot_tangent_forces, 95.0),
        "touchdown_force_p95_n": percentile(touchdown_forces, 95.0),
        "touchdown_force_max_n": max(touchdown_forces, default=None),
        "nonfoot_contact_step_fraction": nonfoot_contact_steps / max(motion_steps, 1),
        "nonfoot_contact_counts": dict(nonfoot_contacts),
        "nonfoot_normal_force_p95_n": percentile(nonfoot_forces, 95.0),
        "nonfoot_normal_force_max_n": max(nonfoot_forces, default=None),
        "torque_saturation_rate": saturation_samples / max(saturation_total, 1),
    }
    print("STAIR_SUMMARY " + json.dumps(summary, separators=(",", ":")), flush=True)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
