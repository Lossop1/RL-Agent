from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


LEGS = ["FL", "FR", "RL", "RR"]


def group_key_columns(df: pd.DataFrame, include_segment: bool = True) -> list[str]:
    """Return the identity columns for one independent rollout attempt."""
    candidates = ["case_id", "env_id", "episode_id"]
    if include_segment:
        candidates.append("cmd_segment_id")
    return [c for c in candidates if c in df.columns]


def command_segment_key_columns(df: pd.DataFrame) -> list[str]:
    """Return one command-attempt key; resets inside a segment remain one attempt."""
    return [c for c in ["case_id", "env_id", "cmd_segment_id"] if c in df.columns]


def finite_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str | Path) -> Any:
    import yaml
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_quantiles(x: Iterable[float], qs=(0.5, 0.9, 0.95, 0.99)) -> dict[str, float | None]:
    arr = pd.to_numeric(pd.Series(list(x)), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return {q_name(q): None for q in qs}
    out = {}
    for q in qs:
        out[q_name(q)] = float(np.quantile(arr, q))
    return out


def q_name(q: float) -> str:
    if abs(q - 0.5) < 1e-12:
        return "p50"
    return f"p{int(round(q * 1000)) / 10:g}" if (q * 100) % 1 else f"p{int(q * 100)}"


def stats(x: Iterable[float], qs=(0.5, 0.9, 0.95, 0.99)) -> dict[str, float | int | None]:
    s = pd.to_numeric(pd.Series(list(x)), errors="coerce").dropna()
    if len(s) == 0:
        return {"n": 0, "mean": None, "min": None, "max": None, **{q_name(q): None for q in qs}}
    d: dict[str, float | int | None] = {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "min": float(s.min()),
        "max": float(s.max()),
    }
    d.update(safe_quantiles(s, qs))
    return d


def bool_series(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    v = df[col]
    if v.dtype == bool:
        return v.fillna(default)
    return pd.to_numeric(v, errors="coerce").fillna(1 if default else 0).astype(float) != 0


def numeric_col(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def mode_col(df: pd.DataFrame) -> str:
    if "cmd_target_mode" in df.columns:
        return "cmd_target_mode"
    if "cmd_mode" in df.columns:
        return "cmd_mode"
    return "__mode"


def command_components(df: pd.DataFrame, prefer_target: bool = True) -> tuple[pd.Series, pd.Series, pd.Series]:
    pfx = "cmd_target_" if prefer_target and all(c in df.columns for c in ["cmd_target_vx", "cmd_target_vy", "cmd_target_wz"]) else "cmd_"
    return numeric_col(df, pfx + "vx", 0.0), numeric_col(df, pfx + "vy", 0.0), numeric_col(df, pfx + "wz", 0.0)


def derive_mode_from_cmd(vx: float, vy: float, wz: float, eps: float = 0.05) -> str:
    lx = abs(vx) > eps
    ly = abs(vy) > eps
    az = abs(wz) > eps
    if not lx and not ly and not az:
        return "stand"
    if (lx or ly) and az:
        return "mixed"
    if lx and not ly and not az:
        return "forward" if vx > 0 else "backward"
    if ly and not lx and not az:
        return "lateral"
    if az and not lx and not ly:
        return "yaw"
    return "mixed"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_record(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return normalize_record(df)


def normalize_record(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "case_id" not in df.columns:
        df["case_id"] = 0
    if "env_id" not in df.columns:
        df["env_id"] = 0
    if "episode_id" not in df.columns:
        df["episode_id"] = 0
    if "cmd_segment_id" not in df.columns:
        df["cmd_segment_id"] = 0
    if "time" not in df.columns:
        if "step" in df.columns and "control_dt" in df.columns:
            df["time"] = numeric_col(df, "step", 0.0) * numeric_col(df, "control_dt", 0.0)
        elif "step" in df.columns:
            df["time"] = numeric_col(df, "step", 0.0)
        else:
            df["time"] = np.arange(len(df), dtype=float)
    if "__mode" not in df.columns:
        if "cmd_target_mode" in df.columns:
            df["__mode"] = df["cmd_target_mode"].astype(str)
        elif "cmd_mode" in df.columns:
            df["__mode"] = df["cmd_mode"].astype(str)
        else:
            vx, vy, wz = command_components(df, prefer_target=True)
            df["__mode"] = [derive_mode_from_cmd(a, b, c) for a, b, c in zip(vx, vy, wz)]
    # Quaternion -> roll/pitch/yaw when needed.
    if not all(c in df.columns for c in ["base_roll", "base_pitch", "base_yaw"]):
        qcols = ["base_quat_w", "base_quat_x", "base_quat_y", "base_quat_z"]
        if all(c in df.columns for c in qcols):
            qw = numeric_col(df, "base_quat_w", 1.0)
            qx = numeric_col(df, "base_quat_x", 0.0)
            qy = numeric_col(df, "base_quat_y", 0.0)
            qz = numeric_col(df, "base_quat_z", 0.0)
            sinr = 2 * (qw * qx + qy * qz)
            cosr = 1 - 2 * (qx * qx + qy * qy)
            roll = np.arctan2(sinr, cosr)
            sinp = 2 * (qw * qy - qz * qx)
            pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))
            siny = 2 * (qw * qz + qx * qy)
            cosy = 1 - 2 * (qy * qy + qz * qz)
            yaw = np.arctan2(siny, cosy)
            df["base_roll"] = np.degrees(roll)
            df["base_pitch"] = np.degrees(pitch)
            df["base_yaw"] = np.degrees(yaw)
    return df
