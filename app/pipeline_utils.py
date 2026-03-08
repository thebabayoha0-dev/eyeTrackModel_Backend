from __future__ import annotations

import ast
import json
import math
import os
import re
import sys
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


TASK_FILE_ORDER: List[Tuple[str, str]] = [
    ("antisaccade", "antisaccade_features.csv"),
    ("fixation", "fixation_features.csv"),
    ("prosaccade", "prosaccade_features.csv"),
    ("smooth_pursuit", "smooth_pursuit_features.csv"),
]


def safe_joblib_load(path: str):
    """Load joblib models across NumPy 1.x / 2.x internal module name changes.

    Some pickles reference private NumPy module paths such as
    ``numpy._core.multiarray`` while others use ``numpy.core.multiarray``.
    On mismatched environments, direct attribute access like
    ``np._core.multiarray`` can fail. Importing the modules explicitly and
    aliasing them in ``sys.modules`` is more reliable.
    """
    import importlib

    aliases = {
        "numpy._core": ["numpy._core", "numpy.core"],
        "numpy._core.multiarray": ["numpy._core.multiarray", "numpy.core.multiarray"],
        "numpy._core._multiarray_umath": ["numpy._core._multiarray_umath", "numpy.core._multiarray_umath"],
    }

    for alias_name, candidates in aliases.items():
        for candidate in candidates:
            try:
                sys.modules[alias_name] = importlib.import_module(candidate)
                break
            except Exception:
                continue

    try:
        return joblib.load(path)
    except Exception as exc:
        numpy_ver = getattr(np, "__version__", "unknown")
        try:
            import sklearn
            sklearn_ver = getattr(sklearn, "__version__", "unknown")
        except Exception:
            sklearn_ver = "unknown"
        raise RuntimeError(
            "Model load failed. This usually means the runtime NumPy / scikit-learn "
            f"versions do not match the versions used when the model was saved. "
            f"Current runtime: numpy={numpy_ver}, scikit-learn={sklearn_ver}. "
            "Recreate the virtual environment and install the pinned requirements. "
            f"Original error: {exc}"
        ) from exc


@lru_cache(maxsize=2)
def load_model(model_path: str):
    return safe_joblib_load(model_path)


@lru_cache(maxsize=4)
def extract_expected_feature_names_from_model_file(model_path: str) -> List[str]:
    """Fallback: extract feature names directly from the joblib binary.

    This is helpful when the model does not expose feature_names_in_ at the
    top-level pipeline object.
    """
    with open(model_path, "rb") as f:
        data = f.read().decode("latin1", errors="ignore")
    pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:__[A-Za-z0-9_]+){2,}")
    return sorted(set(pattern.findall(data)))


def get_model_expected_features(model: Any, model_path: str) -> List[str]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    named_steps = getattr(model, "named_steps", {})
    for step_name in ("imputer", "preprocessor", "rf"):
        step = named_steps.get(step_name)
        if step is not None and hasattr(step, "feature_names_in_"):
            return list(step.feature_names_in_)

    return extract_expected_feature_names_from_model_file(model_path)


def _coerce_jsonish(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, float) and math.isnan(value):
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        for parser in (
            lambda s: json.loads(s),
            lambda s: json.loads(s.replace("'", '"')),
            ast.literal_eval,
        ):
            try:
                parsed = parser(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
    return {}


def _boolish_to_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return 1.0
    if text in {"false", "0", "no", "n"}:
        return 0.0
    try:
        return float(text)
    except Exception:
        return float("nan")


def _num(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _with_payload_dict(events_df: pd.DataFrame) -> pd.DataFrame:
    df = events_df.copy()
    if "payload" in df.columns:
        df["_payload_dict"] = df["payload"].apply(_coerce_jsonish)
    else:
        df["_payload_dict"] = [{} for _ in range(len(df))]
    return df


def _payload_get(row: pd.Series, key: str, default: Any = None) -> Any:
    payload = row.get("_payload_dict", {})
    if isinstance(payload, dict):
        return payload.get(key, default)
    return default


def clean_events_dataframe(events_df: pd.DataFrame) -> pd.DataFrame:
    df = _with_payload_dict(events_df)
    if "t_ms" not in df.columns:
        for alt in ("t", "timestamp_ms", "ts_ms", "time_ms"):
            if alt in df.columns:
                df = df.rename(columns={alt: "t_ms"})
                break
    if "name" not in df.columns and "type" in df.columns:
        df = df.rename(columns={"type": "name"})
    df["t_ms"] = pd.to_numeric(df.get("t_ms"), errors="coerce")
    return df.sort_values("t_ms").reset_index(drop=True)


def clean_gaze_dataframe(gaze_df: pd.DataFrame) -> pd.DataFrame:
    df = gaze_df.copy()
    for alt in ("t", "timestamp_ms", "ts_ms", "time_ms"):
        if "t_ms" not in df.columns and alt in df.columns:
            df = df.rename(columns={alt: "t_ms"})
    for col in ("t_ms", "x", "y"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    if "valid" in df.columns:
        valid_norm = df["valid"].apply(
            lambda v: v if isinstance(v, bool) else str(v).strip().lower() in {"true", "1", "yes"}
        )
        df["valid"] = valid_norm.astype(bool)
    else:
        df["valid"] = True

    return df.sort_values("t_ms").reset_index(drop=True)


def _game_window(events_df: pd.DataFrame, game_num: int) -> Optional[Tuple[float, float]]:
    starts = events_df[
        (events_df["name"] == "game_start")
        & (events_df["_payload_dict"].apply(lambda d: d.get("game") == game_num))
    ]
    ends = events_df[
        (events_df["name"] == "game_end")
        & (events_df["_payload_dict"].apply(lambda d: d.get("game") == game_num))
    ]

    if not starts.empty and not ends.empty:
        return float(starts.iloc[0]["t_ms"]), float(ends.iloc[-1]["t_ms"])

    task_events = events_df[events_df["_payload_dict"].apply(lambda d: d.get("game") == game_num)]
    if task_events.empty:
        return None
    return float(task_events["t_ms"].min()), float(task_events["t_ms"].max())


def _peak_speed_for_window(gaze_df: pd.DataFrame, start_ms: float, end_ms: float) -> float:
    seg = gaze_df[(gaze_df["t_ms"] >= start_ms) & (gaze_df["t_ms"] <= end_ms) & (gaze_df["valid"])]
    seg = seg.dropna(subset=["t_ms", "x", "y"]).sort_values("t_ms")
    if len(seg) < 2:
        return float("nan")

    dt = np.diff(seg["t_ms"].to_numpy())
    dx = np.diff(seg["x"].to_numpy())
    dy = np.diff(seg["y"].to_numpy())
    valid = dt > 0
    if not np.any(valid):
        return float("nan")

    speed_px_per_sec = np.sqrt(dx[valid] ** 2 + dy[valid] ** 2) / (dt[valid] / 1000.0)
    if len(speed_px_per_sec) == 0:
        return float("nan")
    return float(np.nanmax(speed_px_per_sec))


def _build_trial_features(
    events_df: pd.DataFrame,
    gaze_df: pd.DataFrame,
    game_num: int,
    response_name: str = "response",
    fixed_window_ms: float = 1200.0,
) -> pd.DataFrame:
    target_df = events_df[
        (events_df["name"] == "target_on")
        & (events_df["_payload_dict"].apply(lambda d: d.get("game") == game_num))
    ].copy()

    if target_df.empty:
        return pd.DataFrame(columns=["trial_id", "correct", "peak_speed"])

    target_df["trial_id"] = target_df["_payload_dict"].apply(lambda d: d.get("trial_id"))
    target_df["target_on_t_ms"] = pd.to_numeric(target_df["t_ms"], errors="coerce")
    target_df = target_df.sort_values("target_on_t_ms").reset_index(drop=True)
    target_df["next_target_t_ms"] = target_df["target_on_t_ms"].shift(-1)

    response_df = events_df[
        (events_df["name"] == response_name)
        & (events_df["_payload_dict"].apply(lambda d: d.get("game") == game_num))
    ].copy()
    if response_df.empty:
        response_df = pd.DataFrame(columns=["trial_id", "response_t_ms", "rt_ms", "correct"])
    else:
        response_df["trial_id"] = response_df["_payload_dict"].apply(lambda d: d.get("trial_id"))
        response_df["response_t_ms"] = pd.to_numeric(response_df["t_ms"], errors="coerce")
        response_df["rt_ms"] = response_df["_payload_dict"].apply(lambda d: _num(d.get("rt_ms")))
        response_df["correct"] = response_df["_payload_dict"].apply(lambda d: _boolish_to_float(d.get("correct")))
        response_df = response_df[["trial_id", "response_t_ms", "rt_ms", "correct"]]

    trial_end_df = events_df[
        (events_df["name"] == "trial_end")
        & (events_df["_payload_dict"].apply(lambda d: d.get("game") == game_num))
    ].copy()
    if trial_end_df.empty:
        trial_end_df = pd.DataFrame(columns=["trial_id", "trial_end_t_ms", "end_correct"])
    else:
        trial_end_df["trial_id"] = trial_end_df["_payload_dict"].apply(lambda d: d.get("trial_id"))
        trial_end_df["trial_end_t_ms"] = pd.to_numeric(trial_end_df["t_ms"], errors="coerce")
        trial_end_df["end_correct"] = trial_end_df["_payload_dict"].apply(lambda d: _boolish_to_float(d.get("correct")))
        trial_end_df = trial_end_df[["trial_id", "trial_end_t_ms", "end_correct"]]

    merged = target_df[["trial_id", "target_on_t_ms", "next_target_t_ms"]].merge(
        response_df, on="trial_id", how="left"
    ).merge(trial_end_df, on="trial_id", how="left")

    def peak_speed(row: pd.Series) -> float:
        start_ms = _num(row["target_on_t_ms"])
        end_candidates = [start_ms + fixed_window_ms]
        for candidate in (row.get("response_t_ms"), row.get("trial_end_t_ms"), row.get("next_target_t_ms")):
            val = _num(candidate)
            if not math.isnan(val):
                end_candidates.append(val)
        end_ms = min([x for x in end_candidates if not math.isnan(x)])
        if math.isnan(start_ms) or math.isnan(end_ms) or end_ms <= start_ms:
            return float("nan")
        return _peak_speed_for_window(gaze_df, start_ms, end_ms)

    merged["correct"] = merged["correct"].where(~pd.isna(merged["correct"]), merged.get("end_correct"))
    merged["peak_speed"] = merged.apply(peak_speed, axis=1)
    return merged[["trial_id", "correct", "peak_speed"]]


def build_fixation_features(events_df: pd.DataFrame, gaze_df: pd.DataFrame) -> pd.DataFrame:
    window = _game_window(events_df, 4)
    if window is None:
        return pd.DataFrame(columns=["t0_ms", "t1_ms", "n", "dispersion", "pct_missing"])

    t0_ms, t1_ms = window
    game_gaze = gaze_df[(gaze_df["t_ms"] >= t0_ms) & (gaze_df["t_ms"] <= t1_ms)].copy()
    valid = game_gaze[game_gaze["valid"]].dropna(subset=["x", "y"]).copy()
    total_count = int(len(game_gaze))
    valid_count = int(len(valid))

    pct_missing = float("nan")
    if total_count > 0:
        pct_missing = float(1.0 - (valid_count / total_count))

    dispersion = float("nan")
    if valid_count > 0:
        center_x = float(valid["x"].median())
        center_y = float(valid["y"].median())
        radial = np.sqrt((valid["x"] - center_x) ** 2 + (valid["y"] - center_y) ** 2)
        dispersion = float(radial.mean())

    return pd.DataFrame(
        [
            {
                "t0_ms": float(t0_ms),
                "t1_ms": float(t1_ms),
                "n": float(valid_count),
                "dispersion": dispersion,
                "pct_missing": pct_missing,
            }
        ]
    )


def build_smooth_pursuit_features(events_df: pd.DataFrame, gaze_df: pd.DataFrame) -> pd.DataFrame:
    window = _game_window(events_df, 5)
    if window is None:
        return pd.DataFrame(columns=["t0_ms", "t1_ms", "n_samples", "corr_x", "mae", "rmse"])

    t0_ms, t1_ms = window
    pursuit_events = events_df[events_df["name"] == "pursuit_target_pos"].copy()
    target_rows: List[Dict[str, float]] = []

    for _, row in pursuit_events.iterrows():
        payload = row.get("_payload_dict", {})
        if not isinstance(payload, dict):
            continue
        tx = payload.get("screen_x", payload.get("x"))
        local_t = payload.get("t_ms")
        event_t = row.get("t_ms")
        if tx is None or pd.isna(tx):
            continue

        target_t_ms = _num(event_t)
        local_t_ms = _num(local_t)
        if not math.isnan(local_t_ms):
            target_t_ms = float(t0_ms + local_t_ms)

        target_rows.append({"t_ms": float(target_t_ms), "tx": float(tx)})

    target_df = pd.DataFrame(target_rows).dropna().sort_values("t_ms")
    game_gaze = gaze_df[(gaze_df["t_ms"] >= t0_ms) & (gaze_df["t_ms"] <= t1_ms) & (gaze_df["valid"])]
    game_gaze = game_gaze.dropna(subset=["x"]).sort_values("t_ms")

    corr_x = mae = rmse = float("nan")
    if len(game_gaze) > 1 and len(target_df) > 1:
        gaze_times = game_gaze["t_ms"].to_numpy(dtype=float)
        gaze_x = game_gaze["x"].to_numpy(dtype=float)
        target_x = np.interp(gaze_times, target_df["t_ms"].to_numpy(dtype=float), target_df["tx"].to_numpy(dtype=float))

        diff = gaze_x - target_x
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        if np.std(gaze_x) > 1e-6 and np.std(target_x) > 1e-6:
            corr_x = float(np.corrcoef(gaze_x, target_x)[0, 1])

    return pd.DataFrame(
        [
            {
                "t0_ms": float(t0_ms),
                "t1_ms": float(t1_ms),
                "n_samples": float(len(game_gaze)),
                "corr_x": corr_x,
                "mae": mae,
                "rmse": rmse,
            }
        ]
    )


def extract_task_feature_tables(events_df: pd.DataFrame, gaze_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    events_clean = clean_events_dataframe(events_df)
    gaze_clean = clean_gaze_dataframe(gaze_df)

    return {
        "antisaccade": _build_trial_features(events_clean, gaze_clean, game_num=3),
        "fixation": build_fixation_features(events_clean, gaze_clean),
        "prosaccade": _build_trial_features(events_clean, gaze_clean, game_num=2),
        "smooth_pursuit": build_smooth_pursuit_features(events_clean, gaze_clean),
    }


def summarize_feature_table(df: pd.DataFrame, prefix: str) -> Dict[str, float]:
    """Mirror the notebook training logic exactly, but tolerate object columns that are numeric in practice."""
    work = df.copy()
    for col in ["trial_id", "id", "timestamp", "time"]:
        if col in work.columns:
            work = work.drop(columns=[col])

    for col in work.columns:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    numeric_df = work.select_dtypes(include=[np.number]).copy()
    if numeric_df.shape[1] == 0:
        return {}

    features: Dict[str, float] = {}
    for col in numeric_df.columns:
        series = pd.to_numeric(numeric_df[col], errors="coerce").astype(float)
        features[f"{prefix}__{col}__mean"] = float(series.mean()) if series.notna().any() else float("nan")
        features[f"{prefix}__{col}__std"] = float(series.std(ddof=0)) if series.notna().any() else float("nan")
        features[f"{prefix}__{col}__min"] = float(series.min()) if series.notna().any() else float("nan")
        features[f"{prefix}__{col}__max"] = float(series.max()) if series.notna().any() else float("nan")
    return features


def build_summary_row(task_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    row: Dict[str, float] = {}
    for task_name, _filename in TASK_FILE_ORDER:
        row.update(summarize_feature_table(task_tables.get(task_name, pd.DataFrame()), task_name))
    return pd.DataFrame([row])


def prepare_features_for_model(feature_df: pd.DataFrame, model: Any, model_path: str) -> pd.DataFrame:
    X = feature_df.copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    expected = get_model_expected_features(model, model_path)
    for col in expected:
        if col not in X.columns:
            X[col] = np.nan
    X = X[expected]

    if X.shape[1] == 0:
        raise ValueError("No usable model features after alignment.")
    return X


def _score_to_severity(score: int) -> str:
    if score <= 4:
        return "No ADHD"
    if score == 5:
        return "Mild ADHD"
    if score <= 7:
        return "ADHD"
    return "High ADHD"


def _build_prediction_result(model: Any, X: pd.DataFrame, warnings: Optional[List[str]] = None) -> Dict[str, Any]:
    pred = model.predict(X)
    label = int(pred[0])

    adhd_conf = 0.5
    control_conf = 0.5
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0]
        classes = list(getattr(model, "classes_", [0, 1]))
        adhd_idx = classes.index(1) if 1 in classes else min(1, len(prob) - 1)
        control_idx = classes.index(0) if 0 in classes else 0
        adhd_conf = float(prob[adhd_idx])
        control_conf = float(prob[control_idx])

    score = max(0, min(10, round(adhd_conf * 10)))
    return {
        "prediction": "ADHD" if label == 1 else "Control",
        "confidence": round(adhd_conf if label == 1 else control_conf, 4),
        "score": score,
        "severity": _score_to_severity(score),
        "warnings": warnings or [],
    }


def collect_pipeline_warnings(task_tables: Dict[str, pd.DataFrame]) -> List[str]:
    warnings: List[str] = []

    pros = task_tables.get("prosaccade", pd.DataFrame())
    anti = task_tables.get("antisaccade", pd.DataFrame())
    fix = task_tables.get("fixation", pd.DataFrame())
    pursuit = task_tables.get("smooth_pursuit", pd.DataFrame())

    if pros.empty:
        warnings.append("Prosaccade features are empty.")
    elif pros["peak_speed"].notna().sum() == 0:
        warnings.append("Prosaccade peak_speed could not be computed from the gaze stream.")

    if anti.empty:
        warnings.append("Antisaccade features are empty.")
    elif anti["peak_speed"].notna().sum() == 0:
        warnings.append("Antisaccade peak_speed could not be computed from the gaze stream.")

    if fix.empty:
        warnings.append("Fixation window is missing.")
    elif float(fix.iloc[0].get("n", 0)) <= 0:
        warnings.append("Fixation contains zero valid gaze samples.")

    if pursuit.empty:
        warnings.append("Smooth pursuit window is missing.")
    elif float(pursuit.iloc[0].get("n_samples", 0)) <= 0:
        warnings.append("Smooth pursuit contains zero valid gaze samples.")

    return warnings


def run_classification_from_summary(feature_df: pd.DataFrame, model_path: str) -> Dict[str, Any]:
    model = load_model(model_path)
    X = prepare_features_for_model(feature_df, model, model_path)
    return _build_prediction_result(model, X)


def run_end_to_end_from_raw(
    events_df: pd.DataFrame,
    gaze_df: pd.DataFrame,
    model_path: str,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    task_tables = extract_task_feature_tables(events_df, gaze_df)
    summary_df = build_summary_row(task_tables)
    model = load_model(model_path)
    X = prepare_features_for_model(summary_df, model, model_path)
    warnings = collect_pipeline_warnings(task_tables)
    result = _build_prediction_result(model, X, warnings=warnings)
    return task_tables, summary_df, result


def load_csv_upload(file_bytes: bytes) -> pd.DataFrame:
    if not file_bytes:
        return pd.DataFrame()
    return pd.read_csv(BytesIO(file_bytes))
