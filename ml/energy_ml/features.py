"""Leakage-safe features shared by training and serving.

Every feature for an issuance at index ``t`` uses only values at indices ``<= t``
(the target's calendar position is used only as a deterministic clock feature).
Training uses the vectorized ``feature_frame``; serving uses ``last_row_features``
on the in-memory history. ``tests/test_features.py`` asserts they agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES: tuple[str, ...] = (
    "horizon_steps",
    "current",
    "lag1",
    "lag12",
    "diff1",
    "mean12",
    "std12",
    "yesterday_at_target",
    "yesterday_change",
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
)

_MIN_ROLLING = 3
_WINDOW = 12


def steps_per_day(cadence_seconds: float) -> int:
    steps = 86400.0 / cadence_seconds
    if abs(steps - round(steps)) > 1e-9:
        raise ValueError("cadence must divide one day evenly")
    return int(round(steps))


def _clock(target_times: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    seconds = (
        target_times.hour * 3600 + target_times.minute * 60 + target_times.second
    ).to_numpy(dtype=float)
    tod = 2 * np.pi * seconds / 86400.0
    dow = 2 * np.pi * target_times.dayofweek.to_numpy(dtype=float) / 7.0
    return {
        "tod_sin": np.sin(tod),
        "tod_cos": np.cos(tod),
        "dow_sin": np.sin(dow),
        "dow_cos": np.cos(dow),
        "is_weekend": (target_times.dayofweek.to_numpy() >= 5).astype(float),
    }


def feature_frame(
    values: np.ndarray, index: pd.DatetimeIndex, horizon: int, cadence_seconds: float
) -> pd.DataFrame:
    """Features for every issuance time ``t`` predicting ``values[t + horizon]`` (vectorized)."""
    if horizon < 1:
        raise ValueError("horizon must be at least one step")
    day = steps_per_day(cadence_seconds)
    if horizon > day:
        raise ValueError("horizon cannot exceed one day")
    series = pd.Series(np.asarray(values, dtype=float))
    rolling = series.rolling(_WINDOW, min_periods=_MIN_ROLLING)
    target_times = index + pd.Timedelta(seconds=horizon * cadence_seconds)
    yesterday_at_target = series.shift(day - horizon)  # value at t + h - day (<= t - 1)
    yesterday_now = series.shift(day)
    columns = {
        "horizon_steps": np.full(len(series), float(horizon)),
        "current": series.to_numpy(),
        "lag1": series.shift(1).to_numpy(),
        "lag12": series.shift(_WINDOW).to_numpy(),
        "diff1": (series - series.shift(1)).to_numpy(),
        "mean12": rolling.mean().to_numpy(),
        "std12": rolling.std().to_numpy(),
        "yesterday_at_target": yesterday_at_target.to_numpy(),
        "yesterday_change": (yesterday_at_target - yesterday_now).to_numpy(),
        **_clock(target_times),
    }
    return pd.DataFrame(columns, columns=list(FEATURE_NAMES))


def last_row_features(
    values: np.ndarray,
    last_time: pd.Timestamp,
    horizons: np.ndarray,
    cadence_seconds: float,
) -> pd.DataFrame:
    """Features for the most recent index of ``values`` at each horizon (serving path)."""
    v = np.asarray(values, dtype=float)
    m = len(v)
    if m < 1:
        raise ValueError("history must contain the current value")
    day = steps_per_day(cadence_seconds)
    horizons = np.asarray(horizons, dtype=int)
    nan = float("nan")
    cur = v[-1]
    lag1 = v[-2] if m >= 2 else nan
    lag12 = v[-1 - _WINDOW] if m >= 1 + _WINDOW else nan
    window = v[-min(m, _WINDOW) :]
    if len(window) >= _MIN_ROLLING:
        mean12 = float(window.mean())
        std12 = float(window.std(ddof=1))
    else:
        mean12 = std12 = nan
    yday_now = v[m - 1 - day] if m - 1 - day >= 0 else nan

    yday_target = np.full(len(horizons), nan)
    for i, h in enumerate(horizons):
        position = m - 1 + int(h) - day
        if position >= 0:
            yday_target[i] = v[position]
    target_times = pd.DatetimeIndex(
        [last_time + pd.Timedelta(seconds=int(h) * cadence_seconds) for h in horizons]
    )
    columns = {
        "horizon_steps": horizons.astype(float),
        "current": np.full(len(horizons), cur),
        "lag1": np.full(len(horizons), lag1),
        "lag12": np.full(len(horizons), lag12),
        "diff1": np.full(len(horizons), cur - lag1 if m >= 2 else nan),
        "mean12": np.full(len(horizons), mean12),
        "std12": np.full(len(horizons), std12),
        "yesterday_at_target": yday_target,
        "yesterday_change": yday_target - yday_now,
        **_clock(target_times),
    }
    return pd.DataFrame(columns, columns=list(FEATURE_NAMES))
