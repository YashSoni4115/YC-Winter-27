"""Calibrated multi-horizon forecasts for the energy decision layer.

Design
------
* One gradient-boosted quantile model per (signal, quantile); the forecast horizon is
  a feature, so a single model serves every step ahead (direct multi-horizon).
* Split-conformal calibration (CQR, Romano et al. 2019) per (signal, horizon) on a
  chronologically LATER slice than the one the models were fit on. Coverage is
  therefore calibrated marginally under exchangeability; real time series are not
  exchangeable, so always re-check empirical coverage on a held-out period.
* ``MLForecastProvider`` implements ``ForecastProviderProtocol``. It keeps a rolling
  history of the observations it has been shown (never anything later) and refuses
  horizons or cadences it was not trained for instead of back-filling.

Documented persistence assumptions (not learned): deferrable fraction, deferrable
deadlines, grid availability/limits and grid stress are carried forward from the
current observation. Public traces do not provide SLAs or outage models.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from energy_decision_layer.contracts import (
    ForecastBundleV1,
    ForecastPointV1,
    ForecastSignal,
    PredictionIntervalV1,
    ProvenanceV1,
    SiteObservationV1,
)

from .features import FEATURE_NAMES, feature_frame, last_row_features, steps_per_day
from .synthetic import AMBIENT, PRICE, SIGNALS, WORKLOAD

MODEL_NAME = "energy-ml-quantile-gbm"
MODEL_VERSION = "0.1.0"
_LOWER_BOUND = {WORKLOAD: 0.0, PRICE: -1.0e6, AMBIENT: -273.15}
_SIGNAL_ENUM = {
    WORKLOAD: ForecastSignal.WORKLOAD_DEMAND,
    PRICE: ForecastSignal.ELECTRICITY_PRICE,
    AMBIENT: ForecastSignal.AMBIENT_TEMPERATURE,
}


@dataclass
class QuantileForecastModel:
    """Trainable, serializable multi-horizon quantile forecaster."""

    cadence_seconds: float = 300.0
    max_horizon_steps: int = 12  # number of forecast points, including the known current one
    lower_quantile: float = 0.1
    upper_quantile: float = 0.9
    max_iter: int = 150
    learning_rate: float = 0.08
    max_depth: int = 6
    min_samples_leaf: int = 40
    random_state: int = 0
    trained_on_synthetic: bool = True
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    conformal_margin: dict[str, np.ndarray] = field(default_factory=dict)
    shrink: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ fitting
    @property
    def nominal_coverage(self) -> float:
        return self.upper_quantile - self.lower_quantile

    @property
    def horizons(self) -> np.ndarray:
        return np.arange(1, self.max_horizon_steps)

    def _validate_frame(self, frame: pd.DataFrame) -> None:
        missing = [s for s in SIGNALS if s not in frame.columns]
        if missing:
            raise ValueError(f"frame is missing signals: {missing}")
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError("frame must have a timezone-aware DatetimeIndex")
        if not frame.index.is_monotonic_increasing:
            raise ValueError("frame must be chronologically sorted")
        gaps = np.diff(frame.index.asi8) / 1e9
        if not np.allclose(gaps, self.cadence_seconds):
            raise ValueError("frame must have a regular index matching cadence_seconds")
        if frame[list(SIGNALS)].isna().any().any():
            raise ValueError("frame contains missing values; impute explicitly before training")
        if self.max_horizon_steps < 2:
            raise ValueError("max_horizon_steps must be at least 2")
        if self.max_horizon_steps > steps_per_day(self.cadence_seconds):
            raise ValueError("max_horizon_steps cannot exceed one day")

    def _new_estimator(self, quantile: float) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )

    def fit(self, frame: pd.DataFrame, calibration_fraction: float = 0.25) -> QuantileForecastModel:
        """Fit on the early part of ``frame``; conformally calibrate on the later part."""
        self._validate_frame(frame)
        if not 0.05 <= calibration_fraction <= 0.5:
            raise ValueError("calibration_fraction must be in [0.05, 0.5]")
        n = len(frame)
        cut = int(n * (1.0 - calibration_fraction))
        if cut < 2 * steps_per_day(self.cadence_seconds) or n - cut < 400:
            raise ValueError("need >= 2 days of training data and >= 400 calibration intervals")

        index = frame.index
        quantiles = {"lower": self.lower_quantile, "point": 0.5, "upper": self.upper_quantile}
        self.models, self.conformal_margin, self.shrink = {}, {}, {}
        for signal in SIGNALS:
            values = frame[signal].to_numpy(dtype=float)
            xs, ys = [], []
            for h in self.horizons:
                feats = feature_frame(values[:cut], index[:cut], int(h), self.cadence_seconds)
                xs.append(feats.iloc[: cut - h])
                ys.append(values[h:cut] - values[: cut - h])  # change from the current value
            x_train, y_train = pd.concat(xs, ignore_index=True), np.concatenate(ys)
            self.models[signal] = {
                name: self._new_estimator(q).fit(x_train, y_train) for name, q in quantiles.items()
            }

            # The calibration slice is split chronologically. First half: choose how
            # much of the predicted change to trust (shrink toward persistence) per
            # horizon. Second half: conformal widening on the shrunk bands.
            mid = cut + (n - cut) // 2
            self.shrink[signal] = np.ones(len(self.horizons))
            weights = np.linspace(0.0, 1.0, 11)
            for i, h in enumerate(self.horizons):
                feats = feature_frame(values, index, int(h), self.cadence_seconds)
                x_sel = feats.iloc[cut : mid - h]
                y_sel = values[cut + h : mid]
                base = x_sel["current"].to_numpy(dtype=float)
                delta = self.models[signal]["point"].predict(x_sel)
                errors = [np.mean(np.abs(y_sel - (base + w * delta))) for w in weights]
                self.shrink[signal][i] = float(weights[int(np.argmin(errors))])

            margins = np.zeros(len(self.horizons))
            for i, h in enumerate(self.horizons):
                feats = feature_frame(values, index, int(h), self.cadence_seconds)
                x_cal = feats.iloc[mid : n - h]
                y_cal = values[mid + h : n]
                lo, _, hi = self._raw_band(signal, x_cal)
                scores = np.maximum(lo - y_cal, y_cal - hi)
                level = min(1.0, np.ceil((len(scores) + 1) * self.nominal_coverage) / len(scores))
                margins[i] = float(np.quantile(scores, level, method="higher"))
            self.conformal_margin[signal] = margins

        self.metadata = {
            "n_train": cut,
            "n_calibration": n - cut,
            "train_end_utc": index[cut - 1].isoformat(),
            "feature_names": list(FEATURE_NAMES),
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
        }
        return self

    # --------------------------------------------------------------- prediction
    def _raw_band(
        self, signal: str, features: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Models predict the CHANGE from the current value, so a forecast starts at
        # persistence and can only move away from it where the data support it.
        m = self.models[signal]
        base = features["current"].to_numpy(dtype=float)
        horizon = features["horizon_steps"].to_numpy(dtype=int)
        weight = self.shrink[signal][horizon - 1] if signal in self.shrink else 1.0
        point = base + weight * m["point"].predict(features)
        lower = np.minimum(base + weight * m["lower"].predict(features), point)
        upper = np.maximum(base + weight * m["upper"].predict(features), point)
        return lower, point, upper

    def _calibrated(
        self, signal: str, features: pd.DataFrame, horizons: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lower, point, upper = self._raw_band(signal, features)
        margin = self.conformal_margin[signal][horizons - 1]
        floor = _LOWER_BOUND[signal]
        point = np.maximum(point, floor)
        lower = np.minimum(np.maximum(lower - margin, floor), point)
        upper = np.maximum(upper + margin, point)
        return lower, point, upper

    def predict_vectorized(
        self, signal: str, values: np.ndarray, index: pd.DatetimeIndex, horizon: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calibrated (lower, point, upper) for every issuance time (offline evaluation)."""
        feats = feature_frame(values, index, horizon, self.cadence_seconds)
        return self._calibrated(signal, feats, np.full(len(feats), horizon))

    def predict_last(
        self, history: dict[str, np.ndarray], last_time: pd.Timestamp
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Calibrated (lower, point, upper) for horizons 1..H-1 from the latest history."""
        horizons = self.horizons
        out = {}
        for signal in SIGNALS:
            feats = last_row_features(history[signal], last_time, horizons, self.cadence_seconds)
            out[signal] = self._calibrated(signal, feats, horizons)
        return out

    # ----------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @staticmethod
    def load(path: str | Path) -> QuantileForecastModel:
        model = joblib.load(path)
        if not isinstance(model, QuantileForecastModel):
            raise TypeError(f"{path} does not contain a QuantileForecastModel")
        return model


class MLForecastProvider:
    """``ForecastProviderProtocol`` implementation backed by ``QuantileForecastModel``."""

    def __init__(self, model: QuantileForecastModel) -> None:
        if not model.models:
            raise ValueError("model has not been fitted")
        self.model = model
        capacity = steps_per_day(model.cadence_seconds) + model.max_horizon_steps + 24
        self._times: deque[pd.Timestamp] = deque(maxlen=capacity)
        self._values: dict[str, deque[float]] = {s: deque(maxlen=capacity) for s in SIGNALS}

    # The provider only ever stores observations it has been handed, i.e. the
    # past and present. Nothing about later intervals is available to it.
    def _ingest(self, o: SiteObservationV1) -> None:
        now = pd.Timestamp(o.timestamp_utc)
        step = timedelta(seconds=self.model.cadence_seconds)
        if self._times:
            last = self._times[-1]
            if now == last:
                return  # repeated call for the same interval
            if now < last or now - last != step:
                self._times.clear()  # new episode or gap: lags would be misleading
                for series in self._values.values():
                    series.clear()
        self._times.append(now)
        self._values[WORKLOAD].append(o.workload_demand_compute_units_per_h)
        self._values[PRICE].append(o.electricity_price_usd_per_mwh)
        self._values[AMBIENT].append(o.ambient_temperature_c)

    def prime(self, past: pd.DataFrame) -> None:
        """Pre-load PAST observations (e.g. yesterday) so lag features exist on day one.

        ``past`` must be a regular, UTC-indexed frame of the three signals that ends
        exactly one cadence before the first observation that will be forecast;
        otherwise the next observation resets the history. Only data from before
        the forecast issuance time may be passed in.
        """
        if not isinstance(past.index, pd.DatetimeIndex) or past.index.tz is None:
            raise ValueError("past must have a timezone-aware DatetimeIndex")
        self._times.clear()
        for series in self._values.values():
            series.clear()
        step = pd.Timedelta(seconds=self.model.cadence_seconds)
        previous = None
        for ts, row in past.iterrows():
            if previous is not None and ts - previous != step:
                raise ValueError("primed history must be regular")
            self._times.append(ts)
            for signal in SIGNALS:
                self._values[signal].append(float(row[signal]))
            previous = ts

    def forecast(
        self,
        observation: SiteObservationV1,
        *,
        horizon_steps: int,
        cadence_seconds: float,
    ) -> ForecastBundleV1:
        m, o = self.model, observation
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be positive")
        if horizon_steps > m.max_horizon_steps:
            raise ValueError(
                f"horizon_steps={horizon_steps} exceeds the trained horizon "
                f"({m.max_horizon_steps}); refusing to extrapolate or back-fill"
            )
        if abs(cadence_seconds - m.cadence_seconds) > 1e-6 or abs(
            o.timestep_seconds - m.cadence_seconds
        ) > 1e-6:
            raise ValueError("cadence does not match the cadence this model was trained for")

        self._ingest(o)
        history = {s: np.fromiter(self._values[s], dtype=float) for s in SIGNALS}
        bands = m.predict_last(history, pd.Timestamp(o.timestamp_utc)) if horizon_steps > 1 else {}

        points = []
        for i in range(horizon_steps):
            target = o.timestamp_utc + timedelta(seconds=i * cadence_seconds)
            common: dict[str, Any] = {
                "target_timestamp_utc": target,
                "deferrable_fraction": o.incoming_deferrable_fraction,
                "deferrable_deadline_steps": o.incoming_deferrable_deadline_steps,
                "grid_available": o.grid_available,
                "grid_import_limit_mw": o.grid_import_limit_mw,
                "grid_export_limit_mw": o.grid_export_limit_mw,
                "grid_stress_status": o.grid_stress_status,
            }
            if i == 0:  # the current interval's exogenous values are already observed
                values = {
                    WORKLOAD: o.workload_demand_compute_units_per_h,
                    PRICE: o.electricity_price_usd_per_mwh,
                    AMBIENT: o.ambient_temperature_c,
                }
                intervals: tuple[PredictionIntervalV1, ...] = ()
            else:
                values, rows = {}, []
                for signal in SIGNALS:
                    lower, point, upper = bands[signal]
                    values[signal] = float(point[i - 1])
                    rows.append(
                        PredictionIntervalV1(
                            signal=_SIGNAL_ENUM[signal],
                            lower_value=float(lower[i - 1]),
                            upper_value=float(upper[i - 1]),
                            coverage_probability=m.nominal_coverage,
                        )
                    )
                intervals = tuple(rows)
            points.append(ForecastPointV1(**common, **values, prediction_intervals=intervals))

        return ForecastBundleV1(
            forecast_id=uuid4(),
            site_id=o.site_id,
            issued_at_utc=o.timestamp_utc,
            valid_from_utc=points[0].target_timestamp_utc,
            valid_until_utc=points[-1].target_timestamp_utc,
            cadence_seconds=cadence_seconds,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            points=tuple(points),
            provenance=ProvenanceV1(
                source_name="energy-ml quantile GBM (+conformal)",
                ingested_at_utc=o.timestamp_utc,
                is_synthetic=m.trained_on_synthetic,
            ),
        )
