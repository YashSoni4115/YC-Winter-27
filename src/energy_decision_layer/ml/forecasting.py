"""Chronological gradient-boosted quantile models on the canonical ETL schema.

Only past rows enter lag features. Missing features are forward-filled from the
past; residual missing values use medians fitted on the training split. Artifacts
use pickle and must only be loaded from a trusted local training run.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import sklearn
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score

from energy_decision_layer.contracts import (
    ForecastBundleV1,
    ForecastPointV1,
    ForecastSignal,
    GridStressStatus,
    PredictionIntervalV1,
    ProvenanceV1,
    SiteObservationV1,
)
from energy_decision_layer.data.models import QualityFlag, SimulationIntervalV1

SIGNALS = (
    ForecastSignal.WORKLOAD_DEMAND,
    ForecastSignal.AMBIENT_TEMPERATURE,
    ForecastSignal.ELECTRICITY_PRICE,
)
Array = NDArray[np.float64]


def interval_values(row: SimulationIntervalV1) -> Array:
    if row.price_currency not in {None, "USD"}:
        raise ValueError("Forecast training requires explicit USD conversion before ingestion")
    if row.quality_mask & int(QualityFlag.OUT_OF_PHYSICAL_RANGE | QualityFlag.SOURCE_QC_FAILED):
        return np.full(3, np.nan)
    return np.array(
        [
            row.workload_demand_compute_units / row.duration_hours
            if row.workload_demand_compute_units is not None
            else np.nan,
            row.ambient_temperature_c,
            row.grid_import_price_per_mwh,
        ],
        dtype=float,
    )


def features(history: Array, target: datetime, season: int) -> Array:
    """Calendar plus strictly past lags/rolling means and missingness flags."""
    hour = target.hour + target.minute / 60
    values: list[float] = [
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * target.weekday() / 7),
        np.cos(2 * np.pi * target.weekday() / 7),
    ]
    for column in range(3):
        past = history[:, column]
        valid = past[np.isfinite(past)]
        latest = float(valid[-1]) if len(valid) else np.nan
        for lag in (1, 2, 3, 12, season):
            v = float(past[-lag]) if len(past) >= lag else np.nan
            values.extend([v if np.isfinite(v) else latest, float(not np.isfinite(v))])
        for window in (3, 12, season):
            w = past[-window:]
            w = w[np.isfinite(w)]
            values.extend(
                [float(np.mean(w)) if len(w) else latest, float(np.std(w)) if len(w) else 0.0]
            )
    return np.array(values, dtype=float)


def regression_metrics(
    actual: Array, nominal: Array, lower: Array | None = None, upper: Array | None = None
) -> dict[str, Any]:
    if not len(actual):
        return {"status": "unavailable: no observed targets"}
    result = {
        "mae": float(np.mean(np.abs(actual - nominal))),
        "rmse": float(np.sqrt(np.mean((actual - nominal) ** 2))),
    }
    if lower is not None and upper is not None:
        result.update(
            coverage=float(np.mean((actual >= lower) & (actual <= upper))),
            mean_interval_width=float(np.mean(upper - lower)),
        )
    return result


class ForecastModel:
    """One-step quantile predictors, recursively rolled out at inference."""

    def __init__(self) -> None:
        self.models: list[list[Any]] = []
        self.classifier: Any = None
        self.report: dict[str, Any] = {}
        self.version = "untrained"
        self.medians: Array = np.empty(0)
        self.cadence_seconds = 300.0
        self.season = 288
        self.training_end = datetime.min.replace(tzinfo=timezone.utc)
        self.available_after = self.training_end
        self.synthetic = True
        self.site_id = ""
        self.residual_bounds = np.zeros(3)

    @classmethod
    def fit(
        cls, rows: Sequence[SimulationIntervalV1], *, seed: int = 42, n_estimators: int = 40
    ) -> ForecastModel:
        if len(rows) < 60:
            raise ValueError("At least 60 chronological intervals are required")
        m = cls()
        ordered = list(rows)
        m.site_id = ordered[0].site_id
        m.cadence_seconds = ordered[0].duration_hours * 3600
        m.season = max(1, round(86400 / m.cadence_seconds))
        for i, row in enumerate(ordered):
            if (
                row.site_id != m.site_id
                or abs(row.duration_hours * 3600 - m.cadence_seconds) > 1e-6
            ):
                raise ValueError("Training rows must share site and regular cadence")
            if i and ordered[i - 1].interval_end_utc != row.interval_start_utc:
                raise ValueError("Training rows must be sorted, contiguous, and unique")
        y = np.array([interval_values(r) for r in ordered])
        x = np.array(
            [features(y[:i], ordered[i].interval_start_utc, m.season) for i in range(1, len(rows))]
        )
        y = y[1:]
        train_end, valid_end = int(len(y) * 0.6), int(len(y) * 0.8)
        # Targets from validation/test never enter fitted preprocessing or models.
        m.medians = np.array(
            [
                float(np.median(v[np.isfinite(v)])) if np.isfinite(v).any() else 0.0
                for v in x[:train_end].T
            ]
        )
        x = np.where(np.isfinite(x), x, m.medians)
        m.training_end = ordered[train_end].interval_end_utc
        m.available_after = ordered[valid_end].interval_end_utc
        m.synthetic = any(
            bool(r.quality_mask & int(QualityFlag.SYNTHETIC))
            or bool(r.metadata.get("is_synthetic"))
            for r in ordered
        )
        m.report = {
            "seed": seed,
            "data_kind": "synthetic" if m.synthetic else "ingested",
            "method": "gradient-boosted quantile regression; recursive horizon",
            "split": {
                "train": train_end,
                "validation": valid_end - train_end,
                "test": len(y) - valid_end,
            },
            "training_end_utc": m.training_end.isoformat(),
            "calibration_end_utc": m.available_after.isoformat(),
            "available_after_utc": m.available_after.isoformat(),
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "n_estimators": n_estimators,
            "quantiles": [0.1, 0.5, 0.9],
            "cadence_seconds": m.cadence_seconds,
            "coverage_probability": 0.8,
            "interval_limitations": (
                "One-step empirical held-out calibration, not a probabilistic guarantee; "
                "recursive multi-step intervals need horizon-specific calibration on site data. "
                "Persistence fallback bands are heuristic."
            ),
            "signals": {},
        }
        for j, signal in enumerate(SIGNALS):
            mask = np.isfinite(y[:train_end, j])
            if mask.sum() < 20:
                raise ValueError(f"Insufficient observed training labels for {signal.value}")
            models = [
                GradientBoostingRegressor(
                    loss="quantile",
                    alpha=q,
                    n_estimators=n_estimators,
                    max_depth=2,
                    random_state=seed,
                ).fit(x[:train_end][mask], y[:train_end, j][mask])
                for q in (0.1, 0.5, 0.9)
            ]
            m.models.append(models)
            vm = np.isfinite(y[train_end:valid_end, j])
            vp = np.sort(
                np.array([model.predict(x[train_end:valid_end]) for model in models]), axis=0
            )[1]
            m.residual_bounds[j] = (
                float(np.quantile(np.abs(y[train_end:valid_end, j][vm] - vp[vm]), 0.9))
                if vm.any()
                else 0.0
            )
            scores: dict[str, Any] = {}
            for label, lo, hi in (
                ("validation", train_end, valid_end),
                ("test", valid_end, len(y)),
            ):
                mask = np.isfinite(y[lo:hi, j])
                predictions = np.sort(
                    np.array([model.predict(x[lo:hi]) for model in models]), axis=0
                )
                # Calibration uses validation only; report both raw and deployed test bands.
                raw = regression_metrics(
                    y[lo:hi, j][mask],
                    predictions[1][mask],
                    predictions[0][mask],
                    predictions[2][mask],
                )
                if label == "test":
                    predictions[0] = np.minimum(
                        predictions[0], predictions[1] - m.residual_bounds[j]
                    )
                    predictions[2] = np.maximum(
                        predictions[2], predictions[1] + m.residual_bounds[j]
                    )
                    if j == 0:
                        predictions = np.maximum(predictions, 0.0)
                    if j == 1:
                        predictions = np.maximum(predictions, -273.15)
                scores[label] = regression_metrics(
                    y[lo:hi, j][mask],
                    predictions[1][mask],
                    predictions[0][mask],
                    predictions[2][mask],
                )
                scores[label]["raw_quantile_metrics"] = raw
                scores[label]["interval_kind"] = (
                    "validation-calibrated empirical"
                    if label == "test"
                    else "raw quantiles (calibration set)"
                )
                # Persistence and seasonal naive use only measurements preceding target.
                persistence = x[lo:hi, 4 + j * 16]
                seasonal = x[lo:hi, 4 + j * 16 + 8]
                scores[label]["persistence"] = regression_metrics(
                    y[lo:hi, j][mask], persistence[mask]
                )
                scores[label]["seasonal_naive"] = regression_metrics(
                    y[lo:hi, j][mask], seasonal[mask]
                )
            m.report["signals"][signal.value] = scores
        known = np.array(
            [
                r.grid_stress_status != "unknown"
                and not r.quality_mask
                & int(QualityFlag.OUT_OF_PHYSICAL_RANGE | QualityFlag.SOURCE_QC_FAILED)
                for r in ordered[1:]
            ]
        )
        labels = np.array(
            [r.grid_stress_status not in {"normal", "unknown"} for r in ordered[1:]], dtype=int
        )
        train_known = known[:train_end]
        if len(np.unique(labels[:train_end][train_known])) == 2:
            m.classifier = GradientBoostingClassifier(
                n_estimators=n_estimators, max_depth=2, random_state=seed
            ).fit(x[:train_end][train_known], labels[:train_end][train_known])
            tk = known[valid_end:]
            if tk.any():
                probability = m.classifier.predict_proba(x[valid_end:][tk])[:, 1]
                truth = labels[valid_end:][tk]
                m.report["grid_stress"] = {
                    "status": "trained",
                    "labels": "non-normal known stress status",
                    "accuracy": float(accuracy_score(truth, probability >= 0.5)),
                    "f1": float(f1_score(truth, probability >= 0.5, zero_division=0)),
                    "brier": float(brier_score_loss(truth, probability)),
                    "roc_auc": float(roc_auc_score(truth, probability))
                    if len(np.unique(truth)) == 2
                    else None,
                }
        if "grid_stress" not in m.report:
            m.report["grid_stress"] = {"status": "unavailable: insufficient labelled classes"}
        fingerprint = "".join(r.model_dump_json() for r in ordered)
        m.report["data_sha256"] = hashlib.sha256(fingerprint.encode()).hexdigest()
        m.version = hashlib.sha256(
            (fingerprint + str(seed) + str(n_estimators)).encode()
        ).hexdigest()[:16]
        m.report["version"] = m.version
        m.report["multi_horizon_test"] = m.evaluate_horizons(ordered)
        return m

    def predict(self, history: Array, target: datetime) -> tuple[Array, float | None]:
        if target < self.available_after:
            raise ValueError("Model was trained after forecast issuance")
        x = features(history, target, self.season)
        x = np.where(np.isfinite(x), x, self.medians).reshape(1, -1)
        values = np.sort(
            np.array([[float(reg.predict(x)[0]) for reg in models] for models in self.models]),
            axis=1,
        )
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite model predictions")
        # Validation residual widening is empirical, not a safety guarantee.
        values[:, 0] = np.minimum(values[:, 0], values[:, 1] - self.residual_bounds)
        values[:, 2] = np.maximum(values[:, 2], values[:, 1] + self.residual_bounds)
        values[0] = np.maximum(0.0, values[0])
        values[1] = np.maximum(-273.15, values[1])
        stress = float(self.classifier.predict_proba(x)[0, 1]) if self.classifier else None
        return values, stress

    def evaluate_horizons(
        self,
        rows: Sequence[SimulationIntervalV1],
        *,
        horizons: tuple[int, ...] = (1, 3, 12),
        max_origins: int = 64,
    ) -> dict[str, Any]:
        """Rolling-origin held-out evaluation; recursion cannot see future observations.

        At each issuance, the current interval is already observed, as it is in
        SiteObservationV1. Horizon 1 predicts the NEXT interval, not that nowcast.
        Origins are sampled deterministically to bound evaluation runtime.
        """
        if not horizons or min(horizons) < 1 or max(horizons) > 288 or max_origins < 1:
            raise ValueError("Positive horizons <=288 and origin count required")
        values = np.array([interval_values(row) for row in rows])
        origins = [
            i
            for i, row in enumerate(rows)
            if row.interval_start_utc >= self.available_after and i + max(horizons) < len(rows)
        ]
        if len(origins) > max_origins:
            origins = [origins[int(i)] for i in np.linspace(0, len(origins) - 1, max_origins)]
        if not origins:
            return {"status": "unavailable: insufficient held-out horizon history"}
        actual: dict[int, list[Array]] = {h: [] for h in horizons}
        predicted: dict[int, list[Array]] = {h: [] for h in horizons}
        persistence: dict[int, list[Array]] = {h: [] for h in horizons}
        seasonal: dict[int, list[Array]] = {h: [] for h in horizons}
        for origin in origins:
            past = values[: origin + 1].copy()
            current = np.array(
                [col[np.isfinite(col)][-1] if np.isfinite(col).any() else 0.0 for col in past.T]
            )
            for h in range(1, max(horizons) + 1):
                prediction, _ = self.predict(past, rows[origin + h].interval_start_utc)
                if h in horizons:
                    actual[h].append(values[origin + h])
                    predicted[h].append(prediction)
                    persistence[h].append(current)
                    seasonal_index = origin + h - self.season
                    # Fall back to persistence if no previous day is observable at issuance.
                    sv = values[seasonal_index] if 0 <= seasonal_index <= origin else current
                    seasonal[h].append(np.where(np.isfinite(sv), sv, current))
                past = np.vstack([past, prediction[:, 1]])
        report: dict[str, Any] = {
            "issuance_origins": len(origins),
            "origin_sampling": "deterministic evenly spaced",
            "horizon_semantics": "steps after observed current interval",
            "interval_kind": "recursive quantiles with validation one-step residual widening",
            "horizons": {},
        }
        for h in horizons:
            a, p = np.array(actual[h]), np.array(predicted[h])
            scores = {}
            for j, signal in enumerate(SIGNALS):
                valid = np.isfinite(a[:, j])
                score = regression_metrics(
                    a[valid, j], p[valid, j, 1], p[valid, j, 0], p[valid, j, 2]
                )
                score["persistence"] = regression_metrics(
                    a[valid, j], np.array(persistence[h])[valid, j]
                )
                score["seasonal_naive"] = regression_metrics(
                    a[valid, j], np.array(seasonal[h])[valid, j]
                )
                scores[signal.value] = score
            report["horizons"][str(h)] = scores
        return report

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.pkl").write_bytes(pickle.dumps(self))
        (path / "metadata.json").write_text(
            json.dumps(self.report, indent=2, allow_nan=False) + "\n"
        )

    @classmethod
    def load(cls, directory: str | Path) -> ForecastModel:
        result = pickle.loads((Path(directory) / "model.pkl").read_bytes())
        if not isinstance(result, cls):
            raise ValueError("Not a forecast model artifact")
        return result


class ForecastProvider:
    """Stateful protocol implementation; no access to future history rows."""

    def __init__(
        self, model: ForecastModel | None = None, history: Sequence[SimulationIntervalV1] = ()
    ) -> None:
        self.model = model
        self.history = tuple(history)
        self.observed: dict[datetime, Array] = {}
        self.last_status = "uninitialized"
        self.last_stress_probabilities: list[float | None] = []

    def forecast(
        self, observation: SiteObservationV1, *, horizon_steps: int, cadence_seconds: float
    ) -> ForecastBundleV1:
        o = observation
        if not 1 <= horizon_steps <= 288 or cadence_seconds != o.timestep_seconds:
            raise ValueError("Horizon must be 1..288 with observation cadence")
        current = np.array(
            [
                o.workload_demand_compute_units_per_h,
                o.ambient_temperature_c,
                o.electricity_price_usd_per_mwh,
            ]
        )
        self.observed[o.timestamp_utc] = current
        past = {
            r.interval_start_utc: interval_values(r)
            for r in self.history
            if r.site_id == o.site_id and r.interval_end_utc <= o.timestamp_utc
        }
        past.update({t: v for t, v in self.observed.items() if t <= o.timestamp_utc})
        # Limit history to a contiguous suffix: a missing interval must not compress lags.
        sequence = [current]
        t = o.timestamp_utc - timedelta(seconds=cadence_seconds)
        while t in past and len(sequence) < 576:
            sequence.insert(0, past[t])
            t -= timedelta(seconds=cadence_seconds)
        history = np.array(sequence)
        valid_model = (
            self.model is not None
            and len(history) >= 3
            and self.model.cadence_seconds == cadence_seconds
            and self.model.site_id == o.site_id
            and self.model.available_after <= o.timestamp_utc
        )
        status = "trained" if valid_model else "persistence-fallback: insufficient model/history"
        points = []
        stress_values: list[float | None] = []
        for i in range(horizon_steps):
            target = o.timestamp_utc + timedelta(seconds=i * cadence_seconds)
            stress = None
            if i == 0:
                values = np.column_stack([current, current, current])
            elif valid_model and self.model is not None:
                try:
                    values, stress = self.model.predict(history, target)
                except Exception:
                    valid_model = False
                    status = "persistence-fallback: inference failure"
                    values = np.column_stack([history[-1]] * 3)
            else:
                values = np.column_stack([history[-1]] * 3)
            if i and not valid_model:
                width = np.maximum(np.abs(values[:, 1]) * 0.2, [10.0, 2.0, 10.0]) * np.sqrt(i)
                values[:, 0], values[:, 2] = values[:, 1] - width, values[:, 1] + width
                values[0] = np.maximum(values[0], 0.0)
                values[1] = np.maximum(values[1], -273.15)
            point = ForecastPointV1(
                target_timestamp_utc=target,
                workload_demand_compute_units_per_h=float(values[0, 1]),
                ambient_temperature_c=float(values[1, 1]),
                electricity_price_usd_per_mwh=float(values[2, 1]),
                active_compute_capacity_compute_units_per_h=o.active_compute_capacity_compute_units_per_h,
                deferrable_fraction=o.incoming_deferrable_fraction,
                deferrable_deadline_steps=o.incoming_deferrable_deadline_steps,
                grid_available=o.grid_available,
                grid_import_limit_mw=o.grid_import_limit_mw,
                grid_export_limit_mw=o.grid_export_limit_mw,
                grid_stress_status=(
                    GridStressStatus.ELEVATED if stress >= 0.5 else GridStressStatus.NORMAL
                )
                if stress is not None
                else o.grid_stress_status,
                prediction_intervals=tuple(
                    PredictionIntervalV1(
                        signal=signal,
                        lower_value=float(values[j, 0]),
                        upper_value=float(values[j, 2]),
                        coverage_probability=0.8,
                    )
                    for j, signal in enumerate(SIGNALS)
                ),
            )
            points.append(point)
            stress_values.append(stress)
            if i:
                history = np.vstack([history, values[:, 1]])
        self.last_status, self.last_stress_probabilities = status, stress_values
        return ForecastBundleV1(
            forecast_id=uuid4(),
            site_id=o.site_id,
            issued_at_utc=o.timestamp_utc,
            valid_from_utc=points[0].target_timestamp_utc,
            valid_until_utc=points[-1].target_timestamp_utc,
            cadence_seconds=cadence_seconds,
            model_name="gradient-boosted-quantiles" if status == "trained" else status,
            model_version=self.model.version
            if self.model and status == "trained"
            else "fallback-v1",
            points=tuple(points),
            provenance=ProvenanceV1(
                source_name=status,
                ingested_at_utc=o.timestamp_utc,
                is_synthetic=self.model.synthetic if self.model else True,
            ),
        )


def train_forecast(
    rows: Sequence[SimulationIntervalV1],
    output: str | Path,
    *,
    seed: int = 42,
    n_estimators: int = 40,
) -> dict[str, Any]:
    model = ForecastModel.fit(rows, seed=seed, n_estimators=n_estimators)
    model.save(output)
    return model.report
