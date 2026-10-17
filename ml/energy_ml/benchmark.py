"""Forecast accuracy and downstream control value, on SYNTHETIC data.

    python -m energy_ml.benchmark [--out ml/artifacts/benchmark]

Reports (1) accuracy/coverage by horizon against persistence, and (2) simulator
cost when Devansh's unchanged MILP planner is driven by each forecaster. Scoring,
simulator and planner are not modified. The synthetic generator is the same for
train and test (different random draws), so absolute numbers are optimistic; use
this to compare methods and catch regressions, not to claim real-world gains.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from energy_decision_layer.evaluation.runner import evaluate
from energy_decision_layer.optimizer import MilpPolicy
from energy_decision_layer.testing.doubles import (
    OfflineHindsightForecastDouble,
    PersistenceForecastDouble,
)
from energy_ml.forecaster import MLForecastProvider, QuantileForecastModel
from energy_ml.synthetic import SIGNALS, generate_series, to_site_inputs

TRAIN_START = datetime(2025, 12, 1, tzinfo=timezone.utc)
EPISODE_START = datetime(2026, 1, 1, tzinfo=timezone.utc)  # SiteSimulator's default clock
CONTEXT_STEPS = 2 * 288  # past days available before the episode (for lags only)


def accuracy_table(model: QuantileForecastModel, test: pd.DataFrame) -> list[dict]:
    rows = []
    for signal in SIGNALS:
        values = test[signal].to_numpy()
        for h in (1, 3, 6, 11):
            lo, pt, hi = model.predict_vectorized(signal, values, test.index, h)
            y = values[h:]
            lo, pt, hi = lo[:-h], pt[:-h], hi[:-h]
            persistence = values[:-h]
            usable = slice(CONTEXT_STEPS, None)  # score only rows with full history behind them
            mae_model = float(np.mean(np.abs(pt - y)[usable]))
            mae_persist = float(np.mean(np.abs(persistence - y)[usable]))
            rows.append(
                {
                    "signal": signal,
                    "horizon_steps": h,
                    "mae_model": round(mae_model, 3),
                    "mae_persistence": round(mae_persist, 3),
                    "mae_improvement_pct": round(100 * (1 - mae_model / mae_persist), 1),
                    "coverage": round(float(np.mean(((y >= lo) & (y <= hi))[usable])), 3),
                    "nominal_coverage": model.nominal_coverage,
                    "mean_width": round(float(np.mean((hi - lo)[usable])), 3),
                }
            )
    return rows


def _test_frame(seed: int, steps: int) -> pd.DataFrame:
    n_test = CONTEXT_STEPS + steps + 12
    start = EPISODE_START - timedelta(seconds=300 * CONTEXT_STEPS)
    return generate_series(n_test, seed=seed, start=start)


def control_table(model: QuantileForecastModel, full: pd.DataFrame, steps: int) -> list[dict]:
    episode = full.iloc[CONTEXT_STEPS : CONTEXT_STEPS + steps]
    context = full.iloc[:CONTEXT_STEPS]
    assert episode.index[0].to_pydatetime() == EPISODE_START
    inputs = to_site_inputs(episode)

    def ml(use_bands: bool):
        def make() -> MLForecastProvider:
            provider = MLForecastProvider(model)
            provider.prime(context)
            return provider

        return make, use_bands

    runs = []
    for label, forecaster, bands, hindsight in (
        ("persistence", PersistenceForecastDouble(), True, False),
        ("ml point only", ml(False)[0](), False, False),
        ("ml + calibrated bands", ml(True)[0](), True, False),
        ("ORACLE perfect hindsight (offline bound)",
         OfflineHindsightForecastDouble(tuple(inputs), EPISODE_START), True, True),
    ):
        result = evaluate(
            "synthetic-day",
            inputs,
            lambda b=bands, h=hindsight: MilpPolicy(
                use_prediction_intervals=b, allow_offline_hindsight=h
            ),
            forecaster=forecaster,
            policy_label=label,
        )
        runs.append(asdict(result))
    keep = (
        "policy", "total_operating_cost", "cost_per_compute", "completion_rate",
        "deadline_violations", "unserved_energy_mwh", "peak_grid_mw", "fallback_steps",
        "failed", "failure_reason", "optimization_runtime_seconds",
    )
    return [{k: r[k] for k in keep} for r in runs]


def _run_seed(job: tuple[QuantileForecastModel, int, int]) -> list[dict]:
    model, seed, steps = job
    rows = control_table(model, _test_frame(seed, steps), steps)
    for row in rows:
        row["seed"] = seed
    return rows


def summarize_control(rows: list[dict]) -> pd.DataFrame:
    """Paired comparison: each seed's cost relative to persistence on the same inputs."""
    frame = pd.DataFrame(rows)
    base = frame[frame.policy == "persistence"].set_index("seed")["total_operating_cost"]
    frame["cost_vs_persistence_pct"] = [
        100 * (c / base[s] - 1) for c, s in zip(frame.total_operating_cost, frame.seed, strict=True)
    ]
    grouped = frame.groupby("policy", sort=False).agg(
        mean_cost=("total_operating_cost", "mean"),
        mean_vs_persistence_pct=("cost_vs_persistence_pct", "mean"),
        worst_seed_pct=("cost_vs_persistence_pct", "max"),
        best_seed_pct=("cost_vs_persistence_pct", "min"),
        mean_peak_grid_mw=("peak_grid_mw", "mean"),
        mean_deadline_violations=("deadline_violations", "mean"),
        failed_runs=("failed", "sum"),
        seeds=("seed", "nunique"),
    )
    return grouped.round(3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-days", type=int, default=21)
    parser.add_argument("--episode-steps", type=int, default=288)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--skip-control", action="store_true", help="accuracy table only (fast)")
    parser.add_argument("--control-seeds", type=int, nargs="+", default=[123])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "artifacts" / "benchmark"))
    args = parser.parse_args()

    train = generate_series(args.train_days * 288, seed=0, start=TRAIN_START)
    model = QuantileForecastModel(max_iter=args.max_iter).fit(train)

    accuracy = accuracy_table(model, _test_frame(123, args.episode_steps))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "accuracy.json").write_text(json.dumps(accuracy, indent=2) + "\n")
    print("== forecast accuracy (held-out synthetic period) ==")
    print(pd.DataFrame(accuracy).to_string(index=False))
    if args.skip_control:
        return

    jobs = [(model, seed, args.episode_steps) for seed in args.control_seeds]
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_run_seed, jobs))
    else:
        results = [_run_seed(job) for job in jobs]
    control = [row for rows in results for row in rows]
    (out / "control.json").write_text(json.dumps(control, indent=2) + "\n")
    summary = summarize_control(control)
    summary.to_csv(out / "control_summary.csv")
    print("\n== downstream control: unchanged MILP planner + simulator ==")
    print(f"seeds={args.control_seeds}, episode_steps={args.episode_steps}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
