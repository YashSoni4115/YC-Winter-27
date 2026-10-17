"""Collect oracle demonstrations, fit the behaviour-cloning policy, and evaluate it.

    python -m energy_ml.train_policy --workers 2

Requires a trained forecaster (``python -m energy_ml.train``). Evaluation uses the
UNCHANGED simulator/scorer on held-out SYNTHETIC days and compares the learned policy
with Devansh's rule baseline and MILP planner under the same ML forecaster.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from energy_decision_layer.evaluation.runner import evaluate
from energy_decision_layer.optimizer import MilpPolicy, RulePolicy
from energy_decision_layer.sim import SiteConfig

from .benchmark import CONTEXT_STEPS, EPISODE_START, _test_frame
from .forecaster import MLForecastProvider, QuantileForecastModel
from .integration import model_path
from .policy import POLICY_NAME, LearnedPolicy, collect_demonstrations, fit_policy
from .synthetic import to_site_inputs

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def _split(seed: int, steps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = _test_frame(seed, steps)
    return full.iloc[:CONTEXT_STEPS], full.iloc[CONTEXT_STEPS : CONTEXT_STEPS + steps]


def _demo_job(job: tuple[str, int, int]) -> pd.DataFrame:
    forecaster_path, seed, steps = job
    model = QuantileForecastModel.load(forecaster_path)
    return collect_demonstrations(model, [_split(seed, steps)], episode_start=EPISODE_START)


def _eval_job(job: tuple[str, str, int, int]) -> list[dict]:
    forecaster_path, policy_path, seed, steps = job
    model = QuantileForecastModel.load(forecaster_path)
    context, episode = _split(seed, steps)
    inputs = to_site_inputs(episode)
    learned = LearnedPolicy.load(policy_path)
    runs = (
        ("rule baseline", lambda: RulePolicy(SiteConfig())),
        (POLICY_NAME, lambda: learned),
        ("milp + ml forecast", lambda: MilpPolicy(use_prediction_intervals=False)),
    )
    rows = []
    for label, factory in runs:
        provider = MLForecastProvider(model)
        provider.prime(context)
        r = evaluate("synthetic-day", inputs, factory, forecaster=provider, policy_label=label)
        rows.append(
            {
                "seed": seed,
                "policy": label,
                "objective_cost": r.objective_cost,
                "operating_cost": r.total_operating_cost,
                "completion_rate": r.completion_rate,
                "deadline_violations": r.deadline_violations,
                "unserved_energy_mwh": r.unserved_energy_mwh,
                "physical_constraint_events": r.physical_constraint_events,
                "actuator_adjustments": r.actuator_adjustments,
                "policy_latency_seconds": r.policy_latency_seconds,
                "failed": r.failed,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-seeds", type=int, nargs="+", default=list(range(301, 313)))
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[201, 202, 203, 204])
    parser.add_argument("--episode-steps", type=int, default=288)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", default=str(ARTIFACTS / "policy.joblib"))
    args = parser.parse_args()

    forecaster_path = str(model_path())
    if not Path(forecaster_path).exists():
        raise SystemExit(f"train a forecaster first: python -m energy_ml.train ({forecaster_path})")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        parts = list(
            pool.map(_demo_job, [(forecaster_path, s, args.episode_steps) for s in args.demo_seeds])
        )
    truncated = sum(p.attrs.get("truncated_episodes", 0) for p in parts)
    demos = pd.concat(parts, ignore_index=True)
    policy = fit_policy(demos)
    path = policy.save(args.out)
    print(
        f"collected {len(demos)} demonstrations from {len(args.demo_seeds)} episodes "
        f"({truncated} truncated at a shield rejection) -> {path}"
    )

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = [
            row
            for rows in pool.map(
                _eval_job,
                [(forecaster_path, str(path), s, args.episode_steps) for s in args.eval_seeds],
            )
            for row in rows
        ]
    frame = pd.DataFrame(results)
    base = frame[frame.policy == "rule baseline"].set_index("seed")["objective_cost"]
    frame["objective_vs_rule_pct"] = [
        100 * (c / base[s] - 1) for c, s in zip(frame.objective_cost, frame.seed, strict=True)
    ]
    summary = frame.groupby("policy", sort=False).agg(
        mean_objective_cost=("objective_cost", "mean"),
        mean_vs_rule_pct=("objective_vs_rule_pct", "mean"),
        worst_seed_pct=("objective_vs_rule_pct", "max"),
        mean_completion=("completion_rate", "mean"),
        mean_deadline_violations=("deadline_violations", "mean"),
        mean_unserved_mwh=("unserved_energy_mwh", "mean"),
        mean_adjustments=("actuator_adjustments", "mean"),
        mean_policy_latency_s=("policy_latency_seconds", "mean"),
        failed_runs=("failed", "sum"),
    ).round(4)
    out = Path(args.out).parent / "policy_benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs.json").write_text(json.dumps(results, indent=2) + "\n")
    summary.to_csv(out / "summary.csv")
    print(f"eval seeds={args.eval_seeds}, steps={args.episode_steps}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
