"""Train and save the forecaster.

    python -m energy_ml.train                      # SYNTHETIC data (default)
    python -m energy_ml.train --etl path/to/<dataset-version>/intervals.jsonl --usd-per-currency 0.73

Synthetic training is only for plumbing and method development. Models trained on it
carry ``trained_on_synthetic=True`` and every bundle they emit is labelled synthetic.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from .data import load_etl_dataset
from .forecaster import QuantileForecastModel
from .integration import DEFAULT_MODEL_PATH
from .synthetic import generate_series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etl", help="intervals.jsonl from the ETL; omit for synthetic data")
    parser.add_argument("--usd-per-currency", type=float, default=None)
    parser.add_argument("--days", type=int, default=21, help="synthetic days to generate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--start", default="2025-12-01", help="synthetic start date; keep it before any evaluation period"
    )
    parser.add_argument("--max-horizon-steps", type=int, default=12)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    cadence = 300.0
    if args.etl:
        frame = load_etl_dataset(args.etl, usd_per_currency=args.usd_per_currency)
        if len(frame) > 1:
            cadence = (frame.index[1] - frame.index[0]).total_seconds()
        synthetic = False
    else:
        start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        frame = generate_series(
            args.days * 288, cadence_seconds=cadence, seed=args.seed, start=start
        )
        synthetic = True

    model = QuantileForecastModel(
        cadence_seconds=cadence,
        max_horizon_steps=args.max_horizon_steps,
        max_iter=args.max_iter,
        trained_on_synthetic=synthetic,
    ).fit(frame)
    path = model.save(args.out)
    print(json.dumps({"saved": str(path), "synthetic": synthetic, **model.metadata}, indent=2))


if __name__ == "__main__":
    main()
