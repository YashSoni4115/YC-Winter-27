"""Offline demo, ETL, schema export and reproducible benchmark commands."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--output", default="artifacts/benchmark")
    benchmark.add_argument("--steps", type=int, default=24)
    etl = commands.add_parser("etl")
    etl.add_argument("csv", type=Path)
    etl.add_argument("--output", default="artifacts/dataset")
    etl.add_argument("--synthetic", action="store_true")
    etl.add_argument("--terms", type=Path)
    etl.add_argument("--source-url")
    schema = commands.add_parser("schemas")
    schema.add_argument("--output", default="docs/schemas")
    synthetic = commands.add_parser("synthetic", help="generate labelled normalized fixture data")
    synthetic.add_argument("--output", type=Path, default=Path("artifacts/synthetic.jsonl"))
    synthetic.add_argument("--steps", type=int, default=2016)
    synthetic.add_argument("--seed", type=int, default=7)
    for name in ("train-all", "train-forecast", "train-policy", "train-surrogates"):
        train = commands.add_parser(name)
        train.add_argument("--data", type=Path, help="normalized SimulationIntervalV1 JSONL")
        train.add_argument("--output", type=Path, default=Path("artifacts/models"))
        train.add_argument("--steps", type=int, default=2016, help="synthetic training intervals")
        train.add_argument("--seed", type=int, default=7)
        train.add_argument("--config", type=Path)
    evaluation = commands.add_parser("evaluate-ml")
    evaluation.add_argument("--models", type=Path, default=Path("artifacts/models"))
    evaluation.add_argument("--output", type=Path, default=Path("artifacts/ml-evaluation"))
    evaluation.add_argument("--steps", type=int, default=24)
    evaluation.add_argument("--seed", type=int, default=7)
    demo = commands.add_parser("demo")
    demo.add_argument("--models", type=Path, default=Path("artifacts/models"))
    demo.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    demo.add_argument("--steps", type=int, default=24)
    demo.add_argument("--seed", type=int, default=7)
    serve = commands.add_parser("serve")
    serve.add_argument("--models", type=Path, default=Path("artifacts/models"))
    serve.add_argument("--database", type=Path, default=Path("artifacts/decisions.sqlite3"))
    serve.add_argument("--data", type=Path)
    serve.add_argument("--steps", type=int, default=288)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "synthetic":
        from .workflows import generate_dataset

        rows = generate_dataset(args.output, steps=args.steps, seed=args.seed)
        print(f"{len(rows)} labelled synthetic intervals written to {args.output}")
    elif args.command.startswith("train-"):
        from .workflows import train_components

        result = train_components(
            args.command,
            args.output,
            data=args.data,
            steps=args.steps,
            seed=args.seed,
            config=args.config,
        )
        print(json.dumps(result, indent=2, allow_nan=False))
    elif args.command == "evaluate-ml":
        from .evaluation.integrated import run_ml_evaluation

        report = run_ml_evaluation(args.models, args.output, steps=args.steps, seed=args.seed)
        print(f"Evaluation report: {args.output}/summary.md; failed runs={report['failed_runs']}")
        if report["unexpected_failures"]:
            raise SystemExit("unexpected evaluation failures; inspect ml-evaluation.json")
    elif args.command == "demo":
        from .workflows import run_demo

        report = run_demo(args.models, args.output, steps=args.steps, seed=args.seed)
        print(f"Integrated simulation completed {report['steps']} steps: {args.output}/demo.json")
    elif args.command == "serve":
        import uvicorn

        from .service.api import create_app
        from .workflows import create_runtime

        runtime = create_runtime(args.models, args.database, steps=args.steps, data=args.data)
        try:
            uvicorn.run(create_app(runtime), host=args.host, port=args.port, workers=1)
        finally:
            runtime.close()
    elif args.command == "benchmark":
        from .evaluation.policies import FixedPolicy
        from .evaluation.runner import evaluate, scenarios, write_reports
        from .optimizer import MilpPolicy, RulePolicy
        from .safety import SafetyShield
        from .sim import SiteConfig
        from .testing.doubles import OfflineHindsightForecastDouble

        results = []
        for name, inputs in scenarios(args.steps).items():
            for policy in (RulePolicy, MilpPolicy, FixedPolicy):
                results.append(evaluate(name, inputs, policy))
            oracle = OfflineHindsightForecastDouble(
                inputs, datetime(2026, 1, 1, tzinfo=timezone.utc)
            )
            results.append(
                evaluate(
                    name,
                    inputs,
                    lambda: MilpPolicy(allow_offline_hindsight=True),
                    forecaster=oracle,
                    policy_label="OFFLINE-HINDSIGHT-MILP",
                )
            )
            print(f"Completed scenario: {name}", flush=True)
        write_reports(results, args.output)
        metadata = {
            "seed": 7,
            "config": asdict(SiteConfig()),
            "synthetic": True,
            "horizon_steps": 12,
            "scenario_steps": args.steps,
            "shield": SafetyShield.name,
            "safety_stopped_runs": sum(r.failed and r.shield_rejections > 0 for r in results),
            "unexpected_failures": sum(r.failed and r.shield_rejections == 0 for r in results),
        }
        (Path(args.output) / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"{len(results)} evaluations written to {args.output}")
        if metadata["safety_stopped_runs"]:
            print(f"{metadata['safety_stopped_runs']} runs stopped by safety; inspect metrics.json")
        if metadata["unexpected_failures"]:
            raise SystemExit("one or more evaluations failed; inspect metrics.json")
    elif args.command == "etl":
        import hashlib

        from .data.artifacts import RawArtifactStore
        from .data.models import RedistributionStatus, SourceTerms
        from .data.pipeline import CsvMapping, parse_csv, to_site_inputs, write_dataset

        payload = args.csv.read_bytes()
        if not args.synthetic and (args.terms is None or not args.source_url):
            parser.error(
                "external ETL requires --terms and --source-url, or explicitly --synthetic"
            )
        terms = (
            SourceTerms.model_validate_json(args.terms.read_bytes())
            if args.terms
            else SourceTerms(
                terms_url="urn:project:synthetic-fixture",
                retrieved_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
                terms_sha256=hashlib.sha256(b"project-generated synthetic fixture").hexdigest(),
                redistribution_status=RedistributionStatus.ALLOWED,
                required_attribution="YCW27 synthetic fixture; no retrieved observations",
            )
        )
        store = RawArtifactStore(Path(args.output) / "raw")
        source = "synthetic.fixture" if args.synthetic else "local.external"
        digest = hashlib.sha256(payload).hexdigest()
        manifest = store.root / source / digest / "manifest.json"
        if manifest.exists():
            existing = store.verify(manifest)
            if existing.terms != terms or existing.canonical_url != (
                args.source_url or "urn:synthetic:example"
            ):
                raise ValueError("existing raw artifact has conflicting terms or source URL")
        else:
            store.put(
                payload,
                source_id=source,
                product_id="mapped-csv-v1",
                canonical_url=args.source_url or "urn:synthetic:example",
                retrieved_at_utc=datetime.now(timezone.utc),
                content_type="text/csv",
                parser_version="1.0.0",
                terms=terms,
            )
        rows, report = parse_csv(
            payload,
            site_id="synthetic-site",
            mapping=CsvMapping(synthetic=args.synthetic),
        )
        to_site_inputs(rows)
        write_dataset(rows, report, args.output)
        print(json.dumps(report, indent=2))
    else:
        from . import contracts
        from .data.models import SimulationIntervalV1

        target = Path(args.output)
        target.mkdir(parents=True, exist_ok=True)
        models: list[type[BaseModel]] = [
            contracts.TelemetryV1,
            contracts.SiteObservationV1,
            contracts.ForecastBundleV1,
            contracts.ProposedControlActionV1,
            contracts.ExecutedControlActionV1,
            contracts.SafetyResponseV1,
            SimulationIntervalV1,
        ]
        for model in models:
            (target / f"{model.__name__}.json").write_text(
                json.dumps(model.model_json_schema(), indent=2) + "\n"
            )
        print(f"{len(models)} schemas written to {target}")


if __name__ == "__main__":
    main()
