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
    args = parser.parse_args()
    if args.command == "benchmark":
        from .evaluation.policies import FixedPolicy
        from .evaluation.runner import evaluate, scenarios, write_reports
        from .optimizer import MilpPolicy, RulePolicy
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
            "shield": "DEVELOPMENT-ACCEPT-ALL-NOT-SAFE",
        }
        (Path(args.output) / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"{len(results)} evaluations written to {args.output}")
        if any(r.failed for r in results):
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
