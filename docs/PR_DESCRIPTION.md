The repository previously contained the design README only. This implements Devansh's offline simulation, data, baseline planning/dispatch and evaluation foundation so the ML and safety/backend workstreams can integrate against shared versioned interfaces.

- Add modular site physics and Gymnasium environment, explicit power/energy accounting, workload deadlines, battery losses and generator commitment/ramp constraints.
- Add immutable raw storage, validated UTC time-series ETL/Parquet, public-source parsers and synthetic fixtures.
- Add a rolling HiGHS MILP, rule/fixed/callable baselines, independent shield dispatch, decision/outcome audit hooks and reproducible scenario reports.
- Add JSON Schemas/examples, CI and packaging, plus `docs/TEAM_START.md` and `examples/team_smoke.py` for both teammates.

Validation: 148 tests pass with 89.02% branch-aware coverage, including an isolated wheel installation; Ruff lint/format and strict mypy pass; wheel/sdist build; offline ETL, teammate factory smoke and all 32 24-step scenario/policy evaluations pass. Regression checks cover replay after reset, false-truth-value dependency injection, stale forecasts, optimistic forecasts during outages, solver corruption, critical backlog, malformed CSV and audit failure after actuation.

The supplied accept-all shield, persistence forecaster and in-memory logger are explicitly development-only. Real ML, production safety/backend/controller integration and calibrated plant parameters remain teammate/design-partner dependencies. Hosted CI targets Python 3.10 and 3.12; local verification used Python 3.10. No production deployment is included.
