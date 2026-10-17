# Integrated implementation report — 2026-10-17

## A. Completed features

**Yash — machine learning**

- Gradient-boosted quantile models for compute-demand rate, ambient temperature
  and electricity prices, using the existing normalized intervals and units.
- Calendar, lag, rolling-statistic and missingness features; training-only
  imputation; chronological training/calibration/test partitions. Artifact
  availability includes calibration time, preventing calibration-data leakage.
- 80% nominal empirical intervals, persistence/seasonal-naive comparisons,
  MAE/RMSE/coverage/width and measured recursive 1/3/12-step evaluations.
- Labelled grid-stress classifier with accuracy/F1/Brier/AUC when labels/classes
  exist; explicit unavailability otherwise. Synthetic labels are identified.
- Versioned model artifacts and training metadata, safe inference fallback,
  runtime feedback ingestion and direct `ForecastBundleV1` -> MILP integration.
- Simulator-trained CEM controller: a justified low-dimensional learned-control
  approach. Candidate training actions pass the real shield before simulator
  execution. Rewards include service, electricity, thermal, wear, generator,
  grid, terminal battery inventory and unfinished work effects.
- Advisory cooling/thermal-lag, battery-throughput-wear and generator-response
  surrogates, with whole-episode holdouts, physical prediction bounds, domain
  flags and empirical residual bounds. Predictions are exposed in API decision
  previews/dashboard and never influence authoritative safety checks.
- Reproducible training/search settings, checkpoints and evaluation reports.

**Dharm — safety and application**

- Central coupled-constraint shield uses shared site configuration and the
  existing subsystem physics. It validates telemetry, state/time binding,
  battery reserve/power/ramp, generator availability/commitment/ramp, grid
  import/export, compute capacity, cooling capacity/lag/ramp and next temperature.
- LP projection records original constraint violations and exact modified fields;
  independent physical rechecking follows every modification. Missing, malformed,
  nonfinite, stale or infeasible inputs fail closed. Verdicts preserve original
  proposals and never silently imply that rejected commands executed.
- Typed FastAPI endpoints for state, telemetry, forecasts/status, baseline and
  learned previews, safety validation, approved submissions/manual overrides,
  policy selection, history, metrics, saved evaluations, health and readiness.
- SQLite migrations, durable before/after decision audit, complete telemetry and
  simulator checkpoints, idempotency, thread-safe dispatch and canonical-path
  exclusive database ownership. Committed outcomes resume across restart;
  unresolved execution blocks replay. Audit includes model/policy versions,
  original/final actions, reasons, state/forecast snapshots and actual feedback.
- Independent operator/dispatcher tokens; overrides require operator permission.
  Unconfigured deployments are read-only. Physical mode is rejected.
- Five typed mock-controller interfaces prepare commands atomically. The site
  adapter independently verifies the shield result and advances Devansh's same
  simulator, with errors returned and logged before any simulated actuation.
- Responsive same-origin dashboard displays live API values, forecast bands,
  decisions, execution status, history, surrogate advice, scenario comparisons
  and held-out forecast accuracy. Includes loading/empty/stale/error states,
  clear units/timestamps, accessible controls and authorization gating.

**Devansh — preserved and integrated**

The simulator, subsystem models, normalized schemas, ETL, MILP and dispatcher
were retained. No existing physical model or shared contract was rewritten.
The evaluation runner gained an optional execution-adapter factory and now uses
the real safety shield by default. CLI commands and CI were extended; explicitly
named development doubles remain for contract/fault tests and legacy factory
examples. New packages live within `energy_decision_layer`, not another app.

## B. Integration verification

The complete path is:

`RawArtifactStore / CSV ETL -> SimulationIntervalV1 -> SiteInputs -> SiteSimulator`
`-> SiteObservationV1 -> ForecastProvider -> MILP / LearnedPolicy / RulePolicy`
`-> Dispatcher -> SafetyShield -> MockSiteAdapter -> SiteSimulator`
`-> SQLite decision/outcome + telemetry/checkpoint -> FastAPI -> dashboard`.

Manual submissions use the same dispatcher/shield/adapter path. Forecast failure
may trigger an explicit rule fallback for autonomous policies; a manual request
is validated as submitted and is never replaced by a different policy request.
Surrogate predictions attach to recommendation previews as advisory output.

Verification evidence:

- Full 24-step changing-demand/price/weather demo completed. It includes battery
  charging/discharging, generator startup/ramp, a 4 MW grid restriction and an
  intentionally oversized manual action that was modified before execution.
- API/current telemetry, decision counts, outcome metrics and persisted database
  were checked against simulator feedback. A real trained surrogate is exercised
  in integration tests and the updated demo preview.
- Chromium exercised an actual uvicorn server: observer controls disabled,
  operator authentication, preview versus execution, unsafe manual validation,
  modified override execution, trained forecasts after history warm-up, history
  correspondence, evaluation/accuracy tables, mobile layout and network errors.
- Packaged wheel installed outside the source tree successfully served static
  assets and executed baseline -> shield -> adapter -> simulator -> SQLite -> API;
  both database migrations were present and applied.

## C. Testing and measured results

Local environment: Linux/WSL, Python 3.10.12, NumPy 2.1.3, SciPy 1.15.3,
scikit-learn 1.7.2; exact tested constraints in `requirements-dev.lock`.

Commands actually executed (using `.venv/bin/python`):

```bash
python -m pytest -q
python -m pytest -q --cov=energy_decision_layer --cov-report=xml --junitxml=artifacts/junit.xml
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m mypy src/energy_decision_layer
node --check src/energy_decision_layer/service/static/app.js
python -m energy_decision_layer etl examples/synthetic.csv --synthetic --output artifacts/final-etl
python -m energy_decision_layer train-all --output artifacts/models --config configs/training.json
python -m energy_decision_layer demo --steps 24 --output artifacts/demo-delivery
python -m energy_decision_layer evaluate-ml --steps 24 --output artifacts/ml-evaluation
python -m energy_decision_layer benchmark --steps 24 --output artifacts/foundation-final
python examples/team_smoke.py --steps 6
python examples/browser_smoke.py --output artifacts/browser-final
python -m build
```

The initial unchanged suite passed **148 tests**. The integrated suite passed
**198 tests**, with **91.45% branch-aware coverage**, including the safer default shield selection.
Ruff lint/format, strict mypy (46 source files), JavaScript syntax, package build,
and installed-wheel integration smoke all passed. No assertions or coverage gate
were weakened. New tests cover malformed inputs, leakage boundaries, inference
failure, unsafe learned/manual actions, concurrent idempotency, adapter failure,
restart, failed audit commit and database ownership, as well as the full flow.

Non-failing warnings remain: the original Gym physical-unit/unbounded-space
advisories; Starlette's httpx TestClient deprecation; setuptools' legacy license
metadata deprecation during build. These were not hidden or treated as failures.

The browser initially failed to launch because host graphical libraries were
missing. Chromium dependencies were unpacked into ignored local artifacts for
verification without modifying system packages. A later assertion used incorrect
capitalization for the synthetic label; it was corrected to match the displayed
label and strengthened with table-row checks. Browser verification then passed.
For a fresh Linux machine, use Playwright's documented dependency installer.

**24-step comparable evaluation:** 48 runs, **1,122 executed intervals**.
42 runs complete all 24 steps. Six compound no-grid/no-generator runs stop after
19 safe steps when battery reserve makes further auxiliary supply infeasible.
They remain marked `failed: true` in the underlying metrics: a safe stop is not
successful scenario completion. The summary separately records **six safety
stops, zero unexpected failures, zero grid-limit violations and zero thermal-limit
violations**. The shield was not bypassed to finish an impossible scenario.

Example normal-scenario results, all serving 960 compute units:

| Policy | Operating cost USD | Final battery MWh | Final temperature °C |
|---|---:|---:|---:|
| Rule | 1114.42 | 10.00 | 24.02 |
| MILP with learned forecasts | 969.95 | 10.00 | 26.80 |
| Learned controller | 1050.15 | 9.04 | 24.00 |
| Offline rolling hindsight MILP | 900.00 | 10.00 | 26.80 |

These differ in terminal inventory/thermal headroom. The learned policy did not
beat the MILP in this example, and no general superiority or real-world savings
claim is made. Under severe supply restrictions, lower cost may mean more missed
compute; reports retain objective penalties, service and terminal state.

Forecast one-step held-out synthetic test results (nominal interval coverage 80%):

| Signal | MAE | RMSE | Measured coverage |
|---|---:|---:|---:|
| Compute demand, units/h | 11.262 | 13.967 | 91.56% |
| Ambient temperature, °C | 0.254 | 0.318 | 94.29% |
| Electricity price, USD/MWh | 2.734 | 3.464 | 90.82% |

The interval widening is conservative on this fixture, not exactly calibrated to
80%. Full reports include persistence/seasonal baselines and actual recursive
horizon metrics. Synthetic grid-stress labels follow the generated daily pattern;
high classification scores on them do not establish real-event predictive skill.

Surrogate raw held-out MAE: facility temperature **0.0485°C**, cooling **0.0255 MW**,
battery wear **0.0435 USD/interval**, generator output **0.0103 MW**. The reports
identify empirical interval coverage and the pre-physical-bounds evaluation scope.

The general hindsight run is rolling-horizon, not a proven global optimum.
A separately independently solvable one-interval/no-storage/no-generator fixture
has exact objective **37.50 USD**; MILP reproduced it to floating-point tolerance.
Valid regret is reported for that restricted fixture only.

## D. Outstanding issues and limitations

- All default data/site physics are synthetic engineering fixtures. No real
  telemetry, calibrated SOH/ageing labels, production-trained model, physical
  infrastructure validation, or general learned-policy advantage is claimed.
- Existing external source parsers remain available, but real ingestion requires
  source-specific terms/provenance and currency conversion; ERCOT credentials
  are absent. No new real dataset or secret was committed.
- CEM searches a small parametric controller. Recursive/fallback forecast
  intervals and surrogate residual bounds are empirical and lack certified
  out-of-distribution coverage. Battery degradation is a throughput-cost proxy.
- Infeasible sites intentionally stop at the shield. Pending/uncertain execution
  needs deliberate audit recovery; the service does not auto-replay commands.
- Local roles use environment bearer tokens, not enterprise identity. The service
  is a single local simulation process; POSIX file locking requires Linux/macOS
  (Windows users can run it under WSL). Physical mode has no implementation.
- Hosted CI and Python 3.12 execution were not run locally. CI now contains the
  full pipeline and Chromium job but has not been pushed or deployed.

Recommended next work: calibrate site physics and workload economics with a
reviewed real dataset; test forecast distribution shifts and scenario breadth;
agree on comparable terminal inventory/SLA objectives; integrate enterprise
identity and controller safeguards before proposing any real plant connection.

## E. Run locally

```bash
cd /home/devansh/YCW27/YC-Winter-27
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-dev.lock -e '.[dev]' build
python -m energy_decision_layer train-all --seed 7 --config configs/training.json
python -m energy_decision_layer demo --steps 24 --output artifacts/my-demo
python -m energy_decision_layer evaluate-ml --steps 24
python -m energy_decision_layer serve --models artifacts/models --database artifacts/operator.sqlite3
```

Open **http://127.0.0.1:8000**. Commands are disabled until optional distinct local
`EDL_OPERATOR_TOKEN` / `EDL_DISPATCHER_TOKEN` values are configured and supplied.
They grant simulated actions only. For repeat demos, choose a fresh output
folder. For a distinct scenario, choose a new database.

```bash
python -m pytest -q --cov=energy_decision_layer
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m mypy src/energy_decision_layer
python -m pip install -c requirements-dev.lock -e '.[browser]'
python -m playwright install --with-deps chromium
python examples/browser_smoke.py --output artifacts/my-browser-check
python -m build
```

See [SYSTEM.md](SYSTEM.md) for APIs, data ingestion and environment variables,
[ML.md](ML.md) for methodology, and generated `artifacts/ml-evaluation/summary.md`
for the complete measured comparison. Logs and machine-readable check outcomes are
in `artifacts/verification/`; browser screenshots/results are in
`artifacts/browser-final/`, and the final full demo is in
`artifacts/demo-delivery/`. No remote push or production deployment
was performed.
