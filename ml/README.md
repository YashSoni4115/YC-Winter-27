# energy_ml (Yash's ML workstream)

Forecasting, a learned policy and ML evaluation for the energy decision layer. It plugs into
Devansh's platform only through the public protocols; nothing in the simulator, scorer or
contracts is modified.

## Setup
    pip install -e . -e ml
    cd ml && pytest -q tests

## Commands
    python -m energy_ml.train                       # synthetic by default; --etl intervals.jsonl --usd-per-currency X for real data
    python -m energy_ml.benchmark --control-seeds 201 202 --episode-steps 288 --workers 2
    python -m energy_ml.train_policy --workers 2
    python examples/team_smoke.py --forecaster energy_ml.integration:create_forecaster --policy energy_ml.integration:create_policy

## What is here
- `forecaster.py`: direct multi-horizon quantile gradient boosting (0.1/0.5/0.9) on change-from-current targets, per-horizon shrinkage toward persistence, split-conformal calibration. `MLForecastProvider` implements `ForecastProviderProtocol`.
- `features.py`: leakage-safe features with a training/serving parity test.
- `policy.py`, `train_policy.py`: behaviour cloning of the hindsight-oracle MILP (features from the observation and ML forecast only). Implements `PolicyProtocol`; proposals still go through the shield.
- `data.py`: loader for the ETL `SimulationIntervalV1` JSONL (explicit FX rate required for non-USD).
- `PROMPT_GATHER_CONTEXT.md`: prompt to paste into a Claude Code session to gather context from the repo.

## Results (SYNTHETIC data only, optimistic)
Forecast MAE vs persistence (held-out synthetic period): workload +3% at 1 step to +34% at 11 steps; price 0% to +40%; ambient +30% to +67%. 80% interval coverage 0.77 to 0.86.

Control, 4 held-out synthetic days (seeds 201 to 204, 288 steps), after rebasing onto Devansh's
second push (fail-closed shield). Objective cost, paired per seed vs rule baseline:
| seed | rule | behaviour cloning | MILP + ML forecast |
|---|---|---|---|
| 201 | 11761 | 11357 (-3.4%) | run failed: shield rejection |
| 202 | 10695 | 10624 (-0.7%) | 10187 (-4.7%) |
| 203 | 11782 | 10488 (-11.0%) | 10121 (-14.1%) |
| 204 | 10640 | 10251 (-3.7%) | run failed: shield rejection |

Behaviour cloning: mean -4.7%, worst seed -0.7%, completion 0.999, 0 unserved MWh, 0 failed runs.
MILP + ML forecast failed 2 of 4 runs, so its mean over all seeds is not comparable (failed runs stop early).
Demonstrations: 12 oracle episodes, 7 truncated at a shield rejection (executed prefix kept).

## Limitations
- Synthetic train and test come from the same generator; real-data results will differ.
- Only 4 evaluation days; treat as a smoke result, not a finding.
- The shield now fails closed when no feasible coupled action exists. The hindsight MILP and the MILP planner both hit this on some synthetic days; needs investigation with Devansh (planner vs shield thermal model mismatch is the likely cause).
- Behaviour cloning does not beat the MILP, has slightly more unserved energy, and compounds errors off-distribution. Next: DAgger, then RL on `SiteEnv`.
- Calibrated-interval planning vs point forecasts was not benchmarked across seeds (the run hung and was stopped).
- Devansh's `src/energy_decision_layer/ml/` overlaps with this package and should be reconciled.
