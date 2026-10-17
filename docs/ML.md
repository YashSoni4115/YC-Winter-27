# Learned components and reproducibility

The `energy_decision_layer.ml` package consumes Devansh's sole normalized
`SimulationIntervalV1` schema and emits the existing forecast/proposal contracts.
It does not introduce competing telemetry, action or simulator schemas. MW/MWh,
USD/MWh and degrees Celsius remain the system units. ETL workload totals are
converted to useful compute units/hour before training. CAD prices must be
explicitly converted upstream; the model rejects a non-USD price.

## Forecasting

`ForecastModel.fit(rows, seed=42)` fits separate gradient-boosted 10th, 50th and
90th quantile models for demand, ambient temperature and electricity price.
Features include UTC daily/weekly calendar cycles, lags 1/2/3/12 and one day,
rolling means/stddevs and missing-value indicators. Each training target sees
only preceding rows. Missing lags use preceding valid readings; remaining
missing features use medians fitted to training only. Missing or failed-QC labels
are not converted to zero. Entirely missing holdout signal groups report an
explicit unavailable status.

The chronological split is 60% training, 20% calibration/validation and 20% test.
Training weights and feature medians use the training partition only. Validation
absolute residuals widen quantile bands. The resulting artifact is unavailable
until the calibration partition ends: `training_end_utc`, `calibration_end_utc`
and `available_after_utc` distinguish the timestamps. Test labels never tune
weights or calibration. The report includes MAE, RMSE, interval coverage/width,
persistence and seasonal-naive comparisons; test bands reflect the deployed
one-step calibration. Validation metrics show raw quantiles. `multi_horizon_test`
additionally measures actual recursive 1/3/12-step forecasts at up to 64
deterministically sampled held-out issuance times, with persistence and
seasonal-naive comparisons at each horizon. This reports measured horizon
calibration rather than assuming one-step coverage transfers to a full horizon. Reports record
seed, parameters, data digest, Python/sklearn versions and synthetic provenance.

`ForecastProvider(model, history=rows)` implements the real optimizer protocol.
It discards rows ending after issuance, keeps only a contiguous history suffix,
and absorbs new simulator observations every dispatch cycle. Its first point is
the measured current interval. Later points roll the one-step model forward
recursively. Model/site/cadence/cutoff mismatch, insufficient history or inference
failure produces explicitly labelled persistence fallback with heuristic bands.
No interval coverage guarantee is asserted for recursive horizons or fallback
bands. Training uses calendar values in UTC; local holiday effects are absent.

Known `grid_stress_status` labels train a binary gradient-boosted classifier when
the training split contains both normal and stressed examples. Unknown labels
are excluded. Evaluation includes F1, accuracy, Brier score and ROC AUC when both
test classes exist. `last_stress_probabilities` exposes probabilities alongside
the latest bundle; future forecast points carry the predicted NORMAL/ELEVATED
status. Current telemetry determines the nowcast status and grid limits.
A predicted status does not invent a future outage or import-cap reduction.

Artifacts consist of `model.pkl` and `metadata.json`. **Load pickle artifacts only
from trusted local training runs.** Models are not trained on production site
measurements by default.

## Learned dispatch

`LearnedPolicy` uses a five-parameter price/SOC/cooling controller. Reproducible
cross-entropy search trains the parameters on Devansh's `SiteSimulator`. This
simulation optimization is a learned-control alternative to deep RL for the
small deterministic action space; it makes no policy superiority claim.

Each candidate uses `SafetyShield.evaluate`, and only its authorized
`executed_action` reaches `SiteSimulator.step`. There is no direct physics bypass
in policy training or operational dispatch. The score includes the simulator's
electricity/fuel/start/shutdown/wear/missed-compute objective, explicit thermal,
unserved-energy and grid-limit penalties, a terminal battery replacement cost
and pending-work penalty. These terminal terms prevent free initial-energy or
unfinished-queue exploitation. Training and validation use disjoint contiguous
input partitions and fresh identical initial states. Root evaluation compares
rule, MILP, learned and explicitly labelled rolling oracle policies on matching
seeded scenarios.

`LearnedPolicy.load(path, config=...)` accepts a directory containing `policy.json`
or a JSON file and rejects a checkpoint trained for another site configuration.
A newly constructed policy has `untrained-default` version and is labelled as an
untrained default. A trained checkpoint has a `cem-...` version and includes its
parameters, site configuration, seed and iteration scores. A proposal remains
unauthorized until the independent shield approves it.

## Dynamics surrogates

`train_surrogates(inputs, output, config=..., seed=...)` collects varied controller
trajectories from the existing simulator through the real shield. Extra-trees
regression estimates next facility temperature, realized cooling power, battery
wear cost and generator output. Features include previous thermal/cooling state,
ambient temperature, battery energy/power, generator output/timers/availability
and proposed powers. Whole episodes are split between train/validation/test;
nearby transitions from one episode do not leak across partitions.

`DynamicsSurrogate.load(directory).predict(observation, proposal)` returns named
values with held-out 95th-percentile absolute-residual bounds. It marks inputs
outside its fitted domain and rejects invalid physical states. Predictions and
intervals obey one-step physical ranges, including cooling lag/ramp, generator
outage/timers/ramp, and battery throughput-derived degradation limits. Temperature
is bounded by reachable physics, not clipped to a safety threshold. Test reports
explicitly measure the raw regressor before these inference bounds.

These estimates are advisory and are never used to override authoritative safety
physics. Battery wear is the simulator's linear throughput cost proxy. There is
no unsupported calibrated SOH, temperature-aging, real-engine or chiller model.
Bounds are empirical, not certified protections.

## Commands

From the repository after `python -m pip install -e '.[dev]'`:

```bash
python -m energy_decision_layer synthetic --output artifacts/synthetic.jsonl --steps 2016 --seed 7
python -m energy_decision_layer train-all --data artifacts/synthetic.jsonl --output artifacts/models --seed 7
python -m energy_decision_layer evaluate-ml --models artifacts/models --output artifacts/ml-evaluation --steps 24 --seed 7
python -m energy_decision_layer demo --models artifacts/models --output artifacts/demo --steps 24 --seed 7
python -m pytest tests/ml -q
```

`train-forecast`, `train-policy` and `train-surrogates` train individual components.
`--config configs/training.json` supplies the reproducible training configuration (see
the checked-in file for current fields). Normalized input JSONL can come from
existing ETL artifacts after explicit terms, quality and currency handling.
The default generated fixtures are conspicuously synthetic and need no private
credentials. Models, raw data and evaluation artifacts belong under ignored
`artifacts/`; no generated checkpoint is required in source control.
