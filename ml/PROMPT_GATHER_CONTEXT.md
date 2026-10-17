# Prompt: gather context from Devansh's push and continue the ML workstream

Paste everything below the line into a fresh Claude Code session opened at the root of
`YashSoni4115/YC-Winter-27` (after `git pull`).

---

You are joining an in-progress project as the ML engineer. The repository is an autonomous
"energy decision layer" for AI data-centre sites: it jointly allocates power across
compute, cooling, batteries, on-site generation and the grid. Devansh Jain built the
platform (contracts, simulator, planner, evaluator). I (Yash Soni) own the ML work:
forecasting, a learned policy, surrogate models and ML evaluation.

## Step 1: read, do not edit

Read these in order and keep notes as you go:

1. `docs/TEAM_START.md` (ML onboarding and the first acceptance check)
2. `docs/INTEGRATION.md` and `chat.md` (integration contract and decisions)
3. `docs/DATA_SOURCES.md` (what real data exists and its licences)
4. `energy_decision_layer/contracts/` (the Pydantic v1 contracts; note units and sign conventions)
5. `energy_decision_layer/sim/`, `evaluation/runner.py`, `optimizer/planner.py`
6. `examples/team_smoke.py` (the factory-injection hook: `--forecaster module:factory --policy module:factory`)

Then answer in your own words, briefly:

- Exact signatures of `ForecastProviderProtocol` and `PolicyProtocol`.
- Units and sign conventions (power MW, energy MWh, price USD/MWh; battery positive = discharge; grid positive = import; observation workload is a RATE while ETL interval workload is a TOTAL).
- How the planner uses prediction intervals (upper bound for workload, price and ambient; lower bound for grid limits and capacity).
- What `evaluate()` scores and which files I must NOT change (simulator, scorer, contracts).

## Step 2: verify the baseline

Run the project's own checks and report anything failing before touching code:

```
pytest -q tests/contracts tests/integration tests/optimizer
python examples/team_smoke.py
```

## Step 3: check what already exists in `ml/`

If an `ml/` directory exists, it holds the package `energy_ml`. Read `ml/README.md`, then:

```
pip install -e . -e ml
cd ml && pytest -q tests
python -m energy_ml.train
python -m energy_ml.benchmark --control-seeds 201 202 203 204 --episode-steps 288 --workers 2
python -m energy_ml.train_policy --workers 2
cd .. && python examples/team_smoke.py --forecaster energy_ml.integration:create_forecaster
```

Summarise what the benchmark tables say. Be explicit that all training and test data is
SYNTHETIC (same generator, different draws), so results are optimistic and are not
real-world claims.

## Step 4: continue the work (ranked)

1. Replace synthetic training data with the real ETL output (`python -m energy_ml.train --etl <intervals.jsonl> --usd-per-currency <rate>`) once Devansh's dataset exists; re-run accuracy and control benchmarks and compare.
2. Hold-out discipline: chronological splits only; no random shuffles; no feature may use data after the decision time (the parity and leakage tests enforce this, keep them green).
3. Policy: DAgger (roll out the learned policy, relabel visited states with the oracle MILP, retrain), then RL fine-tuning on `SiteEnv`. Do not ship untested RL code.
4. Surrogate models for the planner (thermal and battery response) only if profiling shows the MILP is the latency bottleneck.
5. Report operating cost, completion rate, deadline violations, unserved energy and actuator adjustments against the rule baseline and the MILP planner, with paired per-seed deltas.

## Rules

- Do not edit the simulator, scorer or contracts. Put everything in `ml/`.
- Learned policies are advisory: their proposals still pass through the dispatcher and safety shield.
- Never claim an improvement the paired benchmark does not show; report regressions plainly.
- Use units in names. Keep timestamps UTC.
- No em dashes in prose you write.
