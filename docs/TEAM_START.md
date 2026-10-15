# First checkpoint for the other two workstreams

Devansh's side supplies an installable package, the physics/Gym environment, data schemas/pipeline, baseline policies/planner, a dispatcher, evaluation metrics, schema examples, offline integration doubles and CI. Both teammates can start against these interfaces immediately. Their production logic remains theirs to implement.

## Shared setup

Start from this PR's branch (`devanshs-branch`) until it is merged into `main`; use your own feature branch for your implementation. Install the tested dependencies and establish a green baseline:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-dev.lock -e '.[dev]' build
python -m pytest -q
python examples/team_smoke.py --steps 6
```

The smoke script runs a complete offline forecast -> planner -> shield -> simulator -> audit -> evaluation exchange. Its default shield is explicitly DEVELOPMENT accept-all; it is not suitable for equipment control. Factory arguments use importable `module:factory` callables taking no arguments and returning one component. If your component needs settings or a trained model, expose a factory that loads your own configuration.

## ML teammate

1. Import `ForecastProviderProtocol`, `ForecastBundleV1`, `ForecastPointV1`, `PredictionIntervalV1`, `SiteObservationV1`, `QueuedWorkloadV1` and `PolicyProtocol` from `energy_decision_layer.contracts`.
2. Implement `forecast(observation, *, horizon_steps, cadence_seconds)` with same-site UTC issuance/horizon/cadence. Include explicitly calibrated uncertainty bands and deferrable work/deadlines when known. Use JSON Schema and `examples/contracts.json` for service clients.
3. First connect your forecast to the baseline planner. Then connect a learned controller through the same proposal protocol or `CallablePolicyAdapter`. RL can train against `energy_decision_layer.sim.env.SiteEnv`; keep the evaluation inputs and scoring unchanged.
4. Replace the default factory in the smoke script; the example import path below is a placeholder for your own package, not a module supplied here:

```bash
python examples/team_smoke.py --forecaster your_ml.integration:create_forecaster
python examples/team_smoke.py --forecaster your_ml.integration:create_forecaster --policy your_ml.integration:create_policy
```

Your first acceptance check: valid versioned forecasts/proposals pass the existing contract tests and a six-step evaluation, without future-data leakage or edits to the simulator/scorer. The baseline treats omitted workload flexibility conservatively; do not infer physical power or SLA deadlines from a utilization trace without documenting the mapping.

## Dharm — safety/backend

1. Import `SafetyShieldProtocol`, `SafetyResponseV1`, `ProposedControlActionV1`, `ExecutedControlActionV1`, `SafetyDecisionStatus`, `SafetyReasonCode` and `DecisionLoggerProtocol` from `energy_decision_layer.contracts`.
2. Implement shield acceptance/modification/rejection and both audit hooks (`record_decision` before execution and `record_outcome` after execution). The library supplies matching-ID, validity and status validation, not the production constraint engine or durable storage.
3. Return no executable action on rejection. A modified action must name exactly the changed setpoints and carry reasons. Preserve the original proposal and response/execution IDs.
4. Inject your components into the smoke script using factories in your own package:

```bash
python examples/team_smoke.py --shield your_backend.integration:create_shield --logger your_backend.integration:create_logger
```

Your first acceptance check: accept, modify, reject and communication/audit failure cases behave correctly; rejected actions do not advance the simulator. If the post-execution audit fails, `DispatchOutcomeLoggingFailed.result` identifies what was already applied. Persist/retry that audit record rather than replaying the action. Backend endpoints, operator controls and real controller delivery remain your workstream.

## Integration gate

```bash
python -m pytest -q tests/contracts tests/integration tests/optimizer
python -m ruff check src tests examples
python -m mypy src/energy_decision_layer
```

Add tests for your real component by injecting it through `Dispatcher` or `evaluate(..., forecaster=..., shield=..., decision_logger=...)`. Components with false truth value are still respected. Do not overwrite the development doubles to make them look like production implementations. Use [INTEGRATION.md](INTEGRATION.md) for field units, signs, timestamps and ownership, and [VERIFICATION.md](VERIFICATION.md) for verified capabilities and remaining modeling limits.
