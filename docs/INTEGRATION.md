# Integrated implementation update — 2026-10-17

Yash and Dharm implementations now live in `ml`, `safety`, `adapters` and `service`,
using the contracts documented below. [SYSTEM.md](SYSTEM.md) describes the runnable
application; [ML.md](ML.md) documents models. Historical statements below that real
components were absent describe the original foundation handoff. The simulator
and physics remain simulation models, and physical equipment integration is absent.

# Team handoff and version 1 contracts

Devansh owns `contracts`, `sim`, `data`, `optimizer`, `evaluation`, `testing`, tests and CI. No real forecasting, learned controller, safety shield, backend, UI, or plant adapter is implemented in this branch. Teammates can develop in separate packages and import these boundaries; they need not edit the physics or scoring code.

For a runnable first checkpoint for each teammate, use [TEAM_START.md](TEAM_START.md) and `python examples/team_smoke.py`.

## Shared boundary

All public exchange models live in `energy_decision_layer.contracts`. Committed [JSON Schemas](schemas) include their nested definitions and can generate clients in other languages. Python models forbid extra fields, reject unsupported schema versions, nonfinite values and naive timestamps, normalize aware timestamps to UTC, and validate cross-field consistency. Python-mode construction is strict: supply enum members, tuples and UUID/datetime objects; HTTP/JSON clients should use `Model.model_validate_json(payload)` so strings/arrays decode correctly. Serialize with `model_dump_json()`. Missing optional readings are `null`, never zero. `schema_version` currently equals `1.0.0`; breaking changes require a new version and migration tests. These are first unreleased v1 contracts, extended during implementation before handoff.

| Boundary | Model | Key semantics |
|---|---|---|
| Incomplete sensor input | `TelemetryV1` | Site/state ID, UTC timestamp, sequence, nullable measurements, provenance, field quality flags/issues |
| Complete control state | `SiteObservationV1` | Site/state ID, start of next interval, cadence, incoming demand, queue deadlines, battery/thermal state, generator timers and current external restrictions |
| Forecast | `ForecastBundleV1` | Issuance no later than control time; first target equals current interval; regular cadence; uncertainty intervals contain nominal values; synthetic/hindsight provenance explicit |
| Proposal | `ProposedControlActionV1` | UUID, site/state binding, creation/effective/expiry, policy metadata, setpoints, optional constraint metadata and fallback reason |
| Authorization | `ExecutedControlActionV1` | Entire original proposal, response/execution UUIDs, authorization time/window, actual authorized setpoints and exact changed-field list |
| Shield response | `SafetyResponseV1` | Matched request/site IDs, decision time, shield/version, accepted/modified/rejected status, reasons, authorized action only for accepted/modified |

Every numeric field includes units in its name. Power is MW, energy MWh, temperature °C, elapsed time seconds at boundaries/hours in physics, useful compute abstract units. Contract prices are **USD/MWh**. Battery positive means discharge, grid positive means import; consumption and generation are nonnegative. The ETL schema retains original currency; converting CAD to USD requires an explicit exchange rate. ETL interval workload is a **total**, whereas observation/forecast demand is a **rate**; conversion divides by interval hours.

`completed_compute_units` and `deadline_violated_compute_units` in observations are cumulative. Step diagnostics contain interval increments. Queued workload deadlines count intervals including the upcoming interval (`1` is due now). `incoming_deferrable_fraction` splits aggregate new demand; omitted forecast fractions default to zero (all due now). Observations expose internal state after the previous interval and exogenous conditions for the upcoming interval. Historical power fields are previous interval feedback, set to zero for currently unavailable sources; score completed interval flows using `info`, not by recomputing them from next-interval availability.

## ML teammate: start here

Implement `ForecastProviderProtocol.forecast(observation, *, horizon_steps, cadence_seconds)` and/or `PolicyProtocol.propose_action(observation, forecast=None)`. The caller expects a forecast starting at `observation.timestamp_utc` with matching site/cadence. Providers must reject unavailable horizons rather than backfill from later observations. Feed only measurements/features available at forecast issuance; fit preprocessing on training data only. `chronological_split` drops intervals straddling the cutoff. Public workload utilization does not supply physical power or SLAs: document any synthetic mapping.

`PersistenceForecastDouble` is a deterministic placeholder. `OfflineHindsightForecastDouble` deliberately reads future scenario demand, price, weather and grid restrictions and labels every bundle. Normal `MilpPolicy` rejects it; only explicit `allow_offline_hindsight=True` enables the offline benchmark. This is a rolling oracle benchmark, not a proven global full-episode optimum, and does not forecast generator failures.

Use `CallablePolicyAdapter` to validate a callable's output. `evaluate(name, inputs, policy_factory, forecaster=provider, shield=shield, decision_logger=logger)` starts a fresh simulator and policy on identical initial conditions and inputs. Use `policy_label` for meaningful model names. RL uses `SiteEnv`; pass `SiteSimulator` to its constructor and apply Gym normalization wrappers as needed. A learned policy should emit the same proposal contract and must use the dispatcher/safety path outside pure Gym experiments.

## Dharm: start here

Implement `SafetyShieldProtocol.evaluate(proposal, observation)` and the audit methods `DecisionLoggerProtocol.record_decision(proposal, response)` before execution plus `record_outcome(proposal, next_observation, diagnostics)` after execution. Inject them into `Dispatcher`. Replace the accept-all development double before connecting any real equipment. The supplied in-memory logger is only an exchange test double; durable storage of these decisions/outcomes, backend endpoints, operator controls and controller retry/confirmation remain yours.

For accepted responses, executed setpoints equal the proposal. For modified responses, supply actual setpoints, exact `modified_fields`, and reasons. For rejection, provide reasons and no executable action. IDs, source proposal, timestamps, validity and status are checked again by the dispatcher. The validity window must cover the entire simulation interval. Simulator local physics still records adjustments or infeasibility; it is not a replacement safety shield. A model bearing an execution ID is a data structure, not cryptographic authorization.

Primary policy/forecast exceptions create an explicitly marked rule fallback which still goes through the shield. Shield errors, bad response binding, rejection and audit failures stop before actuation. If post-execution outcome logging fails, `DispatchOutcomeLoggingFailed` explicitly reports that the action **was applied**: never retry it blindly. Diagnostics include dataclass constraint events; encode them explicitly when persisting JSON. Existing local plant controllers should retain control in live operation; this library does not send a guessed action after a shield failure. There is no automatic retry for a rejected action and no unvalidated cached plan. Current implementation replans every dispatch interval; integrate a validated 15-minute plan cache only if needed and retain state/expiry validation.

## Verification and examples

`tests/integration/` exercises forecast -> planner -> proposal -> shield double -> simulator -> metrics, plus callable policies and offline oracle bundles. `tests/optimizer/` injects policy failures, solver timeout/infeasibility/corruption and shield communication/rejection. `tests/contracts/` validates invalid versions, stale proposals, status semantics and schema snapshots. `examples/contracts.json` contains serializable request/state/response/forecast payloads.

Real teammate implementations are absent and their integration tests remain pending. To onboard: install `.[dev]`, run the suite, implement the relevant protocol in your owned package, replace the double in the integration fixture, and run the same contract and scenario checks. No teammate accounts, branches, remote settings, or messages have been modified from Devansh's side.


## Recheck notes — 2026-10-15

State IDs identify an individual episode and timestep, not just a site/index. Each reset and simulator instance gets a fresh episode UUID. Fixed inputs/seeds reproduce physics and scores, while execution IDs remain unique. A previously approved action cannot be replayed into a reset/different simulator.

The dispatcher validates forecast schema/site/issuance/cadence before calling any policy, including policies that ignore forecast fields. Dependency injection uses explicit `None` checks so a legitimate component with false truth value is never replaced by a development double. Current telemetry availability overrides optimistic forecasts in the planner; critical backlog is due immediately.

`DispatchOutcomeLoggingFailed.result` preserves the applied observation/reward/termination/diagnostics tuple. Retry durable audit storage using that feedback rather than reapplying the command. Evaluation includes that already-applied timestep, operating cost and useful work, flags an `outcome_audit_failures` count and leaves `simulator_failures` unchanged unless physics actually terminated. Metrics record the injected shield's name; generated reports do not claim a real integration used a development shield.
