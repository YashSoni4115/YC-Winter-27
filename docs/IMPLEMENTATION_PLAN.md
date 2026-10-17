# Integration implementation plan (2026-10-17)

Audit: Devansh's contracts, discrete physics, Gym wrapper, immutable raw artifacts,
UTC normalized intervals, SciPy/HiGHS MILP, fail-closed dispatcher, scenario scoring,
and Python CI already exist. Yash/Dharm implementations do not. No database or
frontend framework exists. Existing units are MW/MWh and USD/MWh; retain them.
README sections 5.4 and 6 require hierarchical planning and independent safety.

1. Add a centralized coupled-constraint shield using shared subsystem configurations
   and physics. Project proposals into the feasible envelope, recheck transitions,
   and reject invalid/stale/infeasible states. Keep original proposals in responses.
2. Add chronological gradient boosting forecasting with quantiles and baselines,
   simulator-trained learned control and dynamic surrogates. Persist local artifacts
   with metadata. Use only labelled synthetic training fixtures by default.
3. Add simulation adapters, SQLite migrations/audit/idempotency, and a FastAPI service.
   Reuse Dispatcher for autonomous and manual actions. Recover simulator checkpoints
   on restart; fail closed on incomplete execution. No hardware implementation.
4. Serve a buildless responsive operator dashboard from the same service, consuming
   typed API contracts. Separate authenticated simulation controls from read views.
5. Extend comparable evaluations, fault/error experiments, API/UI tests and CI.
   Run existing and added tests, lint, formatting, typing, packaging and a live demo.

Integration: SimulationIntervalV1 -> SiteInputs -> SiteSimulator observation ->
ForecastBundleV1 -> existing MilpPolicy or learned PolicyProtocol -> Dispatcher ->
SafetyResponseV1 -> mock adapter -> simulator -> telemetry + SQLite -> API -> UI.
The raw physics-only advance() remains for subsystem tests and trajectory generation;
all dispatch/training/application control uses the shield.
