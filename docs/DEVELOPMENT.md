# Running Devansh's foundation

Python 3.10–3.12 is targeted. Use a virtual environment; no system packages or API keys are required for offline work.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-dev.lock -e '.[dev]' build
python -m pytest --cov=energy_decision_layer
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m mypy src/energy_decision_layer
python -m energy_decision_layer etl examples/synthetic.csv --synthetic
python -m energy_decision_layer benchmark --steps 24
python examples/team_smoke.py
python -m energy_decision_layer schemas
python -m build
```

Generated data/reports live in ignored `artifacts/`. Benchmark outputs are `metrics.json`, `metrics.csv`, `summary.md`, and `run.json` (seed, site configuration, horizon and scenario length). See [verification](VERIFICATION.md), [integration](INTEGRATION.md), [physics](PHYSICS.md), [planner](PLANNER.md), and [data sources](DATA_SOURCES.md).

GitHub Actions checks Python 3.10 and 3.12, formatting, lint, strict typing, branch coverage (85% minimum), unit/contract/invariant/integration tests, offline ETL, a small scenario benchmark and wheel/sdist packaging. Manual workflow dispatch additionally runs a 288-step scenario comparison. CI requires no live network data after dependencies install. It uploads diagnostic artifacts even on failure. Workflow definitions have been exercised through their local commands; hosted results require pushing the branch. Branch protection and repository secrets are administrator-owned settings and have not been changed.

Optional dependency groups: `sim`, `data`, `optimizer`, `all`, `dev`. Importing contracts requires only the core dependencies. `sim.env`, planner and tabular commands require their respective extras. No production deployment workflow is supplied because there is no service or approved deployment target in this workstream.

The Gym checker emits advisory warnings about an unnormalized action space and unbounded observation bounds. Units are deliberately physical, and prices/backlogs have no arbitrary global cap. ML can add normalization wrappers using training-only statistics.

## Offline example

```python
from energy_decision_layer.sim import SiteInputs, SiteSimulator
from energy_decision_layer.optimizer import RulePolicy
from energy_decision_layer.optimizer.dispatcher import Dispatcher
from energy_decision_layer.testing.doubles import (
    DevelopmentAcceptAllShield, InMemoryDecisionLogger, PersistenceForecastDouble,
)

sim = SiteSimulator([SiteInputs(nondeferrable_compute_units=40.0)] * 24)
observation, _ = sim.reset(seed=7)
controller = Dispatcher(RulePolicy(sim.config), DevelopmentAcceptAllShield(),
    InMemoryDecisionLogger(), RulePolicy(sim.config), PersistenceForecastDouble())
while not sim.done:
    observation, reward, terminated, truncated, diagnostics = controller.dispatch(observation, sim)
```

This is simulation only. The development shield accepts every structurally valid proposal. Production validation, durable logging, controller delivery, manual override and operator UI belong to Dharm.
