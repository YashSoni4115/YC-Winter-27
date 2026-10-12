# Energy Decision Layer (working title)

A software layer that decides, continuously and autonomously, how electrical power is distributed *inside* an AI data centre site: between compute, cooling, on-site storage, on-site generation, and the grid connection.

> Status: early design draft. Everything below is a proposed design and a set of hypotheses to validate, not a description of a built system.

---

## 1. Problem

AI data centres are increasingly limited by the megawatts they can secure, not by chips or floor space. Grid connections can take years, so more sites are adding batteries and on-site generation. Today, the decisions that tie these pieces together are mostly made by static rules and human operators:

- How much power each rack or GPU cluster may draw
- How hard to run cooling, and when
- When to charge or discharge batteries
- When to start, stop, or ramp on-site generators
- How much to import from, or export to, the grid

Each of these is handled by a different system (compute scheduler, building controls, battery controller, generator controller, utility interface). Nobody optimizes them together. The goal of this project is a single decision layer that does.

**Core objective:** maximize useful compute (tokens, jobs, revenue) per available megawatt, subject to reliability, thermal, and contractual limits.

## 2. Scope

**In scope**
- Real-time and day-ahead power allocation across compute, cooling, storage, generation, and grid import/export
- Workload-aware power management (pausing, shifting, or capping jobs)
- Forecasting of load, ambient conditions, prices, and grid events
- Safety constraints, human override, audit logging, simulation

**Out of scope (for now)**
- Data centre siting and design
- Chip-level or inference-kernel optimization
- Electricity market bidding and trading
- Replacing the existing low-level protection systems (breakers, UPS firmware, chiller safeties)

## 3. System context

```
                    +--------------------+
                    |  Utility / Market  |   prices, curtailment signals,
                    |  / Grid Operator   |   interconnection limits
                    +---------+----------+
                              |
        +---------------------v----------------------+
        |              ENERGY DECISION LAYER         |
        |                                            |
        |  Ingest -> State -> Forecast -> Optimize   |
        |                         |                  |
        |                  Safety Shield             |
        |                         |                  |
        |                      Actuate               |
        +---+--------+--------+--------+--------+----+
            |        |        |        |        |
       +----v---+ +--v-----+ +v------+ +v-----+ +v--------+
       |Compute | |Cooling | |Battery| |On-site| |Operators|
       |sched.  | |plant   | |system | |gen    | |(humans) |
       +--------+ +--------+ +-------+ +-------+ +---------+
```

The layer sits *above* existing controllers. It sends setpoints and requests, and never bypasses their local safety logic.

## 4. Power domains

| Domain | What it is | What we can control | Key limits |
|---|---|---|---|
| Grid | Utility connection | Import/export setpoint (if allowed) | Contracted capacity, curtailment events, ramp limits |
| On-site generation | Turbines, engines, fuel cells, solar | Commit/dispatch, output level | Min up/down time, ramp rate, fuel supply, emissions caps |
| Storage | Batteries (possibly thermal storage) | Charge/discharge power | State of charge, power rating, cycle life, temperature |
| Compute | GPU clusters and racks | Power caps, job pausing/shifting, scheduling | SLAs, job priority, checkpointing cost |
| Cooling | Chillers, pumps, CDUs, fans | Setpoints, staging | Thermal limits, lag/inertia, water use |
| Losses | Conversion and distribution | Mostly fixed by design | Efficiency curves vary with load |

## 5. Architecture

### 5.1 Telemetry and integration
- Adapters for building and plant controls (BACnet, Modbus, OPC UA, vendor APIs)
- Adapters for the compute side (cluster scheduler, GPU telemetry and power-cap interfaces)
- Adapters for storage and generation controllers
- External feeds: utility signals, prices, weather
- All signals normalized into one time-series schema with units, quality flags, and timestamps

### 5.2 State estimation and digital twin
- Fuses noisy and missing sensor data into a consistent site state
- Maintains a lightweight model of the site: power flows, thermal response, battery state, generator state
- Used both for live decisions and for offline simulation (the same model, two uses)

### 5.3 Forecasting
- Compute load (queued jobs, priorities, typical durations)
- Ambient temperature and humidity (drives cooling cost)
- Electricity price and grid stress events
- Renewable output if solar or wind is present
- Forecasts carry uncertainty, and the optimizer uses it

### 5.4 Decision engine (hierarchical)

| Level | Horizon | Cadence | Example decisions | Likely method |
|---|---|---|---|---|
| Planner | Hours to a day | Every 15 to 60 min | Generator commitment, battery schedule, workload shifting plan | Mixed-integer or model-predictive optimization |
| Dispatcher | Minutes | Every 1 to 5 min | Power allocation between compute and cooling, import level | Model-predictive control |
| Real-time controller | Seconds | Sub-second to seconds | Smoothing load swings, battery response, GPU power caps | Fast rule-based or learned policy inside tight bounds |

Learned policies (for example reinforcement learning) are an option at the dispatcher and real-time levels, but they only ever act inside the safety envelope in 5.5. An optimization-based baseline comes first, because it is easier to explain, test, and trust.

### 5.5 Safety shield
A separate component that checks every proposed action before it is sent.
- Hard constraints: thermal limits, electrical limits, minimum battery reserve, generator ramp and start limits, grid import cap
- Rate limits on how fast any setpoint can change
- Redundancy rules (for example, keep N+1 headroom)
- Rejects or clips unsafe actions and logs why
- Independent from the optimizer so a bug or a bad model cannot bypass it

### 5.6 Actuation
- Translates decisions into setpoints and requests for each downstream system
- Confirms that commands were applied and watches for the expected response
- Falls back to the previous safe state if a command fails

### 5.7 Human interface and audit
- Live view of site state, current plan, and the reason for each action
- Operator override at any time, with a clear "return to manual" control
- Full decision log: inputs, proposed action, shield result, outcome

## 6. Objective and constraints

**Objective (configurable weighting):**
- Maximize compute output per megawatt (for example tokens per watt, or revenue per MW)
- Minimize energy cost (fuel, grid, battery wear)
- Minimize emissions (optional)
- Minimize risk of SLA breach

**Constraints:** everything in the "Key limits" column of section 4, plus per-workload SLAs and contractual grid terms.

The relative weights are a product decision and will differ by customer.

## 7. Autonomy levels

Operators are risk averse, so trust is built in stages.

| Level | Behavior |
|---|---|
| 0. Observe | Read-only. Compute what the layer *would* do and compare to what operators did |
| 1. Advise | Recommend actions with explanations. Humans apply them |
| 2. Supervised | Layer acts automatically in a limited scope (for example one subsystem), humans approve changes outside it |
| 3. Autonomous | Layer controls the full scope inside the safety envelope, humans monitor and can override |

Each level needs measured evidence from the one below before moving up.

## 8. Failure modes and fallbacks

| Failure | Response |
|---|---|
| Telemetry loss or bad data | Hold last safe setpoints, widen margins, alert operators |
| Optimizer fails or times out | Use last valid plan, then fall back to conservative static rules |
| Forecast badly wrong | Dispatcher replans with fresh state, reserves absorb error |
| Actuation not confirmed | Retry once, then revert and alert |
| Layer loses connection entirely | Downstream controllers keep their local logic. Layer is never required for safe operation |

Design principle: if this layer disappears, the site keeps running safely on its existing controls.

## 9. Evaluation

- **Efficiency:** compute per MW, effective PUE, conversion and cooling losses
- **Economics:** energy cost per unit of compute, battery and generator wear
- **Reliability:** SLA violations, thermal excursions, shield rejections, manual overrides
- **Quality of decisions:** gap to a perfect-hindsight optimum, forecast error impact
- **Trust:** share of recommendations accepted at Level 1

Baseline for comparison: what rule-based controls and human operators do today on the same scenarios.

## 10. Validation plan

1. **Simulator:** model a site with compute load, cooling response, battery, a generator with realistic ramp limits, and a grid connection. Use public price and weather data.
2. **Baselines:** implement rule-based and "human-like" dispatch policies.
3. **Compare:** optimizer vs baselines on cost per unit of compute, constraint violations, and robustness to forecast error.
4. **Replay:** run on historical or real telemetry from a design partner at Level 0.
5. **Pilot:** Level 1, then Level 2 on a single subsystem.

## 11. Roadmap

- **Phase 1: Simulator and baseline optimizer.** Site model, forecasting stubs, planner plus dispatcher, evaluation harness.
- **Phase 2: Safety shield and explanation layer.** Constraint engine, decision logging, operator view.
- **Phase 3: Integrations.** First real adapters (compute telemetry, one plant controller), Level 0 shadow mode.
- **Phase 4: Learned components.** Forecast improvements and optional learned policies inside the shield.
- **Phase 5: Pilot.** Advise, then supervised control with a design partner.

## 12. Open questions

- Which site type is the best first target: grid-connected with batteries, hybrid, or fully off-grid?
- How much control access will operators actually grant, and how fast?
- How much workload flexibility exists in practice (batch vs latency-sensitive inference)?
- Who owns each control interface, and what are the integration costs per site?
- How does this coexist with vendor-bundled controls from equipment makers?
- What liability and insurance questions come with autonomous control of power systems?

## 13. Suggested repository layout

```
/docs            design notes, decision records
/sim             site simulator and scenarios
/forecast        forecasting models
/optimizer       planner, dispatcher, real-time controller
/safety          constraint definitions and shield
/adapters        integrations for downstream systems and data feeds
/ui              operator interface
/eval            metrics and baseline comparisons
```