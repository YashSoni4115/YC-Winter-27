# Site model and Gym semantics

The default synthetic hybrid site has 10 MW compute (1 MW idle, 900 useful units/hour at full power), 4 MW electrical cooling, 20 MWh battery, 8 MW generator and 15 MW grid import. Parameters are dataclasses under `sim`; these are engineering fixtures, not calibrated hardware specifications. There is no renewable subsystem in this topology. Export is optional in physics; the baseline planner conservatively uses imports only.

Subsystem `step` methods are deterministic, return new immutable state/results and expose requested versus actual values and reasons. Integrated `SiteSimulator.reset` returns `(SiteObservationV1, info)`; `step(ExecutedControlActionV1)` returns `(observation, reward, terminated, truncated, info)` and validates site/state/time binding. `advance(SiteAction)` is the lower-level physics entry for tests and Gym only. `reset` restores initial queues, battery, generator, thermal state and index. No stochastic process is currently enabled, so seeds do not alter a fixed scenario. Execution state IDs include a fresh episode UUID on each reset; physical state transitions and scores remain reproducible. Unsupported reset options and stepping after episode end raise errors.

## Compute and deadlines

Capacity is `max(0, (power-idle)/(max_power-idle)) * max_units_per_hour * hours`, limited by active fleet capacity. Below idle consumes energy but completes nothing. Incoming critical work is due in the current interval. Deferrable batches use earliest-deadline-first service after critical work; unused batch deadlines decrease by one, and expired remainder becomes missed work. No unfinished work silently disappears. Work pending at a truncated episode is backlog, not an invented deadline violation. Compare useful output together with misses and pending work.

## Battery and generator

For bus power `b`, stored energy changes by `-b*hours/eta_discharge` when discharging and `-b*hours*eta_charge` when charging. Conversion loss equals bus energy absorbed minus stored energy change. Never add that loss to bus load again. Rated power, SOC-derived bounds, and ramp limits constrain requests. When reserve and ramp constraints conflict, energy bounds take precedence and adjustments are visible. Insufficient bus supply can emergency-curtail charging with an explicit diagnostic. Degradation is cell throughput times configured cost. Equivalent full cycles divide cell throughput by twice usable energy.

Generator output is zero while unavailable/off, otherwise within stable min/max; online ramp and minimum up/down timers apply. Startup jumps to minimum stable output and may ramp one interval further. Shutdown after minimum uptime is an explicit commitment jump to zero, not an online ramp; a forced outage also overrides timers/ramp. When minimum uptime prevents shutdown, the unit ramps down toward stable minimum. Fuel thermal MWh equals electrical MWh divided by electrical efficiency. Fuel, variable, startup and shutdown costs are separate.

## Cooling and thermal model

COP decreases with ambient temperature between configured bounds. Cooling electrical response is `previous + (target-previous)*(1-exp(-hours/tau))`, followed by the configured ramp cap; `tau=0` is immediate response. The applied endpoint power approximates the interval-average electrical and thermal effect. This is a documented discrete model, not a continuous chiller transient solver; varying cadence changes that approximation.

`C * (T_next-T) = hours * (heat_fraction*compute - COP*cooling - UA*(T-ambient))`.

Thermal violations are measured without clamping temperature. Site configuration rejects timesteps beyond the stable envelope integration interval (`hours*UA <= C`). Default cadence is 5 minutes. Detailed airflow, humidity, water use, temperature-dependent battery efficiency and distribution losses await calibration; no unsupported physical fidelity is claimed.

## Bus balance, infeasibility and economics

Grid requests clip to connection limits and availability. Exports/charging cannot consume unavailable supply. Auxiliary and cooling receive priority over compute; requested loads beyond available supply are shed and reported. Surplus bus energy enters an explicit modeled spill/dump sink. There is no invented balancing energy.

`grid + generator + battery = auxiliary_served + cooling_served + compute_served + spill`.

Diagnostics include residual MW, unserved MWh, spilled MWh, actual action, deadline misses, thermal violations and subsystem adjustments. Response lag is informational; an unsafe proposal that local physics clips is distinct from an actual temperature excursion or loss of service. Reward is negative operating cost minus configured missed-work penalty. Operating cost includes signed grid imports/exports, generator costs and battery wear. Thermal excursions and unserved load are separately reported, not hidden inside operating cost. Product objective weights and safety requirements need team/customer agreement before deployment.

Exhausting input intervals truncates the episode; temperature below absolute zero or above 80°C terminates it as a catastrophic model/site failure. A runtime numerical failure raises an exception and the evaluation runner marks failure. This is an offline simulator, not an emergency controller.

## Gym arrays

`SiteEnv.action_space` is a float64 Box ordered as compute MW, cooling MW, battery MW, generator MW, grid MW, with configured nameplate limits. Nonfinite/out-of-space actions raise `ValueError`; local dynamic constraints still apply. Observations are float64 arrays: incoming useful units/hour, facility °C, battery MWh, generator MW, price USD/MWh, ambient °C, pending deferrable units. Observation bounds are unbounded because prices/backlogs have no global physical cap; callers can normalize. The richer typed observation remains available on `env.simulator`. The Gym path applies physics directly and makes no safety authorization claim.
