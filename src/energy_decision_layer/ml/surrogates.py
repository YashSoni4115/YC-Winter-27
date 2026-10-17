"""Advisory one-step dynamics surrogates fitted to shielded simulator trajectories.

Empirical held-out residual bounds are not certified uncertainty guarantees.
Predictions never authorize actions and are never used by the safety shield.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesRegressor

from energy_decision_layer.contracts import ProposedControlActionV1, SiteObservationV1
from energy_decision_layer.safety import SafetyShield
from energy_decision_layer.sim import SiteConfig, SiteSimulator
from energy_decision_layer.sim.battery import BatteryState, BatterySystem
from energy_decision_layer.sim.common import SiteInputs
from energy_decision_layer.sim.generator import GeneratorState
from energy_decision_layer.sim.thermal import ThermalState, ThermalSystem

from .forecasting import regression_metrics
from .policy import LOWER, UPPER, LearnedPolicy

OUTPUTS = (
    "facility_temperature_c",
    "cooling_power_mw",
    "battery_degradation_cost_usd",
    "generator_power_mw",
)


def dynamics_features(o: SiteObservationV1, p: ProposedControlActionV1) -> NDArray[np.float64]:
    return np.array(
        [
            o.facility_temperature_c,
            o.cooling_power_mw,
            o.ambient_temperature_c,
            p.compute_power_mw,
            p.cooling_power_mw,
            o.battery_energy_mwh,
            o.battery_power_mw,
            p.battery_power_mw,
            o.generator_power_mw,
            float(o.generator_is_on),
            float(o.generator_consecutive_on_steps),
            float(o.generator_consecutive_off_steps),
            p.generator_power_mw,
            float(o.generator_available),
        ],
        dtype=float,
    )


class DynamicsSurrogate:
    def __init__(self, config: SiteConfig | None = None) -> None:
        self.config = config or SiteConfig()
        self.model: Any = None
        self.error_bounds = np.zeros(4)
        self.domain_lower = np.zeros(14)
        self.domain_upper = np.zeros(14)
        self.report: dict[str, Any] = {}

    def predict(
        self, observation: SiteObservationV1, action: ProposedControlActionV1
    ) -> dict[str, Any]:
        c, o = self.config, observation
        if self.model is None:
            raise ValueError("Surrogate is not trained")
        if o.site_id != c.site_id or o.timestep_seconds != c.timestep_seconds:
            raise ValueError("Surrogate site/cadence mismatch")
        if action.based_on_state_id != o.state_id or action.site_id != o.site_id:
            raise ValueError("Surrogate action/state binding mismatch")
        if (
            not c.battery.minimum_energy_mwh <= o.battery_energy_mwh <= c.battery.maximum_energy_mwh
            or o.cooling_power_mw > c.thermal.cooling_max_power_mw
            or o.generator_power_mw > c.generator.maximum_power_mw
            or action.compute_power_mw > c.compute.max_power_mw
        ):
            raise ValueError("Surrogate input is outside physical operating bounds")
        x = dynamics_features(o, action)
        if not np.isfinite(x).all():
            raise ValueError("Invalid surrogate input")
        # Explicit domain status lets consumers refuse extrapolated advisory predictions.
        out_of_domain = bool(
            np.any(x < self.domain_lower - 1e-9) or np.any(x > self.domain_upper + 1e-9)
        )
        values = np.asarray(self.model.predict(x.reshape(1, -1))[0], dtype=float)
        dt = o.timestep_seconds / 3600
        thermal = ThermalSystem(c.thermal)
        ts = ThermalState(o.facility_temperature_c, o.cooling_power_mw)
        cool_lo = thermal.constrain_power(ts, 0.0, dt)[0]
        cool_hi = thermal.constrain_power(ts, c.thermal.cooling_max_power_mw, dt)[0]
        temp_lo = thermal.step(
            ts, c.thermal.cooling_max_power_mw, action.compute_power_mw, o.ambient_temperature_c, dt
        ).state.indoor_temperature_c
        temp_hi = thermal.step(
            ts, 0.0, action.compute_power_mw, o.ambient_temperature_c, dt
        ).state.indoor_temperature_c
        battery = BatterySystem(c.battery)
        bs = BatteryState(o.battery_energy_mwh, o.battery_power_mw)
        max_wear = max(
            battery.step(bs, -c.battery.maximum_charge_power_mw, dt).degradation_cost,
            battery.step(bs, c.battery.maximum_discharge_power_mw, dt).degradation_cost,
        )
        gs = GeneratorState(
            o.generator_is_on,
            o.generator_power_mw,
            o.generator_consecutive_on_steps,
            o.generator_consecutive_off_steps,
        )
        if not o.generator_available or (
            not gs.is_on and gs.consecutive_off_steps < c.generator.minimum_down_steps
        ):
            gen_lo = gen_hi = 0.0
        elif gs.is_on:
            gen_lo = max(
                c.generator.minimum_power_mw,
                gs.power_mw - c.generator.ramp_down_rate_mw_per_hour * dt,
            )
            gen_hi = min(
                c.generator.maximum_power_mw,
                gs.power_mw + c.generator.ramp_up_rate_mw_per_hour * dt,
            )
            if (
                action.generator_power_mw <= 0
                and gs.consecutive_on_steps >= c.generator.minimum_up_steps
            ):
                gen_lo = gen_hi = 0.0
        else:
            gen_lo = 0.0 if action.generator_power_mw <= 0 else c.generator.minimum_power_mw
            gen_hi = (
                0.0
                if action.generator_power_mw <= 0
                else min(
                    c.generator.maximum_power_mw,
                    c.generator.minimum_power_mw + c.generator.ramp_up_rate_mw_per_hour * dt,
                )
            )
        lo = np.array([temp_lo, cool_lo, 0.0, gen_lo])
        hi = np.array([temp_hi, cool_hi, max_wear, gen_hi])
        values = np.clip(values, lo, hi)
        return {
            "advisory_only": True,
            "outside_training_domain": out_of_domain,
            "uncertainty": "held-out 95th percentile absolute residual; not safety certified",
            "predictions": {
                name: {
                    "value": float(values[j]),
                    "lower": float(max(lo[j], values[j] - self.error_bounds[j])),
                    "upper": float(min(hi[j], values[j] + self.error_bounds[j])),
                }
                for j, name in enumerate(OUTPUTS)
            },
        }

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "surrogate.pkl").write_bytes(pickle.dumps(self))
        (path / "metadata.json").write_text(
            json.dumps(self.report, indent=2, allow_nan=False) + "\n"
        )

    @classmethod
    def load(cls, directory: str | Path) -> DynamicsSurrogate:
        model = pickle.loads((Path(directory) / "surrogate.pkl").read_bytes())
        if not isinstance(model, cls):
            raise ValueError("Not a dynamics surrogate artifact")
        return model


def train_surrogates(
    inputs: Sequence[SiteInputs],
    output: str | Path,
    *,
    config: SiteConfig | None = None,
    seed: int = 42,
    episodes: int = 12,
) -> dict[str, Any]:
    if len(inputs) < 4 or episodes < 4:
        raise ValueError("Need >=4 intervals and >=4 independent episodes")
    c = config or SiteConfig()
    rng = np.random.default_rng(seed)
    shield = SafetyShield(c)
    rows, targets, episode_ids = [], [], []
    for episode in range(episodes):
        sim = SiteSimulator(inputs, c)
        o, _ = sim.reset(seed=seed + episode)
        policy = LearnedPolicy(c, rng.uniform(LOWER, UPPER).tolist())
        for _ in inputs:
            response = shield.evaluate(policy.propose_action(o), o)
            if response.executed_action is None:
                break
            a = response.executed_action
            # Train on actual shield-authorized commands, not infeasible raw proposals.
            p = a.source_action.model_copy(
                update={
                    name: getattr(a, name)
                    for name in (
                        "compute_power_mw",
                        "cooling_power_mw",
                        "battery_power_mw",
                        "generator_power_mw",
                        "grid_power_mw",
                    )
                }
            )
            x = dynamics_features(o, p)
            o, _, terminated, truncated, info = sim.step(a)
            rows.append(x)
            targets.append(
                [
                    o.facility_temperature_c,
                    info["applied_action"]["cooling_power_mw"],
                    info["battery_degradation_cost"],
                    info["applied_action"]["generator_power_mw"],
                ]
            )
            episode_ids.append(episode)
            if terminated or truncated:
                break
    if len(rows) < 12:
        raise ValueError("Too few safe simulator transitions for surrogate training")
    x, y = np.array(rows), np.array(targets)
    ids = np.array(episode_ids)
    train_mask, validation_mask = (
        ids < int(episodes * 0.6),
        ((ids >= int(episodes * 0.6)) & (ids < int(episodes * 0.8))),
    )
    test_mask = ids >= int(episodes * 0.8)
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("Safety rejections left an empty surrogate split")
    surrogate = DynamicsSurrogate(c)
    surrogate.model = ExtraTreesRegressor(
        n_estimators=64, min_samples_leaf=2, random_state=seed, n_jobs=1
    ).fit(x[train_mask], y[train_mask])
    surrogate.domain_lower, surrogate.domain_upper = x[train_mask].min(0), x[train_mask].max(0)
    vp = surrogate.model.predict(x[validation_mask])
    surrogate.error_bounds = np.quantile(np.abs(y[validation_mask] - vp), 0.95, axis=0)
    tp = surrogate.model.predict(x[test_mask])
    report: dict[str, Any] = {
        "seed": seed,
        "data_kind": "synthetic shielded simulator trajectories",
        "method": "extra-trees multi-output one-step regression",
        "test_metric_scope": "raw held-out regressor before inference physical bounds",
        "split_unit": "independent whole episode, no transition leakage",
        "samples": {
            "train": int(train_mask.sum()),
            "validation": int(validation_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "test": {
            name: regression_metrics(
                y[test_mask, j],
                tp[:, j],
                tp[:, j] - surrogate.error_bounds[j],
                tp[:, j] + surrogate.error_bounds[j],
            )
            for j, name in enumerate(OUTPUTS)
        },
        "error_bounds_95": dict(zip(OUTPUTS, surrogate.error_bounds.tolist(), strict=True)),
        "site_config": asdict(c),
        "advisory_only": True,
        "limitations": (
            "Battery wear follows simulator throughput proxy; no calibrated SOH/aging labels"
        ),
    }
    surrogate.report = report
    surrogate.save(output)
    return report
