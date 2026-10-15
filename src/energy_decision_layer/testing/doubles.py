"""These doubles implement exchange shapes, NOT safety or forecasting assurance."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from energy_decision_layer.contracts import (
    ExecutedControlActionV1,
    ForecastBundleV1,
    ForecastPointV1,
    ProposedControlActionV1,
    ProvenanceV1,
    SafetyDecisionStatus,
    SafetyReasonCode,
    SafetyResponseV1,
    SiteObservationV1,
)
from energy_decision_layer.sim.common import SiteInputs


class DevelopmentAcceptAllShield:
    """UNSAFE outside offline simulation: accepts all structurally valid proposals."""

    def evaluate(
        self, proposal: ProposedControlActionV1, observation: SiteObservationV1
    ) -> SafetyResponseV1:
        response_id = uuid4()
        action = ExecutedControlActionV1(
            execution_id=uuid4(),
            source_action=proposal,
            safety_response_id=response_id,
            authorization_status=SafetyDecisionStatus.ACCEPTED,
            authorized_at_utc=observation.timestamp_utc,
            effective_at_utc=proposal.effective_at_utc,
            expires_at_utc=proposal.expires_at_utc,
            **{
                name: getattr(proposal, name)
                for name in (
                    "compute_power_mw",
                    "cooling_power_mw",
                    "battery_power_mw",
                    "generator_power_mw",
                    "grid_power_mw",
                    "renewable_curtailment_mw",
                )
            },
        )
        return SafetyResponseV1(
            response_id=response_id,
            action_id=proposal.action_id,
            site_id=proposal.site_id,
            decided_at_utc=observation.timestamp_utc,
            shield_name="DEVELOPMENT-ACCEPT-ALL-NOT-SAFE",
            shield_version="0.1.0",
            status=SafetyDecisionStatus.ACCEPTED,
            reason_codes=(SafetyReasonCode.WITHIN_CONSTRAINTS,),
            executed_action=action,
        )


class PersistenceForecastDouble:
    def forecast(
        self, observation: SiteObservationV1, *, horizon_steps: int, cadence_seconds: float
    ) -> ForecastBundleV1:
        o = observation
        if horizon_steps < 1:
            raise ValueError("horizon must be positive")
        points = tuple(
            ForecastPointV1(
                target_timestamp_utc=o.timestamp_utc + timedelta(seconds=i * cadence_seconds),
                workload_demand_compute_units_per_h=o.workload_demand_compute_units_per_h,
                deferrable_fraction=o.incoming_deferrable_fraction,
                deferrable_deadline_steps=o.incoming_deferrable_deadline_steps,
                electricity_price_usd_per_mwh=o.electricity_price_usd_per_mwh,
                ambient_temperature_c=o.ambient_temperature_c,
                grid_available=o.grid_available,
                grid_import_limit_mw=o.grid_import_limit_mw,
            )
            for i in range(horizon_steps)
        )
        return ForecastBundleV1(
            forecast_id=uuid4(),
            site_id=o.site_id,
            issued_at_utc=o.timestamp_utc,
            valid_from_utc=points[0].target_timestamp_utc,
            valid_until_utc=points[-1].target_timestamp_utc,
            cadence_seconds=cadence_seconds,
            model_name="DEVELOPMENT-persistence",
            model_version="0.1.0",
            points=points,
            provenance=ProvenanceV1(
                source_name="synthetic-persistence-double",
                ingested_at_utc=o.timestamp_utc,
                is_synthetic=True,
            ),
        )


class InMemoryDecisionLogger:
    def __init__(self) -> None:
        self.records: list[tuple[ProposedControlActionV1, SafetyResponseV1]] = []
        self.outcomes: list[dict[str, Any]] = []

    def record_decision(
        self, proposal: ProposedControlActionV1, response: SafetyResponseV1
    ) -> None:
        self.records.append((proposal, response))

    def record_outcome(
        self,
        proposal: ProposedControlActionV1,
        observation: SiteObservationV1,
        diagnostics: Mapping[str, Any],
    ) -> None:
        self.outcomes.append(
            dict(
                action_id=str(proposal.action_id),
                state_id=observation.state_id,
                diagnostics=dict(diagnostics),
            )
        )


class OfflineHindsightForecastDouble:
    """Offline oracle benchmark only; reads future scenario inputs intentionally."""

    def __init__(self, inputs: tuple[SiteInputs, ...], start: datetime) -> None:
        self.inputs, self.start = inputs, start

    def forecast(
        self, observation: SiteObservationV1, *, horizon_steps: int, cadence_seconds: float
    ) -> ForecastBundleV1:
        baseline = PersistenceForecastDouble().forecast(
            observation, horizon_steps=horizon_steps, cadence_seconds=cadence_seconds
        )
        points = []
        for i, point in enumerate(baseline.points):
            x = self.inputs[min(observation.step_index + i, len(self.inputs) - 1)]
            payload = point.model_dump()
            payload.update(
                workload_demand_compute_units_per_h=(
                    x.nondeferrable_compute_units + x.deferrable_compute_units
                )
                * 3600
                / cadence_seconds,
                deferrable_fraction=x.deferrable_compute_units
                / max(1e-12, x.nondeferrable_compute_units + x.deferrable_compute_units),
                deferrable_deadline_steps=x.deferrable_deadline_steps,
                electricity_price_usd_per_mwh=x.grid_import_price_per_mwh,
                ambient_temperature_c=x.ambient_temperature_c,
                grid_available=x.grid_available,
                grid_import_limit_mw=x.grid_import_limit_mw,
            )
            points.append(ForecastPointV1.model_validate(payload))
        data = baseline.model_dump()
        data.update(
            points=tuple(points), is_perfect_hindsight=True, model_name="OFFLINE-PERFECT-HINDSIGHT"
        )
        return ForecastBundleV1.model_validate(data)
