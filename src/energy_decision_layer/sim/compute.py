"""Compute workload queue and electrical-to-useful-work conversion."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .common import (
    NUMERICAL_EPSILON,
    ConstraintViolation,
    clip,
    require_nonnegative,
    require_positive,
)


@dataclass(frozen=True)
class ComputeConfig:
    """Static compute-fleet parameters.

    Useful throughput is linear between ``idle_power_mw`` and
    ``max_power_mw``.  Power below the idle threshold is allowed but performs
    no useful work, making inefficient actions visible in the metrics.
    """

    max_power_mw: float = 10.0
    idle_power_mw: float = 1.0
    max_useful_compute_units_per_hour: float = 900.0
    initial_deferrable_compute_units: float = 0.0
    initial_deferrable_deadline_steps: int = 4

    def __post_init__(self) -> None:
        require_positive("max_power_mw", self.max_power_mw)
        require_nonnegative("idle_power_mw", self.idle_power_mw)
        if self.idle_power_mw >= self.max_power_mw:
            raise ValueError("idle_power_mw must be smaller than max_power_mw")
        require_positive(
            "max_useful_compute_units_per_hour",
            self.max_useful_compute_units_per_hour,
        )
        require_nonnegative(
            "initial_deferrable_compute_units",
            self.initial_deferrable_compute_units,
        )
        if self.initial_deferrable_deadline_steps < 1:
            raise ValueError("initial_deferrable_deadline_steps must be at least one")


@dataclass(frozen=True)
class WorkloadBatch:
    remaining_compute_units: float
    steps_to_deadline: int

    def __post_init__(self) -> None:
        require_nonnegative("remaining_compute_units", self.remaining_compute_units)
        if self.steps_to_deadline < 1:
            raise ValueError("steps_to_deadline must be at least one")


@dataclass(frozen=True)
class ComputeState:
    deferrable_batches: tuple[WorkloadBatch, ...] = ()
    cumulative_completed_compute_units: float = 0.0
    cumulative_deadline_missed_compute_units: float = 0.0
    previous_power_mw: float = 0.0

    @property
    def deferrable_backlog_compute_units(self) -> float:
        return sum(batch.remaining_compute_units for batch in self.deferrable_batches)


@dataclass(frozen=True)
class ComputeStepResult:
    state: ComputeState
    requested_power_mw: float
    actual_power_mw: float
    available_capacity_compute_units: float
    completed_compute_units: float
    completed_nondeferrable_compute_units: float
    completed_deferrable_compute_units: float
    deadline_missed_compute_units: float
    utilization_fraction: float
    energy_per_completed_compute_unit_mwh: float | None
    violations: tuple[ConstraintViolation, ...]


class ComputeSystem:
    def __init__(self, config: ComputeConfig) -> None:
        self.config = config

    def initial_state(self) -> ComputeState:
        batches: tuple[WorkloadBatch, ...] = ()
        if self.config.initial_deferrable_compute_units > NUMERICAL_EPSILON:
            batches = (
                WorkloadBatch(
                    self.config.initial_deferrable_compute_units,
                    self.config.initial_deferrable_deadline_steps,
                ),
            )
        return ComputeState(deferrable_batches=batches)

    def constrain_power(
        self, requested_power_mw: float
    ) -> tuple[float, tuple[ConstraintViolation, ...]]:
        actual = clip(requested_power_mw, 0.0, self.config.max_power_mw)
        violations: list[ConstraintViolation] = []
        if abs(actual - requested_power_mw) > NUMERICAL_EPSILON:
            violations.append(
                ConstraintViolation(
                    component="compute",
                    code="power_limit",
                    message="Compute setpoint was clipped to the fleet power range.",
                    requested_value=requested_power_mw,
                    actual_value=actual,
                    unit="MW",
                )
            )
        return actual, tuple(violations)

    def capacity_for_power(self, power_mw: float, duration_hours: float) -> float:
        require_positive("duration_hours", duration_hours)
        power = clip(power_mw, 0.0, self.config.max_power_mw)
        dynamic_fraction = max(
            0.0,
            (power - self.config.idle_power_mw)
            / (self.config.max_power_mw - self.config.idle_power_mw),
        )
        return dynamic_fraction * self.config.max_useful_compute_units_per_hour * duration_hours

    def step(
        self,
        state: ComputeState,
        requested_power_mw: float,
        duration_hours: float,
        nondeferrable_compute_units: float,
        deferrable_compute_units: float,
        deferrable_deadline_steps: int,
        *,
        maximum_delivered_power_mw: float | None = None,
    ) -> ComputeStepResult:
        """Advance the queue by one interval.

        Non-deferrable work is due in this interval and is served first.
        Deferrable batches use earliest-deadline-first service.  A batch with
        ``steps_to_deadline == 1`` must be completed in this call.
        """

        require_positive("duration_hours", duration_hours)
        require_nonnegative("nondeferrable_compute_units", nondeferrable_compute_units)
        require_nonnegative("deferrable_compute_units", deferrable_compute_units)
        if deferrable_deadline_steps < 1:
            raise ValueError("deferrable_deadline_steps must be at least one")

        constrained, local_violations = self.constrain_power(requested_power_mw)
        violations = list(local_violations)
        actual = constrained
        if maximum_delivered_power_mw is not None:
            require_nonnegative("maximum_delivered_power_mw", maximum_delivered_power_mw)
            actual = min(actual, maximum_delivered_power_mw)
            if actual < constrained - NUMERICAL_EPSILON:
                violations.append(
                    ConstraintViolation(
                        component="compute",
                        code="load_shed",
                        message="Compute power was shed to maintain site power balance.",
                        severity="warning",
                        requested_value=constrained,
                        actual_value=actual,
                        unit="MW",
                    )
                )

        capacity = self.capacity_for_power(actual, duration_hours)
        nondef_completed = min(nondeferrable_compute_units, capacity)
        remaining_capacity = capacity - nondef_completed
        nondef_missed = nondeferrable_compute_units - nondef_completed

        batches = list(state.deferrable_batches)
        if deferrable_compute_units > NUMERICAL_EPSILON:
            batches.append(WorkloadBatch(deferrable_compute_units, deferrable_deadline_steps))
        batches.sort(key=lambda batch: batch.steps_to_deadline)

        next_batches: list[WorkloadBatch] = []
        deferrable_completed = 0.0
        deferrable_missed = 0.0
        for batch in batches:
            served = min(batch.remaining_compute_units, remaining_capacity)
            remaining_capacity -= served
            deferrable_completed += served
            remaining = batch.remaining_compute_units - served
            if remaining <= NUMERICAL_EPSILON:
                continue
            next_deadline = batch.steps_to_deadline - 1
            if next_deadline <= 0:
                deferrable_missed += remaining
            else:
                next_batches.append(WorkloadBatch(remaining, next_deadline))

        missed = nondef_missed + deferrable_missed
        completed = nondef_completed + deferrable_completed
        if missed > NUMERICAL_EPSILON:
            violations.append(
                ConstraintViolation(
                    component="compute",
                    code="deadline_missed",
                    message="Workload expired before it could be completed.",
                    severity="critical",
                    requested_value=(
                        nondeferrable_compute_units
                        + deferrable_compute_units
                        + state.deferrable_backlog_compute_units
                    ),
                    actual_value=completed,
                    unit="compute_units",
                )
            )

        next_state = ComputeState(
            deferrable_batches=tuple(next_batches),
            cumulative_completed_compute_units=(
                state.cumulative_completed_compute_units + completed
            ),
            cumulative_deadline_missed_compute_units=(
                state.cumulative_deadline_missed_compute_units + missed
            ),
            previous_power_mw=actual,
        )
        utilization = completed / capacity if capacity > NUMERICAL_EPSILON else 0.0
        energy_per_unit = (
            actual * duration_hours / completed if completed > NUMERICAL_EPSILON else None
        )
        return ComputeStepResult(
            state=next_state,
            requested_power_mw=requested_power_mw,
            actual_power_mw=actual,
            available_capacity_compute_units=capacity,
            completed_compute_units=completed,
            completed_nondeferrable_compute_units=nondef_completed,
            completed_deferrable_compute_units=deferrable_completed,
            deadline_missed_compute_units=missed,
            utilization_fraction=utilization,
            energy_per_completed_compute_unit_mwh=energy_per_unit,
            violations=tuple(violations),
        )


def total_backlog(batches: Iterable[WorkloadBatch]) -> float:
    """Return queued work; useful to adapters that do not own state classes."""

    return sum(batch.remaining_compute_units for batch in batches)
