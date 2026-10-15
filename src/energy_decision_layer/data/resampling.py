"""Signal-aware interval resampling with explicit conservation semantics."""

from __future__ import annotations

import itertools
import math
from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from energy_decision_layer.data.models import QualityFlag, SignalSemantics


@dataclass(frozen=True, slots=True)
class IntervalSample:
    start_utc: datetime
    end_utc: datetime
    value: float
    quality_mask: int = 0

    def __post_init__(self) -> None:
        for name, timestamp in (("start_utc", self.start_utc), ("end_utc", self.end_utc)):
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
            if timestamp.utcoffset() != timedelta(0):
                raise ValueError(f"{name} must be UTC")
        if self.end_utc <= self.start_utc:
            raise ValueError("sample end must be after start")
        missing = bool(self.quality_mask & int(QualityFlag.SOURCE_MISSING))
        if not math.isfinite(self.value) and not (missing and math.isnan(self.value)):
            raise ValueError("sample value must be finite unless explicitly marked missing")
        if self.quality_mask < 0:
            raise ValueError("quality_mask cannot be negative")

    @property
    def duration_seconds(self) -> float:
        return (self.end_utc - self.start_utc).total_seconds()


def _overlap_seconds(
    left_start: datetime, left_end: datetime, right_start: datetime, right_end: datetime
) -> float:
    return max(0.0, (min(left_end, right_end) - max(left_start, right_start)).total_seconds())


def resample_intervals(
    samples: Iterable[IntervalSample],
    *,
    start_utc: datetime,
    end_utc: datetime,
    interval: timedelta,
    semantics: SignalSemantics,
    max_state_age: timedelta | None = None,
) -> list[IntervalSample]:
    """Resample ordered or unordered samples into exact target intervals.

    - interval totals are allocated in proportion to temporal overlap;
    - means/rates are duration-weighted;
    - events use the maximum overlapping severity;
    - state/instantaneous values use the most recently observed value at the
      target start and never back-fill from the future.

    Missing coverage produces ``NaN`` plus ``SOURCE_MISSING`` rather than a
    misleading zero. For totals, partial coverage is allocated but marked
    missing so callers can choose a stricter policy.
    """

    if start_utc.tzinfo is None or start_utc.utcoffset() != timedelta(0):
        raise ValueError("start_utc must be timezone-aware UTC")
    if end_utc.tzinfo is None or end_utc.utcoffset() != timedelta(0):
        raise ValueError("end_utc must be timezone-aware UTC")
    if end_utc <= start_utc:
        raise ValueError("end_utc must be after start_utc")
    if max_state_age is not None and max_state_age < timedelta(0):
        raise ValueError("max_state_age cannot be negative")
    if interval.total_seconds() <= 0:
        raise ValueError("interval must be positive")
    span = (end_utc - start_utc).total_seconds()
    if not math.isclose(span % interval.total_seconds(), 0.0, abs_tol=1e-9):
        raise ValueError("requested range must be an exact multiple of interval")

    ordered = sorted(samples, key=lambda sample: (sample.start_utc, sample.end_utc))
    if semantics in {SignalSemantics.INTERVAL_TOTAL, SignalSemantics.INTERVAL_MEAN}:
        for previous, current in itertools.pairwise(ordered):
            if current.start_utc < previous.end_utc:
                raise ValueError(
                    "overlapping source intervals must be deduplicated before resampling"
                )
    output: list[IntervalSample] = []
    target_start = start_utc.astimezone(timezone.utc)
    starts = [sample.start_utc for sample in ordered]
    cursor = 0
    active: list[IntervalSample] = []
    while target_start < end_utc:
        target_end = target_start + interval
        if semantics in {SignalSemantics.STATE, SignalSemantics.INSTANTANEOUS}:
            position = bisect_right(starts, target_start)
            selected = ordered[position - 1 : position] if position else []
        else:
            active = [sample for sample in active if sample.end_utc > target_start]
            while cursor < len(ordered) and ordered[cursor].start_utc < target_end:
                if ordered[cursor].end_utc > target_start:
                    active.append(ordered[cursor])
                cursor += 1
            selected = active
        output.append(
            _aggregate_target(
                selected,
                target_start,
                target_end,
                semantics=semantics,
                max_state_age=max_state_age,
            )
        )
        target_start = target_end
    return output


def _aggregate_target(
    samples: list[IntervalSample],
    target_start: datetime,
    target_end: datetime,
    *,
    semantics: SignalSemantics,
    max_state_age: timedelta | None,
) -> IntervalSample:
    if semantics in {SignalSemantics.STATE, SignalSemantics.INSTANTANEOUS}:
        eligible = [sample for sample in samples if sample.start_utc <= target_start]
        if eligible:
            selected = max(eligible, key=lambda sample: sample.start_utc)
            age = target_start - selected.start_utc
            if max_state_age is None or age <= max_state_age:
                mask = selected.quality_mask
                if selected.start_utc < target_start:
                    mask |= int(QualityFlag.IMPUTED_FORWARD)
                return IntervalSample(target_start, target_end, selected.value, mask)
        return _missing(target_start, target_end)

    overlaps: list[tuple[IntervalSample, float]] = []
    for sample in samples:
        overlap = _overlap_seconds(sample.start_utc, sample.end_utc, target_start, target_end)
        if overlap > 0:
            overlaps.append((sample, overlap))
    if not overlaps:
        return _missing(target_start, target_end)

    mask = 0
    for sample, _ in overlaps:
        mask |= sample.quality_mask
    covered_seconds = _union_coverage_seconds(overlaps, target_start, target_end)
    target_seconds = (target_end - target_start).total_seconds()
    if covered_seconds < target_seconds - 1e-9:
        mask |= int(QualityFlag.SOURCE_MISSING)

    if semantics is SignalSemantics.INTERVAL_TOTAL:
        value = sum(
            sample.value * overlap / sample.duration_seconds for sample, overlap in overlaps
        )
    elif semantics is SignalSemantics.INTERVAL_MEAN:
        weighted_seconds = sum(overlap for _, overlap in overlaps)
        value = sum(sample.value * overlap for sample, overlap in overlaps) / weighted_seconds
    elif semantics is SignalSemantics.EVENT:
        value = max(sample.value for sample, _ in overlaps)
    else:  # pragma: no cover - Enum exhaustiveness guard
        raise ValueError(f"unsupported semantics: {semantics}")
    return IntervalSample(target_start, target_end, value, mask)


def _union_coverage_seconds(
    overlaps: list[tuple[IntervalSample, float]], target_start: datetime, target_end: datetime
) -> float:
    spans = sorted(
        (max(sample.start_utc, target_start), min(sample.end_utc, target_end))
        for sample, _ in overlaps
    )
    merged: list[tuple[datetime, datetime]] = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return sum((end - start).total_seconds() for start, end in merged)


def _missing(start: datetime, end: datetime) -> IntervalSample:
    return IntervalSample(start, end, math.nan, int(QualityFlag.SOURCE_MISSING))
