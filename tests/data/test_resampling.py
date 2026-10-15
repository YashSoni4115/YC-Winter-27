import math
from datetime import datetime, timedelta, timezone

import pytest

from energy_decision_layer.data.models import QualityFlag, SignalSemantics
from energy_decision_layer.data.resampling import IntervalSample, resample_intervals

UTC = timezone.utc
START = datetime(2026, 1, 1, tzinfo=UTC)


def test_interval_totals_are_conserved_when_downsampling() -> None:
    source = [IntervalSample(START, START + timedelta(hours=1), 10.0)]
    result = resample_intervals(
        source,
        start_utc=START,
        end_utc=START + timedelta(hours=1),
        interval=timedelta(minutes=15),
        semantics=SignalSemantics.INTERVAL_TOTAL,
    )
    assert [row.value for row in result] == pytest.approx([2.5, 2.5, 2.5, 2.5])
    assert sum(row.value for row in result) == pytest.approx(10.0)


def test_interval_means_are_duration_weighted() -> None:
    source = [
        IntervalSample(START, START + timedelta(minutes=15), 10.0),
        IntervalSample(START + timedelta(minutes=15), START + timedelta(hours=1), 30.0),
    ]
    [result] = resample_intervals(
        source,
        start_utc=START,
        end_utc=START + timedelta(hours=1),
        interval=timedelta(hours=1),
        semantics=SignalSemantics.INTERVAL_MEAN,
    )
    assert result.value == pytest.approx(25.0)


def test_state_forward_fill_never_uses_future_observation() -> None:
    source = [IntervalSample(START + timedelta(minutes=30), START + timedelta(hours=1), 2.0)]
    result = resample_intervals(
        source,
        start_utc=START,
        end_utc=START + timedelta(hours=1),
        interval=timedelta(minutes=30),
        semantics=SignalSemantics.STATE,
        max_state_age=timedelta(hours=1),
    )
    assert math.isnan(result[0].value)
    assert result[0].quality_mask & int(QualityFlag.SOURCE_MISSING)
    assert result[1].value == 2.0


def test_stale_state_is_missing() -> None:
    source = [IntervalSample(START, START + timedelta(minutes=5), 1.0)]
    result = resample_intervals(
        source,
        start_utc=START,
        end_utc=START + timedelta(hours=1),
        interval=timedelta(minutes=30),
        semantics=SignalSemantics.STATE,
        max_state_age=timedelta(minutes=10),
    )
    assert result[0].value == 1.0
    assert math.isnan(result[1].value)


def test_partial_coverage_is_explicitly_flagged() -> None:
    source = [IntervalSample(START, START + timedelta(minutes=30), 4.0)]
    [result] = resample_intervals(
        source,
        start_utc=START,
        end_utc=START + timedelta(hours=1),
        interval=timedelta(hours=1),
        semantics=SignalSemantics.INTERVAL_TOTAL,
    )
    assert result.value == pytest.approx(4.0)
    assert result.quality_mask & int(QualityFlag.SOURCE_MISSING)


def test_overlapping_numeric_intervals_must_be_deduplicated_first() -> None:
    source = [
        IntervalSample(START, START + timedelta(hours=1), 1.0),
        IntervalSample(START + timedelta(minutes=30), START + timedelta(hours=1), 2.0),
    ]
    with pytest.raises(ValueError, match="deduplicated"):
        resample_intervals(
            source,
            start_utc=START,
            end_utc=START + timedelta(hours=1),
            interval=timedelta(hours=1),
            semantics=SignalSemantics.INTERVAL_MEAN,
        )


def test_event_aggregation_uses_maximum_severity() -> None:
    source = [
        IntervalSample(START, START + timedelta(minutes=20), 1.0),
        IntervalSample(START + timedelta(minutes=20), START + timedelta(minutes=40), 3.0),
    ]
    [result] = resample_intervals(
        source,
        start_utc=START,
        end_utc=START + timedelta(hours=1),
        interval=timedelta(hours=1),
        semantics=SignalSemantics.EVENT,
    )
    assert result.value == 3.0
    assert result.quality_mask & int(QualityFlag.SOURCE_MISSING)
