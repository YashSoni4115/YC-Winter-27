import numpy as np
import pandas as pd
import pytest

from energy_ml.features import FEATURE_NAMES, feature_frame, last_row_features
from energy_ml.synthetic import WORKLOAD, generate_series


@pytest.mark.parametrize("t", [0, 1, 2, 5, 12, 13, 100, 287, 288, 289, 400, 700])
def test_serving_features_match_training_features(t):
    frame = generate_series(800, seed=3)
    values, index = frame[WORKLOAD].to_numpy(), frame.index
    horizons = np.arange(1, 12)
    served = last_row_features(values[: t + 1], index[t], horizons, 300.0)
    for i, h in enumerate(horizons):
        trained = feature_frame(values, index, int(h), 300.0).iloc[t]
        np.testing.assert_allclose(
            served.iloc[i][list(FEATURE_NAMES)].to_numpy(dtype=float),
            trained[list(FEATURE_NAMES)].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
            err_msg=f"mismatch at t={t}, h={h}",
        )


def test_features_never_depend_on_values_after_issuance():
    frame = generate_series(600, seed=1)
    values, index = frame[WORKLOAD].to_numpy().copy(), frame.index
    t = 400
    before = feature_frame(values, index, 6, 300.0).iloc[t].to_numpy(dtype=float)
    values[t + 1 :] = 1e9  # corrupt the entire future
    after = feature_frame(values, index, 6, 300.0).iloc[t].to_numpy(dtype=float)
    np.testing.assert_array_equal(before, after)


def test_rejects_bad_horizon_and_cadence():
    idx = pd.date_range("2026-01-01", periods=10, freq="5min", tz="UTC")
    with pytest.raises(ValueError):
        feature_frame(np.zeros(10), idx, 0, 300.0)
    with pytest.raises(ValueError):
        feature_frame(np.zeros(10), idx, 1, 7 * 60.0)  # does not divide a day
