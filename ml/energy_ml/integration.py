"""Zero-argument factories for ``examples/team_smoke.py`` and ``evaluate``.

    python examples/team_smoke.py --forecaster energy_ml.integration:create_forecaster

The model path comes from ``ENERGY_ML_MODEL`` (default ``ml/artifacts/forecaster.joblib``).
Create it with ``python -m energy_ml.train``.
"""

from __future__ import annotations

import os
from pathlib import Path

from .forecaster import MLForecastProvider, QuantileForecastModel

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "forecaster.joblib"


def model_path() -> Path:
    return Path(os.environ.get("ENERGY_ML_MODEL", DEFAULT_MODEL_PATH))


def policy_path() -> Path:
    return Path(os.environ.get("ENERGY_ML_POLICY", DEFAULT_MODEL_PATH.with_name("policy.joblib")))


def create_policy():  # -> LearnedPolicy (imported lazily: it pulls in the simulator stack)
    from .policy import LearnedPolicy

    path = policy_path()
    if not path.exists():
        raise FileNotFoundError(
            f"no trained policy at {path}; run `python -m energy_ml.train_policy` first "
            "or set ENERGY_ML_POLICY"
        )
    return LearnedPolicy.load(path)


def create_forecaster() -> MLForecastProvider:
    path = model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"no trained forecaster at {path}; run `python -m energy_ml.train` first "
            "or set ENERGY_ML_MODEL"
        )
    return MLForecastProvider(QuantileForecastModel.load(path))
