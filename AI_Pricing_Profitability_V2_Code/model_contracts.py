"""Contracts shared by forecasting implementations and their consumers."""

from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd


class DemandPredictor(Protocol):
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return one nonnegative demand estimate per feature row."""


class DemandEstimatorProvider(Protocol):
    def build_estimators(self) -> Mapping[str, Any]:
        """Build fresh named estimators for one model-comparison run."""