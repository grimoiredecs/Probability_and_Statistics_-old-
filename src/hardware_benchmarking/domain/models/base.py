"""Shared contract for every classical regression strategy."""

from abc import ABC
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class ModelStrategy(ABC):
    """A named, inspectable wrapper around one scikit-learn estimator."""

    def __init__(self, name: str, params: Dict[str, Any] | None = None):
        self.name = name
        self.params = params or {}
        self.model: Any = None

    def fit(self, X: np.ndarray, y: pd.Series) -> "ModelStrategy":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importances(self, feature_names: List[str]) -> Optional[pd.DataFrame]:
        if hasattr(self.model, "feature_importances_"):
            values = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            values = np.abs(self.model.coef_)
        else:
            return None
        return pd.DataFrame({"Feature": feature_names, "Importance": values}).sort_values("Importance", ascending=False)
