import numpy as np

from src.hardware_benchmarking.infrastructure.registry import ModelRegistry


class FixedPredictor:
    def predict(self, X):
        return np.array([110.0, 180.0])


def test_holdout_evaluation_reports_four_interpretable_metrics(tmp_path):
    metrics = ModelRegistry(models_dir=str(tmp_path)).test_model(
        FixedPredictor(), np.zeros((2, 1)), np.array([100.0, 200.0])
    )

    assert set(metrics) == {"R2_Score", "RMSE", "MAE", "MAPE_Percent"}
    assert metrics["MAE"] == 15.0
    assert metrics["MAPE_Percent"] == 10.0
