import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


class ModelRegistry:
    """
    Phase 6: Model Testing, Quality Gate Verification & Registry Promotion Module.
    Tests out-of-sample candidate model performance, verifies quality gates,
    persists model binaries (.joblib), and registers champion models in models/registry.json.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.registry_path = os.path.join(self.models_dir, "registry.json")

    def test_model(
        self, model: Any, X_test: np.ndarray, y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluates out-of-sample performance metrics.
        """
        preds = model.predict(X_test)
        r2 = float(r2_score(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        mape_percent = float(mean_absolute_percentage_error(y_test, preds) * 100)

        return {"R2_Score": r2, "RMSE": rmse, "MAE": mae, "MAPE_Percent": mape_percent}

    def verify_quality_gate(
        self, metrics: Dict[str, float], min_r2: float = 0.70, max_rmse: float = 0.50, max_mape_percent: float | None = None
    ) -> Tuple[bool, List[str]]:
        """
        Checks if candidate model satisfies quality gate criteria.
        """
        reasons = []
        passed = True

        if metrics["R2_Score"] < min_r2:
            passed = False
            reasons.append(f"R2 score ({metrics['R2_Score']:.4f}) below gate threshold ({min_r2:.4f})")

        if metrics["RMSE"] > max_rmse:
            passed = False
            reasons.append(f"RMSE ({metrics['RMSE']:.4f}) exceeds maximum threshold ({max_rmse:.4f})")

        if max_mape_percent is not None and metrics["MAPE_Percent"] > max_mape_percent:
            passed = False
            reasons.append(f"MAPE ({metrics['MAPE_Percent']:.2f}%) exceeds maximum threshold ({max_mape_percent:.2f}%)")

        return passed, reasons

    def register_and_promote_champion(
        self,
        domain_key: str,
        model_name: str,
        model: Any,
        preprocessor: Any,
        feature_names: List[str],
        metrics: Dict[str, float],
    ) -> bool:
        """
        Promotes candidate model to champion registry if it beats existing champion.
        """
        # Load registry if exists
        registry = {}
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r") as f:
                try:
                    registry = json.load(f)
                except Exception:
                    registry = {}

        existing_champion = registry.get(domain_key)
        promoted = False

        # A model trained with a different feature contract is not comparable.
        # This also invalidates champions created before a leakage fix.
        feature_contract_changed = (
            existing_champion is not None
            and existing_champion.get("feature_names") != feature_names
        )
        if existing_champion is None or feature_contract_changed or metrics["R2_Score"] > existing_champion["metrics"]["R2_Score"]:
            promoted = True

            # Save model binary
            model_filename = f"{domain_key}_champion.joblib"
            prep_filename = f"{domain_key}_preprocessor.joblib"

            model_path = os.path.join(self.models_dir, model_filename)
            prep_path = os.path.join(self.models_dir, prep_filename)

            joblib.dump(model, model_path)
            joblib.dump(preprocessor, prep_path)

            registry[domain_key] = {
                "model_name": model_name,
                "domain_key": domain_key,
                "metrics": metrics,
                "model_artifact": model_path,
                "preprocessor_artifact": prep_path,
                "feature_names": feature_names,
                "feature_contract_changed": feature_contract_changed,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            with open(self.registry_path, "w") as f:
                json.dump(registry, f, indent=2)

        elif "MAPE_Percent" not in existing_champion.get("metrics", {}):
            # One-time backward-compatible metric-contract migration. The
            # champion artifact remains unchanged; only its audited evaluation
            # record gains the newly introduced hold-out metric.
            existing_champion["metrics"].update(metrics)
            with open(self.registry_path, "w") as f:
                json.dump(registry, f, indent=2)

        return promoted
