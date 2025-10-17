"""Small MLflow adapter: tracking is optional locally, never silently faked."""

import os
from pathlib import Path
from typing import Any


class RunTracker:
    def __init__(self, domain_key: str, run_id: str):
        self.enabled = False
        self._mlflow = None
        try:
            import mlflow
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("hardware-benchmarking")
            mlflow.start_run(run_name=f"{domain_key}-{run_id}")
            mlflow.set_tags({"domain": domain_key, "pipeline": "pipes-and-filters", "run_id": run_id})
            self._mlflow, self.enabled = mlflow, True
        except Exception as exc:  # local pipeline must still work without MLflow
            print(f"[Tracking] MLflow unavailable; continuing without remote tracking: {exc}")

    def log_params(self, params: dict[str, Any]) -> None:
        if self.enabled:
            self._mlflow.log_params({key: str(value) for key, value in params.items()})

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if self.enabled:
            self._mlflow.log_metrics(metrics)

    def log_artifact(self, path: str | Path) -> None:
        if self.enabled and Path(path).exists():
            self._mlflow.log_artifact(str(path))

    def finish(self, failed: bool = False) -> None:
        if self.enabled:
            self._mlflow.end_run(status="FAILED" if failed else "FINISHED")
