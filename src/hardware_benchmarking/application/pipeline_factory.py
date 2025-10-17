"""Configuration-driven composition root for the active pipeline."""

from pathlib import Path
from uuid import uuid4

from ..config import DomainSettings, Settings
from ..pipeline import Pipeline, PipelineContext
from ..pipeline.filters import (DataCleaningFilter, EDAFilter, FeatureEngineeringFilter,
                      ModelEvaluationFilter, ModelTrainingFilter,
                      RegistryPromotionFilter, ReportExportFilter, ValidationFilter)
from ..infrastructure.tracking import RunTracker


def build_pipeline(settings: Settings, domain: DomainSettings, run_id: str | None = None) -> tuple[Pipeline, PipelineContext]:
    context = PipelineContext(domain_key=domain.domain_key, raw_filepath=domain.raw_filepath,
                              target_col=domain.target_col, run_id=run_id or uuid4().hex)
    artifact_root = Path(settings.directories["outputs"]) / "runs" / context.run_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    context.artifact_dir = str(artifact_root)
    context.tracker = RunTracker(domain.domain_key, context.run_id)
    context.tracker.log_params({"raw_filepath": domain.raw_filepath, "target": domain.target_col,
                                "min_r2": domain.min_r2, "max_rmse": domain.max_rmse})
    pipeline = (
        ValidationFilter()
        | EDAFilter(output_dir=str(artifact_root / "eda"))
        | DataCleaningFilter()
        | FeatureEngineeringFilter(store_dir=str(Path(settings.directories["feature_store"]) / "runs" / context.run_id))
        | ModelTrainingFilter(
            strategies=settings.strategies,
            cv_folds=int(settings.evaluation["cv_folds"]),
            random_state=int(settings.evaluation["random_state"]),
        )
        | ModelEvaluationFilter(
            min_r2=domain.min_r2,
            max_rmse=domain.max_rmse,
            max_mape_percent=settings.objectives.get(domain.domain_key, {}).get("max_mape_percent"),
        )
        | RegistryPromotionFilter()
        | ReportExportFilter(reports_dir=str(Path(settings.directories["outputs"]) / "reports"))
    )
    return pipeline, context
