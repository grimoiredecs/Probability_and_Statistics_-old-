"""Reader-facing, reproducible report artifacts for completed pipeline runs."""

from __future__ import annotations

from pathlib import Path
from shutil import copy2
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class RunReportGenerator:
    """Exports Markdown and compact visuals without making a notebook a runtime dependency."""

    def __init__(self, reports_dir: str = "outputs/reports") -> None:
        self.reports_dir = Path(reports_dir)
        sns.set_theme(style="whitegrid", palette="deep")

    @staticmethod
    def _markdown_table(frame: pd.DataFrame) -> str:
        display = frame.copy()
        for column in display.select_dtypes(include=[np.number]).columns:
            display[column] = display[column].map(lambda value: f"{value:.4f}")
        headers = [str(column) for column in display.columns]
        rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
        separator = ["---"] * len(headers)
        return "\n".join(
            ["| " + " | ".join(headers) + " |", "| " + " | ".join(separator) + " |"]
            + ["| " + " | ".join(row) + " |" for row in rows]
        )

    @staticmethod
    def _save_model_comparison(cv_summary: pd.DataFrame, path: Path) -> None:
        ordered = cv_summary.sort_values("CV_R2_Mean", ascending=True)
        figure, axis = plt.subplots(figsize=(10, max(5, len(ordered) * 0.5)))
        sns.barplot(data=ordered, x="CV_R2_Mean", y="Model", hue="Model", legend=False, ax=axis)
        axis.set(title="Expanding-window cross-validation performance", xlabel="Mean R²", ylabel="Model strategy")
        figure.tight_layout()
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def _save_prediction_diagnostics(y_true: pd.Series, y_pred: np.ndarray, directory: Path) -> tuple[str, str]:
        values = np.asarray(y_true)
        residuals = values - y_pred
        lower, upper = min(values.min(), y_pred.min()), max(values.max(), y_pred.max())

        actual_path = directory / "actual_vs_predicted.png"
        figure, axis = plt.subplots(figsize=(7, 6))
        sns.scatterplot(x=values, y=y_pred, alpha=0.7, ax=axis)
        axis.plot([lower, upper], [lower, upper], "--", color="#c0392b", label="Perfect prediction")
        axis.set(title="Held-out test: actual vs predicted", xlabel="Actual target", ylabel="Predicted target")
        axis.legend()
        figure.tight_layout()
        figure.savefig(actual_path, dpi=180, bbox_inches="tight")
        plt.close(figure)

        residual_path = directory / "residuals.png"
        figure, axis = plt.subplots(figsize=(7, 5))
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.7, ax=axis)
        axis.axhline(0, linestyle="--", color="#c0392b")
        axis.set(title="Held-out test residual diagnostics", xlabel="Predicted target", ylabel="Actual − predicted")
        figure.tight_layout()
        figure.savefig(residual_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return actual_path.name, residual_path.name

    @staticmethod
    def _save_feature_importance(strategy: Any, feature_names: list[str], directory: Path) -> str | None:
        importance = strategy.get_feature_importances(feature_names)
        if importance is None or importance.empty:
            return None
        top = importance.head(15).sort_values("Importance", ascending=True)
        path = directory / "feature_importance.png"
        figure, axis = plt.subplots(figsize=(10, 7))
        sns.barplot(data=top, x="Importance", y="Feature", hue="Feature", legend=False, ax=axis)
        axis.set(title="Top 15 model feature importances", xlabel="Importance", ylabel="Feature")
        figure.tight_layout()
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return path.name

    @staticmethod
    def _copy_eda_images(context: Any, assets_dir: Path) -> list[str]:
        if not context.artifact_dir:
            return []
        source_dir = Path(context.artifact_dir) / "eda"
        copied: list[str] = []
        for image in sorted(source_dir.glob("*.png")):
            destination = assets_dir / image.name
            copy2(image, destination)
            copied.append(destination.name)
        return copied

    def generate(self, context: Any) -> Path:
        report_dir = self.reports_dir / context.run_id / context.domain_key
        assets_dir = report_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        comparison_path = assets_dir / "model_comparison.png"
        self._save_model_comparison(context.cv_summary, comparison_path)
        predictions = context.fitted_models[context.champion_model_name].predict(context.X_test)
        actual_name, residual_name = self._save_prediction_diagnostics(context.y_test, predictions, assets_dir)
        feature_name = self._save_feature_importance(
            context.fitted_models[context.champion_model_name], context.feature_names, assets_dir
        )
        eda_images = self._copy_eda_images(context, assets_dir)

        metrics = context.champion_metrics
        gate_status = "PASSED — champion eligible for registry promotion" if context.gate_passed else "FAILED — champion was not promoted"
        reasons = "; ".join(context.reasons) if context.reasons else "All configured quality gates passed."
        report = [
            f"# {context.domain_key.upper()} hardware benchmark model report",
            "",
            f"- **Run ID:** `{context.run_id}`",
            f"- **Dataset:** `{context.raw_filepath}`",
            f"- **Target:** `{context.target_col}`",
            f"- **Champion:** {context.champion_model_name}",
            f"- **Pipeline duration:** {(context.execution_time_sec or (time.time() - context.start_timestamp)):.3f}s",
            "",
            "## Decision",
            "",
            f"**Quality gate: {gate_status}.** {reasons}",
            "",
            "## Held-out future test performance",
            "",
            "| R² | RMSE | MAE | MAPE |",
            "| --- | --- | --- | --- |",
            f"| {metrics['R2_Score']:.4f} | {metrics['RMSE']:.4f} | {metrics['MAE']:.4f} | {metrics['MAPE_Percent']:.2f}% |",
            "",
            "The held-out partition contains the newest 20% of records. Model selection uses five expanding temporal folds on the earlier training partition; this avoids learning from future hardware releases.",
            "",
            "## Candidate comparison",
            "",
            "![Cross-validation model comparison](assets/model_comparison.png)",
            "",
            self._markdown_table(context.cv_summary),
            "",
            "## Held-out diagnostics",
            "",
            f"![Actual versus predicted](assets/{actual_name})",
            "",
            f"![Residual diagnostics](assets/{residual_name})",
            "",
        ]
        if feature_name:
            report.extend(["## Model interpretation", "", f"![Feature importance](assets/{feature_name})", ""])
        if eda_images:
            report.extend(["## Data overview", ""])
            report.extend([f"![Exploratory data analysis: {image}](assets/{image})\n" for image in eda_images])
        report.extend([
            "## Data quality and lineage",
            "",
            f"- Raw rows validated: {context.validation_report.get('record_count', 'n/a')}",
            f"- Cleaned rows: {len(context.cleaned_data) if context.cleaned_data is not None else 'n/a'}",
            f"- Feature snapshot: `{context.artifact_dir}`",
            "- Numeric imputation, scaling, categorical encoding, and feature hashing are fit only on training folds.",
            "- Dynamic entity fields are feature-hashed, so unseen product families can be represented without changing the feature contract.",
            "",
            "## Reproduce",
            "",
            "```bash\nuv run python scripts/run_pipeline.py\n```",
            "",
            "For the containerized run and MLflow UI, use `docker compose up -d --build`, then run `docker compose exec -T api-server python scripts/run_pipeline.py`.",
        ])
        report_path = report_dir / "report.md"
        report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
        return report_path
