# Notebooks

Notebooks are for exploration, visual explanation, and reproducible reports.
They must import the production package and must not duplicate cleaning,
feature engineering, model training, registry promotion, or ingestion logic.

| Notebook | Purpose |
| --- | --- |
| `01_data_understanding.ipynb` | Inspect raw datasets and create Seaborn EDA charts. |
| `02_pipeline_experiments.ipynb` | Run the configured pipeline and inspect its recorded metrics. |
| `03_results_report.ipynb` | Present registered champions and link to MLflow runs/artifacts. |

Run Jupyter from the repository root so `src/` is importable. Prototype ideas
here first, then move accepted logic into `src/hardware_benchmarking/` and cover it with
tests.
