# Hardware Benchmarking MLOps System

An end-to-end classical machine-learning system for estimating CPU base frequency and GPU core speed from hardware specifications. It is organised as a Mark Richards-inspired Pipes-and-Filters pipeline: each stage has one responsibility, exchanges a typed context object with the next stage, and can be tested or replaced independently.

The focus is practical: reliable data preparation, leakage-aware temporal evaluation, model governance, and evidence a reviewer can inspect. Deep learning is out of scope.

## Architecture

The runnable application lives in `src/hardware_benchmarking/`. It separates the HTTP boundary, application composition, domain model strategies, pipeline mechanics, and infrastructure concerns.

```text
CLI / FastAPI ingestion
        |
application.pipeline_factory
        |
Validation -> EDA -> Cleaning -> Features -> Training -> Evaluation -> Registry -> Report
        |                                           |              |             |
infrastructure.data                         domain.models    MLflow / files   Markdown + plots
```

| Area | Responsibility |
| --- | --- |
| `api/` | FastAPI prediction endpoints and validated CSV ingestion boundary. |
| `application/` | Configuration-driven composition of a pipeline run. |
| `pipeline/` | Orchestrator, `PipelineContext`, and eight isolated filters. |
| `domain/models/` | Strategy and Factory abstractions for classical regressors. |
| `infrastructure/` | Data cleaning, Parquet feature snapshots, MLflow tracking, model registry, and reporting adapters. |

This is production-oriented local architecture, not a claim of distributed production infrastructure. The ingestion trigger uses FastAPI background work for a local demonstration; a deployed multi-instance system would require a durable queue/worker, remote artifact storage, authentication, secret management, and monitoring.

More detail: [architecture notes](docs/architecture.md).

## Methodology

### Data and features

The pipeline validates the raw CPU and GPU CSV schema before processing. Cleaning normalises text and numeric specification fields, removes invalid targets and duplicates, and preserves missing values until the fitted preprocessing stage.

Feature tables are persisted as versioned Parquet snapshots with metadata and a content hash. Stable categorical fields use one-hot encoding. Open-ended fields—CPU product collection and GPU name—use fixed-width feature hashing, allowing unseen entities without altering the feature contract.

Imputation, scaling, encoding, and hashing are fitted on training data only. The same preprocessing object is applied at inference time. The GPU feature set intentionally excludes any feature derived from `Core_Speed`, the prediction target.

### Evaluation and selection

Hardware release dates are converted to `available_at`. The oldest 80% of observations form development data; the newest 20% are a one-time held-out test set. Candidate selection uses five expanding-window temporal folds on the development partition. This is stricter than a random split and better reflects predicting later hardware from earlier releases.

| Metric | Purpose |
| --- | --- |
| R² | Explained variation relative to a mean baseline. |
| RMSE | Error magnitude with additional penalty for large misses. |
| MAE | Typical absolute prediction error in the target unit. |
| MAPE | Relative error, useful for comparing errors across target scales. |

Candidates are ranked by mean cross-validated R², with RMSE as the tie-breaker. The selected model alone is evaluated on the untouched test partition. It is promoted only if the configured R², RMSE, and MAPE gates pass. See [evaluation protocol](docs/evaluation.md) and [MLOps design notes](docs/chip-huyen-mlops.md).

### Models

The registry includes OLS, Ridge, Lasso, Elastic Net, Bayesian Ridge, Huber, RANSAC, Tweedie GLM, Random Forest, Gradient Boosting, and Extra Trees. They share a common Strategy contract and are instantiated through a Factory, so additional classical estimators do not affect pipeline orchestration.

## Results

The table below is from the latest local pipeline run under the temporal protocol above. It replaces the older, optimistic random-split figures; those were not comparable after the leakage and split corrections.

| Dataset | Selected model | Mean temporal CV R² | Held-out R² | Held-out RMSE | Held-out MAE | Held-out MAPE | Promotion decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Intel CPUs | Extra Trees | 0.6107 | 0.7813 | 0.3590 GHz | 0.2659 GHz | 21.37% | Passed |
| All GPUs | Extra Trees | 0.6124 | 0.2847 | 317.9676 MHz | 218.1585 MHz | 18.11% | Failed |

The CPU model meets its configured quality gates. The GPU model does not: temporal generalisation deteriorates on later releases and breaches its R², RMSE, and MAPE thresholds. It is therefore not promoted. This is an intentional guardrail, not a missing green tick added for decoration.

Every run exports reviewable Markdown reports with the full candidate table, EDA, feature importance where supported, prediction diagnostics, and quality-gate rationale:

- [Curated report index](docs/reports/README.md)
- [CPU report](docs/reports/cpu/report.md)
- [GPU report](docs/reports/gpu/report.md)

## Run locally

Requirements: Python 3.13 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python scripts/run_pipeline.py
open outputs/reports/latest.md  # generated for this run
```

To run the API locally:

```bash
uv run uvicorn src.hardware_benchmarking.api.app:app --port 8000
```

The interactive API schema is available at `http://localhost:8000/docs`.

## Run with Docker and MLflow

```bash
docker compose up -d --build
docker compose exec -T api-server python scripts/run_pipeline.py
```

Then open:

- Portfolio summary: `http://localhost:8000/portfolio`
- API documentation: `http://localhost:8000/docs`
- MLflow experiment tracking: `http://localhost:5001`

Stop the local stack with `docker compose down`.

## Project layout

```text
src/hardware_benchmarking/    Active application package
data/raw/                     Source CPU and GPU datasets
config/settings.yaml          Paths, model strategies, split, and quality-gate configuration
feature_store/                Versioned local Parquet feature snapshots
models/                       Local champion registry and serialized artifacts
outputs/runs/                 Per-run EDA and pipeline artifacts
outputs/reports/              Generated run reports (ignored by Git)
docs/reports/                 Curated report snapshot for GitHub readers
notebooks/                    Optional exploration; not required by the runtime pipeline
tests/                        Unit and pipeline-level checks
```

## Reproducibility checks

```bash
uv run pytest -q
```

The current suite covers the API, cleaning, feature contract, model strategies, evaluation metrics, reporting index, and CPU pipeline execution.

## License

MIT. See [LICENSE](LICENSE).
