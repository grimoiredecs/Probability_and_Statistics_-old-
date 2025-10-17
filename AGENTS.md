Technologies: R, Python, Pandas, NumPy, scikit-learn, Matplotlib
• Built a hardware benchmarking analysis workflow using R, later extended into a Python 3 data-processing pipeline.
• Applied statistical learning methods to analyze benchmark results, compare hardware performance, and identify performance
patterns.
• Processed experimental benchmark data and generated visualizations to support performance interpretation and reporting.


I want to implement the full MLOps pipeline with Automated Workflow though
From:
Data Validation
Exploratory Data Analysis
Data Cleaning (with imputation too if possible)
Feature Storing
Model Training
Model Testing

And looping back

(note we will evaluate on the classical ML stuff, no Deep Learnin)

Remodel and refactor ths project to make it:

Follow Pipeline Software Architecture according to Mark Richards' book (make it modular by design, in terms of algorithms and models and such)
I've turned on docker, so please use any MLOps tools if possible to make it more polish and production ready
Make it such that new Data may get inserted and trigger the pipeline again for real MLOps flow

## Current project context

The intended active implementation is the Python package under
`src/hardware_benchmarking/`. It implements a Mark Richards-inspired Pipes & Filters
workflow:

1. `ValidationFilter`
2. `EDAFilter`
3. `DataCleaningFilter`
4. `FeatureEngineeringFilter`
5. `ModelTrainingFilter`
6. `ModelEvaluationFilter`
7. `RegistryPromotionFilter`

The main entry points are:

- `scripts/run_pipeline.py` for CPU and GPU retraining.
- `src/hardware_benchmarking/api/app.py` for FastAPI inference and CSV ingestion.
- `data/raw/Intel_CPUs.csv` and `data/raw/All_GPUs.csv` for raw inputs.
- `config/settings.yaml` for intended pipeline configuration.

Use `src/hardware_benchmarking/` as the source of truth. Its production layout is:
`api/` (HTTP boundary), `application/` (use cases/composition),
`pipeline/` (pipes, context, and filters), `domain/models/` (Strategy and
Factory), and `infrastructure/` (data, registry, and tracking adapters).
Historical duplicate implementations are isolated under `archive/legacy-python/`
and must not receive features or be imported by the active application.

## MLOps requirements and guardrails

- Keep the project limited to classical machine-learning models; no deep
  learning is in scope.
- Preserve the modular Pipes & Filters design. Filters should communicate
  through `PipelineContext`; model additions belong behind the Strategy and
  Factory abstractions.
- Load paths, quality gates, selected strategies, and other runtime choices
  from `config/settings.yaml` rather than duplicating hard-coded values.
- Treat the current CSV feature store as an interim local store, not a true
  versioned feature store. New work should add immutable run/version metadata,
  dataset hashes, schema information, and raw-data lineage.
- MLflow is configured in Docker Compose but is not yet integrated in code.
  When adding it, log runs, parameters, metrics, artifacts, dataset/version
  lineage, and promoted model metadata; do not claim tracking exists before
  those records are actually written.
- Ingestion must validate an upload before mutating the canonical raw dataset.
  Use schema compatibility checks, atomic writes, duplicate handling, and a
  durable/recoverable execution record before presenting it as production
  automation. FastAPI `BackgroundTasks` is acceptable for local development,
  not durable orchestration.
- Avoid tests that write model, registry, feature-store, or EDA artifacts into
  tracked working directories. Use temporary test directories and fixtures.

## Data-science correctness: mandatory safeguards

- Never engineer a predictor from the prediction target. In particular,
  `Core_to_Mem_Speed_Ratio = Core_Speed / Memory_Speed` leaks the GPU target
  `Core_Speed` and MUST NOT be an input feature when predicting `Core_Speed`.
  Remove it or replace it with a feature available at inference time, then
  retrain and re-baseline GPU metrics.
- Training, validation, and inference must use identical feature definitions.
  Do not train with a target-derived feature and substitute a constant in the
  API.
- Split data before fitting imputation, scaling, and encoding. During
  cross-validation, wrap preprocessing plus estimator in a scikit-learn
  `Pipeline` so each fold fits preprocessing only on its training portion.
- Report held-out metrics only after leakage checks and use reproducible split
  seeds. Do not rely on previously recorded GPU metrics until the leakage fix
  is verified.

## Repository hygiene

- The MLOps refactor is currently largely untracked by Git. Before considering
  it deliverable, review and intentionally commit the source, configuration,
  Docker, scripts, tests, and documentation that form the runnable system.
- Keep generated models, feature snapshots, temporary uploads, EDA images, and
  IDE/OS files out of source commits unless a deliberately versioned artifact
  is required.
- Keep the README consistent with the active package path and `data/raw/`
  locations. Verify commands from a clean clone and container before claiming
  production readiness.
