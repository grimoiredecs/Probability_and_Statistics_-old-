# Production architecture

`src/hardware_benchmarking` is the only runnable Python application package. The
repository uses Pipes & Filters for the retraining workflow and a conventional
boundary-oriented layout around it.

```text
HTTP API / CLI
     |
application/pipeline_factory.py
     |
pipeline/orchestrator.py
     |
validation -> eda -> cleaning -> features -> training -> evaluation -> registry -> report
     |                                                                  |          |
infrastructure/data                                                   registry + tracking   reporting
     |
domain/models
```

## Boundaries

- `api/` accepts HTTP requests, validates transport concerns, and never
  implements ML logic.
- `application/` composes a configured use case; it owns no data parsing or
  estimator implementation.
- `pipeline/` contains the reusable Pipes & Filters mechanics and filters.
- `domain/models/` owns the classical-model Strategy/Factory contract.
- `infrastructure/` implements side effects: CSV persistence, feature snapshots,
  model artifacts, and MLflow.

## Operational reality

Docker Compose is suitable for local development and demonstrations. A real
multi-instance deployment still needs a durable queue/worker for retraining,
object storage or a database for artifacts, secrets management, authentication,
observability, and a deployment platform. `BackgroundTasks` deliberately
remains a development-scale trigger, not a distributed job scheduler.
