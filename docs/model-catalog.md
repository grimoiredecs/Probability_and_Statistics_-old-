# Classical model catalogue

The pipeline evaluates a deliberately mixed set of classical regressors using
the same leakage-safe preprocessing and five-fold cross-validation. The winner
is selected by mean cross-validated R² and then checked once on the held-out
test split.

| Family | Model | Why it is included | Practical interpretation |
| --- | --- | --- | --- |
| Linear | OLS | Unregularised baseline | Tests whether the relationship is broadly linear. |
| Linear | Ridge | L2 regularisation | Stable when encoded hardware categories are correlated. |
| Linear | Lasso | L1 regularisation | Produces a sparse, interpretable feature set. |
| Linear | Elastic Net | L1 + L2 regularisation | More stable than Lasso for correlated specifications. |
| Bayesian | Bayesian Ridge | Posterior regularised linear model | Useful principled baseline when coefficient uncertainty matters. |
| Robust | Huber | Downweights large residuals | Defends against noisy or incorrectly scraped specifications. |
| Robust | RANSAC | Fits a consensus set | Diagnostic candidate for grossly corrupted rows; it may legitimately lose on clean data. |
| GLM | Tweedie | Positive continuous-response model | Suitable alternative when target distributions are positive and skewed. |
| Ensemble | Random Forest / Extra Trees | Non-linear, interaction-aware trees | Strong practical baselines for heterogeneous hardware data. |
| Ensemble | Gradient Boosting | Sequential error correction | Captures smooth non-linear effects with compact ensembles. |

RANSAC and Tweedie are evaluated rather than assumed to be best. A candidate
that cannot fit a future input is recorded as skipped without taking down the
entire retraining run. Tree models use one worker by default to avoid hidden
process fan-out inside a container; increase parallelism only after setting
explicit resource limits in deployment.
