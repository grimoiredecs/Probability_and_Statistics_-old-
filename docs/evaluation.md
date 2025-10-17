# Evaluation protocol

Each model candidate is evaluated using the same fitted preprocessing pipeline
inside five expanding-window temporal folds. The oldest 80% of observations by
`available_at` form development data; the newest 20% remain untouched until the
final test. The candidate with the highest mean cross-validated R² wins; lower
CV RMSE breaks an exact tie. Only that selected candidate is evaluated once on
the untouched hold-out set and may be promoted.

| Metric | What it answers | Direction |
| --- | --- | --- |
| R² | How much target variation does the model explain relative to predicting the mean? | Higher is better. |
| RMSE | How large are errors when large misses should be penalised strongly? | Lower is better. |
| MAE | What is the typical absolute prediction error in GHz or MHz? | Lower is better. |
| MAPE | How large is the typical error relative to the observed target? | Lower is better. |

Cross-validation reports mean R², its standard deviation, a normal-approximate
95% confidence interval, mean RMSE, mean MAE, and mean MAPE. Held-out results
report the same four core metrics. R² and RMSE remain the promotion quality
gates; MAE and MAPE make results easier to interpret and audit.

The targets are validated as positive before training, so MAPE is meaningful.
If the project later supports targets at or near zero, replace MAPE with a
zero-safe measure such as symmetric MAPE or median absolute error.
