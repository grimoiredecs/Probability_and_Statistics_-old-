# MLOps refactor notes

The pipeline uses `available_at` derived from CPU launch quarters and GPU release
dates. Feature tables are persisted as Parquet, with a hash and version manifest.
The first 80% of observations by availability time are development data; the
latest 20% are the untouched test set. Candidate validation uses expanding-window
time-series cross-validation.

Static categories (`Manufacturer`, `Vertical_Segment`, `Notebook_GPU`) use one-hot
encoding. Open-ended entity labels (`Product_Collection`, GPU `Name`) use a fixed
32-bin feature hash, so unseen hardware does not become an all-zero category.

The current data is static product metadata. Dynamic benchmark observations are
represented by `BenchmarkObservation` in `domain/contracts.py`, but require a
future timestamped benchmark feed before they should be used as online features.
