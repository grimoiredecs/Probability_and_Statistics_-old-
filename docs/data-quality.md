# Data cleaning and imputation policy

The cleaner performs deterministic, domain-safe operations before splitting:

1. Unicode, whitespace, and known missing-value tokens are normalised.
2. Hardware specifications are parsed from unit-bearing text; MHz frequency is
   converted to GHz for CPU frequency fields.
3. Exact duplicate rows are removed.
4. Non-positive resource quantities and targets become missing; rows without a
   valid target are removed.
5. Extreme but physically possible hardware specifications are retained rather
   than arbitrarily clipped.

Feature imputation is intentionally **not** performed in the cleaner. After
the train/test split, the preprocessing pipeline fits group medians by CPU
`Vertical_Segment` or GPU `Manufacturer`, then falls back to a training-fold
global median. Categorical values use the most-frequent training-fold value.
Numeric missingness indicators are retained so models can learn whether a
specification was absent.

This applies in cross-validation as well: each fold fits its own imputation
statistics. That keeps test and validation observations out of all learned
cleaning statistics.
