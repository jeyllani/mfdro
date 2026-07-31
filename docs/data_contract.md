# Data contract

MFDRO separates source-data engineering from signal estimation. It does not
infer whether an observation is economically tradable and it never fills a
selected missing return.

## Daily panel

The path interface expects a wide `pandas.DataFrame`:

- rows are unique, increasing observation dates without `NaT`;
- columns are unique asset identifiers;
- at least one asset column is present;
- values are numeric simple returns;
- `-1` is valid, for example for a total loss;
- values below `-1` and infinite values are invalid.

The source may be globally sparse when a point-in-time membership map is
supplied. The selected asset-by-date matrix at each formation must nevertheless
be full. A selected missing return is always a hard failure.

Timezone-naive and timezone-aware indexes are supported. A reference calendar
must use the same timezone as the source index.

## Calendar completeness

Without `reference_calendar`, MFDRO verifies that every requested calendar
month is represented. It cannot infer a date that is absent from the source
index itself.

With an authoritative calendar, the selected daily window must match every
expected observation date exactly. This detects a partial first month or a
missing trading day.

When `formation_dates` is omitted, the calendar also defines the expected last
observation of every represented month. A source that stops before that date
fails before estimation. Explicit formation dates override the default
month-end schedule and may intentionally represent an intra-month decision.

## Asset order

Asset order is part of the numerical contract. With finitely many random
Sliced-Wasserstein directions, a coordinate permutation can change a realized
estimate unless the same permutation is applied to the directions.

When every empirical measure is a DataFrame, the first configured frequency
defines canonical asset order and later measures are reordered by label. With
unlabelled arrays, the caller is responsible for identical column order.

The path audit records a SHA-256 digest of the typed asset order. Label type and
textual value both enter this identity.

## Frequency aggregation

Simple returns are compounded:

```text
R_period = product(1 + r_t) - 1
```

The default rules are `W-FRI` and `ME`. The final aggregation bin may be shorter
than a complete calendar period when a formation date falls inside the bin.
This preserves the reference point-in-time convention.

`min_observations` applies per asset and aggregation bin. If the requested
minimum is not met for every selected asset, the partial cross-section is
rejected.

## Window semantics

For a formation date in month `M` and lookback `K`, the estimator uses calendar
months `M-K+1` through `M`, inclusive, and no observation later than the
formation date.

The successful-window audit records:

- requested start and actual end;
- sample size at every frequency;
- asset count and asset-order digest;
- matrix-fullness and no-future-observation checks;
- configuration digest and effective seed.

Formations omitted under `on_insufficient="skip"` appear in
`SignalPath.skipped` with a reason and explanatory detail.

`skip` applies to insufficient calendar coverage, authoritative-calendar
mismatch, and too few observations in a constructed frequency. Invalid
membership, selected missing data, non-finite values, and malformed
configuration remain hard failures. All three result DataFrames keep stable
columns when empty.

Use `validate_path_inputs` to inspect the same window and aggregation checks
without computing barycenters or transport dispersion.
