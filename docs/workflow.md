# End-to-end workflow

MFDRO begins after source-data engineering and ends before portfolio policy.
Keeping those boundaries visible is essential to interpreting the result.

## Research pipeline

```text
prices and corporate actions
    ↓ user-owned transformation
point-in-time daily simple returns + membership + calendar
    ↓ MFDRO validation and aggregation
frequency-specific empirical distributions
    ↓ scaling, barycenter, transport dispersion
rho + sqrt_rho + audit artifacts
    ↓ user-owned calibration
radius, regime, or model-combination policy
    ↓ optimizer and simulator
orders, holdings, costs, and performance
```

## Before MFDRO

The caller decides and documents:

- price field, adjustment convention, currency, and timestamp convention;
- how simple returns are calculated;
- delistings, corporate actions, stale prices, and non-trading observations;
- the point-in-time investable universe for every formation month;
- the authoritative observation calendar;
- source snapshots, licences, and checksums.

MFDRO cannot reconstruct these choices from a return matrix. A full matrix can
still contain survivorship bias or future information.

## Inside MFDRO

For every formation date, the path API:

1. selects the configured calendar-month lookback;
2. selects the formation month's asset membership, if supplied;
3. rejects missing or invalid selected values;
4. checks the authoritative calendar, when supplied;
5. compounds base returns into each configured frequency;
6. places the empirical measures on the configured scale;
7. constructs the configured center and estimates dispersion;
8. records result identity, sample sizes, seed, and window evidence.

Call `validate_path_inputs` before step 6 to inspect every window without paying
for transport geometry.

## After MFDRO

`rho` must be transformed by an explicit downstream research policy. Examples
include a time-varying ambiguity radius, a regime label, or a model weight. The
mapping must be estimated using only information available at its decision
time.

A portfolio workflow must additionally specify optimization constraints,
solver behavior, execution delay, prices, costs, and accounting. MFDRO can feed
general tools such as CVXPY, cvxportfolio, vectorbt, skfolio, or an internal
engine; it does not require a particular downstream package.

## Minimum reproducible hand-off

Pass the next researcher:

- the source transformation code and frozen source identifiers;
- the daily return panel, membership ledger, and calendar used;
- `config.json` or equivalent configuration JSON;
- the complete saved `SignalPath` directory;
- exact MFDRO, Python, and dependency versions;
- the downstream policy and backtest configuration.

The package can reproduce its own calculation from those inputs. It cannot
make replaceable market-data downloads historically immutable.
