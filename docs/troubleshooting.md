# Diagnostics and troubleshooting

Run preflight diagnostics before a large estimate:

```python
diagnostics = engine.validate_path_inputs(
    returns,
    lookback_months=36,
    formation_dates=formation_dates,
    memberships=memberships,
    reference_calendar=trading_calendar,
    seed_namespace="research_universe",
)

print(diagnostics.summary())
print(diagnostics.formations.query("status != 'ready'"))
```

Preflight uses the same window and frequency preparation as `estimate_path`,
but does not compute barycenters or transport distances.

## Insufficient versus invalid

`on_insufficient="skip"` applies to ordinary data availability:

- warm-up without all requested calendar months;
- disagreement with an authoritative calendar;
- too few observations after a configured aggregation.

These formations are expected in many rolling experiments and are returned
with a reason. Structural or numerical defects raise immediately because
silently skipping them could hide a research error.

## Common failures

### “selected return matrix contains missing values”

The membership active at that formation selected an asset/date cell containing
`NaN`. MFDRO does not impute. Correct the upstream data or provide a historically
valid membership that excludes the asset; do not forward-fill returns.

### “reference calendar does not match selected observations”

The supplied calendar and source disagree within the window. Inspect timezone,
first/last date, exchange holidays, and whether the panel contains only dates
common to all assets. The reference calendar must describe the observations
expected for this experiment.

### “aggregated frequencies require resampling rules”

`estimate_path` constructs later frequencies from the base panel, so every
`FrequencySpec` after the first needs `rule`. Rule-free measures are supported
only when passed directly to `estimate`.

### “too few observations”

Each empirical measure needs at least two rows. Increase the lookback, shorten
the longest frequency, or revise `min_observations`. Inspect `n_<frequency>` in
the preflight table before changing the design.

### Free-support size exceeds the pooled sample

`barycenter_size` cannot exceed the available pooled empirical observations.
Use fewer center atoms, a longer window, or the projected preset. A large
support is not automatically a better statistical model.

### Results change when columns are reordered

Labelled DataFrames are aligned by name. Raw arrays have no labels and must use
the same asset order at every frequency. With finitely many random projections,
coordinate order is part of the realized numerical experiment.

### Re-running gives a different result

Compare `config.digest`, effective seed, formation month, seed namespace, asset
order, sample sizes, package version, and dependency environment. The config
digest alone does not include data or dependencies.

## Progress without a package dependency

The path API accepts any callback. For example:

```python
from mfdro import PathProgress

def report(update: PathProgress) -> None:
    print(f"{update.completed}/{update.total}: {update.date.date()} — {update.status}")

path = engine.estimate_path(
    returns,
    lookback_months=36,
    progress_callback=report,
)
```

This works with a simple function, a logger, a notebook widget, or an adapter to
`tqdm`; MFDRO does not impose a terminal UI dependency.

## Exception classes

- `ConfigurationError`: inconsistent scientific choices or frequency fields;
- `DataContractError`: data, calendar, membership, or persisted-artifact defect;
- `MFDROError`: common package base class for both.

Catch the specific class whenever the recovery action differs.
