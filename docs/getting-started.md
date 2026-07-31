# Getting started

This page takes a deterministic synthetic return panel through validation,
estimation, inspection, and export. No network or external dataset is needed.

## Install

=== "Local source"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -e .
    ```

=== "PyPI (after the first release)"

    ```bash
    python -m pip install mfdro
    ```

MFDRO requires Python 3.10 or newer. Install notebook dependencies with
`python -m pip install -e ".[notebooks]"` when working from a clone.

## 1. Prepare daily simple returns

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(20250301)
dates = pd.bdate_range("2018-01-01", "2022-12-30")
factor = rng.normal(0.0002, 0.008, size=(len(dates), 1))
noise = rng.normal(0.0, 0.006, size=(len(dates), 6))
returns = pd.DataFrame(
    factor + noise,
    index=dates,
    columns=[f"asset_{index:02d}" for index in range(6)],
)
```

Rows are observation dates, columns are assets, and values are **simple
returns**, not prices or log returns. Read the [data contract](data_contract.md)
before substituting production data.

## 2. Choose an estimator

The projected preset is fast enough for exploration and explicit about being
an approximation:

```python
from mfdro import MultiFrequencySignal, SignalConfig

config = SignalConfig.projected(
    n_projections=100,
    n_quantiles=100,
    random_state=20250301,
)
engine = MultiFrequencySignal(config)
```

`SignalConfig.reference()` returns the free-support reference configuration.
Use the full constructor when changing scaling, weights, distance, or frequency
definitions. The [configuration guide](choosing-configuration.md) explains the
scientific trade-offs.

## 3. Run a preflight check

```python
diagnostics = engine.validate_path_inputs(
    returns,
    lookback_months=36,
    seed_namespace="synthetic_example",
)

print(diagnostics.summary())
print(diagnostics.formations.head())
```

This constructs and validates every rolling window without solving transport
problems. `diagnostics.n_ready` tells you how many requested formations can be
estimated; ordinary warm-up appears as an insufficient formation instead of an
exception.

## 4. Estimate the path

```python
path = engine.estimate_path(
    returns,
    lookback_months=36,
    on_insufficient="skip",
    seed_namespace="synthetic_example",
)
```

The compact accessors cover common analysis:

```python
print(path.rho.tail())
print(path.sqrt_rho.tail())
print(path.summary())
```

The complete evidence remains in three stable tables:

| Attribute | Contents |
|---|---|
| `path.estimates` | `rho`, `sqrt_rho`, weights, seeds, sample sizes, and config identity |
| `path.audit` | Window boundaries, asset identity, matrix checks, and sample sizes |
| `path.skipped` | Requested formations not estimated, with reason and detail |

```python
print(path.estimates[["date", "rho", "sqrt_rho"]].tail())
print(path.audit[["date", "n_daily", "n_weekly", "n_monthly"]].tail())
print(path.skipped[["date", "reason"]].head())
```

## 5. Preserve the experiment

```python
config.write_json("artifacts/config.json")
path.save("artifacts/signal_path")

# Later, possibly in another process:
from mfdro import SignalPath

restored = SignalPath.load("artifacts/signal_path")
assert restored.config.digest == config.digest
```

`SignalPath.save` refuses a non-empty destination by default. Pass
`overwrite=True` only when replacement is intentional. The export contains
JSON tables, the configuration, a versioned manifest, and checksums.

## Point estimate from precomputed measures

Use `estimate` when frequency measures are built by another controlled
pipeline:

```python
estimate = engine.estimate(
    {
        "daily": daily_matrix,
        "weekly": weekly_matrix,
        "monthly": monthly_matrix,
    },
    seed=1234,
)

print(estimate.rho)
print(estimate.to_series())
```

Labelled DataFrames are aligned to the first measure's columns. Unlabelled
arrays must already share exactly the same asset order.

## Next steps

- Follow the [end-to-end workflow](workflow.md) for real research data.
- Continue through the [scientific notebook suite](examples.md).
- Learn how to [read every output field](outputs.md).
