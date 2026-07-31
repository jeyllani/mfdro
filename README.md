# MFDRO

MFDRO estimates reproducible, point-in-time disagreement between multivariate
return distributions observed at multiple frequencies. It constructs empirical
measures, places them on comparable scales, estimates a Wasserstein center, and
returns a non-negative squared dispersion `rho` together with a complete audit
trail.

> **Status:** alpha research software. The numerical core is tested, but the
> public API is not yet frozen and the project has not yet been released on
> PyPI.

## Why MFDRO

- Two or more user-configurable frequencies; three is only the default.
- Free-support or projected-quantile barycenter construction.
- Sliced or exact discrete transport dispersion where compatible.
- Strict point-in-time rolling windows, memberships, and optional calendars.
- Preflight diagnostics before expensive numerical work.
- Deterministic seeds, versioned configuration JSON, and portable result
  bundles with checksums.
- Typed public API, stable empty schemas, analytical tests, and a 95% branch
  coverage gate.

MFDRO does **not** acquire production data, infer investable universes,
calibrate an ambiguity radius, optimize portfolios, or simulate trades. It is a
focused signal component designed to work with whichever data, optimizer, and
backtesting system a research project chooses.

## Installation

From a local clone:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

After the first PyPI release:

```bash
python -m pip install mfdro
```

For the bundled notebooks:

```bash
python -m pip install -e ".[notebooks]"
```

## Quick start

```python
import numpy as np
import pandas as pd

from mfdro import MultiFrequencySignal, SignalConfig

rng = np.random.default_rng(20250301)
dates = pd.bdate_range("2018-01-01", "2022-12-30")
returns = pd.DataFrame(
    rng.normal(0.0, 0.01, size=(len(dates), 6)),
    index=dates,
    columns=[f"asset_{index:02d}" for index in range(6)],
)

config = SignalConfig.projected(
    n_projections=100,
    n_quantiles=100,
    random_state=20250301,
)
engine = MultiFrequencySignal(config)

diagnostics = engine.validate_path_inputs(returns, lookback_months=36)
assert diagnostics.is_usable

path = engine.estimate_path(
    returns,
    lookback_months=36,
    on_insufficient="skip",
    seed_namespace="synthetic_example",
)

print(path.rho.tail())
print(path.audit[["date", "n_daily", "n_weekly", "n_monthly"]].tail())
print(path.skipped[["date", "reason"]].head())

config.write_json("artifacts/config.json")
path.save("artifacts/signal_path")
```

`rho` is a squared cross-frequency dispersion—not an expected return, trading
direction, or automatically calibrated DRO radius. Any mapping into a portfolio
policy is a separate point-in-time research decision.

## More than three frequencies

```python
from mfdro import FrequencySpec, SignalConfig

config = SignalConfig(
    frequency_specs=(
        FrequencySpec("daily", 1.0),
        FrequencySpec("weekly", 5.0, rule="W-FRI"),
        FrequencySpec("biweekly", 10.0, rule="2W-FRI"),
        FrequencySpec("monthly", 21.0, rule="ME"),
        FrequencySpec("quarterly", 63.0, rule="QE"),
    )
)
```

Horizons, resampling rules, boundary conventions, minimum observations, weights,
and numerical settings all enter the configuration digest.

## Documentation

The MkDocs site covers the [data contract](docs/data_contract.md),
[configuration choices](docs/choosing-configuration.md),
[output schemas](docs/outputs.md), [reproducibility](docs/reproducibility.md),
and [backtesting integration](docs/backtesting.md). The notebook suite covers
frequency construction, geometry, point-in-time controls, reproducible
artifacts, and an optional Yahoo Finance case study.

Build the site locally:

```bash
python -m pip install -e ".[docs]"
python -m mkdocs serve
```

Then open `http://127.0.0.1:8000`.

## Development

```bash
python -m pip install -e ".[test,docs,dev,notebooks]"
python -m pytest --cov=mfdro --cov-report=term-missing
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m mypy src
python -m mkdocs build --strict
python -m build
python -m twine check dist/*
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and
[CHANGELOG.md](CHANGELOG.md) before preparing a change or release.
