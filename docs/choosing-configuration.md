# Choosing a configuration

There is no universally optimal configuration. Choose settings from the
scientific question, then hold them fixed or treat them as a documented
sensitivity analysis.

## Start with a preset

=== "Exploration"

    ```python
    from mfdro import SignalConfig

    config = SignalConfig.projected(
        n_projections=100,
        n_quantiles=100,
        random_state=20250301,
    )
    ```

    `projected_quantile` avoids a multivariate free-support solve. It is useful
    for examples, pipeline development, and broad robustness grids.

=== "Reference"

    ```python
    from mfdro import SignalConfig

    config = SignalConfig.reference()
    ```

    The reference preset uses a free-support barycenter, sliced distance, 50
    center atoms, and 200 directions. It is a declared project reference, not a
    claim of statistical optimality.

Configurations are immutable. Derive a validated variant without repeating
unchanged fields:

```python
weighted = config.with_updates(frequency_weighting="sample_size")
```

## Decision table

| Question | Main options | Practical consequence |
|---|---|---|
| How should horizons be made comparable? | `power`, `realized_volatility` | Changes the geometry before any barycenter calculation |
| How is the center represented? | `free_support`, `projected_quantile` | Free support stores one multivariate center; projected quantiles define a center direction by direction |
| How is dispersion evaluated? | `sliced`, `exact` | Exact transport is typically much more expensive and requires `free_support` |
| How much does each frequency affect dispersion? | uniform, sample size, log sample size, explicit | Changes `lambda_k`, not the center weights |
| How much does each frequency affect the center? | uniform or `barycenter_weights` | Changes `beta_k`, independently of dispersion weights |
| How much Monte Carlo precision? | `n_projections` | More directions usually reduce projection noise and increase runtime |
| How fine is the projected quantile grid? | `n_quantiles` | More grid points increase resolution and work |
| How large is the free-support center? | `barycenter_size` | More atoms increase flexibility, memory, and optimal-transport work |

## Declare a full experiment

```python
from mfdro import FrequencySpec, SignalConfig

config = SignalConfig(
    frequency_specs=(
        FrequencySpec("daily", 1.0),
        FrequencySpec("weekly", 5.0, rule="W-FRI", min_observations=3),
        FrequencySpec("monthly", 21.0, rule="ME", min_observations=10),
    ),
    scaling="power",
    scaling_exponent=0.5,
    frequency_weighting="explicit",
    explicit_frequency_weights=(1.0, 1.0, 1.0),
    barycenter="free_support",
    barycenter_size=50,
    barycenter_weights=(1.0, 1.0, 1.0),
    barycenter_random_state=0,
    distance="sliced",
    n_projections=500,
    n_quantiles=200,
    random_state=20250301,
    barycenter_max_iter=30,
    barycenter_tolerance=1e-4,
)
```

All positive weights are normalized internally. Frequency horizons are not
inferred from offset aliases: `horizon=21.0` is a scientific effective-horizon
choice, while `rule="ME"` is a calendar aggregation choice.

## Recommended robustness checks

At minimum, inspect sensitivity to:

- projection count and random seed;
- lookback length;
- frequency grid and effective horizons;
- power versus volatility scaling;
- barycenter construction;
- center and dispersion weights;
- minimum observations in partial aggregation bins.

Use the same effective seeds across comparable projected specifications when
common random numbers are part of the design. Do not select the best-looking
configuration from the final backtest without accounting for that selection.

## Save the exact choice

```python
config.write_json("config.json")
restored = SignalConfig.read_json("config.json")

assert restored == config
assert restored.digest == config.digest
```

Configuration JSON has an explicit schema version and strict fields. The
SHA-256 digest identifies the serialized scientific choices; it does not
identify source data or numerical dependency versions.
