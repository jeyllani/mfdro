# Frequency configuration

`FrequencySpec` makes every empirical frequency explicit and auditable.

## Five-frequency example

```python
from mfdro import FrequencySpec, SignalConfig

frequencies = (
    FrequencySpec("daily", 1.0),
    FrequencySpec(
        "weekly",
        5.0,
        rule="W-FRI",
        closed="right",
        label="right",
    ),
    FrequencySpec(
        "biweekly",
        10.0,
        rule="2W-FRI",
        closed="right",
        label="right",
    ),
    FrequencySpec(
        "monthly",
        21.0,
        rule="ME",
        closed="right",
        label="right",
    ),
    FrequencySpec(
        "quarterly",
        63.0,
        rule="QE",
        closed="right",
        label="right",
    ),
)

config = SignalConfig(
    frequency_specs=frequencies,
    barycenter="free_support",
    barycenter_size=50,
    distance="sliced",
    n_projections=500,
    n_quantiles=200,
)
```

MFDRO accepts any configured number of frequencies greater than or equal to
two; three is a default, not a limit. Runtime grows with their number and sample
sizes, the barycenter support, and the number of random projections.

## Field meanings

| Field | Scientific role |
|---|---|
| `name` | Stable key in inputs, outputs, and audit columns |
| `horizon` | Effective number of base periods used by power scaling |
| `rule` | pandas offset alias used to construct the measure |
| `closed` | Side included in each resampling interval |
| `label` | Bin edge used as the resulting timestamp |
| `origin` | Anchor for fixed-frequency bins |
| `offset` | Optional shift applied to the bin anchor |
| `min_observations` | Minimum base observations required in each bin |

The first specification is the unaggregated input and has `rule=None`. Every
later specification used by `estimate_path` requires a resampling rule.

Every constructed measure must contain at least two observations. Long
frequencies therefore require a sufficiently long lookback window. Frequency
names start with a letter and contain only letters, digits, or underscores so
generated output columns remain predictable.

## Frequency weights

Uniform dispersion:

```python
SignalConfig(
    frequency_specs=frequencies,
    frequency_weighting="uniform",
)
```

Weights proportional to sample size:

```python
SignalConfig(
    frequency_specs=frequencies,
    frequency_weighting="sample_size",
)
```

Explicit weights:

```python
SignalConfig(
    frequency_specs=frequencies,
    frequency_weighting="explicit",
    explicit_frequency_weights=(5.0, 4.0, 3.0, 2.0, 1.0),
)
```

Weights are normalized internally and every supplied value must be strictly
positive. `barycenter_weights` is separate from dispersion weights so a
dispersion sensitivity does not silently change the center.

## Precomputed measures

Custom names and horizons may be used without resampling rules for a point
estimate constructed elsewhere:

```python
from mfdro import FrequencySpec, MultiFrequencySignal, SignalConfig

config = SignalConfig(
    frequency_specs=(
        FrequencySpec("one_day", 1.0),
        FrequencySpec("five_day", 5.0),
        FrequencySpec("twenty_day", 20.0),
    ),
)

estimate = MultiFrequencySignal(config).estimate(
    {
        "one_day": one_day_matrix,
        "five_day": five_day_matrix,
        "twenty_day": twenty_day_matrix,
    }
)
```

The first specification always represents the base measure. Rule-free later
specifications are valid for `estimate` because the caller supplies those
measures. The path builder requires a rule on every later specification because
it must construct them.

## Boundary convention

`closed` and `label` determine which observations enter a bin and how the bin
is dated. They enter the configuration digest.

The final bin may be incomplete when the formation date falls inside it. Use
`min_observations` to reject a bin that is too short for an experiment.

Offset aliases are validated by the installed pandas version when the
configuration is constructed. Pin pandas with the rest of the environment for
archival reproduction.

## Serialize a frequency grid

The complete grid is part of `SignalConfig.to_dict()` and `config.write_json()`.
Individual specifications also support strict round trips:

```python
payload = frequencies[1].to_dict()
restored = FrequencySpec.from_dict(payload)
assert restored == frequencies[1]
```
