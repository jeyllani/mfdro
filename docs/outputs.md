# Understanding outputs

MFDRO separates the estimate itself from the evidence that the rolling window
was constructed as requested.

## Point estimate

`MultiFrequencySignal.estimate` returns `SignalEstimate`.

| Field | Meaning |
|---|---|
| `rho` | Configured non-negative squared cross-frequency dispersion |
| `sqrt_rho` | Square root of `rho`, in the scaled return unit |
| `seed` | Effective uint32 seed used by projected calculations |
| `config_digest` | SHA-256 identity of the complete `SignalConfig` |
| `frequency_weights` | Normalized dispersion weights, in configured order |
| `barycenter_weights` | Normalized center weights, in configured order |
| `sample_sizes` | Observation count for every empirical measure |
| `n_assets` | Shared multivariate dimension |
| `asset_labels` | Canonical labels when DataFrames were supplied |
| `support` | Optional free-support center, only when requested and applicable |

`to_record()` flattens the fields into audit-friendly names;
`to_series()` creates a labelled pandas Series.

## Walk-forward path

`estimate_path` returns `SignalPath` with three tables.

### `estimates`

One row per successful formation. Alongside `rho`, `sqrt_rho`, seed, config
identity, and asset count, every frequency contributes:

- `lambda_<name>`: normalized dispersion weight;
- `barycenter_lambda_<name>`: normalized center weight;
- `n_<name>`: empirical sample size.

Convenience views return copies:

```python
path.rho
path.sqrt_rho
path.successful_dates
path.dispersion_weights
path.center_weights
```

### `audit`

One aligned row per successful estimate. It records requested and observed
window endpoints, lookback, selected asset count, typed asset-order digest,
matrix fullness, no-future-observation status, config digest, seed, and every
frequency sample size.

The audit is evidence about package-visible inputs. It does not prove that the
source or membership was historically available.

### `skipped`

One row per requested formation omitted under `on_insufficient="skip"`.
Machine-readable reasons are:

| Reason | Meaning |
|---|---|
| `non_contiguous_months` | The complete calendar-month lookback was not represented |
| `reference_calendar_mismatch` | Observed dates differ from the supplied authoritative calendar |
| `insufficient_frequency_observations` | At least one configured aggregation produced too few usable bins |

Malformed configuration, invalid membership, selected missing values, and
non-finite values remain hard errors rather than skipped observations.

## Empty outputs are stable

All three tables retain their documented columns even if every requested
formation is skipped. This makes downstream concatenation and schema checks
predictable.

## Save and verify a path

```python
destination = path.save("artifacts/path")
restored = SignalPath.load(destination)
```

The directory contains:

```text
path/
├── audit.json
├── config.json
├── estimates.json
├── manifest.json
└── skipped.json
```

The manifest records the path-format version, package version, row counts,
configuration digest, and a SHA-256 checksum for each payload. Loading verifies
checksums, schema, row alignment, and configuration identity before returning
an object.

The JSON-table format is portable and avoids pickle deserialization. For a
long-lived archive, also retain source-data identities and the complete Python
environment; JSON persistence does not make upstream data reproducible.
