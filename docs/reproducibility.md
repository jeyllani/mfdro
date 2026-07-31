# Reproducibility

## Configuration identity

`SignalConfig.digest` is a SHA-256 digest of the complete scientific
configuration, including every frequency and resampling convention.

```python
config = SignalConfig()
print(config.digest)
```

Changing a projection count, horizon, boundary convention, or weight changes
the digest.

Persist the versioned configuration rather than reconstructing it from prose:

```python
config.write_json("config.json")
config = SignalConfig.read_json("config.json")
```

## Random projections

Every estimate records its effective seed. A path seed is derived from:

```text
base random state | namespace | formation month
```

Namespaces are non-empty strings without `|`. Use the same namespace, asset
order, and formation month to reproduce one direction stream.

## Common random numbers

When the experimental design calls for common random numbers, estimate
competing specifications with the same effective seed. This reduces Monte
Carlo noise in their difference.

Finite sliced estimates are not trajectory-wise invariant to column order
under a fixed seed. DataFrames are aligned by label; unlabelled arrays leave
that responsibility to the caller.

## Audit outputs

The path audit includes:

- requested window start and actual end;
- formation timestamp;
- frequency-specific sample sizes;
- selected asset count and typed-order digest;
- matrix-fullness and no-future-observation flags;
- configuration digest and effective seed.

Skipped formations remain in `SignalPath.skipped`, so an omitted result differs
from a date that was never requested.

`SignalPath.save("path")` stores the three tables, configuration, package
version, row counts, and checksums. `SignalPath.load("path")` verifies those
artifacts before loading them. This detects accidental file changes; it is not
a cryptographic signature of the researcher or source vendor.

## Environment boundary

A configuration digest identifies scientific choices, not third-party
numerical implementations. Exact archival reproduction should also retain the
MFDRO version, Python version, and dependency environment.

Configuration and path formats have separate schema versions. A future package
may require an explicit migration for an old artifact rather than silently
guessing its meaning.

## What MFDRO cannot prove

MFDRO cannot determine whether a membership, correction, or source file was
actually known at the stated historical date. Preserve independently:

- source-data checksums and transformation manifests;
- the point-in-time membership ledger;
- the authoritative calendar;
- package and dependency versions;
- estimate, audit, and skipped outputs;
- downstream radius, optimizer, and backtest configurations.
