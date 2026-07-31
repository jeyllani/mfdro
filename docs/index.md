# MFDRO

<div class="mfdro-hero" markdown>

**Point-in-time multi-frequency distribution disagreement, with the evidence
needed to audit every estimate.**

MFDRO turns a daily panel of simple returns into a reproducible path of
Wasserstein dispersion estimates. It keeps the scientific configuration,
frequency construction, random seeds, window checks, and skipped formations
visible.

[Get started](getting-started.md){ .md-button .md-button--primary }
[Understand the signal](design.md){ .md-button }

</div>

<div class="mfdro-pipeline">
daily returns → empirical frequency measures → common scale → barycenter → ρ and √ρ → audit trail
</div>

## Install

```bash
python -m pip install mfdro
```

MFDRO supports Python 3.10 and newer. See [Getting started](getting-started.md)
for a pinned installation and a complete first estimate.

## What you get

<div class="grid cards" markdown>

-   **A small, explicit API**

    Start from `MultiFrequencySignal`, choose a validated `SignalConfig`, and
    receive labelled pandas outputs. Use a preset for exploration or declare
    every scientific choice yourself.

    [First estimate →](getting-started.md)

-   **Two to many frequencies**

    Daily, weekly, monthly is only the default. Any ordered grid with at least
    two measures is supported; resampling boundaries and effective horizons
    are explicit.

    [Configure frequencies →](frequency-configuration.md)

-   **Point-in-time diagnostics**

    Inspect the complete formation schedule before paying for transport
    geometry. Warm-up, calendar mismatch, and insufficient aggregation samples
    are machine-readable.

    [Diagnose inputs →](troubleshooting.md)

-   **Portable research artifacts**

    Save estimates, audits, skipped dates, the exact configuration, a manifest,
    and SHA-256 checksums without using unsafe pickle files.

    [Preserve results →](outputs.md)

</div>

## Scope, without ambiguity

MFDRO estimates cross-frequency distribution disagreement. `rho` is a
non-negative **squared dispersion** and `sqrt_rho` is its square root. Neither
is an expected return, a buy/sell recommendation, nor an automatically
calibrated DRO radius.

The package deliberately does not download or clean production data, infer an
investable universe, solve a portfolio problem, or simulate execution. Those
steps have different assumptions and belong in explicit downstream components.
See the [end-to-end workflow](workflow.md) and [backtesting boundary](backtesting.md).

!!! warning "Alpha research software"

    The implementation is tested and typed, but the public API is not frozen
    across minor releases. Pin an exact package version and retain exported
    results for any research archive.

## Choose your next page

- New to the package: [Getting started](getting-started.md)
- Bringing real data: [Data contract](data_contract.md)
- Choosing numerical settings: [Choosing a configuration](choosing-configuration.md)
- Integrating another backtester or optimizer: [Backtesting integration](backtesting.md)
