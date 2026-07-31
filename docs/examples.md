# Examples and notebooks

The repository ships one script and six notebooks. Each notebook has one
scientific purpose and remains small enough to audit cell by cell.

## Install notebook dependencies

```bash
python -m pip install -e ".[notebooks]"
python -m jupyter lab examples/notebooks
```

## Offline scientific sequence

The first five notebooks require no network and use fixed random seeds:

| Notebook | Scientific focus |
|---|---|
| `01_synthetic_walk_forward.ipynb` | Complete first path, diagnostics, audit, and persistence |
| `02_frequency_lab.ipynb` | Compounding, 2/3/5-frequency grids, sample sizes, and boundaries |
| `03_geometry_and_sensitivity.ipynb` | Scaling, separate weights, projected convergence, free support, and exact distance |
| `04_point_in_time_workflow.ipynb` | Calendars, dynamic memberships, skips, hard failures, and future invariance |
| `05_reproducibility_and_artifacts.ipynb` | Config/data/environment identities, deterministic reruns, manifests, and checksums |

Read them in numerical order. Each visualization answers a stated numerical or
data-contract question; none reports synthetic portfolio performance.

## Optional Yahoo Finance walkthrough

`examples/notebooks/06_yfinance_case_study.ipynb` shows an explicitly configured
`yfinance.download` request, adjusted-close extraction, coverage inspection,
simple-return construction without filling, and the MFDRO path workflow.

!!! warning "Convenience data is not a reproducibility fixture"

    The notebook requires network access. Yahoo can revise data or service
    behavior, and a fixed present-day ticker list is not a point-in-time
    universe. Preserve governed source snapshots and historically valid
    memberships before using this workflow for research claims.

The yfinance project is independent of MFDRO. Review its
[download documentation](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html),
[project notice](https://github.com/ranaroussi/yfinance), and the applicable
Yahoo terms before use.

## Command-line script

Run the smallest offline example without Jupyter:

```bash
python examples/synthetic_signal.py
```

## Validation policy

Notebook JSON, cell identifiers, code syntax, and absence of stored outputs are
checked automatically. All five offline workflows execute in CI. The Yahoo
Finance notebook is not executed there because a live third-party response
cannot be a stable package test.
