# MFDRO notebook suite

The notebooks form one progressive, scientific walkthrough:

| Notebook | Purpose | Network | CI execution |
|---|---|---:|---:|
| `01_synthetic_walk_forward.ipynb` | First complete path, diagnostics, audit, and persistence | No | Yes |
| `02_frequency_lab.ipynb` | Compounding, frequency grids, sample sizes, and boundaries | No | Yes |
| `03_geometry_and_sensitivity.ipynb` | Scaling, weights, projections, barycenters, and distances | No | Yes |
| `04_point_in_time_workflow.ipynb` | Calendars, dynamic memberships, skips, and hard failures | No | Yes |
| `05_reproducibility_and_artifacts.ipynb` | Configuration identity, deterministic reruns, and checksums | No | Yes |
| `06_yfinance_case_study.ipynb` | Optional demonstration of the external data boundary | Yes | No |

From the package root:

```bash
python -m pip install -e ".[notebooks]"
python -m pip install jupyterlab
jupyter lab examples/notebooks
```

Start with notebook 01 and continue in numerical order. The five offline
notebooks contain assertions and are executed automatically. The Yahoo Finance
case study is validated structurally but is never a package test because a live
vendor response is not a reproducible fixture.

Notebooks are committed without stored output. This keeps reviews readable and
forces examples to run against the reader's installed environment.
