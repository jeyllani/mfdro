# Performance

Runtime depends more on the numerical mode and multivariate dimension than on
the number of output rows alone.

## Main cost drivers

| Setting or input | Effect |
|---|---|
| Number of formation dates | Repeats the complete rolling calculation |
| Number of assets | Increases multivariate projection and transport work |
| Number and length of frequencies | Adds empirical measures and samples |
| `n_projections` | Approximately linear work in sliced/projected calculations |
| `n_quantiles` | Increases projected-quantile work and resolution |
| `barycenter_size` | Increases free-support optimization and transport matrices |
| `barycenter_max_iter` | Caps free-support iterations |
| `distance="exact"` | Solves discrete transport rather than projected approximations |

Exact complexity also depends on the POT, NumPy, scikit-learn, and linear
algebra implementations in the installed environment.

## Practical workflow

1. Use `validate_path_inputs` to eliminate input problems without geometry.
2. Develop the pipeline with `SignalConfig.projected` and a modest number of
   projections and quantiles.
3. Benchmark a representative subset of dates and assets.
4. Increase precision settings as a documented convergence analysis.
5. Run the declared research configuration and retain its exact config JSON.

Do not tune settings only until a backtest looks attractive. Numerical
convergence and economic model selection are separate questions.

## Progress reporting

Pass `progress_callback` to integrate a logger or progress bar. The callback is
invoked after each estimated or skipped formation and receives completed count,
total count, date, status, and optional skip reason.

## Parallelism

The public path API currently evaluates formation dates sequentially. This
preserves a simple deterministic execution model and avoids imposing a process
backend. If an external research system parallelizes independent point
estimates, it must preserve formation-specific seeds, asset order, configuration
identity, and deterministic output ordering.

## Memory

The source daily panel, selected rolling window, frequency measures, and
transport work arrays coexist during an estimate. Free-support exact transport
can be materially heavier than sliced or projected modes. Benchmark with the
largest intended asset universe and lookback rather than extrapolating only
from a toy example.
