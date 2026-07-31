# Backtesting integration

MFDRO is a signal component, not a backtesting engine.

## Intended pipeline

```text
point-in-time returns and memberships
    → MFDRO rho and audit
    → radius or regime policy
    → portfolio optimizer
    → target weights
    → next tradable execution
    → holdings, costs, and return ledger
    → performance and inference
```

The separation is intentional. Data eligibility, optimization, execution, and
accounting rules vary by market and should not be hidden inside the signal
estimator.

## Timing rule

An estimate dated `t` may use observations through `t`. A backtest must not
assume that the resulting weights were held during the same observations.
Execution must occur at a price and timestamp that were actually available
after the signal and optimization completed.

Do not apply a blind row shift without checking the trading calendar. Map each
formation timestamp to an explicit decision timestamp and next tradable
execution timestamp.

## Typical integration

```python
path = engine.estimate_path(
    returns,
    lookback_months=36,
    formation_dates=formation_dates,
    memberships=memberships,
    reference_calendar=trading_calendar,
    on_insufficient="raise",
    seed_namespace="research_universe",
)

signal = path.estimates.set_index("date")["rho"]
radius = radius_policy(signal)  # User-defined and point-in-time.
target_weights = optimizer.solve(radius=radius, data=optimizer_inputs)
ledger = simulator.run(target_weights, execution_schedule)
```

`radius_policy`, `optimizer`, and `simulator` are deliberately not supplied by
MFDRO.

## Complementary packages

- [CVXPY](https://www.cvxpy.org/) is appropriate for expressing a custom convex
  DRO portfolio problem and selecting an open-source or commercial solver.
- [cvxportfolio](https://www.cvxportfolio.com/) provides portfolio policies,
  constraints, transaction-cost models, and a simulator.
- [vectorbt](https://vectorbt.dev/) is useful for fast vectorized diagnostics,
  order simulation, fees, slippage, and parameter sweeps.
- [skfolio](https://skfolio.org/) provides portfolio estimators and time-aware
  model-selection tools such as walk-forward and purged cross-validation.
- [Zipline Reloaded](https://zipline.ml4trading.io/) provides an event-driven
  engine with market calendars and data bundles.

These packages should remain optional integrations, not mandatory MFDRO
dependencies.

## Minimum backtest ledger

A rigorous downstream ledger should record at least:

- signal formation timestamp;
- information cutoff;
- asset membership and order;
- radius-policy inputs and output;
- optimizer status and solver;
- target and realized weights;
- execution timestamp and price convention;
- turnover, fees, slippage, and holding costs;
- delisting and corporate-action treatment;
- gross and net returns;
- configuration and source-data identities.

MFDRO's audit should be retained alongside this ledger rather than reduced to a
single merged return series.
