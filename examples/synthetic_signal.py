"""Run a small, deterministic point-in-time signal example."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mfdro import MultiFrequencySignal, SignalConfig


def main() -> None:
    rng = np.random.default_rng(20250301)
    dates = pd.bdate_range("2018-01-01", "2022-12-30")
    common_factor = rng.normal(0.0002, 0.008, size=(len(dates), 1))
    idiosyncratic = rng.normal(0.0, 0.006, size=(len(dates), 6))
    returns = pd.DataFrame(
        common_factor + idiosyncratic,
        index=dates,
        columns=[f"asset_{index:02d}" for index in range(6)],
    )

    engine = MultiFrequencySignal(
        SignalConfig.projected(
            n_projections=100,
            n_quantiles=100,
            random_state=20250301,
        )
    )
    diagnostics = engine.validate_path_inputs(returns, lookback_months=36)
    print(diagnostics.summary())
    path = engine.estimate_path(returns, lookback_months=36)

    print(path.estimates[["date", "rho", "sqrt_rho", "n_assets"]].tail())
    print(path.audit[["date", "n_daily", "n_weekly", "n_monthly"]].tail())
    print(path.skipped[["date", "reason"]].head())


if __name__ == "__main__":
    main()
