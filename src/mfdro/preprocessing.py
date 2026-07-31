"""Return compounding and multi-frequency measure construction."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import cast

import numpy as np
import pandas as pd

from .exceptions import DataContractError
from .frequency import DEFAULT_FREQUENCY_SPECS, FrequencySpec


def validate_daily_returns(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy a full, wide daily simple-return panel.

    Raises :class:`DataContractError` for invalid dates, duplicate assets,
    missing or non-finite values, and simple returns below ``-1``.
    """

    if not isinstance(daily_returns, pd.DataFrame):
        raise DataContractError("daily_returns must be a pandas DataFrame.")
    if not isinstance(daily_returns.index, pd.DatetimeIndex):
        raise DataContractError("daily_returns must have a DatetimeIndex.")
    if daily_returns.empty:
        raise DataContractError("daily_returns cannot be empty.")
    if daily_returns.index.hasnans:
        raise DataContractError("The daily index cannot contain missing dates.")
    if daily_returns.index.has_duplicates or not daily_returns.index.is_monotonic_increasing:
        raise DataContractError("The daily index must be unique and increasing.")
    if daily_returns.columns.has_duplicates:
        raise DataContractError("Asset columns must be unique.")
    if len(daily_returns.columns) == 0:
        raise DataContractError("daily_returns must contain at least one asset column.")

    try:
        values = daily_returns.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataContractError("Daily returns must be numeric.") from exc
    if not np.isfinite(values).all():
        raise DataContractError("The selected daily panel must be full and finite.")
    if (values < -1.0).any():
        raise DataContractError("Simple returns below -100% are invalid.")
    converted = daily_returns.astype(np.float64)
    return cast(pd.DataFrame, converted)


def compound_returns(
    daily_returns: pd.DataFrame,
    rule: str,
    *,
    min_observations: int = 1,
    closed: str | None = None,
    label: str | None = None,
    origin: str = "start_day",
    offset: str | None = None,
) -> pd.DataFrame:
    """Compound simple returns into non-overlapping calendar periods.

    The final period may be incomplete, which is appropriate when the input
    ends on a point-in-time formation date. Empty calendar bins are removed;
    partially missing cross-sections are rejected.

    Parameters
    ----------
    daily_returns:
        Full daily simple-return panel.
    rule:
        pandas offset alias such as ``"W-FRI"`` or ``"ME"``.
    min_observations:
        Minimum base observations required per asset and bin.
    closed, label, origin, offset:
        Explicit pandas resampling boundary conventions.

    Returns
    -------
    pandas.DataFrame
        Compounded returns with the original asset order.
    """

    raw_minimum: object = min_observations
    if isinstance(raw_minimum, bool) or not isinstance(raw_minimum, Integral):
        raise ValueError("min_observations must be a positive integer.")
    if int(raw_minimum) < 1:
        raise ValueError("min_observations must be a positive integer.")
    min_observations = int(raw_minimum)
    if not isinstance(rule, str) or not rule.strip():
        raise ValueError("rule must be a non-empty pandas offset alias.")
    daily = validate_daily_returns(daily_returns)
    compounded = (1.0 + daily).resample(
        rule,
        closed=closed,
        label=label,
        origin=origin,
        offset=offset,
    ).prod(min_count=min_observations) - 1.0
    compounded = compounded.dropna(how="all")
    if compounded.empty:
        raise DataContractError(f"Resampling rule {rule!r} produced no observations.")
    if compounded.isna().any().any():
        raise DataContractError(
            f"Resampling rule {rule!r} produced a partially missing cross-section."
        )
    return cast(pd.DataFrame, compounded.loc[:, daily.columns])


def build_frequency_measures(
    daily_returns: pd.DataFrame,
    *,
    frequency_specs: Sequence[FrequencySpec] = DEFAULT_FREQUENCY_SPECS,
) -> dict[str, pd.DataFrame]:
    """Build an ordered mapping of base and compounded empirical measures.

    Returns an insertion-ordered mapping whose first value is a validated copy
    of the base panel and whose later values are compounded measures.
    """

    daily = validate_daily_returns(daily_returns)
    specs = tuple(frequency_specs)

    _validate_frequency_specs(specs)
    measures = {specs[0].name: daily}
    for spec in specs[1:]:
        if spec.rule is None:  # pragma: no cover - guarded by _validate_frequency_specs
            raise DataContractError(f"{spec.name}: missing aggregation rule.")
        measures[spec.name] = compound_returns(
            daily,
            spec.rule,
            min_observations=spec.min_observations,
            closed=spec.closed,
            label=spec.label,
            origin=spec.origin,
            offset=spec.offset,
        )
    return measures


def _validate_frequency_specs(specs: Sequence[FrequencySpec]) -> None:
    if len(specs) < 2:
        raise DataContractError("At least two frequency specifications are required.")
    if not all(isinstance(spec, FrequencySpec) for spec in specs):
        raise DataContractError("Every frequency definition must be a FrequencySpec.")
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise DataContractError("Frequency names must be unique.")
    if specs[0].rule is not None:
        raise DataContractError("The first frequency must be the unaggregated base panel.")
    missing_rules = [spec.name for spec in specs[1:] if spec.rule is None]
    if missing_rules:
        raise DataContractError(
            f"Aggregated frequencies require a resampling rule: {missing_rules}."
        )
