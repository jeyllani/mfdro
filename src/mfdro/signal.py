"""High-level point and walk-forward signal APIs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from io import StringIO
from numbers import Integral
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .config import SignalConfig
from .exceptions import DataContractError
from .geometry import (
    barycenter_weights,
    dispersion_weights,
    exact_dispersion,
    free_support_barycenter,
    projected_quantile_dispersion,
    scale_measures,
    sliced_dispersion,
)
from .preprocessing import build_frequency_measures
from .random import stable_seed, validate_seed
from .validation import prepare_measures

FloatArray = NDArray[np.float64]
PATH_FORMAT_VERSION = 1


class SkipReason(str, Enum):
    """Machine-readable reason why a requested formation was not estimated."""

    NON_CONTIGUOUS_MONTHS = "non_contiguous_months"
    REFERENCE_CALENDAR_MISMATCH = "reference_calendar_mismatch"
    INSUFFICIENT_FREQUENCY_OBSERVATIONS = "insufficient_frequency_observations"


@dataclass(frozen=True, slots=True)
class PathProgress:
    """One progress notification emitted by :meth:`estimate_path`."""

    completed: int
    total: int
    date: pd.Timestamp
    status: Literal["estimated", "skipped"]
    reason: SkipReason | None = None


ProgressCallback = Callable[[PathProgress], None]


@dataclass(frozen=True, slots=True)
class SignalEstimate:
    """Auditable result of one multi-frequency signal estimate.

    ``rho`` is the configured squared dispersion and ``sqrt_rho`` its square
    root. The remaining fields identify the numerical experiment. ``support``
    is returned only when requested for a free-support estimate.
    """

    rho: float
    sqrt_rho: float
    seed: int
    config_digest: str
    frequencies: tuple[str, ...]
    frequency_weights: tuple[float, ...]
    barycenter_weights: tuple[float, ...]
    sample_sizes: tuple[int, ...]
    n_assets: int
    asset_labels: tuple[object, ...] | None
    support: FloatArray | None = field(default=None, repr=False, compare=False)

    def to_record(self) -> dict[str, object]:
        """Flatten the estimate into a machine-readable record."""

        record: dict[str, object] = {
            "rho": self.rho,
            "sqrt_rho": self.sqrt_rho,
            "seed": self.seed,
            "config_digest": self.config_digest,
            "n_assets": self.n_assets,
        }
        for index, frequency in enumerate(self.frequencies):
            record[f"lambda_{frequency}"] = self.frequency_weights[index]
            record[f"barycenter_lambda_{frequency}"] = self.barycenter_weights[index]
            record[f"n_{frequency}"] = self.sample_sizes[index]
        return record

    def to_series(self) -> pd.Series:
        """Return the flattened estimate as a labelled pandas Series."""

        return pd.Series(self.to_record(), name="signal_estimate")


@dataclass(frozen=True, slots=True)
class PathDiagnostics:
    """Non-numerical readiness report for a requested walk-forward path."""

    formations: pd.DataFrame
    config: SignalConfig
    source_start: pd.Timestamp
    source_end: pd.Timestamp
    n_observations: int
    n_assets: int

    @property
    def n_formations(self) -> int:
        """Return the number of inspected formations."""

        return len(self.formations)

    @property
    def n_ready(self) -> int:
        """Return the number of formations ready for numerical estimation."""

        return int(self.formations["status"].eq("ready").sum())

    @property
    def n_insufficient(self) -> int:
        """Return the number of insufficient formations."""

        return self.n_formations - self.n_ready

    @property
    def is_usable(self) -> bool:
        """Return whether at least one formation can be estimated."""

        return self.n_ready > 0

    def summary(self) -> dict[str, object]:
        """Return a compact machine-readable diagnostic summary."""

        return {
            "source_start": self.source_start,
            "source_end": self.source_end,
            "n_observations": self.n_observations,
            "n_assets": self.n_assets,
            "n_formations": self.n_formations,
            "n_ready": self.n_ready,
            "n_insufficient": self.n_insufficient,
            "is_usable": self.is_usable,
            "config_digest": self.config.digest,
        }


@dataclass(frozen=True, slots=True)
class SignalPath:
    """Signal estimates and window-level audit produced walk-forward.

    The three DataFrames retain stable columns even when empty. Convenience
    accessors expose the most common series without discarding the complete
    audit tables.
    """

    estimates: pd.DataFrame
    audit: pd.DataFrame
    config: SignalConfig
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        if len(self.estimates) != len(self.audit):
            raise ValueError("Signal and audit paths must have the same number of rows.")
        if (
            not self.estimates.empty
            and not self.audit.empty
            and not self.estimates["date"].equals(self.audit["date"])
        ):
            raise ValueError("Signal and audit dates must be aligned exactly.")

    @property
    def rho(self) -> pd.Series:
        """Return a copy of ``rho`` indexed by successful formation date."""

        if self.estimates.empty:
            return pd.Series(
                index=pd.DatetimeIndex([], name="date"),
                dtype=np.float64,
                name="rho",
            )
        result = self.estimates.set_index("date")["rho"].copy()
        result.name = "rho"
        return cast(pd.Series, result)

    @property
    def sqrt_rho(self) -> pd.Series:
        """Return a copy of ``sqrt_rho`` indexed by successful formation date."""

        if self.estimates.empty:
            return pd.Series(
                index=pd.DatetimeIndex([], name="date"),
                dtype=np.float64,
                name="sqrt_rho",
            )
        result = self.estimates.set_index("date")["sqrt_rho"].copy()
        result.name = "sqrt_rho"
        return cast(pd.Series, result)

    @property
    def successful_dates(self) -> pd.DatetimeIndex:
        """Return successful formation dates in path order."""

        return cast(
            pd.DatetimeIndex,
            pd.DatetimeIndex(self.estimates["date"], name="date"),
        )

    @property
    def skipped_dates(self) -> pd.DatetimeIndex:
        """Return skipped formation dates in path order."""

        return cast(
            pd.DatetimeIndex,
            pd.DatetimeIndex(self.skipped["date"], name="date"),
        )

    @property
    def dispersion_weights(self) -> pd.DataFrame:
        """Return dates and normalized dispersion weights."""

        columns = ["date", *(f"lambda_{name}" for name in self.config.frequencies)]
        return cast(pd.DataFrame, self.estimates.loc[:, columns].copy())

    @property
    def center_weights(self) -> pd.DataFrame:
        """Return dates and normalized barycenter weights."""

        columns = [
            "date",
            *(f"barycenter_lambda_{name}" for name in self.config.frequencies),
        ]
        return cast(pd.DataFrame, self.estimates.loc[:, columns].copy())

    def summary(self) -> dict[str, object]:
        """Return a compact path summary without reducing the audit trail."""

        return {
            "n_estimates": len(self.estimates),
            "n_skipped": len(self.skipped),
            "first_estimate": (None if self.estimates.empty else self.estimates["date"].iloc[0]),
            "last_estimate": (None if self.estimates.empty else self.estimates["date"].iloc[-1]),
            "frequencies": self.config.frequencies,
            "config_digest": self.config.digest,
        }

    def save(self, directory: str | Path, *, overwrite: bool = False) -> Path:
        """Persist results, audit, skipped formations, config, and checksums.

        The portable JSON-table format avoids unsafe pickle deserialization and
        does not require an optional Parquet engine.
        """

        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be a boolean.")
        destination = Path(directory).expanduser().resolve()
        if destination.exists() and not destination.is_dir():
            raise FileExistsError(f"Signal path destination is not a directory: {destination}")
        if destination.exists() and any(destination.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Signal path destination is not empty: {destination}. "
                "Pass overwrite=True to replace MFDRO files."
            )
        destination.mkdir(parents=True, exist_ok=True)

        payloads = {
            "config.json": self.config.to_json(),
            "estimates.json": _frame_to_json(self.estimates),
            "audit.json": _frame_to_json(self.audit),
            "skipped.json": _frame_to_json(self.skipped),
        }
        for name, payload in payloads.items():
            destination.joinpath(name).write_text(payload, encoding="utf-8")
        manifest = {
            "format_version": PATH_FORMAT_VERSION,
            "package_version": _package_version(),
            "config_digest": self.config.digest,
            "rows": {
                "estimates": len(self.estimates),
                "audit": len(self.audit),
                "skipped": len(self.skipped),
            },
            "sha256": {
                name: hashlib.sha256(payload.encode("utf-8")).hexdigest()
                for name, payload in payloads.items()
            },
        }
        destination.joinpath("manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, directory: str | Path) -> SignalPath:
        """Load and integrity-check a path written by :meth:`save`."""

        source = Path(directory).expanduser().resolve()
        manifest_path = source.joinpath("manifest.json")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise DataContractError("Signal path manifest is missing or invalid.") from exc
        if not isinstance(manifest, Mapping):
            raise DataContractError("Signal path manifest must contain one object.")
        if manifest.get("format_version") != PATH_FORMAT_VERSION:
            raise DataContractError(
                f"Unsupported signal path format version: {manifest.get('format_version')!r}."
            )
        raw_hashes = manifest.get("sha256")
        if not isinstance(raw_hashes, Mapping):
            raise DataContractError("Signal path manifest does not contain file checksums.")

        payloads: dict[str, str] = {}
        for name in ("config.json", "estimates.json", "audit.json", "skipped.json"):
            try:
                payload = source.joinpath(name).read_text(encoding="utf-8")
            except FileNotFoundError as exc:
                raise DataContractError(f"Signal path file is missing: {name}.") from exc
            expected_hash = raw_hashes.get(name)
            observed_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            if expected_hash != observed_hash:
                raise DataContractError(f"Signal path checksum differs for {name}.")
            payloads[name] = payload

        config = SignalConfig.from_json(payloads["config.json"])
        if manifest.get("config_digest") != config.digest:
            raise DataContractError("Signal path configuration digest does not match its manifest.")
        result = cls(
            estimates=_frame_from_json(payloads["estimates.json"]),
            audit=_frame_from_json(payloads["audit.json"]),
            config=config,
            skipped=_frame_from_json(payloads["skipped.json"]),
        )
        _validate_loaded_path(result)
        return result


@dataclass(frozen=True, slots=True)
class _PathContext:
    source: pd.DataFrame
    source_index: pd.DatetimeIndex
    lookback_months: int
    dates: list[pd.Timestamp]
    memberships: dict[pd.Period, list[object]] | None
    calendar: pd.DatetimeIndex | None
    seed_namespace: str


@dataclass(frozen=True, slots=True)
class _PreparedWindow:
    formation_date: pd.Timestamp
    formation_month: pd.Period
    start_date: pd.Timestamp
    assets: list[object]
    window: pd.DataFrame
    measures: dict[str, pd.DataFrame]


@dataclass(frozen=True, slots=True)
class _InsufficientWindow:
    formation_date: pd.Timestamp
    formation_month: pd.Period
    start_date: pd.Timestamp
    n_assets: int
    reason: SkipReason
    detail: str
    sample_sizes: dict[str, int]

    def to_skipped_record(self, config_digest: str, lookback_months: int) -> dict[str, object]:
        return _skipped_record(
            formation_date=self.formation_date,
            formation_month=self.formation_month,
            start_date=self.start_date,
            lookback_months=lookback_months,
            reason=self.reason.value,
            detail=self.detail,
            config_digest=config_digest,
        )


class MultiFrequencySignal:
    """Estimate a reproducible geometric disagreement signal."""

    def __init__(self, config: SignalConfig | None = None):
        """Create an estimator from a validated scientific configuration."""

        if config is not None and not isinstance(config, SignalConfig):
            raise TypeError("config must be a SignalConfig instance or None.")
        self.config = SignalConfig.reference() if config is None else config

    def estimate(
        self,
        measures: Mapping[str, object],
        *,
        seed: int | None = None,
        include_support: bool = False,
    ) -> SignalEstimate:
        """Estimate one aligned collection of empirical measures.

        DataFrames are aligned by asset label; unlabelled arrays must already
        share column order. ``seed`` controls projected directions, while
        ``include_support`` returns a defensive copy of a free-support center.
        """

        if not isinstance(include_support, bool):
            raise TypeError("include_support must be a boolean.")
        prepared = prepare_measures(measures, self.config.frequencies)
        arrays = scale_measures(prepared.arrays, self.config)
        weights = dispersion_weights(arrays, self.config)
        center_weights = barycenter_weights(self.config)
        effective_seed = self.config.random_state if seed is None else validate_seed(seed, "seed")

        support: FloatArray | None = None
        if self.config.barycenter == "projected_quantile":
            rho = projected_quantile_dispersion(
                arrays,
                weights,
                center_weights,
                self.config,
                effective_seed,
            )
        else:
            support = free_support_barycenter(arrays, self.config, center_weights)
            if self.config.distance == "sliced":
                rho = sliced_dispersion(
                    arrays,
                    support,
                    weights,
                    self.config,
                    effective_seed,
                )
            else:
                rho = exact_dispersion(arrays, support, weights)

        if not math.isfinite(rho) or rho < 0:
            raise RuntimeError("Signal estimation did not produce a finite non-negative value.")
        return SignalEstimate(
            rho=float(rho),
            sqrt_rho=float(math.sqrt(rho)),
            seed=effective_seed,
            config_digest=self.config.digest,
            frequencies=self.config.frequencies,
            frequency_weights=tuple(float(value) for value in weights),
            barycenter_weights=tuple(float(value) for value in center_weights),
            sample_sizes=prepared.sample_sizes,
            n_assets=prepared.n_assets,
            asset_labels=prepared.asset_labels,
            support=support.copy() if include_support and support is not None else None,
        )

    def validate_path_inputs(
        self,
        daily_returns: pd.DataFrame,
        *,
        lookback_months: int,
        formation_dates: Sequence[object] | None = None,
        memberships: Mapping[object, Sequence[object]] | None = None,
        reference_calendar: Sequence[object] | None = None,
        seed_namespace: str = "signal",
    ) -> PathDiagnostics:
        """Inspect every requested window without computing transport geometry.

        Hard contract violations still raise :class:`DataContractError`.
        Ordinary warm-up, calendar, and frequency-sample insufficiencies are
        returned as diagnostic rows so users can inspect the complete schedule.
        """

        context = _prepare_path_context(
            self.config,
            daily_returns,
            lookback_months=lookback_months,
            formation_dates=formation_dates,
            memberships=memberships,
            reference_calendar=reference_calendar,
            seed_namespace=seed_namespace,
        )
        rows: list[dict[str, object]] = []
        for formation_date in context.dates:
            outcome = _prepare_window(self.config, context, formation_date)
            row: dict[str, object] = {
                "date": outcome.formation_date,
                "formation_month": outcome.formation_month.to_timestamp("M"),
                "start_date": outcome.start_date,
                "n_assets": (
                    len(outcome.assets)
                    if isinstance(outcome, _PreparedWindow)
                    else outcome.n_assets
                ),
            }
            if isinstance(outcome, _PreparedWindow):
                row.update({"status": "ready", "reason": None, "detail": None})
                sample_sizes = {name: len(frame) for name, frame in outcome.measures.items()}
            else:
                row.update(
                    {
                        "status": "insufficient",
                        "reason": outcome.reason.value,
                        "detail": outcome.detail,
                    }
                )
                sample_sizes = outcome.sample_sizes
            for name in self.config.frequencies:
                row[f"n_{name}"] = sample_sizes.get(name)
            rows.append(row)

        formations = pd.DataFrame(rows, columns=_diagnostic_columns(self.config.frequencies))
        return PathDiagnostics(
            formations=formations,
            config=self.config,
            source_start=context.source_index.min(),
            source_end=context.source_index.max(),
            n_observations=len(context.source),
            n_assets=len(context.source.columns),
        )

    def estimate_path(
        self,
        daily_returns: pd.DataFrame,
        *,
        lookback_months: int,
        formation_dates: Sequence[object] | None = None,
        memberships: Mapping[object, Sequence[object]] | None = None,
        reference_calendar: Sequence[object] | None = None,
        on_insufficient: Literal["skip", "raise"] = "skip",
        seed_namespace: str = "signal",
        progress_callback: ProgressCallback | None = None,
    ) -> SignalPath:
        """Estimate a monthly point-in-time signal path from a daily panel.

        ``progress_callback`` receives one immutable :class:`PathProgress`
        notification after every estimated or skipped formation. No progress
        dependency or terminal output is imposed by the package.
        """

        if on_insufficient not in {"skip", "raise"}:
            raise ValueError("on_insufficient must be 'skip' or 'raise'.")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable or None.")
        context = _prepare_path_context(
            self.config,
            daily_returns,
            lookback_months=lookback_months,
            formation_dates=formation_dates,
            memberships=memberships,
            reference_calendar=reference_calendar,
            seed_namespace=seed_namespace,
        )
        estimate_rows: list[dict[str, object]] = []
        audit_rows: list[dict[str, object]] = []
        skipped_rows: list[dict[str, object]] = []
        total = len(context.dates)

        for completed, formation_date in enumerate(context.dates, start=1):
            outcome = _prepare_window(self.config, context, formation_date)
            if isinstance(outcome, _InsufficientWindow):
                if on_insufficient == "raise":
                    raise DataContractError(outcome.detail)
                skipped_rows.append(
                    outcome.to_skipped_record(self.config.digest, context.lookback_months)
                )
                _notify_progress(
                    progress_callback,
                    PathProgress(
                        completed=completed,
                        total=total,
                        date=formation_date,
                        status="skipped",
                        reason=outcome.reason,
                    ),
                )
                continue

            seed = stable_seed(
                self.config.random_state,
                context.seed_namespace,
                str(outcome.formation_month),
            )
            estimate = self.estimate(outcome.measures, seed=seed)
            estimate_rows.append(
                {
                    "date": formation_date,
                    "formation_month": outcome.formation_month.to_timestamp("M"),
                    **estimate.to_record(),
                }
            )
            window_index = cast(pd.DatetimeIndex, outcome.window.index)
            audit_record: dict[str, object] = {
                "date": formation_date,
                "formation_month": outcome.formation_month.to_timestamp("M"),
                "start_date": outcome.start_date,
                "window_end": window_index.max(),
                "lookback_months": context.lookback_months,
                "n_assets": len(outcome.assets),
                "asset_order_digest": _asset_order_digest(outcome.assets),
                "no_future_observations": bool(window_index.max() <= formation_date),
                "matrix_is_full": bool(outcome.window.notna().all().all()),
                "config_digest": self.config.digest,
                "seed": seed,
            }
            for name, frame in outcome.measures.items():
                audit_record[f"n_{name}"] = len(frame)
            audit_rows.append(audit_record)
            _notify_progress(
                progress_callback,
                PathProgress(
                    completed=completed,
                    total=total,
                    date=formation_date,
                    status="estimated",
                ),
            )

        estimates = pd.DataFrame(
            estimate_rows,
            columns=_estimate_columns(self.config.frequencies),
        )
        audit = pd.DataFrame(
            audit_rows,
            columns=_audit_columns(self.config.frequencies),
        )
        skipped = pd.DataFrame(skipped_rows, columns=_skipped_columns())
        if not estimates.empty:
            estimates = estimates.sort_values("date", kind="stable").reset_index(drop=True)
            audit = audit.sort_values("date", kind="stable").reset_index(drop=True)
        if not skipped.empty:
            skipped = skipped.sort_values("date", kind="stable").reset_index(drop=True)
        return SignalPath(
            estimates=estimates,
            audit=audit,
            config=self.config,
            skipped=skipped,
        )


def _prepare_path_context(
    config: SignalConfig,
    daily_returns: pd.DataFrame,
    *,
    lookback_months: int,
    formation_dates: Sequence[object] | None,
    memberships: Mapping[object, Sequence[object]] | None,
    reference_calendar: Sequence[object] | None,
    seed_namespace: str,
) -> _PathContext:
    source = _validate_source_panel(daily_returns)
    raw_lookback: object = lookback_months
    if isinstance(raw_lookback, bool) or not isinstance(raw_lookback, Integral):
        raise ValueError("lookback_months must be a positive integer.")
    if int(raw_lookback) < 1:
        raise ValueError("lookback_months must be a positive integer.")
    canonical_lookback = int(raw_lookback)
    if not isinstance(seed_namespace, str) or not seed_namespace or "|" in seed_namespace:
        raise ValueError("seed_namespace must be a non-empty string without '|'.")
    missing_rules = [spec.name for spec in config.frequency_grid[1:] if spec.rule is None]
    if missing_rules:
        raise DataContractError(
            f"Walk-forward aggregated frequencies require resampling rules: {missing_rules}."
        )

    source_index = cast(pd.DatetimeIndex, source.index)
    calendar = _normalise_reference_calendar(reference_calendar, source_index)
    dates = _formation_dates(source_index, formation_dates, calendar)
    return _PathContext(
        source=source,
        source_index=source_index,
        lookback_months=canonical_lookback,
        dates=dates,
        memberships=_normalise_memberships(memberships),
        calendar=calendar,
        seed_namespace=seed_namespace,
    )


def _prepare_window(
    config: SignalConfig,
    context: _PathContext,
    formation_date: pd.Timestamp,
) -> _PreparedWindow | _InsufficientWindow:
    formation_month = _month_period(formation_date)
    start_month = formation_month - (context.lookback_months - 1)
    start_date = start_month.start_time
    if context.source_index.tz is not None:
        start_date = start_date.tz_localize(context.source_index.tz)

    assets: list[object]
    if context.memberships is None:
        assets = list(context.source.columns)
    else:
        if formation_month not in context.memberships:
            raise DataContractError(f"No membership was provided for {formation_month}.")
        assets = context.memberships[formation_month]
    _validate_assets(assets, context.source.columns, formation_month)
    window = context.source.loc[
        (context.source.index >= start_date) & (context.source.index <= formation_date),
        assets,
    ]
    window_index = cast(pd.DatetimeIndex, window.index)
    expected_months = pd.period_range(start_month, formation_month, freq="M")
    observed_months = _monthly_periods(window_index).unique().sort_values()
    base_sizes = {config.frequencies[0]: len(window)}
    if not observed_months.equals(expected_months):
        detail = (
            f"{formation_month}: expected {context.lookback_months} contiguous calendar "
            f"months, observed {len(observed_months)}."
        )
        return _InsufficientWindow(
            formation_date,
            formation_month,
            start_date,
            len(assets),
            SkipReason.NON_CONTIGUOUS_MONTHS,
            detail,
            base_sizes,
        )
    if context.calendar is not None:
        if formation_date not in context.calendar:
            raise DataContractError(
                f"{formation_month}: formation date is absent from the reference calendar."
            )
        expected_dates = context.calendar[
            (context.calendar >= start_date) & (context.calendar <= formation_date)
        ]
        if not window_index.equals(expected_dates):
            detail = (
                f"{formation_month}: observed dates do not match the reference calendar; "
                f"expected={len(expected_dates)}, observed={len(window_index)}."
            )
            return _InsufficientWindow(
                formation_date,
                formation_month,
                start_date,
                len(assets),
                SkipReason.REFERENCE_CALENDAR_MISMATCH,
                detail,
                base_sizes,
            )
    if window.empty or window_index.max() != formation_date:
        raise DataContractError(
            f"{formation_month}: the window does not end on the formation date."
        )
    if window.isna().any().any():
        raise DataContractError(
            f"{formation_month}: the selected point-in-time matrix is not full."
        )

    try:
        measures = build_frequency_measures(window, frequency_specs=config.frequency_grid)
    except DataContractError as exc:
        return _InsufficientWindow(
            formation_date,
            formation_month,
            start_date,
            len(assets),
            SkipReason.INSUFFICIENT_FREQUENCY_OBSERVATIONS,
            f"{formation_month}: frequency construction is insufficient: {exc}",
            base_sizes,
        )
    sample_sizes = {name: len(frame) for name, frame in measures.items()}
    insufficient = {name: size for name, size in sample_sizes.items() if size < 2}
    if insufficient:
        detail = (
            f"{formation_month}: every frequency requires at least two observations; "
            f"insufficient={insufficient}."
        )
        return _InsufficientWindow(
            formation_date,
            formation_month,
            start_date,
            len(assets),
            SkipReason.INSUFFICIENT_FREQUENCY_OBSERVATIONS,
            detail,
            sample_sizes,
        )
    return _PreparedWindow(
        formation_date=formation_date,
        formation_month=formation_month,
        start_date=start_date,
        assets=assets,
        window=window,
        measures=measures,
    )


def _validate_source_panel(daily_returns: pd.DataFrame) -> pd.DataFrame:
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
    if np.isinf(values).any():
        raise DataContractError("Infinite source returns are not permitted.")
    finite = values[np.isfinite(values)]
    if (finite < -1.0).any():
        raise DataContractError("Simple returns below -100% are invalid.")
    converted = daily_returns.astype(np.float64)
    return cast(pd.DataFrame, converted)


def _formation_dates(
    index: pd.DatetimeIndex,
    formation_dates: Sequence[object] | None,
    reference_calendar: pd.DatetimeIndex | None,
) -> list[pd.Timestamp]:
    if formation_dates is None:
        source_months = _monthly_periods(index)
        if reference_calendar is None:
            values = pd.Series(index=index, data=index).groupby(source_months).max().tolist()
        else:
            calendar_months = _monthly_periods(reference_calendar)
            values = []
            for month in source_months.unique().sort_values():
                candidates = reference_calendar[calendar_months == month]
                if candidates.empty:
                    raise DataContractError(
                        f"The reference calendar does not cover source month {month}."
                    )
                values.append(candidates.max())
    else:
        if isinstance(formation_dates, (str, bytes)):
            raise DataContractError("formation_dates must be a sequence of dates.")
        try:
            values = [pd.Timestamp(value) for value in formation_dates]  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError) as exc:
            raise DataContractError("formation_dates contains an invalid date.") from exc
    converted_dates = [pd.Timestamp(value) for value in values]
    if any(pd.isna(date) for date in converted_dates):
        raise DataContractError("formation_dates cannot contain missing dates.")
    if any(str(date.tz) != str(index.tz) for date in converted_dates):
        raise DataContractError("formation_dates and daily_returns must use the same timezone.")
    dates = sorted(set(converted_dates))
    if any(date not in index for date in dates):
        missing = [str(date.date()) for date in dates if date not in index]
        raise DataContractError(
            f"Every formation date must be an observed date; missing={missing[:5]}."
        )
    periods = [_month_period(date) for date in dates]
    if len(periods) != len(set(periods)):
        raise DataContractError("At most one formation date is permitted per month.")
    return dates


def _normalise_memberships(
    memberships: Mapping[object, Sequence[object]] | None,
) -> dict[pd.Period, list[object]] | None:
    if memberships is None:
        return None
    if not isinstance(memberships, Mapping):
        raise DataContractError("memberships must be a mapping from month to assets.")
    normalised: dict[pd.Period, list[object]] = {}
    for key, assets in memberships.items():
        try:
            period = (
                key if isinstance(key, pd.Period) else _month_period(pd.Timestamp(key))  # type: ignore[arg-type]
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise DataContractError("A membership key is not a valid month.") from exc
        period = period.asfreq("M")
        if period in normalised:
            raise DataContractError(f"Duplicate membership month {period}.")
        if isinstance(assets, (str, bytes)):
            raise DataContractError(f"{period}: membership assets must be a sequence of labels.")
        try:
            normalised[period] = list(assets)
        except TypeError as exc:
            raise DataContractError(
                f"{period}: membership assets must be a sequence of labels."
            ) from exc
    return normalised


def _normalise_reference_calendar(
    reference_calendar: Sequence[object] | None,
    source_index: pd.DatetimeIndex,
) -> pd.DatetimeIndex | None:
    if reference_calendar is None:
        return None
    if isinstance(reference_calendar, (str, bytes)):
        raise DataContractError("reference_calendar must be a sequence of dates.")
    try:
        converted = pd.to_datetime(pd.Index(list(reference_calendar)))
        calendar = cast(pd.DatetimeIndex, pd.DatetimeIndex(converted))
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataContractError("reference_calendar contains an invalid date.") from exc
    if calendar.empty:
        raise DataContractError("reference_calendar cannot be empty.")
    if calendar.hasnans:
        raise DataContractError("reference_calendar cannot contain missing dates.")
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise DataContractError("reference_calendar must be unique and increasing.")
    if calendar.tz != source_index.tz:
        raise DataContractError("reference_calendar and daily_returns must use the same timezone.")
    return calendar


def _validate_assets(
    assets: Sequence[object],
    source_columns: pd.Index,
    formation_month: pd.Period,
) -> None:
    if not assets:
        raise DataContractError(f"{formation_month}: membership is empty.")
    try:
        unique_count = len(set(assets))
    except TypeError as exc:
        raise DataContractError(f"{formation_month}: membership labels must be hashable.") from exc
    if len(assets) != unique_count:
        raise DataContractError(f"{formation_month}: membership contains duplicates.")
    missing = [asset for asset in assets if asset not in source_columns]
    if missing:
        raise DataContractError(
            f"{formation_month}: membership assets are absent from the source: {missing[:5]}."
        )


def _monthly_periods(index: pd.DatetimeIndex) -> pd.PeriodIndex:
    naive = index.tz_localize(None) if index.tz is not None else index
    return naive.to_period("M")


def _month_period(timestamp: pd.Timestamp) -> pd.Period:
    naive = timestamp.tz_localize(None) if timestamp.tz is not None else timestamp
    return naive.to_period("M")


def _estimate_columns(frequencies: Sequence[str]) -> list[str]:
    columns = [
        "date",
        "formation_month",
        "rho",
        "sqrt_rho",
        "seed",
        "config_digest",
        "n_assets",
    ]
    for frequency in frequencies:
        columns.extend(
            [
                f"lambda_{frequency}",
                f"barycenter_lambda_{frequency}",
                f"n_{frequency}",
            ]
        )
    return columns


def _audit_columns(frequencies: Sequence[str]) -> list[str]:
    columns = [
        "date",
        "formation_month",
        "start_date",
        "window_end",
        "lookback_months",
        "n_assets",
        "asset_order_digest",
        "no_future_observations",
        "matrix_is_full",
        "config_digest",
        "seed",
    ]
    columns.extend(f"n_{frequency}" for frequency in frequencies)
    return columns


def _skipped_columns() -> list[str]:
    return [
        "date",
        "formation_month",
        "start_date",
        "lookback_months",
        "reason",
        "detail",
        "config_digest",
    ]


def _diagnostic_columns(frequencies: Sequence[str]) -> list[str]:
    return [
        "date",
        "formation_month",
        "start_date",
        "status",
        "reason",
        "detail",
        "n_assets",
        *(f"n_{frequency}" for frequency in frequencies),
    ]


def _skipped_record(
    *,
    formation_date: pd.Timestamp,
    formation_month: pd.Period,
    start_date: pd.Timestamp,
    lookback_months: int,
    reason: str,
    detail: str,
    config_digest: str,
) -> dict[str, object]:
    return {
        "date": formation_date,
        "formation_month": formation_month.to_timestamp("M"),
        "start_date": start_date,
        "lookback_months": lookback_months,
        "reason": reason,
        "detail": detail,
        "config_digest": config_digest,
    }


def _asset_order_digest(assets: Sequence[object]) -> str:
    typed_assets = [
        {
            "type": f"{type(asset).__module__}.{type(asset).__qualname__}",
            "value": str(asset),
        }
        for asset in assets
    ]
    payload = json.dumps(typed_assets, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _notify_progress(callback: ProgressCallback | None, update: PathProgress) -> None:
    if callback is not None:
        callback(update)


def _frame_to_json(frame: pd.DataFrame) -> str:
    payload = frame.to_json(orient="table", date_format="iso", date_unit="ns", index=False)
    if payload is None:  # pragma: no cover - pandas returns text without a path
        raise RuntimeError("pandas did not return a JSON representation.")
    return payload + "\n"


def _frame_from_json(payload: str) -> pd.DataFrame:
    try:
        return pd.read_json(StringIO(payload), orient="table")
    except (TypeError, ValueError) as exc:
        raise DataContractError("A persisted signal table is invalid.") from exc


def _validate_loaded_path(path: SignalPath) -> None:
    expected = {
        "estimates": _estimate_columns(path.config.frequencies),
        "audit": _audit_columns(path.config.frequencies),
        "skipped": _skipped_columns(),
    }
    for name, frame in (
        ("estimates", path.estimates),
        ("audit", path.audit),
        ("skipped", path.skipped),
    ):
        if frame.columns.tolist() != expected[name]:
            raise DataContractError(f"Persisted {name} columns do not match the configuration.")
        if not frame.empty and not frame["config_digest"].eq(path.config.digest).all():
            raise DataContractError(f"Persisted {name} rows contain another configuration digest.")


def _package_version() -> str:
    try:
        return version("mfdro")
    except PackageNotFoundError:  # pragma: no cover - source is normally installed in development
        return "unknown"
