"""Input validation and canonical asset alignment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .exceptions import DataContractError

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PreparedMeasures:
    """Canonical numerical representation of frequency-specific measures."""

    frequencies: tuple[str, ...]
    arrays: tuple[FloatArray, ...]
    asset_labels: tuple[object, ...] | None

    @property
    def n_assets(self) -> int:
        return int(self.arrays[0].shape[1])

    @property
    def sample_sizes(self) -> tuple[int, ...]:
        return tuple(int(array.shape[0]) for array in self.arrays)


def prepare_measures(
    measures: Mapping[str, object],
    frequencies: tuple[str, ...],
) -> PreparedMeasures:
    """Validate, align and convert empirical measures to float64 arrays.

    DataFrame inputs are aligned to the exact column order of the first
    configured frequency. Array inputs must already share the same asset order.
    Mixing labelled and unlabelled inputs is rejected because it makes this
    ordering unverifiable.
    """

    if not isinstance(measures, Mapping):
        raise DataContractError("measures must be a mapping from frequency names to matrices.")
    expected = set(frequencies)
    observed = set(measures)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise DataContractError(f"Frequency keys differ: missing={missing}, extra={extra}.")

    values = [measures[name] for name in frequencies]
    labelled = [isinstance(value, pd.DataFrame) for value in values]
    if any(labelled) and not all(labelled):
        raise DataContractError("Do not mix DataFrame and unlabelled array measures.")

    asset_labels: tuple[object, ...] | None = None
    arrays: list[FloatArray] = []
    if all(labelled):
        frames = [value for value in values if isinstance(value, pd.DataFrame)]
        first = frames[0]
        if first.columns.has_duplicates:
            raise DataContractError("Asset labels must be unique.")
        asset_labels = tuple(first.columns)
        first_assets = set(asset_labels)
        for frequency, frame in zip(frequencies, frames, strict=True):
            if frame.index.has_duplicates:
                raise DataContractError(f"{frequency}: observation index contains duplicates.")
            if frame.columns.has_duplicates or set(frame.columns) != first_assets:
                raise DataContractError(f"{frequency}: asset labels do not match.")
            arrays.append(_validate_array(frame.loc[:, list(asset_labels)].to_numpy(), frequency))
    else:
        arrays = [
            _validate_array(np.asarray(value), frequency)
            for frequency, value in zip(frequencies, values, strict=True)
        ]

    n_assets = arrays[0].shape[1]
    if any(array.shape[1] != n_assets for array in arrays[1:]):
        raise DataContractError("All measures must have the same asset dimension.")

    return PreparedMeasures(
        frequencies=frequencies,
        arrays=tuple(arrays),
        asset_labels=asset_labels,
    )


def _validate_array(value: object, frequency: str) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"{frequency}: observations must be numeric.") from exc
    if array.ndim != 2:
        raise DataContractError(f"{frequency}: expected a two-dimensional matrix.")
    if array.shape[0] < 2:
        raise DataContractError(f"{frequency}: at least two observations are required.")
    if array.shape[1] < 1:
        raise DataContractError(f"{frequency}: at least one asset is required.")
    if not np.isfinite(array).all():
        raise DataContractError(f"{frequency}: NaN or infinite values are not permitted.")
    return np.ascontiguousarray(array)
