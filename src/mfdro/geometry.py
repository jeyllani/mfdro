"""Geometry and weighting primitives for the ambiguity signal."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import ot
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from .config import SignalConfig
from .exceptions import DataContractError

FloatArray = NDArray[np.float64]


def scale_measures(
    arrays: Sequence[FloatArray],
    config: SignalConfig,
) -> tuple[FloatArray, ...]:
    """Place empirical measures on the configured comparable scale."""

    if config.scaling == "power":
        return tuple(
            np.asarray(array / (horizon**config.scaling_exponent), dtype=np.float64)
            for array, horizon in zip(arrays, config.horizons, strict=True)
        )

    scaled: list[FloatArray] = []
    for frequency, array in zip(config.frequencies, arrays, strict=True):
        volatility = np.std(array, axis=0, ddof=1)
        if not np.isfinite(volatility).all() or (volatility <= 1e-12).any():
            raise DataContractError(
                f"{frequency}: realized-volatility scaling encountered a zero "
                "or invalid asset volatility."
            )
        scaled.append(np.asarray(array / volatility, dtype=np.float64))
    return tuple(scaled)


def dispersion_weights(
    arrays: Sequence[FloatArray],
    config: SignalConfig,
) -> FloatArray:
    """Return positive frequency weights that sum exactly to one."""

    raw: FloatArray
    if config.frequency_weighting == "uniform":
        raw = np.ones(len(arrays), dtype=np.float64)
    elif config.frequency_weighting == "sample_size":
        raw = np.asarray([len(array) for array in arrays], dtype=np.float64)
    elif config.frequency_weighting == "log_sample_size":
        raw = np.asarray(
            np.log(np.asarray([len(array) for array in arrays], dtype=np.float64)),
            dtype=np.float64,
        )
    else:
        if config.explicit_frequency_weights is None:  # pragma: no cover - config guard
            raise DataContractError("Explicit frequency weights are unavailable.")
        raw = np.asarray(config.explicit_frequency_weights, dtype=np.float64)
    return _normalise_weights(raw)


def barycenter_weights(config: SignalConfig) -> FloatArray:
    """Return barycenter measure weights; uniform is the reference convention."""

    raw: FloatArray
    if config.barycenter_weights is None:
        raw = np.ones(len(config.frequencies), dtype=np.float64)
    else:
        raw = np.asarray(config.barycenter_weights, dtype=np.float64)
    return _normalise_weights(raw)


def initial_support(
    arrays: Sequence[FloatArray],
    size: int,
    measure_weights: FloatArray,
    *,
    random_state: int,
) -> FloatArray:
    """Initialise a free support with measure-balanced weighted k-means++."""

    pool = np.vstack(arrays)
    if size > len(pool):
        raise DataContractError(
            f"barycenter_size={size} exceeds the pooled sample size {len(pool)}."
        )
    sample_weight = np.concatenate(
        [
            np.full(len(array), measure_weights[index] / len(array), dtype=np.float64)
            for index, array in enumerate(arrays)
        ]
    )
    model = KMeans(
        n_clusters=size,
        init="k-means++",
        n_init=1,
        random_state=random_state,
    )
    model.fit(pool, sample_weight=sample_weight)
    return np.asarray(model.cluster_centers_, dtype=np.float64)


def free_support_barycenter(
    arrays: Sequence[FloatArray],
    config: SignalConfig,
    measure_weights: FloatArray,
) -> FloatArray:
    """Compute an equally weighted free-support Wasserstein barycenter."""

    support = initial_support(
        arrays,
        config.barycenter_size,
        measure_weights,
        random_state=config.barycenter_random_state,
    )
    observation_weights = [
        np.full(len(array), 1.0 / len(array), dtype=np.float64) for array in arrays
    ]
    result = ot.lp.free_support_barycenter(
        list(arrays),
        observation_weights,
        X_init=support,
        b=np.full(config.barycenter_size, 1.0 / config.barycenter_size),
        weights=measure_weights,
        numItermax=config.barycenter_max_iter,
        stopThr=config.barycenter_tolerance,
    )
    result = np.asarray(result, dtype=np.float64)
    if result.shape != (config.barycenter_size, arrays[0].shape[1]):
        raise RuntimeError("The barycenter solver returned an unexpected support shape.")
    if not np.isfinite(result).all():
        raise RuntimeError("The barycenter solver returned non-finite values.")
    return result


def random_directions(
    n_projections: int,
    dimension: int,
    seed: int,
) -> FloatArray:
    """Generate reproducible unit directions on the ambient sphere."""

    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((n_projections, dimension))
    norms = np.linalg.norm(directions, axis=1)
    while (norms == 0).any():
        mask = norms == 0
        directions[mask] = rng.standard_normal((int(mask.sum()), dimension))
        norms = np.linalg.norm(directions, axis=1)
    return np.asarray(directions / norms[:, None], dtype=np.float64)


def sliced_dispersion(
    arrays: Sequence[FloatArray],
    support: FloatArray,
    weights: FloatArray,
    config: SignalConfig,
    seed: int,
) -> float:
    """Estimate weighted sliced-Wasserstein squared dispersion."""

    quantile_grid = np.linspace(0.0, 1.0, config.n_quantiles)
    directions = random_directions(config.n_projections, arrays[0].shape[1], seed)
    total = 0.0
    for direction in directions:
        barycenter_quantiles = np.quantile(support @ direction, quantile_grid)
        total += sum(
            weights[index]
            * float(
                np.mean((np.quantile(array @ direction, quantile_grid) - barycenter_quantiles) ** 2)
            )
            for index, array in enumerate(arrays)
        )
    return _clean_nonnegative(total / config.n_projections)


def projected_quantile_dispersion(
    arrays: Sequence[FloatArray],
    weights: FloatArray,
    measure_weights: FloatArray,
    config: SignalConfig,
    seed: int,
) -> float:
    """Estimate dispersion around direction-wise projected quantile barycenters."""

    quantile_grid = np.linspace(0.0, 1.0, config.n_quantiles)
    directions = random_directions(config.n_projections, arrays[0].shape[1], seed)
    total = 0.0
    for direction in directions:
        quantiles = [np.quantile(array @ direction, quantile_grid) for array in arrays]
        barycenter_quantiles = sum(
            measure_weights[index] * quantile for index, quantile in enumerate(quantiles)
        )
        total += sum(
            weights[index] * float(np.mean((quantile - barycenter_quantiles) ** 2))
            for index, quantile in enumerate(quantiles)
        )
    return _clean_nonnegative(total / config.n_projections)


def exact_dispersion(
    arrays: Sequence[FloatArray],
    support: FloatArray,
    weights: FloatArray,
) -> float:
    """Compute weighted exact discrete squared-Wasserstein dispersion."""

    support_weights = np.full(len(support), 1.0 / len(support))
    total = 0.0
    for index, array in enumerate(arrays):
        observation_weights = np.full(len(array), 1.0 / len(array))
        cost = ot.dist(support, array, metric="sqeuclidean")
        total += weights[index] * float(ot.emd2(support_weights, observation_weights, cost))
    return _clean_nonnegative(total)


def _normalise_weights(raw: FloatArray) -> FloatArray:
    weights = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise DataContractError("Frequency weights must be finite and positive.")
    # Preserve the historical arithmetic whenever the direct sum is safe.
    # Scaling is only a fallback for otherwise valid large positive values.
    with np.errstate(over="ignore", invalid="ignore"):
        total = float(np.sum(weights, dtype=np.float64))
    if np.isfinite(total) and total > 0:
        normalised = weights / total
    else:
        scaled = weights / float(weights.max())
        scaled_total = float(np.sum(scaled, dtype=np.float64))
        if not np.isfinite(scaled_total) or scaled_total <= 0:
            raise DataContractError("Frequency weights could not be normalised safely.")
        normalised = scaled / scaled_total
    if not np.isfinite(normalised).all() or (normalised <= 0).any():
        raise DataContractError("Frequency weights are too unbalanced to remain strictly positive.")
    return np.asarray(normalised, dtype=np.float64)


def _clean_nonnegative(value: float) -> float:
    if not np.isfinite(value):
        raise RuntimeError("The estimated dispersion is not finite.")
    if value < -1e-14:
        raise RuntimeError(f"The estimated dispersion is negative: {value}.")
    return float(max(value, 0.0))
