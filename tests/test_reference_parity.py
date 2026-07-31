from __future__ import annotations

import unittest

import numpy as np
import ot
from sklearn.cluster import KMeans

from mfdro import MultiFrequencySignal, SignalConfig


class NotebookReferenceParityTests(unittest.TestCase):
    def test_free_support_sliced_formula_matches_reference_engine(self) -> None:
        """Lock the package to the independently written notebook convention."""

        rng = np.random.default_rng(314)
        raw = [
            rng.normal(0.0, 0.01, size=(24, 4)),
            rng.normal(0.0, 0.02, size=(8, 4)),
            rng.normal(0.0, 0.04, size=(3, 4)),
        ]
        config = SignalConfig(
            barycenter="free_support",
            barycenter_size=4,
            distance="sliced",
            n_projections=13,
            n_quantiles=17,
            barycenter_max_iter=8,
            barycenter_tolerance=1e-5,
        )
        seed = 987654

        package_result = MultiFrequencySignal(config).estimate(
            dict(zip(config.frequencies, raw, strict=True)),
            seed=seed,
        )
        reference = _independent_notebook_formula(raw, config, seed)

        self.assertAlmostEqual(package_result.rho, reference, places=15)


def _independent_notebook_formula(
    raw: list[np.ndarray],
    config: SignalConfig,
    seed: int,
) -> float:
    arrays = [
        array / (horizon**config.scaling_exponent)
        for array, horizon in zip(raw, config.horizons, strict=True)
    ]
    frequency_weights = np.ones(3) / 3.0

    pool = np.vstack(arrays)
    sample_weight = np.concatenate(
        [np.full(len(array), 1.0 / (3.0 * len(array))) for array in arrays]
    )
    initializer = KMeans(
        n_clusters=config.barycenter_size,
        init="k-means++",
        n_init=1,
        random_state=0,
    )
    initializer.fit(pool, sample_weight=sample_weight)
    support = ot.lp.free_support_barycenter(
        arrays,
        [np.full(len(array), 1.0 / len(array)) for array in arrays],
        X_init=initializer.cluster_centers_,
        b=np.full(config.barycenter_size, 1.0 / config.barycenter_size),
        weights=np.ones(3) / 3.0,
        numItermax=config.barycenter_max_iter,
        stopThr=config.barycenter_tolerance,
    )

    projection_rng = np.random.default_rng(seed)
    quantile_grid = np.linspace(0.0, 1.0, config.n_quantiles)
    total = 0.0
    for _ in range(config.n_projections):
        direction = projection_rng.standard_normal(arrays[0].shape[1])
        direction /= np.linalg.norm(direction)
        barycenter_quantiles = np.quantile(support @ direction, quantile_grid)
        total += sum(
            frequency_weights[index]
            * np.mean((np.quantile(array @ direction, quantile_grid) - barycenter_quantiles) ** 2)
            for index, array in enumerate(arrays)
        )
    return float(total / config.n_projections)


if __name__ == "__main__":
    unittest.main()
