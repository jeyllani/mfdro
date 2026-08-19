from __future__ import annotations

import unittest

import numpy as np

from mfdro import FrequencySpec, MultiFrequencySignal, SignalConfig
from mfdro.exceptions import DataContractError
from mfdro.geometry import (
    barycenter_weights,
    dispersion_weights,
    exact_dispersion,
    exact_dispersion_with_components,
    initial_support,
    projected_quantile_dispersion,
    projected_quantile_dispersion_with_components,
    random_directions,
    scale_measures,
    sliced_dispersion,
    sliced_dispersion_with_components,
)


class GeometryInvariantTests(unittest.TestCase):
    @staticmethod
    def two_frequency_config(**overrides: object) -> SignalConfig:
        options: dict[str, object] = {
            "frequency_specs": (
                FrequencySpec("short", 1.0),
                FrequencySpec("long", 4.0, rule="W-FRI"),
            ),
            "barycenter": "projected_quantile",
            "n_projections": 5,
            "n_quantiles": 7,
        }
        options.update(overrides)
        return SignalConfig(**options)  # type: ignore[arg-type]

    def test_power_scaling_has_known_values_and_does_not_mutate_inputs(self) -> None:
        short = np.array([[1.0, 2.0], [3.0, 4.0]])
        long = np.array([[2.0, 4.0], [6.0, 8.0]])
        original = long.copy()

        scaled = scale_measures(
            (short, long),
            self.two_frequency_config(scaling_exponent=0.5),
        )

        np.testing.assert_array_equal(scaled[0], short)
        np.testing.assert_array_equal(scaled[1], long / 2.0)
        np.testing.assert_array_equal(long, original)

    def test_realized_volatility_scaling_standardizes_every_asset(self) -> None:
        arrays = (
            np.array([[1.0, 2.0], [2.0, 6.0], [3.0, 10.0]]),
            np.array([[2.0, -1.0], [4.0, 0.0], [8.0, 1.0]]),
        )
        config = self.two_frequency_config(scaling="realized_volatility")

        scaled = scale_measures(arrays, config)

        for array in scaled:
            np.testing.assert_allclose(np.std(array, axis=0, ddof=1), np.ones(2))

        with self.assertRaises(DataContractError):
            scale_measures((np.ones((3, 1)), np.arange(3.0)[:, None]), config)

    def test_every_dispersion_weighting_rule_is_normalized_as_documented(self) -> None:
        arrays = (np.zeros((2, 1)), np.zeros((8, 1)))
        cases = {
            "uniform": np.array([0.5, 0.5]),
            "sample_size": np.array([0.2, 0.8]),
            "log_sample_size": np.log([2.0, 8.0]) / np.log([2.0, 8.0]).sum(),
            "explicit": np.array([0.25, 0.75]),
        }

        for rule, expected in cases.items():
            with self.subTest(rule=rule):
                overrides: dict[str, object] = {"frequency_weighting": rule}
                if rule == "explicit":
                    overrides["explicit_frequency_weights"] = (1.0, 3.0)
                actual = dispersion_weights(arrays, self.two_frequency_config(**overrides))
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
                self.assertEqual(float(actual.sum()), 1.0)

    def test_barycenter_weights_and_weighted_quantile_result_have_known_values(self) -> None:
        config = self.two_frequency_config(
            frequency_specs=(
                FrequencySpec("short", 1.0),
                FrequencySpec("long", 1.0, rule="W-FRI"),
            ),
            barycenter_weights=(1.0, 3.0),
        )
        np.testing.assert_allclose(barycenter_weights(config), np.array([0.25, 0.75]))

        result = MultiFrequencySignal(config).estimate(
            {
                "short": np.array([[0.0], [0.0]]),
                "long": np.array([[2.0], [2.0]]),
            },
            seed=7,
        )

        self.assertAlmostEqual(result.rho, 1.25, places=15)

    def test_exact_discrete_dispersion_has_known_value(self) -> None:
        support = np.array([[0.0]])
        arrays = (
            np.array([[1.0], [1.0]]),
            np.array([[2.0], [2.0]]),
        )

        result = exact_dispersion(arrays, support, np.array([0.25, 0.75]))

        self.assertAlmostEqual(result, 3.25, places=15)

    def test_every_geometry_exposes_components_that_reconstruct_rho(self) -> None:
        arrays = (
            np.array([[0.0], [0.0]]),
            np.array([[1.0], [1.0]]),
            np.array([[2.0], [2.0]]),
        )
        support = np.array([[1.0], [1.0]])
        weights = np.array([0.2, 0.3, 0.5])
        center_weights = np.full(3, 1.0 / 3.0)
        config = SignalConfig(
            frequency_specs=(
                FrequencySpec("short", 1.0),
                FrequencySpec("medium", 1.0, rule="W-FRI"),
                FrequencySpec("long", 1.0, rule="ME"),
            ),
            barycenter="projected_quantile",
            n_projections=7,
            n_quantiles=9,
        )
        cases = {
            "projected_quantile": (
                projected_quantile_dispersion_with_components(
                    arrays,
                    weights,
                    center_weights,
                    config,
                    17,
                ),
                projected_quantile_dispersion(
                    arrays,
                    weights,
                    center_weights,
                    config,
                    17,
                ),
            ),
            "free_support_sliced": (
                sliced_dispersion_with_components(arrays, support, weights, config, 17),
                sliced_dispersion(arrays, support, weights, config, 17),
            ),
            "free_support_exact": (
                exact_dispersion_with_components(arrays, support, weights),
                exact_dispersion(arrays, support, weights),
            ),
        }

        for name, ((rho, components), legacy_rho) in cases.items():
            with self.subTest(name=name):
                np.testing.assert_allclose(components, np.array([1.0, 0.0, 1.0]))
                self.assertAlmostEqual(rho, 0.7, places=14)
                self.assertAlmostEqual(rho, float(np.dot(weights, components)), places=14)
                self.assertEqual(legacy_rho, rho)

    def test_random_directions_are_seeded_unit_vectors(self) -> None:
        first = random_directions(17, 4, 123)
        second = random_directions(17, 4, 123)
        changed = random_directions(17, 4, 124)

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, changed))
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), np.ones(17))

    def test_free_support_size_cannot_exceed_the_pooled_sample(self) -> None:
        arrays = (np.array([[0.0], [1.0]]), np.array([[2.0], [3.0]]))

        with self.assertRaises(DataContractError):
            initial_support(
                arrays,
                size=5,
                measure_weights=np.array([0.5, 0.5]),
                random_state=0,
            )


if __name__ == "__main__":
    unittest.main()
