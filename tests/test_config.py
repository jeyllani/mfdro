from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

import numpy as np

from mfdro import (
    ConfigurationError,
    DataContractError,
    FrequencySpec,
    MFDROError,
    MultiFrequencySignal,
    SignalConfig,
)
from mfdro.random import stable_seed


class SignalConfigTests(unittest.TestCase):
    def test_public_exceptions_share_one_package_base_class(self) -> None:
        self.assertTrue(issubclass(ConfigurationError, MFDROError))
        self.assertTrue(issubclass(DataContractError, MFDROError))

    def test_default_configuration_has_stable_identity(self) -> None:
        first = SignalConfig()
        second = SignalConfig()

        self.assertEqual(first.digest, second.digest)
        self.assertEqual(len(first.digest), 64)

    def test_configuration_identity_changes_with_scientific_choice(self) -> None:
        baseline = SignalConfig()
        alternative = SignalConfig(n_projections=baseline.n_projections + 1)

        self.assertNotEqual(baseline.digest, alternative.digest)

    def test_invalid_frequency_contract_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            SignalConfig(
                frequency_specs=(
                    FrequencySpec("daily", 1.0),
                    FrequencySpec("daily", 5.0, rule="W-FRI"),
                )
            )

        with self.assertRaises(ConfigurationError):
            SignalConfig(frequency_specs=(FrequencySpec("daily", 1.0),))

    def test_unknown_scientific_modes_and_invalid_seeds_are_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            SignalConfig(scaling="unknown")  # type: ignore[arg-type]

        with self.assertRaises(ConfigurationError):
            SignalConfig(random_state=-1)

    def test_explicit_weights_require_exact_positive_vector(self) -> None:
        with self.assertRaises(ConfigurationError):
            SignalConfig(frequency_weighting="explicit")

        with self.assertRaises(ConfigurationError):
            SignalConfig(
                frequency_weighting="explicit",
                explicit_frequency_weights=(1.0, 0.0, 1.0),
            )

    def test_stable_seed_is_repeatable_and_label_sensitive(self) -> None:
        first = stable_seed(42, "universe", "2020-01")
        second = stable_seed(42, "universe", "2020-01")
        changed = stable_seed(42, "universe", "2020-02")

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**32)

    def test_configuration_canonicalises_mutable_sequences(self) -> None:
        frequency_specs = [
            FrequencySpec("daily", 1.0),
            FrequencySpec("weekly", 5.0, rule="W-FRI"),
            FrequencySpec("monthly", 21.0, rule="ME"),
        ]
        dispersion = [3.0, 2.0, 1.0]
        center = [1.0, 1.0, 1.0]
        config = SignalConfig(
            frequency_specs=frequency_specs,  # type: ignore[arg-type]
            frequency_weighting="explicit",
            explicit_frequency_weights=dispersion,  # type: ignore[arg-type]
            barycenter_weights=center,  # type: ignore[arg-type]
        )
        digest = config.digest

        frequency_specs[0] = FrequencySpec("changed", 2.0)
        dispersion[0] = 99.0
        center[0] = 99.0

        self.assertEqual(config.frequencies, ("daily", "weekly", "monthly"))
        self.assertEqual(config.horizons, (1.0, 5.0, 21.0))
        self.assertEqual(config.explicit_frequency_weights, (3.0, 2.0, 1.0))
        self.assertEqual(config.barycenter_weights, (1.0, 1.0, 1.0))
        self.assertEqual(config.digest, digest)

    def test_integer_scientific_fields_reject_floats_and_booleans(self) -> None:
        invalid = {
            "barycenter_size": 2.5,
            "n_projections": 2.5,
            "n_quantiles": 2.5,
            "barycenter_max_iter": 2.5,
            "n_projections_bool": True,
        }
        for name, value in invalid.items():
            field = "n_projections" if name == "n_projections_bool" else name
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(ConfigurationError),
            ):
                SignalConfig(**{field: value})  # type: ignore[arg-type]

    def test_large_positive_weights_are_normalised_without_overflow(self) -> None:
        measures = {
            "daily": np.array([[0.0], [1.0]]),
            "weekly": np.array([[1.0], [2.0]]),
            "monthly": np.array([[2.0], [3.0]]),
        }
        config = SignalConfig(
            frequency_specs=(
                FrequencySpec("daily", 1.0),
                FrequencySpec("weekly", 1.0, rule="W-FRI"),
                FrequencySpec("monthly", 1.0, rule="ME"),
            ),
            frequency_weighting="explicit",
            explicit_frequency_weights=(1e308, 1e308, 1e308),
            barycenter="projected_quantile",
            n_projections=3,
            n_quantiles=5,
        )

        result = MultiFrequencySignal(config).estimate(measures)
        np.testing.assert_allclose(result.frequency_weights, np.ones(3) / 3.0)
        self.assertTrue(np.isfinite(result.rho))

    def test_research_seed_stream_is_preserved_and_ambiguous_labels_fail(self) -> None:
        self.assertEqual(stable_seed(20250301, "big_caps", "2010-06"), 1845922314)
        with self.assertRaises(ConfigurationError):
            stable_seed(42, "a|b", "c")

    def test_invalid_configuration_variants_fail_at_construction(self) -> None:
        valid_specs = (
            FrequencySpec("base", 1.0),
            FrequencySpec("week", 5.0, rule="W-FRI"),
        )
        cases: list[tuple[str, dict[str, object]]] = [
            ("frequency string", {"frequency_specs": "daily"}),
            ("frequency non-sequence", {"frequency_specs": 1}),
            ("invalid specification member", {"frequency_specs": (*valid_specs, object())}),
            ("unknown weighting", {"frequency_weighting": "unknown"}),
            ("unknown barycenter", {"barycenter": "unknown"}),
            ("unknown distance", {"distance": "unknown"}),
            ("one frequency", {"frequency_specs": (FrequencySpec("daily", 1.0),)}),
            ("unused explicit weights", {"explicit_frequency_weights": (1.0, 1.0, 1.0)}),
            ("short center weights", {"barycenter_weights": (1.0, 1.0)}),
            ("zero tolerance", {"barycenter_tolerance": 0.0}),
            (
                "incompatible projected distance",
                {"barycenter": "projected_quantile", "distance": "exact"},
            ),
            ("boolean exponent", {"scaling_exponent": True}),
            ("string exponent", {"scaling_exponent": "0.5"}),
            ("non-finite exponent", {"scaling_exponent": float("nan")}),
            ("small quantile grid", {"n_quantiles": 1}),
            ("boolean seed", {"random_state": True}),
            ("large seed", {"random_state": 2**32}),
        ]

        for label, options in cases:
            with self.subTest(label=label), self.assertRaises(ConfigurationError):
                SignalConfig(**options)  # type: ignore[arg-type]

    def test_reference_and_projected_presets_are_explicit(self) -> None:
        reference = SignalConfig.reference()
        projected = SignalConfig.projected(n_projections=17, n_quantiles=19)
        updated = projected.with_updates(frequency_weighting="sample_size")

        self.assertEqual(reference, SignalConfig())
        self.assertEqual(projected.barycenter, "projected_quantile")
        self.assertEqual(projected.n_projections, 17)
        self.assertEqual(projected.n_quantiles, 19)
        self.assertEqual(projected.frequency_grid, projected.frequency_specs)
        self.assertEqual(updated.frequency_weighting, "sample_size")
        self.assertEqual(projected.frequency_weighting, "uniform")
        with self.assertRaises(ConfigurationError):
            projected.with_updates(unknown_setting=True)

    def test_configuration_serialization_round_trip_preserves_identity(self) -> None:
        config = SignalConfig.projected(
            frequency_specs=(
                FrequencySpec("base", 1.0),
                FrequencySpec(
                    "week",
                    5.0,
                    rule="W-FRI",
                    closed="right",
                    label="right",
                    offset="1h",
                    min_observations=3,
                ),
            ),
            n_projections=23,
            n_quantiles=29,
            random_state=7,
        )

        self.assertEqual(SignalConfig.from_dict(config.to_dict()), config)
        self.assertEqual(SignalConfig.from_json(config.to_json()), config)
        self.assertEqual(SignalConfig.from_json(config.to_json()).digest, config.digest)
        with TemporaryDirectory() as directory:
            path = config.write_json(f"{directory}/nested/config.json")
            self.assertTrue(path.is_absolute())
            self.assertEqual(SignalConfig.read_json(path), config)

    def test_serialization_rejects_unknown_schema_fields_and_invalid_json(self) -> None:
        payload = SignalConfig().to_dict()
        payload["schema_version"] = 999
        with self.assertRaises(ConfigurationError):
            SignalConfig.from_dict(payload)

        payload = SignalConfig().to_dict()
        payload["unexpected"] = True
        with self.assertRaises(ConfigurationError):
            SignalConfig.from_dict(payload)

        with self.assertRaises(ConfigurationError):
            SignalConfig.from_json("not json")
        with self.assertRaises(ConfigurationError):
            SignalConfig.from_json("[]")
        with self.assertRaises(ConfigurationError):
            SignalConfig.from_json(1)  # type: ignore[arg-type]
        with self.assertRaises(ConfigurationError):
            SignalConfig.from_dict([])  # type: ignore[arg-type]
        with self.assertRaises(ConfigurationError):
            FrequencySpec.from_dict([])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
