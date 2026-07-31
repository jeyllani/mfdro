from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from mfdro import (
    FrequencySpec,
    MultiFrequencySignal,
    SignalConfig,
    build_frequency_measures,
)
from mfdro.exceptions import ConfigurationError, DataContractError

FIVE_FREQUENCIES = (
    FrequencySpec("daily", 1.0),
    FrequencySpec("weekly", 5.0, rule="W-FRI", closed="right", label="right"),
    FrequencySpec("biweekly", 10.0, rule="2W-FRI", closed="right", label="right"),
    FrequencySpec("monthly", 21.0, rule="ME", closed="right", label="right"),
    FrequencySpec("quarterly", 63.0, rule="QE", closed="right", label="right"),
)


class FrequencySpecificationTests(unittest.TestCase):
    def test_five_frequency_grid_is_canonical_and_digestible(self) -> None:
        config = SignalConfig(frequency_specs=FIVE_FREQUENCIES)

        self.assertEqual(
            config.frequencies,
            ("daily", "weekly", "biweekly", "monthly", "quarterly"),
        )
        self.assertEqual(config.horizons, (1.0, 5.0, 10.0, 21.0, 63.0))
        self.assertEqual(len(config.digest), 64)

    def test_resampling_convention_changes_scientific_digest(self) -> None:
        right = SignalConfig(
            frequency_specs=(
                FrequencySpec("base", 1.0),
                FrequencySpec("week", 7.0, rule="W", closed="right"),
            )
        )
        left = SignalConfig(
            frequency_specs=(
                FrequencySpec("base", 1.0),
                FrequencySpec("week", 7.0, rule="W", closed="left"),
            )
        )

        self.assertNotEqual(right.digest, left.digest)

    def test_invalid_base_frequency_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            SignalConfig(
                frequency_specs=(
                    FrequencySpec("not_base", 1.0, rule="D"),
                    FrequencySpec("weekly", 5.0, rule="W-FRI"),
                )
            )

    def test_walk_forward_requires_rules_for_aggregated_frequencies(self) -> None:
        config = SignalConfig(
            frequency_specs=(
                FrequencySpec("base", 1.0),
                FrequencySpec("provided_weekly", 5.0),
            ),
            barycenter="projected_quantile",
        )
        index = pd.bdate_range("2020-01-01", "2020-03-31")
        returns = pd.DataFrame({"A": np.zeros(len(index))}, index=index)

        with self.assertRaises(DataContractError):
            MultiFrequencySignal(config).estimate_path(
                returns,
                lookback_months=2,
            )

    def test_invalid_frequency_fields_fail_at_construction(self) -> None:
        cases: list[tuple[str, dict[str, object]]] = [
            ("empty name", {"name": "", "horizon": 1.0}),
            ("invalid name", {"name": "weekly-return", "horizon": 1.0}),
            ("boolean horizon", {"name": "x", "horizon": True}),
            ("zero horizon", {"name": "x", "horizon": 0.0}),
            ("empty rule", {"name": "x", "horizon": 1.0, "rule": ""}),
            ("invalid rule", {"name": "x", "horizon": 1.0, "rule": "NOT-A-RULE"}),
            ("closed side", {"name": "x", "horizon": 1.0, "closed": "middle"}),
            ("label side", {"name": "x", "horizon": 1.0, "label": "middle"}),
            ("empty origin", {"name": "x", "horizon": 1.0, "origin": ""}),
            ("invalid origin", {"name": "x", "horizon": 1.0, "origin": "not-a-date"}),
            ("offset type", {"name": "x", "horizon": 1.0, "offset": 1}),
            ("invalid offset", {"name": "x", "horizon": 1.0, "offset": "later"}),
            ("boolean minimum", {"name": "x", "horizon": 1.0, "min_observations": True}),
            ("zero minimum", {"name": "x", "horizon": 1.0, "min_observations": 0}),
        ]

        for label, options in cases:
            with self.subTest(label=label), self.assertRaises(ConfigurationError):
                FrequencySpec(**options)  # type: ignore[arg-type]

    def test_frequency_serialization_is_strict_and_lossless(self) -> None:
        spec = FrequencySpec(
            "weekly",
            5.0,
            rule="W-FRI",
            closed="right",
            label="right",
            origin="2020-01-01",
            offset="12h",
            min_observations=3,
        )

        self.assertEqual(FrequencySpec.from_dict(spec.to_dict()), spec)
        payload = spec.to_dict()
        payload["extra"] = True
        with self.assertRaises(ConfigurationError):
            FrequencySpec.from_dict(payload)


class MultiFrequencyPipelineTests(unittest.TestCase):
    @staticmethod
    def daily_panel() -> pd.DataFrame:
        index = pd.bdate_range("2019-01-01", "2021-12-31")
        t = np.arange(len(index), dtype=float)
        return pd.DataFrame(
            {
                "A": 0.0002 + 0.0002 * np.sin(t / 11.0),
                "B": 0.0001 + 0.0003 * np.cos(t / 17.0),
                "C": -0.0001 + 0.0002 * np.sin(t / 23.0),
            },
            index=index,
        )

    def test_builder_creates_five_ordered_measures(self) -> None:
        measures = build_frequency_measures(
            self.daily_panel(),
            frequency_specs=FIVE_FREQUENCIES,
        )

        self.assertEqual(tuple(measures), tuple(spec.name for spec in FIVE_FREQUENCIES))
        self.assertGreater(len(measures["daily"]), len(measures["weekly"]))
        self.assertGreater(len(measures["weekly"]), len(measures["monthly"]))
        self.assertGreater(len(measures["monthly"]), len(measures["quarterly"]))
        for measure in measures.values():
            self.assertEqual(list(measure.columns), ["A", "B", "C"])
            self.assertFalse(measure.isna().any().any())

    def test_five_frequency_signal_with_explicit_weights(self) -> None:
        weights = (5.0, 4.0, 3.0, 2.0, 1.0)
        config = SignalConfig(
            frequency_specs=FIVE_FREQUENCIES,
            frequency_weighting="explicit",
            explicit_frequency_weights=weights,
            barycenter="projected_quantile",
            n_projections=9,
            n_quantiles=11,
        )
        measures = build_frequency_measures(
            self.daily_panel(),
            frequency_specs=FIVE_FREQUENCIES,
        )

        result = MultiFrequencySignal(config).estimate(measures)

        np.testing.assert_allclose(
            result.frequency_weights,
            np.asarray(weights) / sum(weights),
            rtol=0,
            atol=1e-15,
        )
        self.assertEqual(len(result.sample_sizes), 5)
        self.assertGreaterEqual(result.rho, 0.0)

    def test_walk_forward_audit_expands_to_every_frequency(self) -> None:
        config = SignalConfig(
            frequency_specs=FIVE_FREQUENCIES,
            barycenter="projected_quantile",
            n_projections=7,
            n_quantiles=9,
        )
        engine = MultiFrequencySignal(config)

        path = engine.estimate_path(
            self.daily_panel(),
            lookback_months=12,
        )

        self.assertEqual(len(path.estimates), 25)
        for name in ("daily", "weekly", "biweekly", "monthly", "quarterly"):
            self.assertIn(f"n_{name}", path.audit.columns)
            self.assertIn(f"lambda_{name}", path.estimates.columns)
        self.assertTrue(path.audit["no_future_observations"].all())
        self.assertTrue(path.audit["matrix_is_full"].all())
        self.assertTrue(path.audit["n_monthly"].eq(12).all())


if __name__ == "__main__":
    unittest.main()
