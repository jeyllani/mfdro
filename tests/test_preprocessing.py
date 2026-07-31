from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from mfdro import FrequencySpec, build_frequency_measures, compound_returns
from mfdro.exceptions import DataContractError


class PreprocessingTests(unittest.TestCase):
    def test_simple_returns_are_compounded_not_summed(self) -> None:
        index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        daily = pd.DataFrame({"A": [0.10, -0.05, 0.02]}, index=index)

        monthly = compound_returns(daily, "ME")

        expected = (1.10 * 0.95 * 1.02) - 1.0
        self.assertAlmostEqual(float(monthly.iloc[0, 0]), expected, places=15)

    def test_frequency_builder_preserves_asset_order(self) -> None:
        index = pd.bdate_range("2024-01-02", "2024-02-29")
        daily = pd.DataFrame(
            np.full((len(index), 2), 0.001),
            index=index,
            columns=["B", "A"],
        )

        measures = build_frequency_measures(daily)

        self.assertEqual(tuple(measures), ("daily", "weekly", "monthly"))
        self.assertEqual(list(measures["weekly"].columns), ["B", "A"])
        self.assertEqual(list(measures["monthly"].columns), ["B", "A"])

    def test_missing_daily_value_is_rejected(self) -> None:
        index = pd.bdate_range("2024-01-02", periods=3)
        daily = pd.DataFrame({"A": [0.0, np.nan, 0.0]}, index=index)

        with self.assertRaises(DataContractError):
            build_frequency_measures(daily)

    def test_minus_one_return_is_valid_but_below_minus_one_is_not(self) -> None:
        index = pd.bdate_range("2024-01-02", periods=2)
        valid = pd.DataFrame({"A": [0.0, -1.0]}, index=index)
        invalid = pd.DataFrame({"A": [0.0, -1.01]}, index=index)

        result = compound_returns(valid, "ME")
        self.assertEqual(float(result.iloc[0, 0]), -1.0)
        with self.assertRaises(DataContractError):
            compound_returns(invalid, "ME")

    def test_resampling_contract_rejects_boolean_minimum_and_empty_rule(self) -> None:
        index = pd.bdate_range("2024-01-02", periods=3)
        daily = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=index)

        with self.assertRaises(ValueError):
            compound_returns(daily, "ME", min_observations=True)
        with self.assertRaises(ValueError):
            compound_returns(daily, "")

    def test_daily_panel_structure_and_numeric_contract_are_enforced(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=3)
        duplicate_dates = pd.DatetimeIndex([dates[0], dates[0], dates[1]])
        missing_dates = pd.DatetimeIndex([dates[0], pd.NaT, dates[2]])
        cases: list[tuple[str, object]] = [
            ("not a frame", np.ones((3, 1))),
            ("not dated", pd.DataFrame({"A": [0.0, 0.0, 0.0]})),
            ("empty", pd.DataFrame(index=pd.DatetimeIndex([]), columns=["A"])),
            ("missing date", pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=missing_dates)),
            ("duplicate date", pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=duplicate_dates)),
            ("no assets", pd.DataFrame(index=dates)),
            ("duplicate assets", pd.DataFrame(np.zeros((3, 2)), index=dates, columns=["A", "A"])),
            ("non-numeric", pd.DataFrame({"A": ["bad", "data", "here"]}, index=dates)),
            ("infinite", pd.DataFrame({"A": [0.0, np.inf, 0.0]}, index=dates)),
        ]

        for label, panel in cases:
            with self.subTest(label=label), self.assertRaises(DataContractError):
                build_frequency_measures(panel)  # type: ignore[arg-type]

    def test_resampling_minimum_can_reject_every_bin(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=3)
        daily = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)

        with self.assertRaises(DataContractError):
            compound_returns(daily, "ME", min_observations=10)

    def test_custom_builder_grid_is_declared_in_one_place(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=10)
        daily = pd.DataFrame({"A": np.zeros(len(dates))}, index=dates)
        specs = (
            FrequencySpec("base", 1.0),
            FrequencySpec("week", 5.0, rule="W-FRI"),
        )

        measures = build_frequency_measures(
            daily,
            frequency_specs=specs,
        )
        self.assertEqual(tuple(measures), ("base", "week"))

    def test_invalid_frequency_grid_is_rejected_by_builder(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=10)
        daily = pd.DataFrame({"A": np.zeros(len(dates))}, index=dates)
        invalid_grids: list[tuple[object, ...]] = [
            (FrequencySpec("base", 1.0),),
            (FrequencySpec("base", 1.0), object()),
            (FrequencySpec("base", 1.0), FrequencySpec("base", 5.0, rule="W-FRI")),
            (FrequencySpec("base", 1.0, rule="D"), FrequencySpec("week", 5.0, rule="W-FRI")),
            (FrequencySpec("base", 1.0), FrequencySpec("week", 5.0)),
        ]

        for grid in invalid_grids:
            with self.subTest(grid=grid), self.assertRaises(DataContractError):
                build_frequency_measures(
                    daily,
                    frequency_specs=grid,  # type: ignore[arg-type]
                )


if __name__ == "__main__":
    unittest.main()
