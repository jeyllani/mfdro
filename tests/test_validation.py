from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from mfdro.exceptions import DataContractError
from mfdro.validation import prepare_measures


class MeasureValidationTests(unittest.TestCase):
    def test_dataframe_assets_are_reordered_to_canonical_order(self) -> None:
        daily = pd.DataFrame([[1.0, 10.0], [2.0, 20.0]], columns=["A", "B"])
        weekly = pd.DataFrame([[100.0, 3.0], [200.0, 4.0]], columns=["B", "A"])
        monthly = pd.DataFrame([[300.0, 5.0], [400.0, 6.0]], columns=["B", "A"])

        prepared = prepare_measures(
            {"daily": daily, "weekly": weekly, "monthly": monthly},
            ("daily", "weekly", "monthly"),
        )

        self.assertEqual(prepared.asset_labels, ("A", "B"))
        np.testing.assert_array_equal(
            prepared.arrays[1],
            np.array([[3.0, 100.0], [4.0, 200.0]]),
        )

    def test_mixed_labelled_and_unlabelled_inputs_are_rejected(self) -> None:
        with self.assertRaises(DataContractError):
            prepare_measures(
                {
                    "daily": pd.DataFrame(np.ones((2, 2))),
                    "weekly": np.ones((2, 2)),
                },
                ("daily", "weekly"),
            )

    def test_missing_and_nonfinite_values_are_rejected(self) -> None:
        with self.assertRaises(DataContractError):
            prepare_measures(
                {
                    "daily": np.array([[0.0], [np.nan]]),
                    "weekly": np.array([[0.0], [1.0]]),
                },
                ("daily", "weekly"),
            )

    def test_mapping_keys_and_dataframe_identity_are_enforced(self) -> None:
        valid = pd.DataFrame([[0.0], [1.0]], columns=["A"])
        cases: list[tuple[str, object]] = [
            ("not a mapping", [valid, valid]),
            ("missing key", {"daily": valid}),
            ("extra key", {"daily": valid, "weekly": valid, "monthly": valid}),
            (
                "duplicate columns",
                {
                    "daily": pd.DataFrame(np.ones((2, 2)), columns=["A", "A"]),
                    "weekly": pd.DataFrame(np.ones((2, 2)), columns=["A", "A"]),
                },
            ),
            (
                "duplicate index",
                {
                    "daily": valid,
                    "weekly": pd.DataFrame([[0.0], [1.0]], index=[0, 0], columns=["A"]),
                },
            ),
            (
                "different assets",
                {"daily": valid, "weekly": pd.DataFrame([[0.0], [1.0]], columns=["B"])},
            ),
        ]

        for label, measures in cases:
            with self.subTest(label=label), self.assertRaises(DataContractError):
                prepare_measures(measures, ("daily", "weekly"))  # type: ignore[arg-type]

    def test_array_shape_dimension_and_numeric_contract_are_enforced(self) -> None:
        valid = np.array([[0.0], [1.0]])
        cases: list[tuple[str, dict[str, object]]] = [
            ("one dimensional", {"daily": np.array([0.0, 1.0]), "weekly": valid}),
            ("one observation", {"daily": np.array([[0.0]]), "weekly": valid}),
            ("zero assets", {"daily": np.empty((2, 0)), "weekly": np.empty((2, 0))}),
            ("different dimensions", {"daily": valid, "weekly": np.ones((2, 2))}),
            ("non-numeric", {"daily": [["a"], ["b"]], "weekly": valid}),
            ("infinite", {"daily": np.array([[0.0], [np.inf]]), "weekly": valid}),
        ]

        for label, measures in cases:
            with self.subTest(label=label), self.assertRaises(DataContractError):
                prepare_measures(measures, ("daily", "weekly"))


if __name__ == "__main__":
    unittest.main()
