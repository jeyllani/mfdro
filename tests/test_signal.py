from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from mfdro import (
    FrequencySpec,
    MultiFrequencySignal,
    PathProgress,
    SignalConfig,
    SignalPath,
    SkipReason,
)
from mfdro.exceptions import ConfigurationError, DataContractError
from mfdro.signal import _asset_order_digest

UNIT_FREQUENCY_SPECS = (
    FrequencySpec("daily", 1.0),
    FrequencySpec("weekly", 1.0, rule="W-FRI"),
    FrequencySpec("monthly", 1.0, rule="ME"),
)


class PointSignalTests(unittest.TestCase):
    def test_signal_path_canonicalises_datetime_units_without_mutating_inputs(self) -> None:
        config = SignalConfig(
            frequency_specs=UNIT_FREQUENCY_SPECS,
            barycenter="projected_quantile",
        )

        for timezone in (None, "Europe/Zurich"):
            with self.subTest(timezone=timezone):
                dates = pd.date_range("2020-01-31", periods=2, tz=timezone).as_unit("us")
                estimates = pd.DataFrame({"date": dates})
                audit = pd.DataFrame({"date": dates, "start_date": dates})

                path = SignalPath(estimates=estimates, audit=audit, config=config)

                self.assertTrue(str(path.estimates["date"].dtype).startswith("datetime64[ns"))
                self.assertTrue(str(path.audit["start_date"].dtype).startswith("datetime64[ns"))
                self.assertTrue(str(estimates["date"].dtype).startswith("datetime64[us"))
                self.assertTrue(str(audit["start_date"].dtype).startswith("datetime64[us"))

    def test_estimator_and_result_container_reject_invalid_public_inputs(self) -> None:
        with self.assertRaises(TypeError):
            MultiFrequencySignal("invalid")  # type: ignore[arg-type]

        engine = MultiFrequencySignal(
            SignalConfig(
                frequency_specs=UNIT_FREQUENCY_SPECS,
                barycenter="projected_quantile",
            )
        )
        array = np.array([[0.0], [1.0]])
        with self.assertRaises(TypeError):
            engine.estimate(
                {"daily": array, "weekly": array, "monthly": array},
                include_support=1,  # type: ignore[arg-type]
            )

        with self.assertRaises(ValueError):
            SignalPath(
                estimates=pd.DataFrame({"date": [pd.Timestamp("2020-01-31")]}),
                audit=pd.DataFrame(),
                config=engine.config,
            )
        with self.assertRaises(ValueError):
            SignalPath(
                estimates=pd.DataFrame({"date": [pd.Timestamp("2020-01-31")]}),
                audit=pd.DataFrame({"date": [pd.Timestamp("2020-02-28")]}),
                config=engine.config,
            )

    def test_one_dimensional_quantile_signal_has_known_value(self) -> None:
        measures = {
            "daily": np.array([[0.0], [1.0]]),
            "weekly": np.array([[1.0], [2.0]]),
            "monthly": np.array([[2.0], [3.0]]),
        }
        config = SignalConfig(
            frequency_specs=UNIT_FREQUENCY_SPECS,
            barycenter="projected_quantile",
            n_projections=7,
            n_quantiles=5,
        )

        result = MultiFrequencySignal(config).estimate(measures)

        self.assertAlmostEqual(result.rho, 2.0 / 3.0, places=14)
        self.assertAlmostEqual(result.sqrt_rho**2, result.rho, places=14)
        self.assertEqual(result.to_series().name, "signal_estimate")
        self.assertEqual(float(result.to_series()["rho"]), result.rho)

    def test_identical_measures_have_zero_projected_dispersion(self) -> None:
        array = np.arange(12, dtype=float).reshape(6, 2)
        config = SignalConfig(
            frequency_specs=UNIT_FREQUENCY_SPECS,
            barycenter="projected_quantile",
            n_projections=11,
            n_quantiles=13,
        )

        result = MultiFrequencySignal(config).estimate(
            {"daily": array, "weekly": array.copy(), "monthly": array.copy()}
        )

        self.assertLessEqual(result.rho, 1e-28)

    def test_estimate_is_reproducible(self) -> None:
        rng = np.random.default_rng(12)
        measures = {
            "daily": rng.normal(size=(30, 4)),
            "weekly": rng.normal(size=(8, 4)),
            "monthly": rng.normal(size=(3, 4)),
        }
        config = SignalConfig(
            barycenter="projected_quantile",
            n_projections=20,
            n_quantiles=25,
        )
        engine = MultiFrequencySignal(config)

        first = engine.estimate(measures, seed=77)
        second = engine.estimate(measures, seed=77)

        self.assertEqual(first.rho, second.rho)
        self.assertEqual(first.to_record(), second.to_record())

    def test_dataframe_column_alignment_preserves_numerical_identity(self) -> None:
        rng = np.random.default_rng(4)
        daily = pd.DataFrame(rng.normal(size=(20, 3)), columns=["A", "B", "C"])
        weekly = pd.DataFrame(rng.normal(size=(8, 3)), columns=["A", "B", "C"])
        monthly = pd.DataFrame(rng.normal(size=(3, 3)), columns=["A", "B", "C"])
        config = SignalConfig(
            barycenter="projected_quantile",
            n_projections=15,
            n_quantiles=17,
        )
        engine = MultiFrequencySignal(config)

        canonical = engine.estimate(
            {"daily": daily, "weekly": weekly, "monthly": monthly},
            seed=8,
        )
        reordered = engine.estimate(
            {
                "daily": daily,
                "weekly": weekly[["C", "A", "B"]],
                "monthly": monthly[["B", "C", "A"]],
            },
            seed=8,
        )

        self.assertEqual(canonical.rho, reordered.rho)
        self.assertEqual(reordered.asset_labels, ("A", "B", "C"))

    def test_free_support_and_exact_distance_execute(self) -> None:
        rng = np.random.default_rng(51)
        measures = {
            "daily": rng.normal(size=(12, 2)),
            "weekly": rng.normal(size=(6, 2)),
            "monthly": rng.normal(size=(3, 2)),
        }
        config = SignalConfig(
            frequency_specs=UNIT_FREQUENCY_SPECS,
            barycenter="free_support",
            barycenter_size=3,
            distance="exact",
            barycenter_max_iter=5,
        )

        result = MultiFrequencySignal(config).estimate(measures, include_support=True)

        self.assertGreaterEqual(result.rho, 0.0)
        self.assertIsNotNone(result.support)
        assert result.support is not None
        self.assertEqual(result.support.shape, (3, 2))

    def test_explicit_seed_must_follow_uint32_contract(self) -> None:
        array = np.array([[0.0], [1.0]])
        engine = MultiFrequencySignal(
            SignalConfig(
                frequency_specs=UNIT_FREQUENCY_SPECS,
                barycenter="projected_quantile",
            )
        )
        with self.assertRaises(ConfigurationError):
            engine.estimate(
                {"daily": array, "weekly": array, "monthly": array},
                seed=-1,
            )

    def test_asset_order_digest_preserves_label_types(self) -> None:
        self.assertNotEqual(_asset_order_digest([1]), _asset_order_digest(["1"]))


class SignalPathTests(unittest.TestCase):
    @staticmethod
    def daily_panel() -> pd.DataFrame:
        index = pd.bdate_range("2020-01-01", "2020-04-30")
        x = np.arange(len(index), dtype=float)
        return pd.DataFrame(
            {
                "A": 0.0005 + 0.0001 * np.sin(x / 5.0),
                "B": 0.0002 + 0.0001 * np.cos(x / 7.0),
            },
            index=index,
        )

    @staticmethod
    def engine() -> MultiFrequencySignal:
        return MultiFrequencySignal(
            SignalConfig(
                barycenter="projected_quantile",
                n_projections=9,
                n_quantiles=11,
            )
        )

    def test_path_skips_only_incomplete_warmup(self) -> None:
        path = self.engine().estimate_path(
            self.daily_panel(),
            lookback_months=2,
        )

        self.assertEqual(len(path.estimates), 3)
        self.assertEqual(len(path.skipped), 1)
        self.assertEqual(path.skipped.loc[0, "reason"], "non_contiguous_months")
        self.assertEqual(path.estimates["formation_month"].dt.month.tolist(), [2, 3, 4])
        self.assertTrue(path.audit["no_future_observations"].all())
        self.assertTrue(path.audit["matrix_is_full"].all())
        self.assertTrue((path.audit["window_end"] == path.audit["date"]).all())

    def test_preflight_diagnostics_match_the_estimation_schedule(self) -> None:
        engine = self.engine()
        diagnostics = engine.validate_path_inputs(
            self.daily_panel(),
            lookback_months=2,
        )
        path = engine.estimate_path(self.daily_panel(), lookback_months=2)

        self.assertEqual(diagnostics.n_formations, 4)
        self.assertEqual(diagnostics.n_ready, len(path.estimates))
        self.assertEqual(diagnostics.n_insufficient, len(path.skipped))
        self.assertTrue(diagnostics.is_usable)
        self.assertEqual(diagnostics.summary()["config_digest"], engine.config.digest)
        self.assertEqual(
            diagnostics.formations.loc[0, "reason"],
            SkipReason.NON_CONTIGUOUS_MONTHS.value,
        )

    def test_progress_callback_receives_every_formation_in_order(self) -> None:
        updates: list[PathProgress] = []

        path = self.engine().estimate_path(
            self.daily_panel(),
            lookback_months=2,
            progress_callback=updates.append,
        )

        self.assertEqual(len(updates), len(path.estimates) + len(path.skipped))
        self.assertEqual([update.completed for update in updates], [1, 2, 3, 4])
        self.assertTrue(all(update.total == 4 for update in updates))
        self.assertEqual(updates[0].status, "skipped")
        self.assertEqual(updates[0].reason, SkipReason.NON_CONTIGUOUS_MONTHS)
        self.assertEqual(updates[-1].status, "estimated")
        self.assertIsNone(updates[-1].reason)

    def test_frequency_sample_insufficiency_is_diagnostic_and_skippable(self) -> None:
        config = SignalConfig.projected(
            frequency_specs=(
                FrequencySpec("daily", 1.0),
                FrequencySpec("weekly", 5.0, rule="W-FRI"),
                FrequencySpec("monthly", 21.0, rule="ME", min_observations=30),
            ),
            n_projections=3,
            n_quantiles=5,
        )
        engine = MultiFrequencySignal(config)
        formation_date = self.daily_panel().index[-1]

        diagnostics = engine.validate_path_inputs(
            self.daily_panel(),
            lookback_months=2,
            formation_dates=[formation_date],
        )
        path = engine.estimate_path(
            self.daily_panel(),
            lookback_months=2,
            formation_dates=[formation_date],
        )

        self.assertEqual(diagnostics.n_ready, 0)
        self.assertFalse(diagnostics.is_usable)
        self.assertEqual(
            diagnostics.formations.loc[0, "reason"],
            SkipReason.INSUFFICIENT_FREQUENCY_OBSERVATIONS.value,
        )
        self.assertTrue(path.estimates.empty)
        self.assertEqual(
            path.skipped.loc[0, "reason"],
            SkipReason.INSUFFICIENT_FREQUENCY_OBSERVATIONS.value,
        )

        one_month = self.engine().validate_path_inputs(
            self.daily_panel(),
            lookback_months=1,
            formation_dates=[formation_date],
        )
        self.assertEqual(
            one_month.formations.loc[0, "reason"],
            SkipReason.INSUFFICIENT_FREQUENCY_OBSERVATIONS.value,
        )

    def test_result_conveniences_and_portable_round_trip(self) -> None:
        path = self.engine().estimate_path(self.daily_panel(), lookback_months=2)

        self.assertEqual(path.rho.name, "rho")
        self.assertEqual(path.sqrt_rho.name, "sqrt_rho")
        self.assertEqual(len(path.successful_dates), len(path.estimates))
        self.assertEqual(len(path.skipped_dates), len(path.skipped))
        self.assertEqual(path.dispersion_weights.shape, (3, 4))
        self.assertEqual(path.center_weights.shape, (3, 4))
        self.assertEqual(path.summary()["n_estimates"], 3)

        with TemporaryDirectory() as directory:
            destination = Path(directory, "result")
            saved = path.save(destination)
            loaded = SignalPath.load(saved)

            pd.testing.assert_frame_equal(loaded.estimates, path.estimates)
            pd.testing.assert_frame_equal(loaded.audit, path.audit)
            pd.testing.assert_frame_equal(loaded.skipped, path.skipped)
            self.assertEqual(loaded.config, path.config)
            with self.assertRaises(FileExistsError):
                path.save(destination)
            self.assertEqual(path.save(destination, overwrite=True), destination.resolve())

    def test_persisted_path_detects_tampering(self) -> None:
        path = self.engine().estimate_path(self.daily_panel(), lookback_months=2)

        with TemporaryDirectory() as directory:
            destination = path.save(Path(directory, "result"))
            estimates_path = destination / "estimates.json"
            estimates_path.write_text(
                estimates_path.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            with self.assertRaises(DataContractError):
                SignalPath.load(destination)

    def test_path_persistence_rejects_invalid_destinations_and_manifests(self) -> None:
        path = self.engine().estimate_path(self.daily_panel(), lookback_months=2)

        with TemporaryDirectory() as directory:
            destination = Path(directory, "not-a-directory")
            destination.write_text("occupied", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                path.save(destination)
            with self.assertRaises(TypeError):
                path.save(Path(directory, "result"), overwrite=1)  # type: ignore[arg-type]

        invalid_manifests: list[object] = [[], {"format_version": 999}, {"format_version": 1}]
        for manifest in invalid_manifests:
            with self.subTest(manifest=manifest), TemporaryDirectory() as directory:
                destination = path.save(Path(directory, "result"))
                (destination / "manifest.json").write_text(
                    json.dumps(manifest),
                    encoding="utf-8",
                )
                with self.assertRaises(DataContractError):
                    SignalPath.load(destination)

        with TemporaryDirectory() as directory:
            destination = Path(directory, "missing")
            with self.assertRaises(DataContractError):
                SignalPath.load(destination)

    def test_future_return_changes_do_not_change_earlier_signal(self) -> None:
        daily = self.daily_panel()
        formation_date = daily.index[daily.index.to_period("M") == pd.Period("2020-03")][-1]
        first = self.engine().estimate_path(
            daily,
            lookback_months=2,
            formation_dates=[formation_date],
        )
        altered = daily.copy()
        altered.loc[altered.index > formation_date, :] = 0.50
        second = self.engine().estimate_path(
            altered,
            lookback_months=2,
            formation_dates=[formation_date],
        )

        self.assertEqual(
            float(first.estimates.loc[0, "rho"]),
            float(second.estimates.loc[0, "rho"]),
        )

    def test_dynamic_membership_may_select_full_subset_from_sparse_source(self) -> None:
        daily = self.daily_panel()
        daily["C"] = np.nan
        april_date = daily.index[-1]

        path = self.engine().estimate_path(
            daily,
            lookback_months=2,
            formation_dates=[april_date],
            memberships={"2020-04-01": ["B", "A"]},
        )

        self.assertEqual(int(path.estimates.loc[0, "n_assets"]), 2)

    def test_selected_missing_value_fails_hard(self) -> None:
        daily = self.daily_panel()
        april_date = daily.index[-1]
        daily.loc[daily.index[-5], "A"] = np.nan

        with self.assertRaises(DataContractError):
            self.engine().estimate_path(
                daily,
                lookback_months=2,
                formation_dates=[april_date],
            )

    def test_skipped_windows_are_recorded_with_reason(self) -> None:
        daily = self.daily_panel()
        daily = daily.loc[daily.index.to_period("M") != pd.Period("2020-03")]
        april_date = daily.index[-1]

        path = self.engine().estimate_path(
            daily,
            lookback_months=2,
            formation_dates=[april_date],
            on_insufficient="skip",
        )

        self.assertTrue(path.estimates.empty)
        self.assertTrue(path.audit.empty)
        self.assertTrue(path.rho.empty)
        self.assertTrue(path.sqrt_rho.empty)
        self.assertIsNone(path.summary()["first_estimate"])
        self.assertEqual(len(path.skipped), 1)
        self.assertEqual(path.skipped.loc[0, "reason"], "non_contiguous_months")
        self.assertEqual(
            path.estimates.columns.tolist(),
            [
                "date",
                "formation_month",
                "rho",
                "sqrt_rho",
                "seed",
                "config_digest",
                "n_assets",
                "lambda_daily",
                "barycenter_lambda_daily",
                "n_daily",
                "lambda_weekly",
                "barycenter_lambda_weekly",
                "n_weekly",
                "lambda_monthly",
                "barycenter_lambda_monthly",
                "n_monthly",
            ],
        )
        self.assertIn("asset_order_digest", path.audit.columns)

    def test_reference_calendar_detects_missing_observation_dates(self) -> None:
        complete_calendar = pd.bdate_range("2020-01-01", "2020-02-28")
        observed = complete_calendar[complete_calendar >= pd.Timestamp("2020-01-15")]
        daily = pd.DataFrame(
            {
                "A": np.linspace(0.0, 0.001, len(observed)),
                "B": np.linspace(0.001, 0.0, len(observed)),
            },
            index=observed,
        )

        with self.assertRaises(DataContractError):
            self.engine().estimate_path(
                daily,
                lookback_months=2,
                formation_dates=[observed[-1]],
                reference_calendar=complete_calendar,
                on_insufficient="raise",
            )

        skipped = self.engine().estimate_path(
            daily,
            lookback_months=2,
            formation_dates=[observed[-1]],
            reference_calendar=complete_calendar,
            on_insufficient="skip",
        )
        self.assertTrue(skipped.estimates.empty)
        self.assertEqual(len(skipped.skipped), 1)
        self.assertEqual(
            skipped.skipped.loc[0, "reason"],
            "reference_calendar_mismatch",
        )

    def test_reference_calendar_accepts_a_complete_window(self) -> None:
        daily = self.daily_panel().loc["2020-01":"2020-02"]

        path = self.engine().estimate_path(
            daily,
            lookback_months=2,
            formation_dates=[daily.index[-1]],
            reference_calendar=daily.index,
            on_insufficient="raise",
        )

        self.assertEqual(len(path.estimates), 1)
        self.assertTrue(path.skipped.empty)

    def test_reference_calendar_defines_default_month_end_schedule(self) -> None:
        complete_calendar = pd.bdate_range("2020-01-01", "2020-02-28")
        partial_source = complete_calendar[complete_calendar <= pd.Timestamp("2020-02-20")]
        daily = pd.DataFrame(
            np.zeros((len(partial_source), 2)),
            index=partial_source,
            columns=["A", "B"],
        )

        with self.assertRaises(DataContractError):
            self.engine().estimate_path(
                daily,
                lookback_months=2,
                reference_calendar=complete_calendar,
            )

    def test_timezone_aware_path_is_supported(self) -> None:
        daily = self.daily_panel().tz_localize("Europe/Zurich")

        path = self.engine().estimate_path(
            daily,
            lookback_months=2,
            formation_dates=[daily.index[-1]],
            reference_calendar=daily.index,
            on_insufficient="raise",
        )

        self.assertEqual(len(path.estimates), 1)
        self.assertEqual(
            str(path.estimates.loc[0, "date"].tzinfo),
            str(daily.index.tz),
        )
        with TemporaryDirectory() as directory:
            restored = SignalPath.load(path.save(Path(directory, "path")))
            pd.testing.assert_frame_equal(restored.estimates, path.estimates)

    def test_reference_calendar_rejects_missing_dates(self) -> None:
        calendar = list(self.daily_panel().index)
        calendar[0] = pd.NaT

        with self.assertRaises(DataContractError):
            self.engine().estimate_path(
                self.daily_panel(),
                lookback_months=2,
                reference_calendar=calendar,
            )

    def test_seed_namespace_rejects_ambiguous_delimiter(self) -> None:
        with self.assertRaises(ValueError):
            self.engine().estimate_path(
                self.daily_panel(),
                lookback_months=2,
                seed_namespace="universe|variant",
            )

    def test_path_scalar_options_are_validated_before_estimation(self) -> None:
        cases: list[tuple[str, dict[str, object]]] = [
            ("boolean lookback", {"lookback_months": True}),
            ("zero lookback", {"lookback_months": 0}),
            ("policy", {"lookback_months": 2, "on_insufficient": "ignore"}),
            ("empty namespace", {"lookback_months": 2, "seed_namespace": ""}),
            ("namespace type", {"lookback_months": 2, "seed_namespace": 1}),
        ]

        for label, options in cases:
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.engine().estimate_path(
                    self.daily_panel(),
                    **options,  # type: ignore[arg-type]
                )

        with self.assertRaises(TypeError):
            self.engine().estimate_path(
                self.daily_panel(),
                lookback_months=2,
                progress_callback=1,  # type: ignore[arg-type]
            )

    def test_path_source_structure_and_numeric_contract_are_enforced(self) -> None:
        daily = self.daily_panel()
        missing_index = daily.index.to_list()
        missing_index[0] = pd.NaT
        duplicate_index = daily.index.to_list()
        duplicate_index[1] = duplicate_index[0]
        cases: list[tuple[str, object]] = [
            ("not a frame", daily.to_numpy()),
            ("not dated", daily.reset_index(drop=True)),
            ("empty", daily.iloc[:0]),
            ("missing date", daily.set_axis(pd.DatetimeIndex(missing_index), axis=0)),
            ("duplicate date", daily.set_axis(pd.DatetimeIndex(duplicate_index), axis=0)),
            ("duplicate assets", daily.set_axis(["A", "A"], axis=1)),
            ("non-numeric", daily.assign(A="bad")),
            ("infinite", daily.assign(A=np.inf)),
            ("below minus one", daily.assign(A=-1.01)),
        ]

        for label, source in cases:
            with self.subTest(label=label), self.assertRaises(DataContractError):
                self.engine().estimate_path(
                    source,  # type: ignore[arg-type]
                    lookback_months=2,
                )

    def test_formation_schedule_contract_is_enforced(self) -> None:
        daily = self.daily_panel()
        cases: list[tuple[str, object]] = [
            ("string schedule", "2020-04-30"),
            ("invalid date", [object()]),
            ("missing date", [pd.NaT]),
            ("unobserved date", [pd.Timestamp("2020-05-01")]),
            ("twice in month", [daily.index[-1], daily.index[-2]]),
            ("timezone mismatch", [daily.index[-1].tz_localize("Europe/Zurich")]),
        ]

        for label, dates in cases:
            with self.subTest(label=label), self.assertRaises(DataContractError):
                self.engine().estimate_path(
                    daily,
                    lookback_months=2,
                    formation_dates=dates,  # type: ignore[arg-type]
                )

    def test_membership_contract_is_enforced(self) -> None:
        daily = self.daily_panel()
        formation_date = daily.index[-1]
        cases: list[tuple[str, object]] = [
            ("not a mapping", ["A", "B"]),
            ("invalid month", {object(): ["A"]}),
            ("asset string", {"2020-04": "A"}),
            ("asset non-sequence", {"2020-04": 1}),
            ("absent month", {"2020-03": ["A"]}),
            ("empty", {"2020-04": []}),
            ("duplicate asset", {"2020-04": ["A", "A"]}),
            ("missing asset", {"2020-04": ["C"]}),
            ("unhashable asset", {"2020-04": [["A"]]}),
        ]

        for label, memberships in cases:
            with self.subTest(label=label), self.assertRaises(DataContractError):
                self.engine().estimate_path(
                    daily,
                    lookback_months=2,
                    formation_dates=[formation_date],
                    memberships=memberships,  # type: ignore[arg-type]
                )

    def test_reference_calendar_contract_is_enforced(self) -> None:
        daily = self.daily_panel()
        duplicate = list(daily.index)
        duplicate[1] = duplicate[0]
        cases: list[tuple[str, object]] = [
            ("string calendar", "2020-01-01"),
            ("invalid date", [object()]),
            ("empty", []),
            ("duplicate", duplicate),
            ("timezone mismatch", daily.index.tz_localize("Europe/Zurich")),
            ("missing source month", daily.loc["2020-02":].index),
        ]

        for label, calendar in cases:
            with self.subTest(label=label), self.assertRaises(DataContractError):
                self.engine().estimate_path(
                    daily,
                    lookback_months=2,
                    reference_calendar=calendar,  # type: ignore[arg-type]
                )


if __name__ == "__main__":
    unittest.main()
