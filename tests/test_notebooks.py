from __future__ import annotations

import importlib.util
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

NOTEBOOK_DIRECTORY = Path(__file__).parents[1] / "examples" / "notebooks"
SYNTHETIC_NOTEBOOK = NOTEBOOK_DIRECTORY / "01_synthetic_walk_forward.ipynb"
FREQUENCY_NOTEBOOK = NOTEBOOK_DIRECTORY / "02_frequency_lab.ipynb"
GEOMETRY_NOTEBOOK = NOTEBOOK_DIRECTORY / "03_geometry_and_sensitivity.ipynb"
POINT_IN_TIME_NOTEBOOK = NOTEBOOK_DIRECTORY / "04_point_in_time_workflow.ipynb"
REPRODUCIBILITY_NOTEBOOK = NOTEBOOK_DIRECTORY / "05_reproducibility_and_artifacts.ipynb"
YFINANCE_NOTEBOOK = NOTEBOOK_DIRECTORY / "06_yfinance_case_study.ipynb"
NOTEBOOKS = (
    SYNTHETIC_NOTEBOOK,
    FREQUENCY_NOTEBOOK,
    GEOMETRY_NOTEBOOK,
    POINT_IN_TIME_NOTEBOOK,
    REPRODUCIBILITY_NOTEBOOK,
    YFINANCE_NOTEBOOK,
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AssertionError(f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result


def _load_notebook(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object)
    if not isinstance(payload, dict):  # pragma: no cover - guarded by the fixture shape
        raise AssertionError(f"{path} must contain one JSON object.")
    return payload


def _code_cells(payload: dict[str, object]) -> list[str]:
    cells = payload["cells"]
    assert isinstance(cells, list)
    return [
        "".join(cell["source"])
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_type") == "code"
    ]


class NotebookQualityTests(unittest.TestCase):
    def test_notebooks_are_valid_reviewable_version_four_documents(self) -> None:
        self.assertEqual(
            {path.name for path in NOTEBOOK_DIRECTORY.glob("*.ipynb")},
            {path.name for path in NOTEBOOKS},
        )
        for path in NOTEBOOKS:
            with self.subTest(path=path.name):
                payload = _load_notebook(path)
                self.assertEqual(payload["nbformat"], 4)
                self.assertGreaterEqual(payload["nbformat_minor"], 5)
                cells = payload["cells"]
                assert isinstance(cells, list)
                self.assertGreater(len(cells), 5)
                for cell in cells:
                    assert isinstance(cell, dict)
                    if cell["cell_type"] == "code":
                        self.assertIsNone(cell["execution_count"])
                        self.assertEqual(cell["outputs"], [])
                for index, source in enumerate(_code_cells(payload)):
                    compile(source, f"{path.name}:cell-{index}", "exec")

    @unittest.skipUnless(
        importlib.util.find_spec("nbformat") is not None,
        "formal notebook validation requires the notebook extra",
    )
    def test_notebooks_pass_nbformat_validation(self) -> None:
        import nbformat

        for path in NOTEBOOKS:
            with self.subTest(path=path.name):
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)

    def test_network_boundary_is_machine_readable_and_explained(self) -> None:
        synthetic = _load_notebook(SYNTHETIC_NOTEBOOK)
        yfinance = _load_notebook(YFINANCE_NOTEBOOK)
        synthetic_metadata = synthetic["metadata"]
        yfinance_metadata = yfinance["metadata"]
        assert isinstance(synthetic_metadata, dict)
        assert isinstance(yfinance_metadata, dict)
        self.assertFalse(synthetic_metadata["mfdro"]["network"])
        self.assertTrue(yfinance_metadata["mfdro"]["network"])
        for path in NOTEBOOKS[:-1]:
            metadata = _load_notebook(path)["metadata"]
            assert isinstance(metadata, dict)
            self.assertFalse(metadata["mfdro"]["network"])

        source = YFINANCE_NOTEBOOK.read_text(encoding="utf-8")
        for required in (
            "auto_adjust=True",
            "multi_level_index=True",
            "pct_change(fill_method=None)",
            "not a point-in-time universe",
            "personal use only",
        ):
            self.assertIn(required, source)

    def test_suite_exercises_the_public_research_workflow(self) -> None:
        source = "\n".join(path.read_text(encoding="utf-8") for path in NOTEBOOKS[:-1])
        for public_feature in (
            "FrequencySpec",
            "SignalConfig.projected",
            "SignalConfig.reference",
            "MultiFrequencySignal",
            "build_frequency_measures",
            "compound_returns",
            "validate_path_inputs",
            "estimate_path",
            "reference_calendar",
            "memberships",
            "progress_callback",
            "SignalPath.load",
            "write_json",
            "include_support=True",
            "distance=",
        ):
            self.assertIn(public_feature, source)

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib") is not None,
        "offline notebook execution requires the notebook extra",
    )
    def test_offline_notebooks_execute(self) -> None:
        with patch.dict(
            os.environ,
            {"MPLBACKEND": "Agg", "MPLCONFIGDIR": os.fspath(Path(os.getenv("TMPDIR", "/tmp")))},
        ):
            for path in NOTEBOOKS[:-1]:
                namespace: dict[str, object] = {"__name__": "__notebook_test__"}
                with self.subTest(path=path.name):
                    for index, source in enumerate(_code_cells(_load_notebook(path))):
                        exec(  # noqa: S102 - version-controlled offline notebook fixtures
                            compile(source, f"{path.name}:cell-{index}", "exec"),
                            namespace,
                        )


if __name__ == "__main__":
    unittest.main()
