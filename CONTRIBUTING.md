# Contributing

Contributions that improve correctness, documentation, reproducibility, or
interoperability are welcome once the public repository is opened.

## Development environment

```bash
git clone <repository-url>
cd mfdro
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,docs,dev,notebooks]"
```

## Required checks

```bash
python -m pytest --cov=mfdro --cov-report=term-missing
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m mypy src
python -m mkdocs build --strict
python -m pytest tests/test_notebooks.py
python examples/synthetic_signal.py
python -m build
python -m twine check dist/*
check-wheel-contents dist/*.whl
```

## Scientific changes

Every scientific change should include:

- a unit invariant or analytical fixture;
- an integration test for point-in-time behavior;
- a changelog entry;
- documentation of numerical tolerance and reproducibility consequences;
- evidence that the historical reference convention is either preserved or
  intentionally versioned.

Never weaken a hard data-contract failure merely to make an empirical example
pass.

## Pull requests

Keep changes focused. Explain the user-visible behavior, validation performed,
and any compatibility consequence. Do not commit generated environments,
documentation sites, wheels, caches, or proprietary data.

Public API removals require a deprecation path after the first stable release.
Before version 1, breaking changes must still be explicit in the changelog.
