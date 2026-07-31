# Validation

MFDRO combines analytical invariants, contract failures, integration paths,
and an independently written reference formula. Tests do not download market
data.

## Numerical invariants

- a known one-dimensional dispersion equals `2/3`;
- power and realized-volatility scaling satisfy analytical invariants;
- every frequency-weight rule normalizes to its documented value;
- exact discrete transport has a known one-dimensional value;
- seeded random directions are repeatable unit vectors;
- identical measures have zero projected dispersion;
- free-support sliced dispersion matches an independent notebook-style formula
  at floating-point precision;
- simple returns are compounded rather than summed;
- extreme finite weights normalize without overflow;
- a fixed historical seed derivation remains unchanged.

## Data and point-in-time invariants

- selected missing and non-finite returns fail hard;
- labelled measures align to one canonical asset order;
- changing future returns cannot change an earlier estimate;
- complete windows end exactly on their formation dates;
- authoritative calendars detect missing dates and partial default month ends;
- timezone-aware paths preserve timezone information;
- skipped formations retain reasons and stable output schemas;
- preflight readiness uses the same window preparation as estimation;
- progress callbacks receive every requested formation in order;
- persisted paths round-trip and reject checksum tampering.

## Configuration invariants

- mutable inputs become immutable canonical tuples;
- non-integral counts and invalid seeds fail before computation;
- resampling choices alter the configuration digest;
- two-, three-, and five-frequency grids execute;
- explicit weights contain one positive value per frequency;
- strict JSON configuration round trips preserve identity and reject unknown
  schema fields.

## Local validation

```bash
python -m pip install -e ".[test,docs,dev]"
python -m pytest
python -m ruff check src tests examples
python -m mypy src
python -m compileall -q src tests examples
python -m mkdocs build --strict
python -m build
python -m twine check dist/*
```

Notebook files are structurally validated, duplicate JSON keys are rejected,
and every code cell is compiled. All five deterministic offline notebooks are
executed in notebook CI. The Yahoo Finance case study is intentionally excluded
from numerical CI because a live vendor response is not a reproducible fixture.

The CI coverage gate is 95% with branch coverage enabled. Coverage complements,
but does not replace, the independent formula and point-in-time invariants.

A release must also install the built wheel outside the source tree and run
smoke tests against that installed artifact. Passing tests against editable
sources does not prove that a wheel is complete.

## Numerical tolerance

Seeds and formulas are deterministic within a fixed environment. Cross-platform
or dependency-version comparisons should use documented tolerances because
clustering, linear algebra, and optimal-transport implementations may produce
small floating-point changes.
