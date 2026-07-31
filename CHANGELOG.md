# Changelog

## Unreleased

- Replace the duplicate frequency interfaces with one canonical
  `frequency_specs` contract and rename the projected center to the explicit
  `projected_quantile` mode before the public API is released.
- Add reference and exploration presets plus strict, versioned configuration
  JSON serialization.
- Add non-numerical walk-forward preflight diagnostics and dependency-free
  progress callbacks.
- Add convenient labelled result accessors and portable, checksummed
  `SignalPath` persistence without pickle.
- Validate pandas frequency aliases, names, origins, and offsets when the
  scientific configuration is constructed.
- Expand skipped-window reasons to include insufficient frequency samples.
- Rebuild the documentation around an end-to-end user workflow, configuration
  decisions, outputs, troubleshooting, performance, and research boundaries.
- Add a six-part scientific notebook suite covering frequency construction,
  geometry, point-in-time controls, reproducible artifacts, and an optional
  Yahoo Finance case study, with automated offline execution checks.
- Clarify the mathematical meaning and interpretation boundary of `rho`.
- Add a public backtesting-integration guide and streamline the package roadmap.
- Export package exceptions from the top-level namespace.
- Remove the misleading scikit-learn-style `fit_transform` alias.
- Preserve stable path schemas when every requested formation is skipped.
- Support timezone-aware paths and reject missing calendar dates explicitly.
- Let an authoritative calendar define default monthly formation endpoints.
- Reject malformed numeric panels, schedules, memberships, and API scalar types
  with package-level contract errors.
- Single-source the package version and remove the unused direct SciPy dependency.
- Add analytical geometry invariants and enforce a 95% branch-coverage gate.
- Include tests, documentation, examples, and maintainer guidance in source
  distributions.
- Prepare quality, release, security, and contribution infrastructure.

## 0.1.0.dev1

- Canonicalise mutable configuration inputs into immutable tuples.
- Reject non-integral scientific counts and invalid explicit seeds before computation.
- Normalise extreme positive frequency weights without floating-point overflow.
- Preserve the reference seed stream while rejecting ambiguous seed namespaces.
- Record skipped walk-forward formations with machine-readable reasons.
- Add an optional authoritative calendar for detecting missing observation dates.
- Make asset-order digests sensitive to label types.
- Align the pandas lower bound with the default `ME` resampling alias.
- Extend unit tests for configuration identity, numerical boundaries and path audits.

## 0.1.0.dev0

- Establish the local, installable package foundation.
- Define a validated and serialisable signal configuration.
- Add deterministic scaling, weighting, barycenter and transport primitives.
- Add a point-estimate API and a point-in-time walk-forward path API.
- Add unit and integration regression tests.
- Add declarative frequency grids with explicit resampling boundaries.
- Validate two-to-five-frequency configurations and custom frequency weights.
- Add a strict MkDocs Material site with generated API reference.
