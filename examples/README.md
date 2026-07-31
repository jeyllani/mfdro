# Examples

`synthetic_signal.py` is deterministic, offline, and safe to execute in CI:

```bash
python examples/synthetic_signal.py
```

It demonstrates the primary daily-panel workflow and prints estimates, audits,
and skipped warm-up formations.

The `notebooks/` directory contains a six-part progression covering the first
walk-forward estimate, frequency construction, geometry and sensitivity,
point-in-time controls, reproducible artifacts, and an optional Yahoo Finance
case study. The five offline notebooks are executed in notebook CI; the network
case study is syntax-checked but never downloaded there.

Install their dependencies with `python -m pip install -e ".[notebooks]"`.

Examples that require downloads should remain optional, document provider and
retrieval time, and state adjustment, timezone, membership, and boundary
conventions. External data must never become a unit-test dependency.
