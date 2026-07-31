"""Deterministic random-state helpers."""

from __future__ import annotations

import hashlib
from numbers import Integral

from .exceptions import ConfigurationError


def validate_seed(seed: object, name: str = "seed") -> int:
    """Return a canonical uint32 seed or raise a package configuration error."""

    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ConfigurationError(f"{name} must be an integer in [0, 2**32).")
    if not 0 <= int(seed) < 2**32:
        raise ConfigurationError(f"{name} must be an integer in [0, 2**32).")
    return int(seed)


def stable_seed(base_seed: int, *parts: str) -> int:
    """Derive a platform-independent uint32 seed from unambiguous labels.

    The delimiter and string-only contract preserve the historical research
    seed stream while preventing ambiguous concatenations.
    """

    canonical_base = validate_seed(base_seed, "base_seed")
    if any(not isinstance(part, str) or not part or "|" in part for part in parts):
        raise ConfigurationError("Seed labels must be non-empty strings that do not contain '|'.")
    payload = "|".join([str(canonical_base), *parts])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)
