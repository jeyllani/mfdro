"""Declarative frequency specifications."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal, cast

import pandas as pd

from .exceptions import ConfigurationError

Side = Literal["left", "right"]


@dataclass(frozen=True, slots=True)
class FrequencySpec:
    """Define one empirical return frequency.

    Parameters
    ----------
    name:
        Stable public name used in input mappings and output columns.
    horizon:
        Effective number of base periods used by horizon scaling.
    rule:
        pandas offset alias used to aggregate the base panel. The first
        frequency must use ``None`` because it represents the unaggregated
        input.
    closed, label, origin, offset:
        Explicit pandas resampling conventions. Their values are included in
        the scientific configuration digest.
    min_observations:
        Minimum number of base observations required in an aggregation bin.
    """

    name: str
    horizon: float
    rule: str | None = None
    closed: Side | None = None
    label: Side | None = None
    origin: str = "start_day"
    offset: str | None = None
    min_observations: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.name):
            raise ConfigurationError(
                "A frequency name must start with a letter and contain only letters, "
                "digits, or underscores."
            )
        raw_horizon: object = self.horizon
        if isinstance(raw_horizon, bool) or not isinstance(raw_horizon, Real):
            raise ConfigurationError(f"{self.name}: horizon must be finite and positive.")
        horizon = float(raw_horizon)
        if not math.isfinite(horizon) or horizon <= 0:
            raise ConfigurationError(f"{self.name}: horizon must be finite and positive.")
        object.__setattr__(self, "horizon", horizon)
        if self.rule is not None and (not isinstance(self.rule, str) or not self.rule.strip()):
            raise ConfigurationError(f"{self.name}: rule must be None or a non-empty string.")
        if self.rule is not None:
            try:
                pd.tseries.frequencies.to_offset(self.rule)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    f"{self.name}: rule is not a valid pandas offset alias."
                ) from exc
        if self.closed not in {None, "left", "right"}:
            raise ConfigurationError(f"{self.name}: closed must be 'left', 'right' or None.")
        if self.label not in {None, "left", "right"}:
            raise ConfigurationError(f"{self.name}: label must be 'left', 'right' or None.")
        if not isinstance(self.origin, str) or not self.origin:
            raise ConfigurationError(f"{self.name}: origin must be a non-empty string.")
        if self.origin not in {"epoch", "start", "start_day", "end", "end_day"}:
            try:
                pd.Timestamp(self.origin)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    f"{self.name}: origin must be a pandas resampling origin or timestamp."
                ) from exc
        if self.offset is not None and not isinstance(self.offset, str):
            raise ConfigurationError(f"{self.name}: offset must be a string or None.")
        if self.offset is not None:
            try:
                pd.Timedelta(self.offset)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    f"{self.name}: offset must be convertible to a pandas Timedelta."
                ) from exc
        raw_minimum: object = self.min_observations
        if isinstance(raw_minimum, bool) or not isinstance(raw_minimum, Integral):
            raise ConfigurationError(f"{self.name}: min_observations must be a positive integer.")
        if int(raw_minimum) < 1:
            raise ConfigurationError(f"{self.name}: min_observations must be a positive integer.")
        object.__setattr__(self, "min_observations", int(raw_minimum))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable frequency definition."""

        return {
            "name": self.name,
            "horizon": self.horizon,
            "rule": self.rule,
            "closed": self.closed,
            "label": self.label,
            "origin": self.origin,
            "offset": self.offset,
            "min_observations": self.min_observations,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> FrequencySpec:
        """Construct a frequency definition from :meth:`to_dict` output."""

        if not isinstance(payload, Mapping):
            raise ConfigurationError("A frequency payload must be a mapping.")
        expected = {
            "name",
            "horizon",
            "rule",
            "closed",
            "label",
            "origin",
            "offset",
            "min_observations",
        }
        observed = set(payload)
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ConfigurationError(f"Frequency fields differ: missing={missing}, extra={extra}.")
        return cls(
            name=cast(str, payload["name"]),
            horizon=cast(float, payload["horizon"]),
            rule=cast(str | None, payload["rule"]),
            closed=cast(Side | None, payload["closed"]),
            label=cast(Side | None, payload["label"]),
            origin=cast(str, payload["origin"]),
            offset=cast(str | None, payload["offset"]),
            min_observations=cast(int, payload["min_observations"]),
        )


DEFAULT_FREQUENCY_SPECS: tuple[FrequencySpec, ...] = (
    FrequencySpec(name="daily", horizon=1.0),
    FrequencySpec(name="weekly", horizon=5.0, rule="W-FRI"),
    FrequencySpec(name="monthly", horizon=21.0, rule="ME"),
)
