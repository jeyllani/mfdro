"""Validated, serialisable scientific configuration for the signal engine."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Literal, cast

from .exceptions import ConfigurationError
from .frequency import DEFAULT_FREQUENCY_SPECS, FrequencySpec

Scaling = Literal["power", "realized_volatility"]
FrequencyWeighting = Literal["uniform", "sample_size", "log_sample_size", "explicit"]
Barycenter = Literal["free_support", "projected_quantile"]
Distance = Literal["sliced", "exact"]

CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SignalConfig:
    """Configuration of a multi-frequency ambiguity-signal estimate.

    ``frequency_specs`` is the only frequency interface. Specifications with
    ``rule=None`` can represent measures constructed by the caller; the first
    specification must always be the unaggregated base panel. Walk-forward
    estimation additionally requires a resampling rule on every later
    specification.

    Parameters
    ----------
    frequency_specs:
        Ordered empirical-frequency definitions. At least two are required.
    scaling:
        ``"power"`` for horizon scaling or ``"realized_volatility"`` for
        within-frequency asset standardization.
    scaling_exponent:
        Exponent applied to horizons under power scaling.
    frequency_weighting:
        Rule used to aggregate dispersion across frequencies.
    explicit_frequency_weights:
        One positive dispersion weight per frequency when weighting is explicit.
    barycenter:
        Multivariate free-support or projected-quantile construction.
    barycenter_size:
        Number of free-support atoms. Ignored by ``"projected_quantile"``.
    barycenter_weights:
        Optional positive measure weights used to construct the center.
    barycenter_random_state:
        uint32 seed for weighted k-means++ support initialization.
    distance:
        Sliced approximation or exact discrete squared transport cost.
    n_projections:
        Number of random directions used by projected calculations.
    n_quantiles:
        Quantile-grid size used by projected calculations.
    random_state:
        Base uint32 seed used directly or to derive walk-forward seeds.
    barycenter_max_iter:
        Maximum free-support solver iterations.
    barycenter_tolerance:
        Positive stopping threshold for the free-support solver.

    Notes
    -----
    Sequence inputs are copied into tuples. ``digest`` therefore remains stable
    if caller-owned lists are mutated after construction.
    """

    frequency_specs: tuple[FrequencySpec, ...] = DEFAULT_FREQUENCY_SPECS
    scaling: Scaling = "power"
    scaling_exponent: float = 0.5
    frequency_weighting: FrequencyWeighting = "uniform"
    explicit_frequency_weights: tuple[float, ...] | None = None
    barycenter: Barycenter = "free_support"
    barycenter_size: int = 50
    barycenter_weights: tuple[float, ...] | None = None
    barycenter_random_state: int = 0
    distance: Distance = "sliced"
    n_projections: int = 200
    n_quantiles: int = 200
    random_state: int = 20250301
    barycenter_max_iter: int = 30
    barycenter_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        raw_specs = self._as_tuple(self.frequency_specs, "frequency_specs")
        if not all(isinstance(spec, FrequencySpec) for spec in raw_specs):
            raise ConfigurationError("frequency_specs must contain only FrequencySpec instances.")
        specs = tuple(spec for spec in raw_specs if isinstance(spec, FrequencySpec))
        object.__setattr__(self, "frequency_specs", specs)

        explicit_weights = self._canonical_weights(
            self.explicit_frequency_weights,
            "explicit_frequency_weights",
        )
        center_weights = self._canonical_weights(
            self.barycenter_weights,
            "barycenter_weights",
        )
        object.__setattr__(self, "explicit_frequency_weights", explicit_weights)
        object.__setattr__(self, "barycenter_weights", center_weights)

        exponent = self._finite_float(self.scaling_exponent, "scaling_exponent")
        tolerance = self._finite_float(self.barycenter_tolerance, "barycenter_tolerance")
        object.__setattr__(self, "scaling_exponent", exponent)
        object.__setattr__(self, "barycenter_tolerance", tolerance)

        integer_fields = {
            "barycenter_size": (self.barycenter_size, 1),
            "n_projections": (self.n_projections, 1),
            "n_quantiles": (self.n_quantiles, 2),
            "barycenter_max_iter": (self.barycenter_max_iter, 1),
        }
        for name, (value, minimum) in integer_fields.items():
            object.__setattr__(self, name, self._positive_integer(value, name, minimum))

        object.__setattr__(self, "random_state", self._seed(self.random_state, "random_state"))
        object.__setattr__(
            self,
            "barycenter_random_state",
            self._seed(self.barycenter_random_state, "barycenter_random_state"),
        )

        if self.scaling not in {"power", "realized_volatility"}:
            raise ConfigurationError(f"Unsupported scaling mode: {self.scaling!r}.")
        if self.frequency_weighting not in {
            "uniform",
            "sample_size",
            "log_sample_size",
            "explicit",
        }:
            raise ConfigurationError(
                f"Unsupported frequency weighting: {self.frequency_weighting!r}."
            )
        if self.barycenter not in {"free_support", "projected_quantile"}:
            raise ConfigurationError(f"Unsupported barycenter: {self.barycenter!r}.")
        if self.distance not in {"sliced", "exact"}:
            raise ConfigurationError(f"Unsupported distance: {self.distance!r}.")

        n_frequencies = len(specs)
        if n_frequencies < 2:
            raise ConfigurationError("At least two frequencies are required.")
        if specs[0].rule is not None:
            raise ConfigurationError(
                "The first frequency must represent the unaggregated base panel."
            )
        if len(set(self.frequencies)) != n_frequencies:
            raise ConfigurationError("Frequency names must be unique.")
        if self.frequency_weighting == "explicit":
            self._validate_weights(
                self.explicit_frequency_weights,
                n_frequencies,
                "explicit_frequency_weights",
            )
        elif self.explicit_frequency_weights is not None:
            raise ConfigurationError(
                "explicit_frequency_weights requires frequency_weighting='explicit'."
            )
        if self.barycenter_weights is not None:
            self._validate_weights(self.barycenter_weights, n_frequencies, "barycenter_weights")
        if self.barycenter_tolerance <= 0:
            raise ConfigurationError("barycenter_tolerance must be finite and positive.")
        if self.barycenter == "projected_quantile" and self.distance != "sliced":
            raise ConfigurationError(
                "The projected-quantile barycenter is defined only with sliced distance."
            )

    @property
    def frequencies(self) -> tuple[str, ...]:
        """Return canonical frequency names in numerical coordinate order."""

        return tuple(spec.name for spec in self.frequency_specs)

    @property
    def horizons(self) -> tuple[float, ...]:
        """Return effective base-period horizons in frequency order."""

        return tuple(spec.horizon for spec in self.frequency_specs)

    @property
    def frequency_grid(self) -> tuple[FrequencySpec, ...]:
        """Return the canonical, fully materialised frequency specification."""

        return self.frequency_specs

    @classmethod
    def reference(cls) -> SignalConfig:
        """Return the fully explicit reference research configuration."""

        return cls()

    @classmethod
    def projected(
        cls,
        *,
        frequency_specs: Sequence[FrequencySpec] = DEFAULT_FREQUENCY_SPECS,
        n_projections: int = 100,
        n_quantiles: int = 100,
        random_state: int = 20250301,
    ) -> SignalConfig:
        """Return a lightweight projected-quantile configuration for exploration."""

        return cls(
            frequency_specs=tuple(frequency_specs),
            barycenter="projected_quantile",
            n_projections=n_projections,
            n_quantiles=n_quantiles,
            random_state=random_state,
        )

    def with_updates(self, **changes: object) -> SignalConfig:
        """Return a validated copy with selected fields changed.

        This is the ergonomic route for modifying an immutable preset without
        repeating every unchanged field. Unknown field names fail explicitly.
        """

        try:
            return replace(self, **changes)  # type: ignore[arg-type]
        except TypeError as exc:
            unknown = sorted(set(changes) - set(self.__dataclass_fields__))
            if unknown:
                raise ConfigurationError(f"Unknown configuration fields: {unknown}.") from exc
            raise

    @staticmethod
    def _validate_weights(
        weights: tuple[float, ...] | None,
        expected_length: int,
        name: str,
    ) -> None:
        if weights is None or len(weights) != expected_length:
            raise ConfigurationError(f"{name} must contain one value per frequency.")
        if not all(math.isfinite(weight) and weight > 0 for weight in weights):
            raise ConfigurationError(f"{name} must be finite and strictly positive.")

    @staticmethod
    def _as_tuple(value: object, name: str) -> tuple[object, ...]:
        if isinstance(value, (str, bytes)):
            raise ConfigurationError(f"{name} must be a sequence, not a string.")
        try:
            return tuple(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ConfigurationError(f"{name} must be a finite sequence.") from exc

    @classmethod
    def _canonical_weights(
        cls,
        weights: Sequence[float] | None,
        name: str,
    ) -> tuple[float, ...] | None:
        if weights is None:
            return None
        values = cls._as_tuple(weights, name)
        return tuple(cls._finite_float(value, name) for value in values)

    @staticmethod
    def _finite_float(value: object, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ConfigurationError(f"{name} must be a finite real number.")
        result = float(value)
        if not math.isfinite(result):
            raise ConfigurationError(f"{name} must be a finite real number.")
        return result

    @staticmethod
    def _positive_integer(value: object, name: str, minimum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ConfigurationError(
                f"{name} must be an integer greater than or equal to {minimum}."
            )
        result = int(value)
        if result < minimum:
            raise ConfigurationError(
                f"{name} must be an integer greater than or equal to {minimum}."
            )
        return result

    @staticmethod
    def _seed(seed: object, name: str) -> int:
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ConfigurationError(f"{name} must be an integer in [0, 2**32).")
        if not 0 <= int(seed) < 2**32:
            raise ConfigurationError(f"{name} must be an integer in [0, 2**32).")
        return int(seed)

    def to_dict(self) -> dict[str, object]:
        """Return a versioned, JSON-serialisable scientific configuration."""

        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "frequency_specs": [spec.to_dict() for spec in self.frequency_specs],
            "scaling": self.scaling,
            "scaling_exponent": self.scaling_exponent,
            "frequency_weighting": self.frequency_weighting,
            "explicit_frequency_weights": (
                None
                if self.explicit_frequency_weights is None
                else list(self.explicit_frequency_weights)
            ),
            "barycenter": self.barycenter,
            "barycenter_size": self.barycenter_size,
            "barycenter_weights": (
                None if self.barycenter_weights is None else list(self.barycenter_weights)
            ),
            "barycenter_random_state": self.barycenter_random_state,
            "distance": self.distance,
            "n_projections": self.n_projections,
            "n_quantiles": self.n_quantiles,
            "random_state": self.random_state,
            "barycenter_max_iter": self.barycenter_max_iter,
            "barycenter_tolerance": self.barycenter_tolerance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SignalConfig:
        """Construct a configuration from :meth:`to_dict` output."""

        if not isinstance(payload, Mapping):
            raise ConfigurationError("Configuration payload must be a mapping.")
        expected = {
            "schema_version",
            "frequency_specs",
            "scaling",
            "scaling_exponent",
            "frequency_weighting",
            "explicit_frequency_weights",
            "barycenter",
            "barycenter_size",
            "barycenter_weights",
            "barycenter_random_state",
            "distance",
            "n_projections",
            "n_quantiles",
            "random_state",
            "barycenter_max_iter",
            "barycenter_tolerance",
        }
        observed = set(payload)
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ConfigurationError(
                f"Configuration fields differ: missing={missing}, extra={extra}."
            )
        if payload["schema_version"] != CONFIG_SCHEMA_VERSION:
            raise ConfigurationError(
                f"Unsupported configuration schema version: {payload['schema_version']!r}."
            )
        raw_specs = cls._as_tuple(payload["frequency_specs"], "frequency_specs")
        specs = tuple(
            FrequencySpec.from_dict(cast(Mapping[str, object], item)) for item in raw_specs
        )
        return cls(
            frequency_specs=specs,
            scaling=cast(Scaling, payload["scaling"]),
            scaling_exponent=cast(float, payload["scaling_exponent"]),
            frequency_weighting=cast(FrequencyWeighting, payload["frequency_weighting"]),
            explicit_frequency_weights=cast(
                tuple[float, ...] | None,
                payload["explicit_frequency_weights"],
            ),
            barycenter=cast(Barycenter, payload["barycenter"]),
            barycenter_size=cast(int, payload["barycenter_size"]),
            barycenter_weights=cast(tuple[float, ...] | None, payload["barycenter_weights"]),
            barycenter_random_state=cast(int, payload["barycenter_random_state"]),
            distance=cast(Distance, payload["distance"]),
            n_projections=cast(int, payload["n_projections"]),
            n_quantiles=cast(int, payload["n_quantiles"]),
            random_state=cast(int, payload["random_state"]),
            barycenter_max_iter=cast(int, payload["barycenter_max_iter"]),
            barycenter_tolerance=cast(float, payload["barycenter_tolerance"]),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise the configuration to deterministic JSON text."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, payload: str) -> SignalConfig:
        """Construct a configuration from JSON text."""

        if not isinstance(payload, str):
            raise ConfigurationError("Configuration JSON must be text.")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ConfigurationError("Configuration JSON is invalid.") from exc
        if not isinstance(decoded, Mapping):
            raise ConfigurationError("Configuration JSON must contain one object.")
        return cls.from_dict(decoded)

    def write_json(self, path: str | Path) -> Path:
        """Write deterministic configuration JSON and return the resolved path."""

        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(), encoding="utf-8")
        return destination

    @classmethod
    def read_json(cls, path: str | Path) -> SignalConfig:
        """Read a configuration written by :meth:`write_json`."""

        source = Path(path).expanduser().resolve()
        return cls.from_json(source.read_text(encoding="utf-8"))

    @property
    def digest(self) -> str:
        """Return a stable SHA-256 identity for this scientific configuration."""

        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
