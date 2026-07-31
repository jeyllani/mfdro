"""Public interface for MFDRO."""

from .config import SignalConfig
from .exceptions import ConfigurationError, DataContractError, MFDROError
from .frequency import FrequencySpec
from .preprocessing import build_frequency_measures, compound_returns
from .signal import (
    MultiFrequencySignal,
    PathDiagnostics,
    PathProgress,
    SignalEstimate,
    SignalPath,
    SkipReason,
)

__all__ = [
    "ConfigurationError",
    "DataContractError",
    "FrequencySpec",
    "MFDROError",
    "MultiFrequencySignal",
    "PathDiagnostics",
    "PathProgress",
    "SignalConfig",
    "SignalEstimate",
    "SignalPath",
    "SkipReason",
    "build_frequency_measures",
    "compound_returns",
]

__version__ = "0.1.0"
