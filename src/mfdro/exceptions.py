"""Package-specific exceptions."""


class MFDROError(Exception):
    """Base class for package errors."""


class ConfigurationError(MFDROError, ValueError):
    """Raised when a scientific configuration is internally inconsistent."""


class DataContractError(MFDROError, ValueError):
    """Raised when an input panel violates the point-in-time data contract."""
