"""
Exception hierarchy for the hybrid NTN optimizer.

Convention
----------
Raise the most specific subclass so callers can catch narrowly.
All exceptions carry a human-readable message; include the offending
value wherever helpful.
"""


class NTNOptimizerError(Exception):
    """Base class for all project-specific errors."""


# ---------------------------------------------------------------------------
# Configuration / input errors
# ---------------------------------------------------------------------------

class ConfigurationError(NTNOptimizerError):
    """Raised when a configuration value is missing or invalid."""


class InvalidParameterError(ConfigurationError):
    """Raised when a numeric parameter is out of its valid range."""


# ---------------------------------------------------------------------------
# Constellation / propagation errors
# ---------------------------------------------------------------------------

class ConstellationError(NTNOptimizerError):
    """Base class for constellation-related errors."""


class IncompatibleOrbitTypeError(ConstellationError):
    """Raised when an operation is not valid for the given orbit type."""


class PropagationError(ConstellationError):
    """Raised when orbital propagation fails (e.g. divergence)."""


class VisibilityError(ConstellationError):
    """Raised when a visibility computation cannot be completed."""


# ---------------------------------------------------------------------------
# Link-budget errors
# ---------------------------------------------------------------------------

class LinkBudgetError(NTNOptimizerError):
    """Base class for link-budget computation errors."""


# ---------------------------------------------------------------------------
# Optimization errors
# ---------------------------------------------------------------------------

class OptimizationError(NTNOptimizerError):
    """Raised when the optimizer fails to find a feasible solution."""


class InfeasibleConstraintError(OptimizationError):
    """Raised when the constraint set has no feasible region."""


# ---------------------------------------------------------------------------
# I/O errors
# ---------------------------------------------------------------------------

class DataIOError(NTNOptimizerError):
    """Raised for file-read/write or serialization failures."""