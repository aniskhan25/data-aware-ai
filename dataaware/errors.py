"""The three things that can go wrong.

One exception per kind of mistake, rather than one per module. A reader does not need
to learn ten class names to follow the code.
"""


class ConfigError(ValueError):
    """A configuration is invalid: unknown key, bad value, unresolvable path."""


class DataError(ValueError):
    """Data or an artifact is missing, inconsistent, or unreadable."""


class SummaryError(ValueError):
    """A run summary does not match the schema."""
