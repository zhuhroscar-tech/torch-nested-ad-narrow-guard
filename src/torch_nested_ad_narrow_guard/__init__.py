"""torch-nested-ad-narrow-guard: version and package marker."""
__version__ = "0.1.0"

from .core import (
    safe_nested_slogdet_second_order_jvp,
    safe_jagged_narrow_unbind,
    diagnose,
)  # noqa: F401
