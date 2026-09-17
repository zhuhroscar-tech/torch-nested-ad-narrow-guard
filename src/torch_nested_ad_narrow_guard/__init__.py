"""torch-nested-ad-narrow-guard: version and package marker."""
__version__ = "0.2.0"

from .core import (
    safe_nested_slogdet_second_order_jvp,
    safe_jagged_narrow_unbind,
    safe_jagged_padded_transform,
    diagnose,
)  # noqa: F401
