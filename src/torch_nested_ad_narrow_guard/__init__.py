"""torch-nested-ad-narrow-guard: version and package marker."""
__version__ = "0.5.1"

from .core import (
    safe_nested_slogdet_second_order_jvp,
    safe_nested_householder_product_second_order_jvp,
    safe_layer_norm_second_order_jvp,
    safe_jagged_narrow_unbind,
    safe_jagged_padded_transform,
    safe_autograd_function_higher_order_derivative,
    diagnose,
)  # noqa: F401
