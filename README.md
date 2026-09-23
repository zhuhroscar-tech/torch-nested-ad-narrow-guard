# torch-nested-ad-narrow-guard

This repository has been consolidated into [`torch-correctness-guards`](https://github.com/zhuhroscar-tech/torch-correctness-guards).

Use the umbrella package instead:

```bash
python -m pip install git+https://github.com/zhuhroscar-tech/torch-correctness-guards.git

torch-guard run nested-ad-narrow
```

Python API equivalents are exported from `torch_correctness_guards`:

```python
from torch_correctness_guards import diagnose_nested_ad_narrow
from torch_correctness_guards import safe_nested_slogdet_second_order_jvp
from torch_correctness_guards import safe_nested_householder_product_second_order_jvp
from torch_correctness_guards import safe_layer_norm_second_order_jvp
from torch_correctness_guards import safe_jagged_narrow_unbind
from torch_correctness_guards import safe_jagged_padded_transform
from torch_correctness_guards import safe_autograd_function_higher_order_derivative
```

This source repository is archived to keep the GitHub account focused on fewer, more complete packages. The original functionality is preserved in the umbrella package as the `nested-ad-narrow` guard.
