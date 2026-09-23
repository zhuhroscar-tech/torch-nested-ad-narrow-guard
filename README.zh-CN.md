# torch-nested-ad-narrow-guard

本仓库已合并到 [`torch-correctness-guards`](https://github.com/zhuhroscar-tech/torch-correctness-guards)。

请改用总包：

```bash
python -m pip install git+https://github.com/zhuhroscar-tech/torch-correctness-guards.git

torch-guard run nested-ad-narrow
```

对应的 Python API 已由 `torch_correctness_guards` 导出：

```python
from torch_correctness_guards import diagnose_nested_ad_narrow
from torch_correctness_guards import safe_nested_slogdet_second_order_jvp
from torch_correctness_guards import safe_nested_householder_product_second_order_jvp
from torch_correctness_guards import safe_layer_norm_second_order_jvp
from torch_correctness_guards import safe_jagged_narrow_unbind
from torch_correctness_guards import safe_jagged_padded_transform
from torch_correctness_guards import safe_autograd_function_higher_order_derivative
```

本源仓库已归档，以便 GitHub 账号聚焦于更少、更完整的软件包。原功能已作为 `nested-ad-narrow` guard 保留在总包中。
