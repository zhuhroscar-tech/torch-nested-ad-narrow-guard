# torch-nested-ad-narrow-guard

Diagnoses and guards three real, independently-reproduced correctness bugs
in PyTorch (torch 2.14.0 confirmed on this host, macOS arm64 CPU):

## Bug 1 — nested forward-mode AD silently zeros a slogdet second derivative

**Upstream issue:** [pytorch/pytorch#196697](https://github.com/pytorch/pytorch/issues/196697) (open)

`torch.func.jvp` nested inside another `torch.func.jvp` call (i.e.
forward-over-forward second-order differentiation) silently returns
`0.0` for the second derivative of `torch.linalg.slogdet`, instead of
the mathematically correct nonzero value.

Exact repro (from the issue, reproduced verbatim on this host):

```python
import torch
dtype = torch.float64
def f(t):
    c = torch.tensor([1, 2, 3], dtype=t.dtype)
    a = torch.stack((t + c[1], c[0], c[0], c[2])).reshape(2, 2)
    sign, value = torch.linalg.slogdet(a)
    return value
def jvp1(t):
    return torch.func.jvp(f, (t,), (torch.ones_like(t),))[1]
t = torch.tensor(0.7, dtype=dtype)
_, actual = torch.func.jvp(jvp1, (t,), (torch.ones_like(t),))
# actual   = 0.0
# expected = -0.17853600476096013
```

This is dangerous specifically because it fails *silently* — no
exception, no NaN, no warning — for anyone computing second-order
sensitivities (e.g. curvature/Hessian-vector products) of
log-determinants via nested forward-mode AD.

**Investigated during development** (not just the buggy path): forward-
over-reverse (`torch.func.grad` of `torch.func.jvp`) is **also**
silently wrong on this host — it returns `0.0` too. Only
reverse-mode-first approaches are correct:

| Method | Result on this host |
|---|---|
| forward-over-forward (`jvp` of `jvp`) — **buggy** | `0.0` |
| forward-over-reverse (`grad` of `jvp`) — **also buggy** | `0.0` |
| reverse-over-reverse (`autograd.grad(create_graph=True)` twice) | `-0.17853600476096013` ✅ |
| reverse-over-forward (`jvp` of `grad`) | `-0.17853600476096013` ✅ |
| central finite difference (independent oracle) | `-0.17853600507961032` ✅ (matches to ~1e-9) |

`safe_nested_slogdet_second_order_jvp(f, t)` uses reverse-over-reverse
differentiation and is verified to match the analytic value to
`1e-9`.

## Bug 2 — jagged nested-tensor narrow+unbind includes an unselected element

**Upstream issue:** [pytorch/pytorch#196708](https://github.com/pytorch/pytorch/issues/196708) (open)

`torch.nested.narrow(x, dim, starts, lengths, layout=torch.jagged)`
followed by `.unbind()` silently includes an element **outside** the
requested `[start, start+length)` range for at least one row.

Exact repro (from the issue, reproduced verbatim on this host):

```python
import torch
dtype = torch.float64
t = torch.tensor(0.7, dtype=dtype)
x = torch.stack((t, 2 * t, 3 * t, 4 * t)).reshape(2, 2)
starts = torch.tensor([0, 1], dtype=torch.int64)
lengths = torch.tensor([2, 1], dtype=torch.int64)
nt = torch.nested.narrow(x, 1, starts, lengths, layout=torch.jagged)
parts = nt.unbind()
actual = parts[0].sum() + parts[1].sum()
# actual   = 6.999999999999999  (= 10*t, includes the unselected 3*t)
# expected = 4.8999999999999995 (= 7*t)
```

`safe_jagged_narrow_unbind(dense_x, dim, starts, lengths)` never calls
the buggy `torch.nested.narrow`. Instead it builds the correct
per-row slices directly with plain Python/tensor indexing and returns
them as a list of tensors — verified to match the expected selection
exactly.

## Bug 3 — padded↔jagged nested-tensor roundtrip crashes the backward pass

**Upstream issue:** [pytorch/pytorch#145837](https://github.com/pytorch/pytorch/issues/145837) (open)

A common NestedTensor pattern — convert a jagged nested tensor to
padded form with `torch.nested.to_padded_tensor`, apply some
operation unsupported directly in jagged layout (e.g. adding a
positional embedding), then convert back to jagged with
`torch.nested.narrow(..., layout=torch.jagged)` — raises on
`.backward()`:

```
RuntimeError: Function CloneBackward0 returned an invalid gradient at
index 0 - got [4, j21, 64] but expected shape compatible with [4, j20, 64]
```

Independently reproduced on this host with a 3-sequence jagged batch
(lengths 3/2/4, embedding dim 8):

```python
import torch
import torch.nn as nn

def padded_from_jagged(tensor, pad_value=0.0):
    offsets = tensor.offsets()
    padded = torch.nested.to_padded_tensor(tensor, padding=pad_value)
    return padded, offsets

def jagged_from_padded(tensor, offsets, contiguous=True):
    seq_lens = offsets.diff()
    jagged = torch.nested.narrow(tensor, dim=1, start=0, length=seq_lens, layout=torch.jagged)
    return jagged.contiguous() if contiguous else jagged

values = torch.randn(9, 8, requires_grad=True)
offsets = torch.tensor([0, 3, 5, 9])
x = torch.nested.nested_tensor_from_jagged(values, offsets)

padded, offs = padded_from_jagged(x)
padded = padded + torch.randn(1, padded.shape[1], 8)
jagged = jagged_from_padded(padded, offs)
jagged.values().mean().backward()
# RuntimeError: Function CloneBackward0 returned an invalid gradient ...
```

The forward pass succeeds every time; only `.backward()` fails. This
is a training-loop stopper for a normal, expected workflow (adding
positional information to variable-length sequence batches), not a
silently-wrong-output bug like bugs 1 and 2.

`safe_jagged_padded_transform(nested_x, transform_fn)` never calls
`torch.nested.narrow`. Instead it pads, applies `transform_fn`,
reconstructs the jagged tensor by slicing each row back to its
original length with plain Python/tensor indexing, and
re-concatenates with `torch.cat` — verified on this host to complete
`.backward()` without raising, and to produce forward values that
exactly match an independent no-grad reference computed the same way.

## Install

```bash
pip install "torch-nested-ad-narrow-guard[torch]"
```

## Usage

```bash
torch-nested-ad-narrow-guard          # human-readable report
torch-nested-ad-narrow-guard --json   # machine-readable report
torch-nested-ad-narrow-guard --no-color
```

Exit code `0` if both guard functions match the expected value on
this host's installed torch build, `1` if a guard itself is wrong,
`2` if torch isn't installed.

```python
from torch_nested_ad_narrow_guard import (
    safe_nested_slogdet_second_order_jvp,
    safe_jagged_narrow_unbind,
    safe_jagged_padded_transform,
)
```

## Real limitations

- **Narrow scope by design.** `safe_jagged_narrow_unbind` only
  supports `dim=1` (matching the exact upstream repro) and only
  handles a dense 2D input with per-row starts/lengths — it does not
  attempt to cover every jagged-narrow call shape.
- **`safe_nested_slogdet_second_order_jvp` requires `f` to be
  autograd-differentiable** (no internal `torch.func` transforms) —
  it does not generalize to arbitrary user functions that themselves
  use forward-mode AD internally.
- **`safe_jagged_padded_transform` requires `transform_fn` to preserve
  the padded tensor's shape** (`[batch, max_len, ...]` in, same shape
  out) and only reconstructs along dim=1 (sequence length) — it does
  not support transforms that change batch size or sequence length.
- **These are workarounds, not upstream fixes.** If PyTorch fixes any
  of these issues in a future release, this package's own regression
  tests (`test_slogdet_nested_jvp_silently_returns_zero_on_this_host`,
  `test_narrow_unbind_silently_includes_unselected_element_on_this_host`,
  `test_padded_jagged_roundtrip_breaks_backward_on_this_host`)
  will start failing — that is the intended signal to re-check the
  issue and update this README, not a regression in this package.
- **Verified on torch 2.14.0** (macOS arm64 CPU via local testing,
  ubuntu-latest + macos-latest via CI, both CPU-only). Not tested on
  CUDA/ROCm/MPS-accelerated code paths; the upstream issues report
  the same CPU-reproducible behavior regardless of accelerator
  availability, but this package does not independently verify that.
- Read the full source before relying on this in a security- or
  correctness-critical pipeline; this is a small guard package, not a
  PyTorch patch.

## License

MIT
