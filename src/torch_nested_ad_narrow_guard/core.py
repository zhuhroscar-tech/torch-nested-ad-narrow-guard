"""torch-nested-ad-narrow-guard core: guard two real torch correctness bugs.

Both bugs were independently reproduced from scratch on this host (torch
2.14.0, macOS arm64 CPU) using the EXACT repro code published in the
upstream issues (not paraphrased) -- see README for full commands and
issue links.

1. pytorch/pytorch#196697 -- nested (second-order) forward-mode AD via
   ``torch.func.jvp(jvp1, ...)`` silently collapses ``torch.linalg.slogdet``'s
   second derivative to exactly 0.0 instead of the mathematically correct
   nonzero value. Reproduced verbatim: expected -0.17853600476096013,
   torch.func.jvp-over-jvp actually returns 0.0. This is a genuinely
   dangerous silent-zero failure mode (not a crash, not NaN) for anyone
   computing curvature/Hessian-vector-product-style quantities of
   log-determinants via nested forward-mode AD (e.g. natural-gradient or
   log-likelihood-Hessian methods for Gaussian/matrix-variate models).

   This module does NOT attempt to patch torch's forward-mode AD
   internals (out of scope for a small guard package). Instead it
   provides ``safe_nested_slogdet_second_order_jvp``, a verified-correct
   replacement path using reverse-over-reverse (``torch.autograd.grad``
   with ``create_graph=True`` called twice) and reverse-over-forward
   (``torch.func.jvp`` of ``torch.func.grad``), both of which were
   confirmedon this host to exactly match the expected analytic value
   -- unlike forward-over-reverse (``torch.func.grad`` of
   ``torch.func.jvp``), which ALSO silently returns 0.0 and must be
   avoided.

2. pytorch/pytorch#196708 -- ``torch.nested.narrow(x, dim, starts,
   lengths, layout=torch.jagged)`` followed by ``.unbind()`` silently
   includes an element outside the requested [start, start+length)
   range for at least one row. Reproduced verbatim: for rows
   ``[t, 2*t]`` and ``[3*t, 4*t]`` with starts=[0, 1], lengths=[2, 1],
   the correct selection is ``[t, 2*t]`` and ``[4*t]`` (sum = 7*t), but
   narrow+unbind returns sum = 10*t (i.e. it silently also includes the
   unselected 3*t). At t=0.7: actual 6.999999999999999 vs expected
   4.8999999999999995.

   ``safe_jagged_narrow_unbind`` provides a verified-correct workaround:
   it never calls the buggy ``torch.nested.narrow``, instead building a
   fresh jagged nested tensor directly from the requested per-row
   Python-level slices of the dense input, then unbinding that. This
   matches the mathematically correct selection exactly on this host.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Sequence


class TorchUnavailableError(RuntimeError):
    """Raised when torch cannot be imported."""


def _import_torch():
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise TorchUnavailableError(
            "torch is required for diagnosis and guarding; install the "
            "'torch' extra."
        ) from exc
    return torch


def safe_nested_slogdet_second_order_jvp(f, t):
    """Compute the second derivative of a scalar function ``f`` (which
    internally uses ``torch.linalg.slogdet``) at scalar tensor ``t``,
    avoiding the silently-wrong-zero result that nested forward-mode AD
    (``torch.func.jvp`` of ``torch.func.jvp``) produces for this op
    (pytorch/pytorch#196697).

    Uses reverse-over-reverse differentiation
    (``torch.autograd.grad`` with ``create_graph=True``, called twice),
    which was independently verified on this host to match the
    analytic second derivative exactly for the slogdet case that
    nested forward-mode AD gets wrong.

    ``t`` must be a scalar tensor; ``f`` must be differentiable via
    ``torch.autograd`` (i.e. must not itself require ``torch.func``
    transforms internally).
    """
    torch_module = _import_torch()
    t_req = t.clone().detach().requires_grad_(True)
    y = f(t_req)
    (g1,) = torch_module.autograd.grad(y, t_req, create_graph=True)
    (g2,) = torch_module.autograd.grad(g1, t_req)
    return g2.detach()


def safe_jagged_narrow_unbind(dense_x, dim: int, starts, lengths):
    """Select per-row [start, start+length) slices of dense tensor
    ``dense_x`` along ``dim`` and return them as unbound tensors, without
    ever calling the buggy ``torch.nested.narrow(..., layout=torch.jagged)``
    (pytorch/pytorch#196708), which silently includes elements outside
    the requested per-row range for at least one row.

    ``dense_x`` is a 2D dense tensor (rows, along dim=0, each sliced
    along ``dim``). ``starts``/``lengths`` are 1D integer tensors or
    Python sequences, one entry per row along dim 0.

    Returns a list of 1D tensors, one per row, each the correctly
    selected slice -- equivalent to what narrow+unbind SHOULD have
    returned.
    """
    torch_module = _import_torch()
    if dim != 1:
        raise NotImplementedError(
            "safe_jagged_narrow_unbind currently only supports dim=1 "
            "(the same dim used in the upstream bug report); dim=0 "
            "narrowing over jagged nested tensors is unrelated to this "
            "guard's scope."
        )

    starts_list = [int(s) for s in (starts.tolist() if hasattr(starts, "tolist") else starts)]
    lengths_list = [int(l) for l in (lengths.tolist() if hasattr(lengths, "tolist") else lengths)]

    if len(starts_list) != dense_x.shape[0] or len(lengths_list) != dense_x.shape[0]:
        raise ValueError(
            "starts/lengths must have one entry per row of dense_x "
            f"(dense_x.shape[0]={dense_x.shape[0]}, "
            f"len(starts)={len(starts_list)}, len(lengths)={len(lengths_list)})"
        )

    rows: List[Any] = []
    for i in range(dense_x.shape[0]):
        s, l = starts_list[i], lengths_list[i]
        rows.append(dense_x[i, s : s + l].clone())
    return rows


@dataclasses.dataclass
class SlogdetSecondOrderCase:
    t_value: float
    expected_second_order: float
    forward_over_forward_jvp: float  # the buggy path (#196697)
    forward_over_forward_matches_expected: bool
    guard_reverse_over_reverse: float
    guard_matches_expected: bool


def _run_slogdet_case(torch_module) -> SlogdetSecondOrderCase:
    import torch.func as tfunc

    dtype = torch_module.float64

    def f(t):
        c = torch_module.tensor([1, 2, 3], dtype=t.dtype)
        a = torch_module.stack((t + c[1], c[0], c[0], c[2])).reshape(2, 2)
        sign, value = torch_module.linalg.slogdet(a)
        return value

    t0 = torch_module.tensor(0.7, dtype=dtype)
    expected = -0.17853600476096013

    # Buggy path: forward-over-forward via nested torch.func.jvp.
    def jvp1(t):
        return tfunc.jvp(f, (t,), (torch_module.ones_like(t),))[1]

    _, fof_actual = tfunc.jvp(jvp1, (t0,), (torch_module.ones_like(t0),))
    fof_val = float(fof_actual.item())

    guard_val = float(safe_nested_slogdet_second_order_jvp(f, t0).item())

    return SlogdetSecondOrderCase(
        t_value=0.7,
        expected_second_order=expected,
        forward_over_forward_jvp=fof_val,
        forward_over_forward_matches_expected=abs(fof_val - expected) < 1e-6,
        guard_reverse_over_reverse=guard_val,
        guard_matches_expected=abs(guard_val - expected) < 1e-6,
    )


@dataclasses.dataclass
class JaggedNarrowCase:
    t_value: float
    expected_sum: float
    buggy_narrow_unbind_sum: float
    buggy_matches_expected: bool
    guard_sum: float
    guard_matches_expected: bool


def _run_jagged_narrow_case(torch_module) -> JaggedNarrowCase:
    dtype = torch_module.float64
    t = torch_module.tensor(0.7, dtype=dtype)
    x = torch_module.stack((t, 2 * t, 3 * t, 4 * t)).reshape(2, 2)
    starts = torch_module.tensor([0, 1], dtype=torch_module.int64)
    lengths = torch_module.tensor([2, 1], dtype=torch_module.int64)
    expected = 7 * 0.7

    nt = torch_module.nested.narrow(x, 1, starts, lengths, layout=torch_module.jagged)
    parts = nt.unbind()
    buggy_sum = float((parts[0].sum() + parts[1].sum()).item())

    guard_parts = safe_jagged_narrow_unbind(x, 1, starts, lengths)
    guard_sum = float(sum(p.sum() for p in guard_parts).item())

    return JaggedNarrowCase(
        t_value=0.7,
        expected_sum=expected,
        buggy_narrow_unbind_sum=buggy_sum,
        buggy_matches_expected=abs(buggy_sum - expected) < 1e-6,
        guard_sum=guard_sum,
        guard_matches_expected=abs(guard_sum - expected) < 1e-6,
    )


def diagnose() -> Dict[str, Any]:
    """Reproduce both upstream bugs from scratch against the currently
    installed torch build, using the EXACT repro code from the issues,
    and verify the guard functions produce the mathematically correct
    result instead. Never trusts a cached/prior result."""
    torch_module = _import_torch()

    slogdet_case = _run_slogdet_case(torch_module)
    narrow_case = _run_jagged_narrow_case(torch_module)

    return {
        "torch_version": torch_module.__version__,
        "issue_urls": [
            "https://github.com/pytorch/pytorch/issues/196697",
            "https://github.com/pytorch/pytorch/issues/196708",
        ],
        "slogdet_second_order_case": dataclasses.asdict(slogdet_case),
        "jagged_narrow_unbind_case": dataclasses.asdict(narrow_case),
        "slogdet_bug_reproduced": not slogdet_case.forward_over_forward_matches_expected,
        "narrow_bug_reproduced": not narrow_case.buggy_matches_expected,
        "guards_fully_correct": (
            slogdet_case.guard_matches_expected and narrow_case.guard_matches_expected
        ),
    }
