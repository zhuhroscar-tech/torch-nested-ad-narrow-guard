"""Tests for the CLI entry point: argument parsing, --version, --json,
--no-color, and exit codes -- independent of whether torch is installed.
Mirrors the test_cli.py pattern already used across the fleet (e.g.
rng-leak-audit, causality-audit) for a repo that had none.

The branch-coverage tests below mock ``core.diagnose`` so every CLI
message path (torch-unavailable, the four no-bug "info" lines, and a
guard-mismatch "fail" line) is exercised deterministically, regardless
of whether this host's installed torch build happens to reproduce the
underlying bugs. Real fleet-wide gap found by pytest-cov inspection:
cli.py sat at 79% coverage with the negative branches of each status
line (lines 50-56, 68, 73, 78, 83) and the ``__main__`` guard (line
119) never hit by any existing test."""
from __future__ import annotations

import json
import runpy
import sys

import pytest

from torch_nested_ad_narrow_guard import core
from torch_nested_ad_narrow_guard.cli import main


def _fake_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "slogdet_bug_reproduced": False,
        "householder_bug_reproduced": False,
        "layer_norm_bug_reproduced": False,
        "narrow_bug_reproduced": False,
        "padded_transform_bug_reproduced": False,
        "autograd_function_higher_order_bug_reproduced": False,
        "guards_fully_correct": True,
        "slogdet_second_order_case": {
            "expected_second_order": 1.0,
            "forward_over_forward_jvp": 1.0,
            "guard_reverse_over_reverse": 1.0,
        },
        "householder_product_second_order_case": {
            "expected_second_order": 1.0,
            "forward_over_forward_jvp": 1.0,
            "guard_reverse_over_reverse": 1.0,
        },
        "layer_norm_second_order_case": {
            "expected_second_order": 1.0,
            "forward_over_forward_jvp": 1.0,
            "guard_reverse_over_reverse": 1.0,
        },
        "jagged_narrow_unbind_case": {
            "expected_sum": 1.0,
            "buggy_narrow_unbind_sum": 1.0,
            "guard_sum": 1.0,
        },
        "jagged_padded_transform_case": {
            "backward_raised_on_buggy_path": False,
            "guard_backward_succeeded": True,
            "guard_forward_matches_reference": True,
        },
        "autograd_function_higher_order_case": {
            "expected_derivatives": [108.0, 108.0, 72.0],
            "custom_function_jacfwd_chain": [108.0, 108.0, 72.0],
            "guard_reverse_mode_chain": [108.0, 108.0, 72.0],
        },
    }
    report.update(overrides)
    return report


def test_version_flag(capsys):
    code = main(["--version"])
    out = capsys.readouterr().out
    assert code == 0
    assert "torch-nested-ad-narrow-guard" in out


def test_json_output_is_valid_json_and_reports_guard_status(capsys):
    torch = pytest.importorskip("torch")
    code = main(["--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert "torch_version" in report
    assert report["torch_version"] == torch.__version__
    assert "guards_fully_correct" in report
    assert code in (0, 1)


def test_json_exit_code_matches_guards_fully_correct(capsys):
    pytest.importorskip("torch")
    code = main(["--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert code == (0 if report["guards_fully_correct"] else 1)


def test_text_output_no_color_has_no_ansi_escapes(capsys):
    pytest.importorskip("torch")
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "\x1b[" not in out


def test_text_output_reports_both_cases(capsys):
    pytest.importorskip("torch")
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "slogdet second-order derivative" in out
    assert "jagged narrow+unbind selection sum" in out


def test_torch_unavailable_json_mode_reports_error_and_exit_2(monkeypatch, capsys):
    """cli.py lines 51-52: TorchUnavailableError + --json emits a JSON
    error object and exits 2, regardless of torch install state."""

    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload == {"error": "torch is required for diagnosis"}
    assert code == 2


def test_torch_unavailable_text_mode_reports_fail_headline_and_exit_2(monkeypatch, capsys):
    """cli.py lines 53-56: TorchUnavailableError in text mode prints a
    'fail' status headline (not the JSON branch) and exits 2."""

    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "torch unavailable: torch is required for diagnosis" in out
    assert "[X]" in out
    assert code == 2


def test_no_bugs_present_prints_all_four_info_lines(monkeypatch, capsys):
    """cli.py lines 68, 73, 78, 83-ish: the 'info' (not 'warn') branch
    for each of the tracked bugs when none reproduce on this host."""
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report())
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "slogdet nested-JVP bug did not reproduce" in out
    assert "householder_product nested-JVP bug did not reproduce" in out
    assert "layer_norm nested-JVP bug did not reproduce" in out
    assert "narrow+unbind bug did not reproduce" in out
    assert "padded<->jagged roundtrip bug did not reproduce" in out
    assert "jacfwd-chain-through-custom-Function bug did not reproduce" in out
    assert "slogdet second-derivative bug reproduced" not in out
    assert "householder_product second-derivative bug reproduced" not in out
    assert "layer_norm second-derivative bug reproduced" not in out
    assert "narrow+unbind element-selection bug reproduced" not in out
    assert "padded<->jagged roundtrip breaks backward pass" not in out


def test_all_bugs_present_prints_all_four_warn_lines(monkeypatch, capsys):
    """Positive-branch companion: when all tracked bugs ARE reproduced,
    the 'warn' headlines fire instead."""
    monkeypatch.setattr(
        core,
        "diagnose",
        lambda: _fake_report(
            slogdet_bug_reproduced=True,
            householder_bug_reproduced=True,
            layer_norm_bug_reproduced=True,
            narrow_bug_reproduced=True,
            padded_transform_bug_reproduced=True,
            autograd_function_higher_order_bug_reproduced=True,
        ),
    )
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "slogdet second-derivative bug reproduced" in out
    assert "householder_product second-derivative bug reproduced" in out
    assert "layer_norm second-derivative bug reproduced" in out
    assert "narrow+unbind element-selection bug reproduced" in out
    assert "padded<->jagged roundtrip breaks backward pass" in out
    assert "jacfwd chained 3-deep through custom autograd.Function silently zeroes 2nd/3rd derivative" in out


def test_guard_mismatch_prints_fail_line_and_exit_1(monkeypatch, capsys):
    """cli.py line 83 + 115: guards_fully_correct=False prints the
    'fail' headline (not the 'ok' one) and the process exits 1."""
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(guards_fully_correct=False))
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "at least one guard did NOT match the expected value" in out
    assert "all match the mathematically correct value" not in out
    assert code == 1


def test_module_entry_point_runs_main_and_exits_with_its_code(monkeypatch):
    """cli.py line 119 (``if __name__ == "__main__": sys.exit(main())``):
    running the module as a script must invoke main() and propagate its
    return code via SystemExit, not just be dead code."""
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(guards_fully_correct=False))
    monkeypatch.setattr(sys, "argv", ["torch-nested-ad-narrow-guard", "--no-color"])
    monkeypatch.delitem(sys.modules, "torch_nested_ad_narrow_guard.cli", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("torch_nested_ad_narrow_guard.cli", run_name="__main__")
    assert exc_info.value.code == 1
