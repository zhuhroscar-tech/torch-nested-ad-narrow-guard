"""Command-line interface: run the from-scratch diagnosis of the nested
forward-mode-AD slogdet second-derivative bug (pytorch/pytorch#196697),
the jagged narrow+unbind bug (pytorch/pytorch#196708), and the padded
<-> jagged roundtrip backward-pass crash (pytorch/pytorch#145837)
against the currently installed torch build, using the shared
semantic-color design system.
"""
from __future__ import annotations

import argparse
import json
import sys

from .style import print_fields, resolve_style, section, status_headline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-nested-ad-narrow-guard",
        description=(
            "Diagnose four real torch correctness bugs against the "
            "currently installed torch build: (1) nested (second-order) "
            "forward-mode AD via torch.func.jvp-of-jvp silently returns "
            "0.0 instead of the correct nonzero second derivative of "
            "torch.linalg.slogdet (pytorch/pytorch#196697), (2) the same "
            "failure class for torch.linalg.householder_product's second "
            "derivative (pytorch/pytorch#196698), (3) "
            "torch.nested.narrow(..., layout=torch.jagged) followed by "
            "unbind() silently includes an unselected element "
            "(pytorch/pytorch#196708), and (4) a padded<->jagged "
            "roundtrip that crashes backward (pytorch/pytorch#145837). "
            "Verifies that the safe_* guard functions produce the "
            "mathematically correct result instead. Never trusts a "
            "cached or previously-reported result, always re-runs the "
            "repro on THIS host's actual installed torch version."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-nested-ad-narrow-guard {__version__}")
        return 0

    from .core import TorchUnavailableError, diagnose

    try:
        report = diagnose()
    except TorchUnavailableError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            style = resolve_style(no_color_flag=args.no_color)
            print(status_headline(style, "fail", f"torch unavailable: {exc}"))
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["guards_fully_correct"] else 1

    style = resolve_style(no_color_flag=args.no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["slogdet_bug_reproduced"]:
        print(status_headline(style, "warn", "nested forward-mode AD slogdet second-derivative bug reproduced (#196697)"))
    else:
        print(status_headline(style, "info", "slogdet nested-JVP bug did not reproduce on this host's installed torch build"))

    if report["householder_bug_reproduced"]:
        print(status_headline(style, "warn", "nested forward-mode AD householder_product second-derivative bug reproduced (#196698)"))
    else:
        print(status_headline(style, "info", "householder_product nested-JVP bug did not reproduce on this host's installed torch build"))

    if report["narrow_bug_reproduced"]:
        print(status_headline(style, "warn", "jagged narrow+unbind element-selection bug reproduced (#196708)"))
    else:
        print(status_headline(style, "info", "narrow+unbind bug did not reproduce on this host's installed torch build"))

    if report["padded_transform_bug_reproduced"]:
        print(status_headline(style, "warn", "padded<->jagged roundtrip breaks backward pass (#145837)"))
    else:
        print(status_headline(style, "info", "padded<->jagged roundtrip bug did not reproduce on this host's installed torch build"))

    if report["guards_fully_correct"]:
        print(status_headline(style, "ok", "safe_nested_slogdet_second_order_jvp(), safe_nested_householder_product_second_order_jvp(), safe_jagged_narrow_unbind(), and safe_jagged_padded_transform() all match the mathematically correct value"))
    else:
        print(status_headline(style, "fail", "at least one guard did NOT match the expected value"))

    section("slogdet second-order derivative (t=0.7)")
    c = report["slogdet_second_order_case"]
    print_fields(
        [
            ("expected", f"{c['expected_second_order']!s}"),
            ("forward-over-forward jvp (buggy)", f"{c['forward_over_forward_jvp']!s}"),
            ("guard (reverse-over-reverse)", f"{c['guard_reverse_over_reverse']!s}"),
        ]
    )

    section("householder_product second-order derivative (t=0.7)")
    c = report["householder_product_second_order_case"]
    print_fields(
        [
            ("expected", f"{c['expected_second_order']!s}"),
            ("forward-over-forward jvp (buggy)", f"{c['forward_over_forward_jvp']!s}"),
            ("guard (reverse-over-reverse)", f"{c['guard_reverse_over_reverse']!s}"),
        ]
    )

    section("jagged narrow+unbind selection sum (t=0.7)")
    c = report["jagged_narrow_unbind_case"]
    print_fields(
        [
            ("expected", f"{c['expected_sum']!s}"),
            ("buggy narrow+unbind", f"{c['buggy_narrow_unbind_sum']!s}"),
            ("guard (manual per-row slice)", f"{c['guard_sum']!s}"),
        ]
    )

    section("padded<->jagged roundtrip backward pass (#145837)")
    c = report["jagged_padded_transform_case"]
    print_fields(
        [
            ("buggy path raised RuntimeError on backward", f"{c['backward_raised_on_buggy_path']!s}"),
            ("guard backward succeeded", f"{c['guard_backward_succeeded']!s}"),
            ("guard forward matches reference", f"{c['guard_forward_matches_reference']!s}"),
        ]
    )

    return 0 if report["guards_fully_correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
