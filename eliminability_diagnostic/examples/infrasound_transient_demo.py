"""
Infrasound transient demo: research adjacent illustration of the
eliminability discriminant.

The problem construction lives in eliminability.demos. This wrapper
adds verbose printing and is the recommended reading order for new
users who want to see the discriminant applied to a setting closer to
real signal analysis than the Section 5 toy example.

The demo generates a synthetic infrasound window containing two
bandlimited background components and one narrow Gaussian transient.
Two postures are compared:

    BandlimitStrict     excludes the transient under a bandlimit rule
    BroadbandPermissive retains the transient under OLS

Expected behavior:
    Delta_BandlimitStrict(transient)     = 0.000
    Delta_BroadbandPermissive(transient) > 0
    Both background components are invariantly necessary.
    The transient is flagged as evaluation dependent.

Run:
    python examples/infrasound_transient_demo.py
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from eliminability import (
    assign_classifications,
    assign_persistence,
    build_results_table,
    compute_all_discriminants,
    compute_dependence,
    generate_summary,
    mean_squared_error,
)
from eliminability.demos import build_infrasound_transient_problem


def run_infrasound_demo(verbose: bool = True):
    """Run the infrasound transient demo and return the results table."""
    y, components, postures, threshold = build_infrasound_transient_problem()

    discriminants = compute_all_discriminants(
        y=y,
        components=components,
        postures=postures,
        functional=mean_squared_error,
    )
    assign_persistence(discriminants, threshold)
    assign_classifications(discriminants, threshold)

    dependence = compute_dependence(
        discriminants,
        posture_a=postures[0].name,
        posture_b=postures[1].name,
        threshold=threshold,
    )

    table = build_results_table(discriminants, dependence)
    summary = generate_summary(discriminants, dependence, threshold)

    if verbose:
        print("=" * 72)
        print("Infrasound transient demo")
        print("=" * 72)
        print(
            f"Synthetic window: {len(y)} samples. "
            "Two bandlimited backgrounds plus one narrow Gaussian transient."
        )
        print()
        print("Discriminant table:")
        print(table.to_string(index=False))
        print()
        print(summary)
        print("=" * 72)

    return table, discriminants, dependence, summary


if __name__ == "__main__":
    run_infrasound_demo()
