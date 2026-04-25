"""
Toy signal demo: reproduces the classical numerical illustration of
Section 5 of the manuscript. The problem construction lives in
eliminability.demos; this script adds verbose printing.

Expected outputs:

    Delta_F(tau1) = Delta_G(tau1) = 0.120
    Delta_F(tau2) = Delta_G(tau2) = 0.056
    Delta_F(tau3) = 0.000
    Delta_G(tau3) = 0.129
    D_{F,G}(tau3) = 0.129
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
from eliminability.demos import build_toy_problem


def run_toy_demo(verbose: bool = True):
    y, components, postures, threshold = build_toy_problem()

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
        posture_a="Strict",
        posture_b="Permissive",
        threshold=threshold,
    )

    table = build_results_table(discriminants, dependence)
    summary = generate_summary(discriminants, dependence, threshold)

    if verbose:
        print("=" * 72)
        print("Toy signal demo")
        print("=" * 72)
        print(f"y = {y}")
        print()
        print(table.to_string(index=False))
        print()
        print(summary)
        print("=" * 72)

    return table, discriminants, dependence, summary


if __name__ == "__main__":
    run_toy_demo()
