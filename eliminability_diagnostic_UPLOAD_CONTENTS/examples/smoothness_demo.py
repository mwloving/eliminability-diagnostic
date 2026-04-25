"""
Smoothness demo: soft penalty analogue of the toy demo.

A weighted posture applies a large quadratic penalty to components
tagged as violating the smoothness constraint, pushing their
coefficients toward zero without strictly removing them. As the
violation penalty grows the weighted posture approaches the strict
posture; as it falls to zero it approaches the permissive posture.

The problem construction lives in eliminability.demos.
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
from eliminability.demos import build_smoothness_problem


def run_smoothness_demo(
    lam: float = 5.0, violation_penalty: float = 100.0, verbose: bool = True
):
    y, components, postures, threshold = build_smoothness_problem(
        lam=lam, violation_penalty=violation_penalty
    )

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
        posture_a="SoftPenalty",
        posture_b="Permissive",
        threshold=threshold,
    )

    table = build_results_table(discriminants, dependence)
    summary = generate_summary(discriminants, dependence, threshold)

    if verbose:
        print("=" * 72)
        print(
            f"Smoothness demo (lambda = {lam}, violation penalty = {violation_penalty})"
        )
        print("=" * 72)
        print(f"y = {y}")
        print()
        print(table.to_string(index=False))
        print()
        print(summary)
        print("=" * 72)

    return table, discriminants, dependence, summary


if __name__ == "__main__":
    run_smoothness_demo()
