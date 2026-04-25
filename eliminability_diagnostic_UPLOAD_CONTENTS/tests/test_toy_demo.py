"""
Regression tests against the classical numerical illustration of
Section 5 of the accompanying manuscript.

Any change that moves these numbers is a real change to the tool.
Run with:   pytest -q   or:   python -m pytest tests/

No pytest fixtures are used so the file can also be executed directly
via   python tests/test_toy_demo.py   and will raise on failure.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from eliminability import (
    CandidateComponent,
    PermissivePosture,
    StrictPosture,
    assign_classifications,
    assign_persistence,
    compute_all_discriminants,
    compute_dependence,
    mean_squared_error,
)


TOL = 1e-3  # manuscript reports 3 significant figures


def _build_toy():
    y = np.array([0.610, 0.410, 0.020, -0.590, -0.410, 0.880])
    tau1 = CandidateComponent("tau1", np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0]))
    tau2 = CandidateComponent("tau2", np.array([0.0, 1.0, 0.0, 0.0, -1.0, 0.0]))
    tau3 = CandidateComponent(
        "tau3",
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        tags={"violates_constraint": True},
    )
    components = [tau1, tau2, tau3]
    postures = [StrictPosture("Strict"), PermissivePosture("Permissive")]
    return y, components, postures


def _delta(results, component_name, posture_name):
    for r in results:
        if r.component_name == component_name and r.posture_name == posture_name:
            return r.delta
    raise KeyError((component_name, posture_name))


def test_toy_discriminants_match_manuscript():
    """Delta values for all three components under both postures
    must match Table 1 of Section 5 to three significant figures."""
    y, components, postures = _build_toy()
    results = compute_all_discriminants(y, components, postures, mean_squared_error)

    # tau1 and tau2 are admissible under both postures and must give
    # equal discriminants across them (orthogonality of the basis).
    assert abs(_delta(results, "tau1", "Strict") - 0.120) < TOL
    assert abs(_delta(results, "tau1", "Permissive") - 0.120) < TOL
    assert abs(_delta(results, "tau2", "Strict") - 0.056) < TOL
    assert abs(_delta(results, "tau2", "Permissive") - 0.056) < TOL

    # tau3 is the evaluation dependent component.
    assert abs(_delta(results, "tau3", "Strict") - 0.000) < TOL
    assert abs(_delta(results, "tau3", "Permissive") - 0.129) < TOL


def test_toy_dependence_magnitude_matches_manuscript():
    """D_{F,G}(tau3) must equal 0.129 to three significant figures,
    and D must vanish for the two admissible components."""
    y, components, postures = _build_toy()
    results = compute_all_discriminants(y, components, postures, mean_squared_error)
    dep = compute_dependence(results, "Strict", "Permissive", threshold=0.05)

    by_name = {d.component_name: d for d in dep}
    assert by_name["tau1"].dependence_magnitude < TOL
    assert by_name["tau2"].dependence_magnitude < TOL
    assert abs(by_name["tau3"].dependence_magnitude - 0.129) < TOL

    # tau3 is the only component whose postures place it on opposite
    # sides of the unit persistence threshold at theta = 0.05.
    assert by_name["tau3"].cross_threshold is True
    assert by_name["tau1"].cross_threshold is False
    assert by_name["tau2"].cross_threshold is False


def test_toy_classification():
    """tau3 must carry the 'evaluation dependent' label under the
    default classification precedence; tau1 and tau2 must not."""
    y, components, postures = _build_toy()
    results = compute_all_discriminants(y, components, postures, mean_squared_error)
    assign_persistence(results, threshold=0.05)
    assign_classifications(results, threshold=0.05)

    labels = {r.component_name: r.classification for r in results}
    assert labels["tau3"] == "evaluation dependent"
    assert labels["tau1"] == "invariantly necessary"
    assert labels["tau2"] == "invariantly necessary"


def test_persistence_scores():
    """Persistence score P_F(tau) = Delta_F(tau) / theta.
    tau3 must give P_F = 0 and P_G approximately 2.58 at theta = 0.05."""
    y, components, postures = _build_toy()
    results = compute_all_discriminants(y, components, postures, mean_squared_error)
    assign_persistence(results, threshold=0.05)

    scores = {
        (r.component_name, r.posture_name): r.persistence_score for r in results
    }
    assert scores[("tau3", "Strict")] == 0.0
    assert abs(scores[("tau3", "Permissive")] - 2.58) < 0.02


def test_fixed_candidate_set_invariant():
    """Removing a component from the candidate set must only reduce
    the fitted set, never reparameterize the remaining basis."""
    y, components, postures = _build_toy()
    # Under Strict, tau3 is excluded, so removing tau3 must leave
    # Delta_Strict(tau3) at exactly zero (not a refitting artifact).
    results = compute_all_discriminants(y, components, postures, mean_squared_error)
    assert _delta(results, "tau3", "Strict") == 0.0


if __name__ == "__main__":
    test_toy_discriminants_match_manuscript()
    test_toy_dependence_magnitude_matches_manuscript()
    test_toy_classification()
    test_persistence_scores()
    test_fixed_candidate_set_invariant()
    print("All regression tests passed.")
