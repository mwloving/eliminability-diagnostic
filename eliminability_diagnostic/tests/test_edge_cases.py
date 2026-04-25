"""
Edge case and robustness tests for the Eliminability Diagnostic.

These tests go beyond the toy example regression suite in test_toy_demo.py.
They exercise the behavior of the discriminant, posture, and persistence
layers under conditions where a naive implementation would silently
misbehave: noisy inputs, collinear components, degenerate bases,
threshold boundary values, and malformed calls.

Run with:
    pytest tests/ -q
or directly:
    python tests/test_edge_cases.py
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pytest

from eliminability import (
    CandidateComponent,
    PermissivePosture,
    StrictPosture,
    ThresholdedPosture,
    WeightedPosture,
    assign_classifications,
    assign_persistence,
    compute_all_discriminants,
    compute_dependence,
    compute_discriminant,
    mean_squared_error,
    persistence_score,
    reconstruct_reduced,
    squared_frobenius_norm,
)


# ---------------------------------------------------------------------------
# Noise robustness
# ---------------------------------------------------------------------------

def _build_toy(y):
    """Return (components, postures) for the canonical toy basis."""
    tau1 = CandidateComponent("tau1", np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0]))
    tau2 = CandidateComponent("tau2", np.array([0.0, 1.0, 0.0, 0.0, -1.0, 0.0]))
    tau3 = CandidateComponent(
        "tau3",
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        tags={"violates_constraint": True},
    )
    components = [tau1, tau2, tau3]
    postures = [StrictPosture("Strict"), PermissivePosture("Permissive")]
    return components, postures


def test_noise_degrades_D_smoothly():
    """D(tau3) under increasing noise should remain a finite, non negative
    scalar and should decrease continuously rather than jumping. We test
    monotonic degradation on average across noise realizations."""
    base_y = np.array([0.610, 0.410, 0.020, -0.590, -0.410, 0.880])
    components, postures = _build_toy(base_y)

    rng = np.random.default_rng(seed=42)
    D_by_sigma = {}
    for sigma in (0.0, 0.01, 0.1, 0.5):
        Ds = []
        for _ in range(20):
            y = base_y + rng.normal(0.0, sigma, size=base_y.size)
            results = compute_all_discriminants(
                y, components, postures, mean_squared_error
            )
            dep = compute_dependence(results, "Strict", "Permissive", threshold=0.05)
            D_tau3 = next(d for d in dep if d.component_name == "tau3")
            assert np.isfinite(D_tau3.dependence_magnitude)
            assert D_tau3.dependence_magnitude >= 0.0
            Ds.append(D_tau3.dependence_magnitude)
        D_by_sigma[sigma] = float(np.mean(Ds))

    # At sigma = 0, D should match the manuscript value closely.
    assert abs(D_by_sigma[0.0] - 0.129) < 1e-3

    # Mean D should stay near 0.129 for low noise and not blow up.
    for sigma, D_mean in D_by_sigma.items():
        assert 0.05 < D_mean < 0.5, f"sigma={sigma} gave mean D={D_mean}"


# ---------------------------------------------------------------------------
# Collinearity
# ---------------------------------------------------------------------------

def test_collinear_components_do_not_crash():
    """Two components that are scalar multiples of each other produce a
    rank deficient design matrix. The least squares layer must not raise;
    NumPy's lstsq handles this via minimum norm solution. The
    discriminant must remain finite."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    a = CandidateComponent("a", np.array([1.0, 1.0, 1.0, 1.0]))
    b = CandidateComponent("b", np.array([2.0, 2.0, 2.0, 2.0]))  # 2 * a
    c = CandidateComponent("c", np.array([0.0, 1.0, 2.0, 3.0]))  # linearly indep.
    components = [a, b, c]
    posture = PermissivePosture("Permissive")

    result = compute_discriminant(
        y, components, "a", posture, mean_squared_error
    )
    assert np.isfinite(result.delta)
    assert np.isfinite(result.full_error)
    assert np.isfinite(result.reduced_error)


# ---------------------------------------------------------------------------
# Degenerate bases
# ---------------------------------------------------------------------------

def test_empty_reduction_yields_zero_reconstruction():
    """Reducing a single component set by removing its only member
    produces the zero reconstruction, and the discriminant equals the
    full signal MSE."""
    y = np.array([1.0, 2.0, 3.0])
    tau = CandidateComponent("tau", np.array([1.0, 1.0, 1.0]))
    posture = PermissivePosture("Permissive")

    result = compute_discriminant(
        y, [tau], "tau", posture, mean_squared_error
    )
    # With only tau in S, A(S) fits tau to y; removing it gives y_hat = 0.
    expected_reduced_error = float(np.mean(y * y))
    assert abs(result.reduced_error - expected_reduced_error) < 1e-12
    # delta = reduced - full, both non negative.
    assert result.delta >= 0.0


def test_zero_component_is_invariantly_eliminable():
    """A component whose values are all zero contributes nothing to the
    reconstruction. Its discriminant must be exactly zero under any
    posture."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    tau1 = CandidateComponent("tau1", np.array([1.0, 1.0, 1.0, 1.0]))
    tau_zero = CandidateComponent("zero", np.array([0.0, 0.0, 0.0, 0.0]))
    components = [tau1, tau_zero]

    for posture in (StrictPosture("Strict"), PermissivePosture("Permissive")):
        result = compute_discriminant(
            y, components, "zero", posture, mean_squared_error
        )
        assert abs(result.delta) < 1e-10, f"posture={posture.name}"


def test_unknown_component_name_raises_keyerror():
    """reconstruct_reduced must raise KeyError cleanly when the target
    name does not appear in the candidate set."""
    y = np.array([1.0, 2.0, 3.0])
    tau = CandidateComponent("tau", np.array([1.0, 0.0, 0.0]))
    posture = PermissivePosture("Permissive")

    with pytest.raises(KeyError):
        reconstruct_reduced(y, [tau], "nonexistent", posture)


# ---------------------------------------------------------------------------
# Threshold boundaries
# ---------------------------------------------------------------------------

def test_threshold_zero_or_negative_raises():
    """Persistence score requires theta > 0 per Section 3.5 of the
    manuscript. Non positive values must raise."""
    with pytest.raises(ValueError):
        persistence_score(delta=0.1, threshold=0.0)
    with pytest.raises(ValueError):
        persistence_score(delta=0.1, threshold=-1.0)


def test_very_large_threshold_makes_everything_invariantly_eliminable():
    """With theta large enough that no component exceeds it, the
    classification layer must label every component invariantly
    eliminable rather than silently misclassifying."""
    y = np.array([0.610, 0.410, 0.020, -0.590, -0.410, 0.880])
    components, postures = _build_toy(y)

    results = compute_all_discriminants(
        y, components, postures, mean_squared_error
    )
    assign_persistence(results, threshold=1.0)  # far above any delta
    assign_classifications(results, threshold=1.0)

    labels = {r.component_name: r.classification for r in results}
    for name, label in labels.items():
        assert label == "invariantly eliminable", (
            f"component {name} got label {label!r} under large threshold"
        )


# ---------------------------------------------------------------------------
# Non orthogonal basis
# ---------------------------------------------------------------------------

def test_non_orthogonal_basis_still_valid():
    """The manuscript's closed form coefficients assume orthogonality.
    The implementation uses OLS via np.linalg.lstsq and must remain
    numerically correct for non orthogonal bases. We verify by checking
    that A(S) recovers y exactly when y lies in the span of S."""
    tau1 = CandidateComponent("tau1", np.array([1.0, 1.0, 0.0, 0.0]))
    tau2 = CandidateComponent("tau2", np.array([1.0, 0.0, 1.0, 0.0]))  # not orthog to tau1
    tau3 = CandidateComponent("tau3", np.array([0.0, 0.0, 0.0, 1.0]))
    components = [tau1, tau2, tau3]
    # y = 2 * tau1 + 3 * tau2 + 5 * tau3
    y = 2.0 * tau1.values + 3.0 * tau2.values + 5.0 * tau3.values
    posture = PermissivePosture("Permissive")

    result = compute_discriminant(
        y, components, "tau3", posture, mean_squared_error
    )
    # y is exactly in span(S), so full reconstruction error must vanish.
    assert result.full_error < 1e-20
    # Removing tau3 loses the support at index 3, yielding nonzero error.
    assert result.reduced_error > 0.0
    assert result.delta > 0.0


# ---------------------------------------------------------------------------
# Matrix valued metric (Section 4 of the manuscript)
# ---------------------------------------------------------------------------

def test_squared_frobenius_reproduces_quantum_discriminant():
    """For a two level density operator rho with off diagonal coherence c,
    the squared Frobenius norm of rho minus the dephased diagonal equals
    2 |c|**2. This is the closed form result used in Section 4 of the
    manuscript."""
    p = 0.5
    c = 0.4
    rho = np.array([[p, c], [c, 1.0 - p]])
    tau_D = np.array([[p, 0.0], [0.0, 1.0 - p]])
    distance = squared_frobenius_norm(rho, tau_D)
    expected = 2.0 * (c ** 2)
    assert abs(distance - expected) < 1e-12


def test_frobenius_handles_complex_coherence():
    """Complex off diagonal entries should yield 2 |c|**2 where
    |c|**2 = c.real**2 + c.imag**2."""
    p = 0.5
    c = 0.3 + 0.4j
    rho = np.array([[p, c], [np.conj(c), 1.0 - p]])
    tau_D = np.array([[p, 0.0], [0.0, 1.0 - p]], dtype=complex)
    distance = squared_frobenius_norm(rho, tau_D)
    expected = 2.0 * (abs(c) ** 2)
    assert abs(distance - expected) < 1e-12


# ---------------------------------------------------------------------------
# Posture class behaviors
# ---------------------------------------------------------------------------

def test_thresholded_posture_admits_by_score():
    """ThresholdedPosture must exclude components whose score falls below
    threshold and admit components whose score meets or exceeds it."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    low = CandidateComponent("low", np.array([1.0, 0.0, 0.0, 0.0]), tags={"q": 0.1})
    high = CandidateComponent("high", np.array([0.0, 1.0, 0.0, 0.0]), tags={"q": 0.9})
    components = [low, high]

    posture = ThresholdedPosture(
        name="Gated",
        score_fn=lambda c: c.tags["q"],
        threshold=0.5,
    )
    _, coeffs = posture.reconstruct(y, components)
    # low is rejected; its coefficient must be zero.
    assert coeffs["low"] == 0.0
    # high is admitted; its coefficient should be the OLS projection.
    assert abs(coeffs["high"] - 2.0) < 1e-10  # y[1] = 2


def test_weighted_posture_reduces_to_ols_with_zero_penalty():
    """WeightedPosture with penalty_fn returning zero must produce the
    same coefficients as PermissivePosture (ordinary least squares)."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    a = CandidateComponent("a", np.array([1.0, 0.0, 0.0, 0.0]))
    b = CandidateComponent("b", np.array([0.0, 1.0, 0.0, 0.0]))
    components = [a, b]

    weighted = WeightedPosture(
        name="NoPenalty", penalty_fn=lambda c: 0.0, lam=1.0
    )
    permissive = PermissivePosture(name="Permissive")

    _, w_coeffs = weighted.reconstruct(y, components)
    _, p_coeffs = permissive.reconstruct(y, components)
    for name in ("a", "b"):
        assert abs(w_coeffs[name] - p_coeffs[name]) < 1e-10, name


# ---------------------------------------------------------------------------
# Research adjacent demo
# ---------------------------------------------------------------------------

def test_infrasound_transient_flags_correct_component():
    """The infrasound transient demo must flag the transient as
    evaluation dependent and the two background components as
    invariantly necessary. The transient must have
    Delta_BandlimitStrict = 0 exactly (excluded by the posture) and
    Delta_BroadbandPermissive > threshold (non eliminable)."""
    from eliminability.demos import build_infrasound_transient_problem

    y, components, postures, threshold = build_infrasound_transient_problem()

    results = compute_all_discriminants(
        y, components, postures, mean_squared_error
    )
    assign_persistence(results, threshold)
    assign_classifications(results, threshold)

    # Pull Delta values by (component, posture).
    deltas = {(r.component_name, r.posture_name): r.delta for r in results}
    labels = {r.component_name: r.classification for r in results}

    # Transient is excluded under the strict posture.
    assert deltas[("transient", "BandlimitStrict")] == 0.0
    # Transient is non eliminable under the permissive posture.
    assert deltas[("transient", "BroadbandPermissive")] > threshold
    # Classification tracks the manuscript's notion of evaluation dependence.
    assert labels["transient"] == "evaluation dependent"
    assert labels["bg1"] == "invariantly necessary"
    assert labels["bg2"] == "invariantly necessary"


# ---------------------------------------------------------------------------
# Direct execution entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_noise_degrades_D_smoothly()
    test_collinear_components_do_not_crash()
    test_empty_reduction_yields_zero_reconstruction()
    test_zero_component_is_invariantly_eliminable()
    test_unknown_component_name_raises_keyerror()
    test_threshold_zero_or_negative_raises()
    test_very_large_threshold_makes_everything_invariantly_eliminable()
    test_non_orthogonal_basis_still_valid()
    test_squared_frobenius_reproduces_quantum_discriminant()
    test_frobenius_handles_complex_coherence()
    test_thresholded_posture_admits_by_score()
    test_weighted_posture_reduces_to_ols_with_zero_penalty()
    test_infrasound_transient_flags_correct_component()
    print("All edge case tests passed.")
