"""
Bundled demo problems.

Two primary demos are exposed:

    build_toy_problem
        Reproduces Section 5 of the accompanying manuscript. Six point
        orthogonal basis, one constraint violating spike. Pedagogical.

    build_infrasound_transient_problem
        Synthetic single channel infrasound or seismic window with a
        transient event. Strict posture is a bandlimited admissibility
        rule (only slowly varying candidate waveforms are admitted);
        permissive posture is broadband OLS. The transient appears as
        an evaluation dependent component. This is the research
        adjacent demo: same formalism as the toy, but the candidate
        set and signal geometry match an attribution diagnostic a
        signal analyst would actually encounter.

A third builder, build_smoothness_problem, is retained for backward
compatibility with earlier example scripts.

The examples/ directory contains runnable wrappers that call these
builders and add verbose printing. Those wrappers remain the
recommended reading order for new users.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .data_models import CandidateComponent
from .postures import EvaluationPosture, PermissivePosture, StrictPosture, WeightedPosture


def build_toy_problem() -> Tuple[
    np.ndarray, List[CandidateComponent], List[EvaluationPosture], float
]:
    """
    Toy signal problem from Section 5 of the manuscript.

    Returns
    -------
    (y, components, postures, threshold)
        y : np.ndarray of shape (6,)
        components : list of three CandidateComponent objects, the third
            of which is tagged as violating the strict admissibility rule.
        postures : [StrictPosture, PermissivePosture]
        threshold : 0.05
    """
    y = np.array([0.610, 0.410, 0.020, -0.590, -0.410, 0.880])

    tau1 = CandidateComponent(
        name="tau1",
        values=np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
        tags={"violates_constraint": False},
    )
    tau2 = CandidateComponent(
        name="tau2",
        values=np.array([0.0, 1.0, 0.0, 0.0, -1.0, 0.0]),
        tags={"violates_constraint": False},
    )
    tau3 = CandidateComponent(
        name="tau3",
        values=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        tags={"violates_constraint": True},
    )

    components = [tau1, tau2, tau3]
    postures = [
        StrictPosture(
            name="Strict",
            description=(
                "Enforces the smoothness constraint; constraint violating "
                "components are excluded from the reconciliation."
            ),
        ),
        PermissivePosture(
            name="Permissive",
            description="No admissibility constraint; ordinary least squares over S.",
        ),
    ]
    threshold = 0.05
    return y, components, postures, threshold


def build_infrasound_transient_problem(
    n_samples: int = 128,
    sample_rate_hz: float = 20.0,
    transient_center_s: float = 4.5,
    transient_width_s: float = 0.25,
    transient_amplitude: float = 0.7,
    noise_sigma: float = 0.02,
    seed: int = 7,
) -> Tuple[np.ndarray, List[CandidateComponent], List[EvaluationPosture], float]:
    """
    Synthetic single channel infrasound or seismic window with a transient.

    The generated signal mixes two slowly varying (bandlimited) waveforms
    representing the local acoustic background with a narrow Gaussian
    transient representing a short duration event such as an impulsive
    arrival. A small Gaussian noise floor is added.

    Two postures are compared:

        BandlimitStrict
            Admits only the slowly varying background components. The
            transient is tagged as violating the bandlimit admissibility
            rule and is excluded from the reconciliation. This
            corresponds to an analyst who treats localized energy that
            fails the bandlimit specification as uninformative.

        BroadbandPermissive
            Imposes no bandlimit. Fits all components, including the
            transient, by ordinary least squares. Corresponds to an
            analyst whose evaluator retains localized features for
            attribution purposes.

    The transient is evaluation dependent: its eliminability status
    flips between the two postures under the same input and the same
    candidate set. This is the infrasound and seismic analogue of the
    classical Section 5 result, and it is the setting where the
    discriminant is most directly useful as a diagnostic instrument.

    Parameters
    ----------
    n_samples : int
        Number of discrete time samples in the window (default 128).
    sample_rate_hz : float
        Sampling rate in Hz (default 20.0, giving a 6.4 s window).
    transient_center_s : float
        Center time of the Gaussian transient in seconds.
    transient_width_s : float
        Standard deviation of the Gaussian transient in seconds.
    transient_amplitude : float
        Peak amplitude of the transient.
    noise_sigma : float
        Standard deviation of the additive Gaussian noise floor.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    (y, components, postures, threshold)
        y : np.ndarray of shape (n_samples,)
            Synthetic time series in arbitrary pressure units.
        components : list of CandidateComponent
            Two background components and one transient component.
            The transient carries tags {'violates_constraint': True,
            'kind': 'transient'}.
        postures : [BandlimitStrict, BroadbandPermissive]
        threshold : float
            Tolerance theta, scaled to the expected discriminant range
            for this problem size.
    """
    rng = np.random.default_rng(seed=seed)
    dt = 1.0 / sample_rate_hz
    t = np.arange(n_samples) * dt  # seconds

    # Two slowly varying (low frequency) background components.
    # Frequencies chosen well below Nyquist to count as bandlimited.
    bg1 = np.cos(2.0 * np.pi * 0.30 * t)   # 0.30 Hz background
    bg2 = np.sin(2.0 * np.pi * 0.55 * t)   # 0.55 Hz background

    # Narrow Gaussian transient centered in the window.
    transient = np.exp(-0.5 * ((t - transient_center_s) / transient_width_s) ** 2)

    # Observed signal with additive noise.
    y = (
        0.6 * bg1
        + 0.4 * bg2
        + transient_amplitude * transient
        + rng.normal(0.0, noise_sigma, size=n_samples)
    )

    tau_bg1 = CandidateComponent(
        name="bg1",
        values=bg1,
        tags={"violates_constraint": False, "kind": "background"},
    )
    tau_bg2 = CandidateComponent(
        name="bg2",
        values=bg2,
        tags={"violates_constraint": False, "kind": "background"},
    )
    tau_transient = CandidateComponent(
        name="transient",
        values=transient,
        tags={"violates_constraint": True, "kind": "transient"},
    )
    components = [tau_bg1, tau_bg2, tau_transient]

    postures = [
        StrictPosture(
            name="BandlimitStrict",
            description=(
                "Bandlimit admissibility: only slowly varying background "
                "components are retained. Transients are excluded."
            ),
        ),
        PermissivePosture(
            name="BroadbandPermissive",
            description=(
                "No bandlimit constraint; ordinary least squares over all "
                "candidates including the transient."
            ),
        ),
    ]

    # Threshold chosen to be well below the expected transient
    # discriminant for this problem size. With the defaults above, the
    # transient contributes approximately 1e-2 in MSE terms; theta =
    # 1e-3 gives a clear separation between admissible background
    # components (Delta near zero) and the transient (Delta above theta).
    threshold = 1e-3
    return y, components, postures, threshold


def build_smoothness_problem(lam: float = 5.0, violation_penalty: float = 100.0):
    """
    Smoothness demo problem: soft penalty analogue of the toy demo.

    Retained for backward compatibility with existing example scripts.
    New users are directed to build_infrasound_transient_problem for a
    research adjacent demo.

    Parameters
    ----------
    lam : float
        Regularization strength for the weighted posture.
    violation_penalty : float
        Per component weight applied to tagged violating components.
        As violation_penalty grows, the weighted posture approaches
        the strict posture's hard exclusion behavior; as it falls to
        zero it reduces to the permissive posture.

    Returns
    -------
    (y, components, postures, threshold)
    """
    y = np.array([0.610, 0.410, 0.020, -0.590, -0.410, 0.880])

    tau1 = CandidateComponent(
        name="tau1",
        values=np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
        tags={"violates_constraint": False},
    )
    tau2 = CandidateComponent(
        name="tau2",
        values=np.array([0.0, 1.0, 0.0, 0.0, -1.0, 0.0]),
        tags={"violates_constraint": False},
    )
    tau3 = CandidateComponent(
        name="tau3",
        values=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        tags={"violates_constraint": True},
    )
    components = [tau1, tau2, tau3]

    def tag_penalty(comp):
        return violation_penalty if comp.tags.get("violates_constraint") else 0.0

    postures = [
        WeightedPosture(
            name="SoftPenalty",
            penalty_fn=tag_penalty,
            lam=lam,
            description=(
                "Soft quadratic penalty on constraint violating components. "
                f"Violation weight = {violation_penalty}, lambda = {lam}."
            ),
        ),
        PermissivePosture(
            name="Permissive",
            description="No admissibility constraint; ordinary least squares.",
        ),
    ]
    threshold = 0.05
    return y, components, postures, threshold
