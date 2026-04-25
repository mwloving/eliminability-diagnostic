"""
Eliminability discriminants and cross posture dependence.

Implements the central quantities of the formalism:

    Delta_F(tau) = R(y, A_F(S without tau)) minus R(y, A_F(S))

    D_{F,G}(tau) = | Delta_F(tau) minus Delta_G(tau) |

Per Section 3 of the manuscript, large positive Delta_F(tau) means
tau is non eliminable under F; near zero Delta_F(tau) means tau is
eliminable; negative Delta_F(tau) means removing tau improves the
reconstruction under F.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .data_models import (
    CandidateComponent,
    DiscriminantResult,
    EvaluationDependenceResult,
)
from .metrics import ReconstructionFunctional
from .postures import EvaluationPosture
from .reconstruction import reconstruct_full, reconstruct_reduced


def compute_discriminant(
    y: np.ndarray,
    components: Sequence[CandidateComponent],
    target_name: str,
    posture: EvaluationPosture,
    functional: ReconstructionFunctional,
) -> DiscriminantResult:
    """
    Compute Delta_F(tau) for a single component and posture.

    Parameters
    ----------
    y : np.ndarray
        Target description.
    components : sequence of CandidateComponent
        Fixed candidate set S.
    target_name : str
        Name of the component tau whose discriminant is computed.
    posture : EvaluationPosture
        Evaluation structure F.
    functional : callable
        Reconstruction functional R.

    Returns
    -------
    DiscriminantResult
    """
    y_hat_full, _ = reconstruct_full(y, components, posture)
    full_error = functional(y, y_hat_full)

    y_hat_reduced, _ = reconstruct_reduced(y, components, target_name, posture)
    reduced_error = functional(y, y_hat_reduced)

    delta = reduced_error - full_error
    return DiscriminantResult(
        component_name=target_name,
        posture_name=posture.name,
        full_error=float(full_error),
        reduced_error=float(reduced_error),
        delta=float(delta),
    )


def compute_all_discriminants(
    y: np.ndarray,
    components: Sequence[CandidateComponent],
    postures: Sequence[EvaluationPosture],
    functional: ReconstructionFunctional,
) -> List[DiscriminantResult]:
    """
    Compute Delta_F(tau) for every component and every posture.

    Returns
    -------
    list of DiscriminantResult
        Flat list, one entry per (component, posture) pair, in the order
        (outer loop over components, inner loop over postures).
    """
    results: List[DiscriminantResult] = []
    for comp in components:
        for posture in postures:
            results.append(
                compute_discriminant(y, components, comp.name, posture, functional)
            )
    return results


def compute_dependence(
    results: Sequence[DiscriminantResult],
    posture_a: str,
    posture_b: str,
    threshold: float,
) -> List[EvaluationDependenceResult]:
    """
    Compute D_{F,G}(tau) for every component, for a fixed pair of
    postures.

    Parameters
    ----------
    results : sequence of DiscriminantResult
        Output of compute_all_discriminants. Must include entries for
        both posture_a and posture_b.
    posture_a : str
        Name of posture F.
    posture_b : str
        Name of posture G.
    threshold : float
        Tolerance threshold theta. Used to detect whether the two
        postures place the component on opposite sides of the unit
        persistence threshold.

    Returns
    -------
    list of EvaluationDependenceResult
        One entry per component.
    """
    by_component: Dict[str, Dict[str, DiscriminantResult]] = {}
    for r in results:
        by_component.setdefault(r.component_name, {})[r.posture_name] = r

    out: List[EvaluationDependenceResult] = []
    for comp_name, posture_map in by_component.items():
        if posture_a not in posture_map or posture_b not in posture_map:
            continue
        delta_a = posture_map[posture_a].delta
        delta_b = posture_map[posture_b].delta
        magnitude = abs(delta_a - delta_b)
        cross = _crosses_threshold(delta_a, delta_b, threshold)
        out.append(
            EvaluationDependenceResult(
                component_name=comp_name,
                posture_a=posture_a,
                posture_b=posture_b,
                delta_a=float(delta_a),
                delta_b=float(delta_b),
                dependence_magnitude=float(magnitude),
                cross_threshold=bool(cross),
            )
        )
    return out


def _crosses_threshold(delta_a: float, delta_b: float, threshold: float) -> bool:
    """
    True when the two postures place tau on opposite sides of the unit
    persistence threshold.

    Per Section 3.5 of the manuscript: a component is non eliminable
    under posture F when P_F(tau) > 1, equivalently Delta_F(tau) > theta.
    The sharpest form of evaluation dependence arises when one posture
    yields P > 1 and the other does not. This function returns True in
    exactly that case.
    """
    a_above = delta_a > threshold
    b_above = delta_b > threshold
    return a_above != b_above
