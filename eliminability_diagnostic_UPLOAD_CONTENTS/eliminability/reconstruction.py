"""
Reconstruction orchestration.

Thin layer over the posture.reconstruct interface. Provides named
helpers for the full set reconstruction A_F(S) and the reduced set
reconstruction A_F(S without tau). The fixed candidate set assumption
is enforced here: a reduced reconstruction uses the same posture
applied to S minus one component, without reparameterizing the basis.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from .data_models import CandidateComponent
from .postures import EvaluationPosture


def reconstruct_full(
    y: np.ndarray,
    components: Sequence[CandidateComponent],
    posture: EvaluationPosture,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute the full reconstruction A_F(S) under the given posture.

    Parameters
    ----------
    y : np.ndarray
        Target description.
    components : sequence of CandidateComponent
        The fixed candidate set S.
    posture : EvaluationPosture
        Evaluation structure F.

    Returns
    -------
    (y_hat, coeffs) : tuple
        Reconstructed signal and a dict mapping component name to
        fitted coefficient. Components excluded by the posture appear
        with coefficient zero.
    """
    return posture.reconstruct(y, list(components))


def reconstruct_reduced(
    y: np.ndarray,
    components: Sequence[CandidateComponent],
    target_name: str,
    posture: EvaluationPosture,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute the reduced reconstruction A_F(S without tau) under the
    given posture.

    The component whose name matches target_name is removed from the
    candidate set and the remaining components are passed to the same
    posture's reconstruct method. No reparameterization of the basis
    occurs.

    Parameters
    ----------
    y : np.ndarray
        Target description.
    components : sequence of CandidateComponent
        The fixed candidate set S.
    target_name : str
        Name of the component to remove.
    posture : EvaluationPosture
        Evaluation structure F.

    Returns
    -------
    (y_hat, coeffs) : tuple
        Reconstructed signal and coefficient dict over the remaining
        components.

    Raises
    ------
    KeyError
        If target_name does not appear in components.
    """
    names = [c.name for c in components]
    if target_name not in names:
        raise KeyError(
            f"Component {target_name!r} not found in candidate set "
            f"with names {names}"
        )
    reduced = [c for c in components if c.name != target_name]
    return posture.reconstruct(y, reduced)
