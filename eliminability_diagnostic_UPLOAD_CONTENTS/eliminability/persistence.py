"""
Persistence scoring and classification.

Implements the persistence score

    P_F(tau) = Delta_F(tau) / theta

and a classification procedure that labels each component based on the
pattern of its discriminants across postures.

Classification labels (imported from data_models):

    invariantly eliminable     : eliminable under every posture
    invariantly necessary      : non eliminable under every posture
    evaluation dependent       : eliminable under at least one posture,
                                 non eliminable under at least one other
    beneficially removable     : at least one posture yields Delta < 0
    partially evaluation dependent :
                                 not invariant, but not a clean crossing
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from .data_models import (
    DiscriminantResult,
    LABEL_BENEFICIALLY_REMOVABLE,
    LABEL_EVALUATION_DEPENDENT,
    LABEL_INVARIANTLY_ELIMINABLE,
    LABEL_INVARIANTLY_NECESSARY,
    LABEL_MIXED,
)


def persistence_score(delta: float, threshold: float) -> float:
    """
    Compute P_F(tau) = Delta_F(tau) / theta.

    Raises
    ------
    ValueError
        If threshold is not strictly positive.
    """
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    return float(delta) / float(threshold)


def assign_persistence(
    results: Sequence[DiscriminantResult], threshold: float
) -> None:
    """
    Populate the persistence_score field on each DiscriminantResult in
    place.
    """
    for r in results:
        r.persistence_score = persistence_score(r.delta, threshold)


def classify_component(
    deltas_by_posture: Dict[str, float], threshold: float
) -> str:
    """
    Assign a single classification label summarizing behavior across
    all postures for one component.

    A component is classified relative to its persistence score
    P_F(tau) = Delta_F(tau) / theta. Per Section 3.5 of the manuscript,
    a component with P_F > 1 is non eliminable under F and a component
    with P_F <= 1 is eliminable under F in the sense that its removal
    costs no more than the tolerated amount.

    Precedence:

        1. Any delta strictly below zero yields 'beneficially removable'.
        2. All deltas above threshold yields 'invariantly necessary'.
        3. All deltas at or below threshold yields 'invariantly eliminable'.
        4. At least one delta above threshold and at least one at or
           below threshold yields 'evaluation dependent'.

    Case 4 is the defining condition: the postures disagree about
    whether tau crosses the persistence threshold. When every posture
    agrees that tau is below threshold, it is invariantly eliminable
    under the tolerance, even if the magnitudes differ.

    Parameters
    ----------
    deltas_by_posture : dict
        Mapping from posture name to Delta_F(tau).
    threshold : float
        Tolerance theta.
    """
    deltas = list(deltas_by_posture.values())
    if not deltas:
        raise ValueError("No discriminants supplied for classification")

    if any(d < 0.0 for d in deltas):
        return LABEL_BENEFICIALLY_REMOVABLE

    all_necessary = all(d > threshold for d in deltas)
    all_eliminable = all(d <= threshold for d in deltas)
    any_necessary = any(d > threshold for d in deltas)
    any_eliminable = any(d <= threshold for d in deltas)

    if all_necessary:
        return LABEL_INVARIANTLY_NECESSARY
    if all_eliminable:
        return LABEL_INVARIANTLY_ELIMINABLE
    if any_necessary and any_eliminable:
        return LABEL_EVALUATION_DEPENDENT
    return LABEL_MIXED


def assign_classifications(
    results: Sequence[DiscriminantResult], threshold: float
) -> None:
    """
    Populate the classification field on each DiscriminantResult in
    place. All rows for a given component receive the same label
    because the label summarizes behavior across postures.
    """
    by_component: Dict[str, Dict[str, float]] = {}
    for r in results:
        by_component.setdefault(r.component_name, {})[r.posture_name] = r.delta

    labels = {
        name: classify_component(dmap, threshold)
        for name, dmap in by_component.items()
    }

    for r in results:
        r.classification = labels[r.component_name]
