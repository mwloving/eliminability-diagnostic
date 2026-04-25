"""
Data models for the eliminability diagnostic.

Defines the structural objects carried through the pipeline:
candidate components, discriminant results, and cross posture
dependence results. Kept deliberately thin so that downstream code
can serialize, tabulate, and plot without inspecting internal state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


# Classification labels used in reporting. Written as plain strings so
# that downstream code and serialization remain transparent.
LABEL_INVARIANTLY_ELIMINABLE = "invariantly eliminable"
LABEL_INVARIANTLY_NECESSARY = "invariantly necessary"
LABEL_EVALUATION_DEPENDENT = "evaluation dependent"
LABEL_BENEFICIALLY_REMOVABLE = "beneficially removable"
LABEL_MIXED = "partially evaluation dependent"


@dataclass
class CandidateComponent:
    """
    A single element tau of the candidate set S.

    In the classical vector setting, values is a one dimensional array
    indexed over the discrete grid. In quantum matrix settings (future
    expansion) values may be a two dimensional array representing a
    component of a density operator decomposition.

    Attributes
    ----------
    name : str
        Short identifier used in tables, plots, and reports.
    values : np.ndarray
        Numeric representation of the component. Shape must be
        consistent with the target description y.
    tags : dict, optional
        Free form metadata. Postures may consult tags to determine
        admissibility (for example, tags['violates_smoothness'] = True).
    """

    name: str
    values: np.ndarray
    tags: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values)


@dataclass
class DiscriminantResult:
    """
    Output of computing Delta_F(tau) for a single component under a
    single posture.

    Attributes
    ----------
    component_name : str
        Name of the candidate component tau.
    posture_name : str
        Name of the evaluation structure F.
    full_error : float
        R(y, A_F(S)).
    reduced_error : float
        R(y, A_F(S without tau)).
    delta : float
        Delta_F(tau) = reduced_error minus full_error.
    persistence_score : float, optional
        P_F(tau) = Delta_F(tau) / theta. Populated by persistence module.
    classification : str, optional
        Label assigned across postures after cross comparison.
    """

    component_name: str
    posture_name: str
    full_error: float
    reduced_error: float
    delta: float
    persistence_score: Optional[float] = None
    classification: Optional[str] = None


@dataclass
class EvaluationDependenceResult:
    """
    Output of comparing two postures on a single component.

    Attributes
    ----------
    component_name : str
        Name of the candidate component tau.
    posture_a : str
        Name of the first evaluation structure F.
    posture_b : str
        Name of the second evaluation structure G.
    delta_a : float
        Delta_F(tau).
    delta_b : float
        Delta_G(tau).
    dependence_magnitude : float
        D_{F,G}(tau) = | Delta_F(tau) minus Delta_G(tau) |.
    cross_threshold : bool
        True when the two postures place tau on opposite sides of the
        tolerance threshold theta (one eliminable, one non eliminable).
        This is the sharpest form of evaluation dependence.
    """

    component_name: str
    posture_a: str
    posture_b: str
    delta_a: float
    delta_b: float
    dependence_magnitude: float
    cross_threshold: bool
