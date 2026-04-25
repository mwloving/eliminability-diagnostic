"""
Evaluation postures F, G, ...

Each posture induces an evaluation map M_F : Y -> Z that governs the
reconciliation operator A_F. Postures differ by the admissibility rule,
the weighting rule, and the reconstruction mode. Four representative
types are supplied:

    StrictPosture       : hard exclusion of constraint violating components
    PermissivePosture   : unconstrained projection over all components
    ThresholdedPosture  : inclusion contingent on a per component score
    WeightedPosture     : soft penalty rather than hard exclusion

All posture objects expose the same reconstruct interface so that the
discriminant engine can treat them uniformly. The reconstruction is
performed with the fixed candidate set assumption: the basis is never
reparameterized, and reduced set reconstructions recompute only the
coefficients of the remaining components under the same rule.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data_models import CandidateComponent


class EvaluationPosture:
    """
    Base class for evaluation postures.

    A posture carries three pieces of information:

        is_admissible(component) -> bool
            Whether the component is retained by the reconciliation.
        reconstruct(y, components) -> (y_hat, coeffs)
            Produce a reconstruction of y from the given components
            under this posture.
        description
            Human readable summary for reports and plots.

    Subclasses implement the reconstruct method and, where relevant,
    override is_admissible. The base class default admits every
    component.
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description

    def is_admissible(self, component: CandidateComponent) -> bool:
        return True

    def reconstruct(
        self, y: np.ndarray, components: Sequence[CandidateComponent]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(name={self.name!r})"


def _least_squares_fit(
    y: np.ndarray, components: Sequence[CandidateComponent]
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Ordinary least squares fit of y against the listed components."""
    if len(components) == 0:
        return np.zeros_like(y, dtype=float), {}
    X = np.column_stack([c.values.astype(float) for c in components])
    coeffs, _residuals, _rank, _sv = np.linalg.lstsq(X, y.astype(float), rcond=None)
    y_hat = X @ coeffs
    coeff_map = {c.name: float(coeffs[i]) for i, c in enumerate(components)}
    return y_hat, coeff_map


class StrictPosture(EvaluationPosture):
    """
    Strict posture M_F.

    Enforces a hard admissibility rule. Components for which
    constraint_fn returns False are excluded from the reconciliation
    and assigned zero effective weight. Admissible components are
    fit by ordinary least squares.

    Parameters
    ----------
    name : str
        Posture identifier.
    constraint_fn : callable, optional
        Function mapping a CandidateComponent to a boolean. Defaults
        to consulting the component's tags for 'violates_constraint'.
    description : str, optional
        Human readable summary.
    """

    def __init__(
        self,
        name: str,
        constraint_fn: Optional[Callable[[CandidateComponent], bool]] = None,
        description: str = "",
    ) -> None:
        super().__init__(name=name, description=description)
        if constraint_fn is None:
            constraint_fn = default_tag_constraint
        self.constraint_fn = constraint_fn

    def is_admissible(self, component: CandidateComponent) -> bool:
        return bool(self.constraint_fn(component))

    def reconstruct(
        self, y: np.ndarray, components: Sequence[CandidateComponent]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        admissible = [c for c in components if self.is_admissible(c)]
        y_hat, coeffs = _least_squares_fit(y, admissible)
        # Report zero coefficients for excluded components so that
        # downstream consumers see a complete set keyed by component name.
        for c in components:
            coeffs.setdefault(c.name, 0.0)
        return y_hat, coeffs


class PermissivePosture(EvaluationPosture):
    """
    Permissive posture M_G.

    Imposes no admissibility constraint. All components are retained
    and fit by ordinary least squares.
    """

    def __init__(self, name: str, description: str = "") -> None:
        super().__init__(name=name, description=description)

    def reconstruct(
        self, y: np.ndarray, components: Sequence[CandidateComponent]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        return _least_squares_fit(y, components)


class ThresholdedPosture(EvaluationPosture):
    """
    Thresholded posture.

    Admits a component only if a user supplied score_fn returns a value
    at or above threshold. Admitted components are fit by ordinary
    least squares; rejected components are assigned zero weight.

    Parameters
    ----------
    name : str
    score_fn : callable
        Maps a CandidateComponent to a float score.
    threshold : float
        Minimum score required for admission.
    description : str, optional
    """

    def __init__(
        self,
        name: str,
        score_fn: Callable[[CandidateComponent], float],
        threshold: float,
        description: str = "",
    ) -> None:
        super().__init__(name=name, description=description)
        self.score_fn = score_fn
        self.threshold = float(threshold)

    def is_admissible(self, component: CandidateComponent) -> bool:
        return float(self.score_fn(component)) >= self.threshold

    def reconstruct(
        self, y: np.ndarray, components: Sequence[CandidateComponent]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        admissible = [c for c in components if self.is_admissible(c)]
        y_hat, coeffs = _least_squares_fit(y, admissible)
        for c in components:
            coeffs.setdefault(c.name, 0.0)
        return y_hat, coeffs


class WeightedPosture(EvaluationPosture):
    """
    Weighted posture.

    Replaces hard exclusion with a soft quadratic penalty. The
    reconciliation solves

        argmin over beta of   || X beta minus y ||**2
                            + lambda sum_i w_i beta_i**2

    where w_i = penalty_fn(component_i) is the per component penalty.
    Setting penalty_fn to the zero function reduces this to ordinary
    least squares. Setting penalty_fn arbitrarily large on a component
    drives its coefficient to zero, approaching hard exclusion.

    Parameters
    ----------
    name : str
    penalty_fn : callable
        Maps a CandidateComponent to a non negative penalty weight.
    lam : float, optional
        Global regularization strength lambda. Default 1.0.
    description : str, optional
    """

    def __init__(
        self,
        name: str,
        penalty_fn: Callable[[CandidateComponent], float],
        lam: float = 1.0,
        description: str = "",
    ) -> None:
        super().__init__(name=name, description=description)
        self.penalty_fn = penalty_fn
        self.lam = float(lam)

    def reconstruct(
        self, y: np.ndarray, components: Sequence[CandidateComponent]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if len(components) == 0:
            return np.zeros_like(y, dtype=float), {}
        X = np.column_stack([c.values.astype(float) for c in components])
        y_f = y.astype(float)
        weights = np.array([float(self.penalty_fn(c)) for c in components])
        gram = X.T @ X + self.lam * np.diag(weights)
        rhs = X.T @ y_f
        coeffs = np.linalg.solve(gram, rhs)
        y_hat = X @ coeffs
        coeff_map = {c.name: float(coeffs[i]) for i, c in enumerate(components)}
        return y_hat, coeff_map


def default_tag_constraint(component: CandidateComponent) -> bool:
    """
    Default admissibility check used by StrictPosture when no custom
    constraint function is supplied.

    A component is admissible if its tags dict does not mark it as
    violating. Specifically, it is rejected iff any of the following
    tags is truthy:

        violates_constraint
        violates_smoothness
        violating

    This provides a lightweight way to build demos in which the
    classification of components as admissible or violating is
    declared by the user rather than recomputed from the component
    geometry.
    """
    tags = component.tags
    for key in ("violates_constraint", "violates_smoothness", "violating"):
        if tags.get(key, False):
            return False
    return True


def smoothness_roughness(component: CandidateComponent) -> float:
    """
    Squared L2 norm of the discrete second difference of a one
    dimensional component.

        rho(tau) = sum_{t=2}^{n-1} ( tau(t+1) minus 2 tau(t) plus tau(t-1) )**2

    Used as a scoring or penalty function for smoothness aware postures.
    Larger values indicate rougher components. The discrete second
    difference is computed on interior indices only.
    """
    v = component.values.astype(float).ravel()
    if v.size < 3:
        return 0.0
    second_diff = v[2:] - 2.0 * v[1:-1] + v[:-2]
    return float(np.sum(second_diff * second_diff))
