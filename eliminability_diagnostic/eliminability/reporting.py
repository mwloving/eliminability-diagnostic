"""
Reporting utilities.

Two outputs:

    build_results_table : a pandas DataFrame combining discriminants
                          across postures and dependence magnitudes
                          across posture pairs.

    generate_summary    : short textual interpretation of the table.

The reporting layer is separated from the discriminant computation so
that downstream consumers can format results for the CLI, a notebook,
or a further analysis layer without reworking the numerical core.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .data_models import (
    DiscriminantResult,
    EvaluationDependenceResult,
    LABEL_BENEFICIALLY_REMOVABLE,
    LABEL_EVALUATION_DEPENDENT,
    LABEL_INVARIANTLY_ELIMINABLE,
    LABEL_INVARIANTLY_NECESSARY,
    LABEL_MIXED,
)


def build_results_table(
    discriminants: Sequence[DiscriminantResult],
    dependence: Sequence[EvaluationDependenceResult] = (),
) -> pd.DataFrame:
    """
    Assemble a wide format table with one row per component.

    Columns:
        component
        delta[posture_name]         for each posture present
        P[posture_name]             persistence score per posture
        D[posture_a, posture_b]     dependence magnitude for each
                                    comparison
        classification

    Parameters
    ----------
    discriminants : sequence of DiscriminantResult
    dependence : sequence of EvaluationDependenceResult, optional

    Returns
    -------
    pandas.DataFrame
    """
    rows: Dict[str, Dict[str, object]] = {}
    for r in discriminants:
        row = rows.setdefault(r.component_name, {"component": r.component_name})
        row[f"delta[{r.posture_name}]"] = r.delta
        if r.persistence_score is not None:
            row[f"P[{r.posture_name}]"] = r.persistence_score
        if r.classification is not None:
            row["classification"] = r.classification

    for d in dependence:
        row = rows.setdefault(d.component_name, {"component": d.component_name})
        key = f"D[{d.posture_a}, {d.posture_b}]"
        row[key] = d.dependence_magnitude
        row[f"cross_threshold[{d.posture_a}, {d.posture_b}]"] = d.cross_threshold

    ordered = list(rows.values())
    df = pd.DataFrame(ordered)

    preferred_order: List[str] = ["component"]
    delta_cols = sorted([c for c in df.columns if c.startswith("delta[")])
    p_cols = sorted([c for c in df.columns if c.startswith("P[")])
    d_cols = sorted([c for c in df.columns if c.startswith("D[")])
    cross_cols = sorted([c for c in df.columns if c.startswith("cross_threshold[")])
    preferred_order.extend(delta_cols)
    preferred_order.extend(p_cols)
    preferred_order.extend(d_cols)
    preferred_order.extend(cross_cols)
    if "classification" in df.columns:
        preferred_order.append("classification")
    remaining = [c for c in df.columns if c not in preferred_order]
    df = df[preferred_order + remaining]
    return df


def generate_summary(
    discriminants: Sequence[DiscriminantResult],
    dependence: Sequence[EvaluationDependenceResult],
    threshold: float,
) -> str:
    """
    Produce a short text summary interpreting the results.

    The summary lists components by classification and flags the
    sharpest cases of evaluation dependence, namely those in which the
    two postures place the component on opposite sides of the
    persistence threshold.
    """
    by_component: Dict[str, Dict[str, float]] = {}
    labels: Dict[str, str] = {}
    for r in discriminants:
        by_component.setdefault(r.component_name, {})[r.posture_name] = r.delta
        if r.classification is not None:
            labels[r.component_name] = r.classification

    buckets: Dict[str, List[str]] = {
        LABEL_INVARIANTLY_ELIMINABLE: [],
        LABEL_INVARIANTLY_NECESSARY: [],
        LABEL_EVALUATION_DEPENDENT: [],
        LABEL_BENEFICIALLY_REMOVABLE: [],
        LABEL_MIXED: [],
    }
    for comp, label in labels.items():
        buckets.setdefault(label, []).append(comp)

    lines: List[str] = []
    lines.append(f"Tolerance threshold theta = {threshold:.4g}")
    lines.append("")

    for label in (
        LABEL_INVARIANTLY_ELIMINABLE,
        LABEL_INVARIANTLY_NECESSARY,
        LABEL_EVALUATION_DEPENDENT,
        LABEL_BENEFICIALLY_REMOVABLE,
        LABEL_MIXED,
    ):
        comps = buckets.get(label, [])
        if not comps:
            continue
        lines.append(f"{label}: {', '.join(sorted(comps))}")

    if dependence:
        lines.append("")
        lines.append("Cross posture comparisons:")
        for d in dependence:
            if d.cross_threshold:
                side_a = _side_label(d.delta_a, threshold)
                side_b = _side_label(d.delta_b, threshold)
                lines.append(
                    f"  Component {d.component_name} is evaluation dependent: "
                    f"{side_a} under {d.posture_a}, "
                    f"{side_b} under {d.posture_b}. "
                    f"|Delta_{d.posture_a} minus Delta_{d.posture_b}| "
                    f"= {d.dependence_magnitude:.4g}"
                )
            else:
                lines.append(
                    f"  Component {d.component_name}: "
                    f"Delta_{d.posture_a} = {d.delta_a:.4g}, "
                    f"Delta_{d.posture_b} = {d.delta_b:.4g}, "
                    f"D = {d.dependence_magnitude:.4g}"
                )

    return "\n".join(lines)


def _side_label(delta: float, threshold: float) -> str:
    if delta > threshold:
        return "non eliminable"
    if delta <= 0.0:
        return "eliminable"
    return "marginal"
