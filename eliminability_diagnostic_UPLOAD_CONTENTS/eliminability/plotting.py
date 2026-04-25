"""
Plotting utilities.

Four standard visualizations:

    plot_discriminant_bars         : grouped bar chart of Delta_F(tau)
                                     across components and postures.
    plot_discriminant_heatmap      : rows = components, columns = postures,
                                     values = discriminants.
    plot_dependence_bars           : per component bar chart of
                                     D_{F,G}(tau).
    plot_reconstruction_comparison : overlay of observed signal, full
                                     reconstruction, and reduced
                                     reconstruction.

Kept intentionally minimal: no custom style sheets, no color
arguments, no seaborn dependency. Consumers may modify axes after the
call returns if styling is needed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .data_models import (
    CandidateComponent,
    DiscriminantResult,
    EvaluationDependenceResult,
)
from .postures import EvaluationPosture
from .reconstruction import reconstruct_full, reconstruct_reduced


def _organize_by_component(
    results: Sequence[DiscriminantResult],
) -> Tuple[List[str], List[str], Dict[Tuple[str, str], float]]:
    components = sorted({r.component_name for r in results})
    postures = sorted({r.posture_name for r in results})
    matrix: Dict[Tuple[str, str], float] = {
        (r.component_name, r.posture_name): r.delta for r in results
    }
    return components, postures, matrix


def plot_discriminant_bars(
    results: Sequence[DiscriminantResult],
    ax: Optional[plt.Axes] = None,
    threshold: Optional[float] = None,
) -> plt.Axes:
    """
    Grouped bar chart of Delta_F(tau) across components and postures.

    Parameters
    ----------
    results : sequence of DiscriminantResult
    ax : matplotlib.axes.Axes, optional
        If provided, draw into this axes; otherwise create a new figure.
    threshold : float, optional
        If provided, draw a horizontal reference line at theta.

    Returns
    -------
    matplotlib.axes.Axes
    """
    components, postures, matrix = _organize_by_component(results)
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(components)), 4))

    n_comp = len(components)
    n_post = len(postures)
    group_width = 0.8
    bar_width = group_width / max(n_post, 1)
    x = np.arange(n_comp)

    for i, posture in enumerate(postures):
        heights = [matrix.get((c, posture), 0.0) for c in components]
        offset = (i - (n_post - 1) / 2.0) * bar_width
        ax.bar(x + offset, heights, width=bar_width, label=posture)

    if threshold is not None:
        ax.axhline(threshold, linestyle="--", linewidth=1, color="gray")
        ax.text(
            x[-1] + 0.5,
            threshold,
            rf"$\theta = {threshold:.3g}$",
            color="gray",
            va="bottom",
            ha="right",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=0)
    ax.set_xlabel("component")
    ax.set_ylabel(r"$\Delta_F(\tau)$")
    ax.set_title(r"Eliminability discriminant $\Delta_F(\tau)$ by posture")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(title="posture", frameon=False)
    return ax


def plot_discriminant_heatmap(
    results: Sequence[DiscriminantResult],
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Heatmap with components on rows, postures on columns, cells
    showing Delta_F(tau).
    """
    components, postures, matrix = _organize_by_component(results)
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(4, 1.0 * len(postures) + 2), max(3, 0.5 * len(components) + 2))
        )

    grid = np.array(
        [[matrix.get((c, p), np.nan) for p in postures] for c in components],
        dtype=float,
    )
    im = ax.imshow(grid, aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(postures)))
    ax.set_xticklabels(postures, rotation=30, ha="right")
    ax.set_yticks(range(len(components)))
    ax.set_yticklabels(components)
    ax.set_xlabel("posture")
    ax.set_ylabel("component")
    ax.set_title(r"$\Delta_F(\tau)$ by component and posture")

    for i in range(len(components)):
        for j in range(len(postures)):
            val = grid[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.3g}",
                    ha="center",
                    va="center",
                    color="white" if val < np.nanmax(grid) * 0.5 else "black",
                    fontsize=9,
                )

    plt.colorbar(im, ax=ax, label=r"$\Delta$")
    return ax


def plot_dependence_bars(
    dependence: Sequence[EvaluationDependenceResult],
    ax: Optional[plt.Axes] = None,
    threshold: Optional[float] = None,
) -> plt.Axes:
    """
    Per component bar chart of D_{F,G}(tau) for a single posture pair.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(dependence)), 4))

    names = [d.component_name for d in dependence]
    heights = [d.dependence_magnitude for d in dependence]
    colors = [
        "#d62728" if d.cross_threshold else "#1f77b4" for d in dependence
    ]
    ax.bar(names, heights, color=colors)

    if threshold is not None:
        ax.axhline(threshold, linestyle="--", linewidth=1, color="gray")
        ax.text(
            len(names) - 0.5,
            threshold,
            rf"$\theta = {threshold:.3g}$",
            color="gray",
            va="bottom",
            ha="right",
            fontsize=9,
        )

    pair_label = (
        rf"$F =$ {dependence[0].posture_a}, $G =$ {dependence[0].posture_b}"
        if dependence
        else ""
    )
    ax.set_xlabel("component")
    ax.set_ylabel(r"$D_{F,G}(\tau)$")
    ax.set_title(rf"Evaluation dependence magnitude ({pair_label})")
    ax.axhline(0, color="black", linewidth=0.5)
    return ax


def plot_reconstruction_comparison(
    y: np.ndarray,
    components: Sequence[CandidateComponent],
    postures: Sequence[EvaluationPosture],
    removed_component: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Overlay of observed signal and reconstructions under each posture.

    If removed_component is provided, the reduced reconstruction
    A_F(S without removed_component) is plotted alongside A_F(S) for
    each posture.

    Parameters
    ----------
    y : np.ndarray
        Observed signal.
    components : sequence of CandidateComponent
    postures : sequence of EvaluationPosture
    removed_component : str, optional
        Name of a component to remove when plotting the reduced
        reconstruction. If None, only full reconstructions are drawn.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    t = np.arange(len(y))
    ax.plot(t, y, "o-", color="black", label=r"observed $y$", linewidth=1.5)

    for posture in postures:
        y_hat_full, _ = reconstruct_full(y, components, posture)
        ax.plot(
            t,
            y_hat_full,
            "s--",
            label=rf"$A_{{{posture.name}}}(S)$",
            linewidth=1.2,
            markersize=5,
        )
        if removed_component is not None:
            y_hat_red, _ = reconstruct_reduced(
                y, components, removed_component, posture
            )
            ax.plot(
                t,
                y_hat_red,
                "^:",
                label=f"{posture.name}: reduced without {removed_component}",
                linewidth=1.2,
                markersize=5,
            )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel("signal value")
    title = "Reconstruction comparison"
    if removed_component is not None:
        title += rf" (removed: ${removed_component}$)"
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best", frameon=False)
    ax.axhline(0, color="gray", linewidth=0.5)
    return ax
