"""
Command line entry point for the Eliminability Diagnostic.

Usage:

    python main.py --demo toy
    python main.py --demo smoothness
    python main.py --signal path/to/signal.csv \
                   --components path/to/components.csv \
                   --postures strict permissive \
                   --threshold 0.05 \
                   --outdir results

CSV format:

    signal.csv      : two columns, 't' and 'y', with one row per time index.

    components.csv  : one row per component. Columns are:
                        'name'                    component identifier
                        'violates_constraint'     0 or 1 (1 marks the
                                                  component as violating
                                                  the strict posture's
                                                  constraint)
                        'v1','v2', ..., 'vn'      component values at
                                                  each time index, in
                                                  the same order as the
                                                  signal.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eliminability import (
    CandidateComponent,
    EvaluationPosture,
    PermissivePosture,
    StrictPosture,
    assign_classifications,
    assign_persistence,
    build_results_table,
    compute_all_discriminants,
    compute_dependence,
    generate_summary,
    mean_squared_error,
    plot_dependence_bars,
    plot_discriminant_bars,
    plot_discriminant_heatmap,
    plot_reconstruction_comparison,
)


def _load_signal(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    if "y" not in df.columns:
        raise ValueError(
            f"{path}: signal CSV must contain a 'y' column; got {list(df.columns)}"
        )
    if "t" in df.columns:
        df = df.sort_values("t")
    return df["y"].to_numpy(dtype=float)


def _load_components(path: str, n: int) -> List[CandidateComponent]:
    df = pd.read_csv(path)
    if "name" not in df.columns:
        raise ValueError(f"{path}: components CSV must contain a 'name' column")
    value_cols = [
        c for c in df.columns if c.startswith("v") and c[1:].isdigit()
    ]
    if len(value_cols) != n:
        raise ValueError(
            f"{path}: expected {n} value columns (v1..v{n}), got {len(value_cols)}"
        )
    value_cols_sorted = sorted(value_cols, key=lambda c: int(c[1:]))
    components: List[CandidateComponent] = []
    for _, row in df.iterrows():
        values = np.array([float(row[c]) for c in value_cols_sorted])
        violates = bool(int(row.get("violates_constraint", 0)))
        components.append(
            CandidateComponent(
                name=str(row["name"]),
                values=values,
                tags={"violates_constraint": violates},
            )
        )
    return components


def _build_postures(names: Sequence[str]) -> List[EvaluationPosture]:
    postures: List[EvaluationPosture] = []
    for name in names:
        lname = name.lower()
        if lname in ("strict", "s"):
            postures.append(StrictPosture(name="Strict"))
        elif lname in ("permissive", "p"):
            postures.append(PermissivePosture(name="Permissive"))
        else:
            raise ValueError(
                f"Unknown posture name {name!r}. "
                "Supported: 'strict', 'permissive'. "
                "Custom postures can be constructed programmatically."
            )
    return postures


def _run_custom(
    signal_path: str,
    components_path: str,
    posture_names: Sequence[str],
    threshold: float,
    outdir: str,
) -> None:
    y = _load_signal(signal_path)
    components = _load_components(components_path, n=len(y))
    postures = _build_postures(posture_names)
    if len(postures) < 2:
        raise ValueError("At least two postures are required for cross comparison.")

    discriminants = compute_all_discriminants(
        y=y,
        components=components,
        postures=postures,
        functional=mean_squared_error,
    )
    assign_persistence(discriminants, threshold)
    assign_classifications(discriminants, threshold)

    dependence = compute_dependence(
        discriminants,
        posture_a=postures[0].name,
        posture_b=postures[1].name,
        threshold=threshold,
    )

    table = build_results_table(discriminants, dependence)
    summary = generate_summary(discriminants, dependence, threshold)

    os.makedirs(outdir, exist_ok=True)
    csv_out = os.path.join(outdir, "results.csv")
    table.to_csv(csv_out, index=False)

    print("=" * 72)
    print("Eliminability Diagnostic results")
    print("=" * 72)
    print(f"signal: {signal_path}")
    print(f"components: {components_path}")
    print(f"postures: {[p.name for p in postures]}")
    print(f"threshold theta = {threshold}")
    print()
    print(table.to_string(index=False))
    print()
    print(summary)
    print()
    print(f"Results written to {csv_out}")

    _write_plots(y, components, postures, discriminants, dependence, threshold, outdir)


def _write_plots(
    y,
    components,
    postures,
    discriminants,
    dependence,
    threshold,
    outdir,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_discriminant_bars(discriminants, ax=ax, threshold=threshold)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "discriminant_bars.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_discriminant_heatmap(discriminants, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "discriminant_heatmap.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_dependence_bars(dependence, ax=ax, threshold=threshold)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "dependence_bars.png"), dpi=150)
    plt.close(fig)

    # Reconstruction comparison removing the component with the largest
    # cross posture dependence, which is the natural candidate to
    # visualize.
    if dependence:
        worst = max(dependence, key=lambda d: d.dependence_magnitude)
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_reconstruction_comparison(
            y=y,
            components=components,
            postures=postures,
            removed_component=worst.component_name,
            ax=ax,
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(outdir, "reconstruction_comparison.png"), dpi=150
        )
        plt.close(fig)

    print(f"Plots written to {outdir}")


def _run_demo(name: str, outdir: str) -> None:
    from .demos import (
        build_infrasound_transient_problem,
        build_smoothness_problem,
        build_toy_problem,
    )

    if name == "toy":
        y, components, postures, threshold = build_toy_problem()
    elif name == "infrasound":
        y, components, postures, threshold = build_infrasound_transient_problem()
    elif name == "smoothness":
        y, components, postures, threshold = build_smoothness_problem()
    else:
        raise ValueError(
            f"Unknown demo {name!r}. Supported: 'toy', 'infrasound', 'smoothness'."
        )

    discriminants = compute_all_discriminants(
        y=y,
        components=components,
        postures=postures,
        functional=mean_squared_error,
    )
    assign_persistence(discriminants, threshold)
    assign_classifications(discriminants, threshold)

    dependence = compute_dependence(
        discriminants,
        posture_a=postures[0].name,
        posture_b=postures[1].name,
        threshold=threshold,
    )

    table = build_results_table(discriminants, dependence)
    summary = generate_summary(discriminants, dependence, threshold)

    os.makedirs(outdir, exist_ok=True)
    table.to_csv(os.path.join(outdir, "results.csv"), index=False)

    print("=" * 72)
    print(f"Demo: {name}")
    print("=" * 72)
    print(table.to_string(index=False))
    print()
    print(summary)

    _write_plots(y, components, postures, discriminants, dependence, threshold, outdir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eliminability-diagnostic",
        description=(
            "Detect evaluation dependent eliminability of candidate "
            "components under fixed dynamics and fixed input."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--demo",
        choices=["toy", "infrasound", "smoothness"],
        help="Run a bundled demo instead of supplying CSV files.",
    )
    group.add_argument(
        "--signal",
        help="Path to observed signal CSV (columns 't', 'y').",
    )
    parser.add_argument(
        "--components",
        help=(
            "Path to candidate components CSV. Required when --signal is "
            "used. See main.py docstring for column format."
        ),
    )
    parser.add_argument(
        "--postures",
        nargs="+",
        default=["strict", "permissive"],
        help="Posture names to apply. Supported: strict, permissive.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Tolerance threshold theta (default 0.05).",
    )
    parser.add_argument(
        "--outdir",
        default="results",
        help="Directory to write CSV and PNG outputs (default 'results').",
    )
    args = parser.parse_args(argv)

    if args.demo is not None:
        _run_demo(args.demo, outdir=args.outdir)
        return 0

    if args.signal is not None:
        if args.components is None:
            parser.error("--components is required when --signal is used")
        _run_custom(
            signal_path=args.signal,
            components_path=args.components,
            posture_names=args.postures,
            threshold=args.threshold,
            outdir=args.outdir,
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
