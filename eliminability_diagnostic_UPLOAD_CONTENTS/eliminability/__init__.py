"""
Eliminability Diagnostic.

A diagnostic for evaluation dependent eliminability of candidate
components under fixed dynamics and fixed input. See Loving (2026),
"Evaluation Dependence in Quantum Measurement: A Minimal FRAME
Formalism and Eliminability Discriminant", for the formalism.

Public API:

    CandidateComponent, DiscriminantResult, EvaluationDependenceResult
    EvaluationPosture
    StrictPosture, PermissivePosture, ThresholdedPosture, WeightedPosture
    default_tag_constraint, smoothness_roughness
    mean_squared_error, squared_frobenius_norm
    reconstruct_full, reconstruct_reduced
    compute_discriminant, compute_all_discriminants, compute_dependence
    persistence_score, classify_component
    assign_persistence, assign_classifications
    build_results_table, generate_summary
    plot_discriminant_bars, plot_discriminant_heatmap,
    plot_dependence_bars, plot_reconstruction_comparison
"""

from .data_models import (
    CandidateComponent,
    DiscriminantResult,
    EvaluationDependenceResult,
    LABEL_BENEFICIALLY_REMOVABLE,
    LABEL_EVALUATION_DEPENDENT,
    LABEL_INVARIANTLY_ELIMINABLE,
    LABEL_INVARIANTLY_NECESSARY,
    LABEL_MIXED,
)
from .discriminants import (
    compute_all_discriminants,
    compute_dependence,
    compute_discriminant,
)
from .metrics import mean_squared_error, squared_frobenius_norm
from .persistence import (
    assign_classifications,
    assign_persistence,
    classify_component,
    persistence_score,
)
from .plotting import (
    plot_dependence_bars,
    plot_discriminant_bars,
    plot_discriminant_heatmap,
    plot_reconstruction_comparison,
)
from .postures import (
    EvaluationPosture,
    PermissivePosture,
    StrictPosture,
    ThresholdedPosture,
    WeightedPosture,
    default_tag_constraint,
    smoothness_roughness,
)
from .reconstruction import reconstruct_full, reconstruct_reduced
from .reporting import build_results_table, generate_summary


__all__ = [
    "CandidateComponent",
    "DiscriminantResult",
    "EvaluationDependenceResult",
    "EvaluationPosture",
    "PermissivePosture",
    "StrictPosture",
    "ThresholdedPosture",
    "WeightedPosture",
    "assign_classifications",
    "assign_persistence",
    "build_results_table",
    "classify_component",
    "compute_all_discriminants",
    "compute_dependence",
    "compute_discriminant",
    "default_tag_constraint",
    "generate_summary",
    "mean_squared_error",
    "persistence_score",
    "plot_dependence_bars",
    "plot_discriminant_bars",
    "plot_discriminant_heatmap",
    "plot_reconstruction_comparison",
    "reconstruct_full",
    "reconstruct_reduced",
    "smoothness_roughness",
    "squared_frobenius_norm",
    "LABEL_BENEFICIALLY_REMOVABLE",
    "LABEL_EVALUATION_DEPENDENT",
    "LABEL_INVARIANTLY_ELIMINABLE",
    "LABEL_INVARIANTLY_NECESSARY",
    "LABEL_MIXED",
]

__version__ = "1.0.0"
