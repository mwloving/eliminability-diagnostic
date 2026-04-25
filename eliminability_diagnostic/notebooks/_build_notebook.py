"""
Build the eliminability_demo.ipynb notebook programmatically.
Run once to regenerate the notebook from the canonical cells below.
"""

from __future__ import annotations

import os

import nbformat as nbf


NOTEBOOK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "eliminability_demo.ipynb"
)


CELLS = [
    ("markdown", """\
# Eliminability Diagnostic: Toy Demo

This notebook reproduces the classical numerical illustration of Section 5 of the manuscript
*Evaluation Dependence in Quantum Measurement: A Minimal FRAME Formalism and Eliminability Discriminant*.

A six point observed signal is reconstructed from a three element candidate set under two evaluation postures.
One component of the candidate set violates the strict posture's admissibility constraint.
The eliminability discriminant $\\Delta_F(\\tau)$ registers the reconstruction cost of removing each component under each posture,
and the evaluation dependence magnitude $D_{F,G}(\\tau) = |\\Delta_F(\\tau) - \\Delta_G(\\tau)|$ quantifies cross posture disagreement.

Expected results:

| Component  | $\\Delta_{\\text{Strict}}$ | $\\Delta_{\\text{Permissive}}$ | $D_{F,G}$ | Evaluation dependent? |
|------------|--------------------------|-----------------------------|-----------|-----------------------|
| $\\tau_1$  | 0.120                    | 0.120                       | 0.000     | No                    |
| $\\tau_2$  | 0.056                    | 0.056                       | 0.000     | No                    |
| $\\tau_3$  | 0.000                    | 0.129                       | 0.129     | Yes                   |
"""),
    ("code", """\
import os
import sys

# Allow running the notebook from the notebooks/ directory.
_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib.pyplot as plt

from eliminability import (
    CandidateComponent,
    StrictPosture,
    PermissivePosture,
    mean_squared_error,
    compute_all_discriminants,
    compute_dependence,
    assign_persistence,
    assign_classifications,
    build_results_table,
    generate_summary,
    plot_discriminant_bars,
    plot_discriminant_heatmap,
    plot_dependence_bars,
    plot_reconstruction_comparison,
)
"""),
    ("markdown", """\
## 1. Define the toy problem

The signal is indexed at $t = 1, \\ldots, 6$. The candidate set contains three orthogonal components: two admissible basis elements $\\tau_1, \\tau_2$ and one localized spike $\\tau_3$ at $t = 6$, which is classified as violating the strict posture's smoothness constraint.
"""),
    ("code", """\
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
threshold = 0.05

print("y =", y)
for c in components:
    print(f"  {c.name}: values={c.values}, violates={c.tags['violates_constraint']}")
"""),
    ("markdown", """\
## 2. Define the postures

`StrictPosture` enforces the admissibility tag: constraint violating components are excluded from the reconciliation and assigned zero effective weight.
`PermissivePosture` imposes no admissibility constraint and fits all components by ordinary least squares.
"""),
    ("code", """\
strict = StrictPosture(name="Strict")
permissive = PermissivePosture(name="Permissive")
postures = [strict, permissive]

for p in postures:
    print(f"{p.name}: {type(p).__name__}")
"""),
    ("markdown", """\
## 3. Compute discriminants and dependence

For each component $\\tau$ and each posture $F$ we compute
$$\\Delta_F(\\tau) = R(y, A_F(S \\setminus \\{\\tau\\})) - R(y, A_F(S)),$$
with $R$ the mean squared error. We then compute the cross posture dependence
$$D_{F,G}(\\tau) = |\\Delta_F(\\tau) - \\Delta_G(\\tau)|.$$
"""),
    ("code", """\
discriminants = compute_all_discriminants(
    y=y, components=components, postures=postures, functional=mean_squared_error
)
assign_persistence(discriminants, threshold)
assign_classifications(discriminants, threshold)

dependence = compute_dependence(
    discriminants, posture_a="Strict", posture_b="Permissive", threshold=threshold
)

table = build_results_table(discriminants, dependence)
table
"""),
    ("markdown", """\
## 4. Plain text summary
"""),
    ("code", """\
print(generate_summary(discriminants, dependence, threshold))
"""),
    ("markdown", """\
## 5. Plots

### 5.1 Discriminant bar chart

Bars show $\\Delta_F(\\tau)$ grouped by component. The dashed line marks the tolerance threshold $\\theta$.
The admissible components $\\tau_1, \\tau_2$ receive equal discriminants across postures.
The violating component $\\tau_3$ is non eliminable under Permissive but eliminable under Strict.
"""),
    ("code", """\
fig, ax = plt.subplots(figsize=(8, 4))
plot_discriminant_bars(discriminants, ax=ax, threshold=threshold)
plt.tight_layout()
plt.show()
"""),
    ("markdown", """\
### 5.2 Heatmap

Rows are components, columns are postures, cells show $\\Delta_F(\\tau)$.
The evaluation dependent cell (top right or bottom right, depending on sort order) stands out by color.
"""),
    ("code", """\
fig, ax = plt.subplots(figsize=(6, 4))
plot_discriminant_heatmap(discriminants, ax=ax)
plt.tight_layout()
plt.show()
"""),
    ("markdown", """\
### 5.3 Dependence magnitude

A single bar per component for $D_{F,G}(\\tau)$. Red bars flag components that cross the unit persistence threshold.
"""),
    ("code", """\
fig, ax = plt.subplots(figsize=(8, 4))
plot_dependence_bars(dependence, ax=ax, threshold=threshold)
plt.tight_layout()
plt.show()
"""),
    ("markdown", """\
### 5.4 Reconstruction comparison

Overlay of the observed signal and the reconstructions $A_F(S)$ and $A_F(S \\setminus \\{\\tau_3\\})$ under each posture.
The permissive full reconstruction tracks the spike at $t = 6$; every reconstruction that excludes $\\tau_3$ fails to capture it.
"""),
    ("code", """\
fig, ax = plt.subplots(figsize=(8, 4))
plot_reconstruction_comparison(
    y=y, components=components, postures=postures, removed_component="tau3", ax=ax
)
plt.tight_layout()
plt.show()
"""),
    ("markdown", """\
## 6. Interpretation

The admissible components $\\tau_1$ and $\\tau_2$ yield identical discriminants under Strict and Permissive,
so evaluation invariance holds for them. The constraint violating component $\\tau_3$ yields
$\\Delta_{\\text{Strict}}(\\tau_3) = 0$ and $\\Delta_{\\text{Permissive}}(\\tau_3) \\approx 0.129$.
The persistence scores at $\\theta = 0.05$ are $P_{\\text{Strict}}(\\tau_3) = 0$ and
$P_{\\text{Permissive}}(\\tau_3) \\approx 2.58$, placing $\\tau_3$ on opposite sides of the unit threshold.
By Definition 1 of the manuscript, $\\tau_3$ is evaluation dependent with respect to Strict and Permissive.

The disagreement is attributable solely to the difference in evaluation maps: the signal, the candidate set,
and the reconstruction functional are held fixed across the two postures.
"""),
]


def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []
    for kind, src in CELLS:
        if kind == "markdown":
            cells.append(nbf.v4.new_markdown_cell(src))
        elif kind == "code":
            cells.append(nbf.v4.new_code_cell(src))
        else:
            raise ValueError(f"unknown cell kind {kind!r}")
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    return nb


if __name__ == "__main__":
    nb = build_notebook()
    with open(NOTEBOOK_PATH, "w") as f:
        nbf.write(nb, f)
    print(f"Wrote {NOTEBOOK_PATH}")
