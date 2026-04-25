# Eliminability Diagnostic

A minimal reproducible demonstration of the eliminability discriminant introduced in:

**Matthew W. Loving**
*Evaluation Dependence in Quantum Measurement: A Minimal FRAME Formalism and Eliminability Discriminant*
International Journal of Quantum Foundations (2026)

---

## Overview

This repository provides a computational demonstration of evaluation dependence under fixed dynamics.

The central question is simple:

Can the same component of a state description be considered eliminable under one evaluation posture and non eliminable under another, while the underlying system, input, and dynamics remain unchanged?

This project shows that the answer is yes.

The eliminability discriminant measures the reconstruction cost of removing a candidate component under a specified evaluation structure. When two evaluators assign different discriminants to the same component under fixed conditions, that component is evaluation dependent.

This repository provides minimal working examples that make that structure explicit, testable, and reproducible.

---

## Included Demonstrations

### 1. Quantum Toy Example

A two level quantum system represented by a density operator.

A coherence bearing off diagonal term is:

* eliminable under a classical admissibility evaluator
* non eliminable under a coherence preserving evaluator

The discriminant difference is computed explicitly using the Frobenius norm.

---

### 2. Classical Signal Example

A six point discrete signal with:

* two smooth admissible components
* one localized constraint violating spike

A strict evaluator removes the spike.

A permissive evaluator retains it.

The discriminant shows that the same component is fully eliminable under one evaluator and structurally necessary under the other.

---

### 3. Optional Smoothness Variant

An extended example demonstrating evaluator dependence under alternative smoothness constraints.

---

## Why This Matters

This repository is not intended as an interpretation of quantum mechanics.

It is a diagnostic tool.

It provides a computable criterion for identifying when observer indexed disagreement is structurally significant rather than merely descriptive.

The goal is modest:

to measure evaluation dependence clearly, reproducibly, and without metaphysical overreach.

---

## Quick Start

Install dependencies:

```bash id="3ub9kt"
pip install -r requirements.txt
```

Run the demo:

```bash id="yt8d1k"
python main.py
```

Outputs include:

* numerical discriminant values
* persistence scores
* CSV results
* generated figures
* reproducible examples for comparison

---

## Repository Structure

```text id="r8iywd"
README.md
LICENSE
CITATION.cff
requirements.txt
main.py
examples/
eliminability/
tests/
docs/
notebooks/
results/
```

---

## Citation

If you use or reference this repository, please cite:

Loving, Matthew W.
“Evaluation Dependence in Quantum Measurement: A Minimal FRAME Formalism and Eliminability Discriminant.”
International Journal of Quantum Foundations, 2026.

---

## License

Apache License 2.0

---

## Contact

Matthew W. Loving

This repository is intended as a minimal public demonstration of the eliminability discriminant and does not include proprietary implementations associated with other FRAME related systems.
