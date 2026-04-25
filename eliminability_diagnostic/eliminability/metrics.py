"""
Reconstruction functionals R : Y x Y -> R_{>=0}.

The reconstruction functional measures the discrepancy between a target
description y and a reconstructed signal y_hat. The choice of functional
is part of the problem specification and is held fixed across all
evaluation structures (see Section 3.1 of the accompanying manuscript).

All functionals here satisfy:
    R(y, y) = 0
    R(y, y_hat) >= 0
    R(y, y_hat) monotonically increases with discrepancy under some
    metric on Y.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


ReconstructionFunctional = Callable[[np.ndarray, np.ndarray], float]


def mean_squared_error(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Mean squared error for discrete vector signals.

        R(y, y_hat) = (1/n) sum_t (y(t) - y_hat(t))**2

    Parameters
    ----------
    y : np.ndarray
        Target signal, shape (n,).
    y_hat : np.ndarray
        Reconstructed signal, shape (n,).

    Returns
    -------
    float
        Mean squared error, non negative.
    """
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    if y.shape != y_hat.shape:
        raise ValueError(
            f"Shape mismatch: y has shape {y.shape}, y_hat has shape {y_hat.shape}"
        )
    residual = y - y_hat
    return float(np.mean(residual * residual))


def squared_frobenius_norm(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Squared Frobenius norm distance between two matrices.

        R(rho, sigma) = Tr[ (rho - sigma)^dagger (rho - sigma) ]

    Suitable for matrix valued examples, including two level density
    operators as developed in Section 4 of the manuscript. For a 2 x 2
    density operator with off diagonal coherence c, this returns 2 |c|**2
    when sigma is the dephased diagonal.

    Parameters
    ----------
    rho : np.ndarray
        Target matrix, shape (n, n). May be complex valued.
    sigma : np.ndarray
        Reconstructed matrix, shape (n, n). May be complex valued.

    Returns
    -------
    float
        Squared Frobenius norm of (rho - sigma), non negative and real.
    """
    rho = np.asarray(rho)
    sigma = np.asarray(sigma)
    if rho.shape != sigma.shape:
        raise ValueError(
            f"Shape mismatch: rho has shape {rho.shape}, sigma has shape {sigma.shape}"
        )
    diff = rho - sigma
    val = np.vdot(diff.flatten(), diff.flatten())
    return float(np.real(val))
