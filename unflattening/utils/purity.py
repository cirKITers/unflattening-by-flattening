"""Compute product-state g-purity by closed form or explicit basis sum."""

from __future__ import annotations

import numpy as np

from qml_essentials.algebra import g_purity_from_basis  # noqa: F401  (re-export)


def product_state(theta: np.ndarray) -> np.ndarray:
    """Statevector of |psi(Theta)> = prod_i R_y(theta_i)|0>, Eq. (encoding).

    R_y(t)|0> = cos(t/2)|0> + sin(t/2)|1>; qubit 0 leftmost.
    """
    theta = np.asarray(theta, dtype=float)
    psi = np.array([1.0], dtype=complex)
    for t in theta:
        psi = np.kron(psi, np.array([np.cos(t / 2), np.sin(t / 2)], dtype=complex))
    return psi


def g_purity_closed_form(theta: np.ndarray) -> np.ndarray:
    """Eq. (closedform): sum_k cos^2 t_k + sum_{j<k} sin^2 t_j sin^2 t_k
    prod_{j<l<k} cos^2 t_l.

    ``theta`` has shape (..., n); returns shape (...).  Uses the O(n) recurrence
    W_{k+1} = c_k W_k + s_k, cross = sum_k s_k W_k (W_1 = 0), whose expectation
    under the uniform prior is n-1+2^{-n} (Lemma 2), verified analytically.
    """
    theta = np.asarray(theta, dtype=float)
    c = np.cos(theta) ** 2
    s = np.sin(theta) ** 2
    n = theta.shape[-1]
    W = np.zeros(theta.shape[:-1])
    cross = np.zeros(theta.shape[:-1])
    for k in range(n):
        cross = cross + s[..., k] * W
        W = c[..., k] * W + s[..., k]
    return c.sum(axis=-1) + cross


def analytic_loss_variance(theta: np.ndarray) -> float:
    """Theorem-1 analytic loss variance Var_W[<Z_i>] = P_g(rho(theta)) / dim g.

    ``theta`` is a single angle configuration of shape (n,).
    """
    from unflattening.utils.dla import dim_g  # local import avoids import-order coupling

    return float(g_purity_closed_form(theta)) / dim_g(len(theta))


def offdiag_closed_form(theta: np.ndarray) -> np.ndarray:
    """P_g of the off-diagonal DLA for R_y product states. theta shape (..., n) -> (...)."""
    s, c = np.sin(theta) ** 2, np.cos(theta) ** 2
    n = theta.shape[-1]
    P = np.zeros(theta.shape[:-1])
    for j in range(n):
        prod = np.ones(theta.shape[:-1])      # prod_{j<l<k} c, = 1 at k = j+1
        for k in range(j + 1, n):
            if (k - j) % 2 == 1:
                P = P + s[..., j] * prod * s[..., k]
            prod = prod * c[..., k]
    return P


def offdiag_uniform_mean(n: int) -> float:
    """E_Theta[P_g] under the iid uniform prior: sum_{d odd} (n-d) 2^{-(d+1)} -> n/3 - 5/9."""
    return sum((n - d) * 2.0 ** -(d + 1) for d in range(1, n, 2))
