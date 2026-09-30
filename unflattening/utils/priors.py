"""Sample angle priors and convert rotated data to polar angles."""

from __future__ import annotations

import numpy as np
from scipy.stats import special_ortho_group


def sample_uniform(rng: np.random.Generator, m: int, n: int) -> np.ndarray:
    """m angle configurations theta_i ~ U[0, 2pi), shape (m, n)."""
    return rng.uniform(0.0, 2 * np.pi, size=(m, n))


def sample_clustered(rng: np.random.Generator, m: int, n: int, eps: float) -> np.ndarray:
    """Strongly clustered angles theta_i ~ N(0, eps^2) (so rho ~ |0^n><0^n|)."""
    return rng.normal(0.0, eps, size=(m, n))


def sample_raw(rng: np.random.Generator, m: int, n: int, eps: float) -> np.ndarray:
    """Sample angles near {0, pi} with Gaussian width ``eps``."""
    b = rng.integers(0, 2, size=(m, n)) * np.pi
    return b + rng.normal(0.0, eps, size=(m, n))


def anisotropic_data(rng: np.random.Generator, m: int, d: int, rho: float = 0.97) -> np.ndarray:
    """Sample m feature vectors with within-pair correlation ``rho``."""
    assert d % 2 == 0
    a = rng.normal(size=(m, d // 2))
    b = rho * a + np.sqrt(1 - rho**2) * rng.normal(size=(m, d // 2))
    x = np.empty((m, d))
    x[:, 0::2] = a
    x[:, 1::2] = b
    return x


def polar_angles(x: np.ndarray) -> np.ndarray:
    """Polar angle of each of the n = d/2 coordinate pairs, mapped to [0, 2pi).

    ``x`` has shape (m, d) with d even; returns (m, n).
    """
    m, d = x.shape
    assert d % 2 == 0, "need an even feature dimension to form coordinate pairs"
    pairs = x.reshape(m, d // 2, 2)
    ang = np.arctan2(pairs[..., 1], pairs[..., 0])
    return np.mod(ang, 2 * np.pi)


def isotropic_precondition(rng: np.random.Generator, x: np.ndarray) -> np.ndarray:
    """Rotate all rows by one random SO(d) matrix and return polar angles."""
    d = x.shape[1]
    Q = special_ortho_group.rvs(dim=d, random_state=rng)
    return polar_angles(x @ Q.T)
