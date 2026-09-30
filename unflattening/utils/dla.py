"""Helpers for Pauli-string Lie algebras and statevector validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

import numpy as np

from qml_essentials.algebra import (
    dim_so2n as dim_g,
    lie_closure_paulis,
    matchgate_basis,
    matchgate_generators,
)

if TYPE_CHECKING:
    from qml_essentials.algebra import PauliWord

__all__ = [
    "lie_closure",
    "lie_closure_paulis",
    "matchgate_generators",
    "matchgate_basis",
    "dim_g",
    "xx_yy_generators",
    "pauli_to_bitmasks",
    "word_matrix",
    "apply_word",
    "random_dla_variance",
]


def lie_closure(generators: List[Union[str, "PauliWord"]]) -> set[str]:
    """Return the Lie closure as Pauli strings, omitting global phases."""
    return {w.to_pauli_string() for w in lie_closure_paulis(generators)}


def xx_yy_generators(n: int) -> list[str]:
    """Return nearest-neighbor XX and YY generators on an open chain."""
    gens = ["".join("X" if q in (k, k + 1) else "I" for q in range(n))
            for k in range(n - 1)]
    gens += ["".join("Y" if q in (k, k + 1) else "I" for q in range(n))
             for k in range(n - 1)]
    return gens


# ---- dense bitmask-Pauli helpers (shared by exp_hollow_doping and
#      exp_offdiag_closedform.part_variance) ----------------------------------

I2 = np.eye(2, dtype=complex)
PAULI = {
    "I": I2,
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
}


def pauli_to_bitmasks(s: str) -> tuple[int, int]:
    """'XIZY' -> (x, z) bitmasks, qubit 0 leftmost."""
    x = sum(1 << i for i, c in enumerate(s) if c in "XY")
    z = sum(1 << i for i, c in enumerate(s) if c in "ZY")
    return x, z


def word_matrix(x: int, z: int, n: int) -> np.ndarray:
    M = np.array([[1.0 + 0j]])
    for i in range(n):
        M = np.kron(M, PAULI["IXZY"[((x >> i) & 1) + 2 * ((z >> i) & 1)]])
    return M


def _reverse_bits(v: int, n: int) -> int:
    """Bitmask bit q (qubit q) -> statevector index bit n-1-q (qubit 0 leftmost)."""
    return sum(((v >> q) & 1) << (n - 1 - q) for q in range(n))


def apply_word(psi: np.ndarray, x: int, z: int, n: int) -> np.ndarray:
    """Apply a Pauli word as a signed index permutation in O(2^n) time.

    With Y = iXZ, P|b> = i^n_Y (-1)^popcount(b & z) |b ^ x>.
    """
    xr, zr = _reverse_bits(x, n), _reverse_bits(z, n)
    src = np.arange(psi.size) ^ xr                       # b such that b ^ x = j
    sign = 1.0 - 2.0 * (np.bitwise_count(src & zr) & 1)  # (-1)^{popcount(b & z)}
    return (1j ** bin(x & z).count("1")) * sign * psi[src]


def random_dla_variance(psi0, gen_mats, O, depth: int, M: int,
                        rng: np.random.Generator) -> float:
    """Sample Var[<O>] over M circuits of depth random generator rotations."""
    vals = np.empty(M)
    for m in range(M):
        psi = psi0.astype(complex)
        for g, t in zip(rng.integers(0, len(gen_mats), depth),
                        rng.uniform(0, 2 * np.pi, depth)):
            psi = np.cos(t / 2) * psi - 1j * np.sin(t / 2) * (gen_mats[g] @ psi)
        vals[m] = np.real(psi.conj() @ (O @ psi))
    return float(np.var(vals))
