"""Matchgate dynamical Lie algebra helpers.

`lie_closure`` wraps :func:`qml_essentials.algebra.lie_closure_paulis` 
to return a set of bare Pauli strings which is the unflattening call-site contract
(e.g. ``set(basis) == closure`` in exp_closedform).
"""

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
    "random_dla_variance",
]


def lie_closure(generators: List[Union[str, "PauliWord"]]) -> set[str]:
    """Pauli-string Lie closure as a set of bare strings.

    Delegates to :func:`qml_essentials.algebra.lie_closure_paulis` and returns
    the bare Pauli strings (the global phase is irrelevant to the span).
    """
    return {w.to_pauli_string() for w in lie_closure_paulis(generators)}


def xx_yy_generators(n: int) -> list[str]:
    """Off-diagonal generators {X_k X_{k+1}} u {Y_k Y_{k+1}} on the open n-chain.

    The Pauli-string analogue of :func:`matchgate_generators` for the floor-free
    off-diagonal DLA g_od ~= so(n) (+) so(n); feed to ``lie_closure`` /
    ``lie_closure_paulis`` to build the basis.
    """
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


def random_dla_variance(psi0, gen_mats, O, depth: int, M: int,
                        rng: np.random.Generator) -> float:
    """Var over M deep random-e^g statevector circuits of <O>.

    Each sample applies ``depth`` single-generator rotations exp(-i t/2 G), G
    uniform from ``gen_mats``, t ~ U[0, 2pi), to ``psi0`` (dense 2^n vectors).
    Shared by exp_hollow_doping.validate_variance and
    exp_offdiag_closedform.part_variance.
    """
    vals = np.empty(M)
    for m in range(M):
        psi = psi0.astype(complex)
        for g, t in zip(rng.integers(0, len(gen_mats), depth),
                        rng.uniform(0, 2 * np.pi, depth)):
            psi = np.cos(t / 2) * psi - 1j * np.sin(t / 2) * (gen_mats[g] @ psi)
        vals[m] = np.real(psi.conj() @ (O @ psi))
    return float(np.var(vals))
