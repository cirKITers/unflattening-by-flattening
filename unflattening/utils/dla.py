"""Matchgate dynamical Lie algebra helpers.

`lie_closure`` wraps :func:`qml_essentials.algebra.lie_closure_paulis` 
to return a set of bare Pauli strings which is the unflattening call-site contract
(e.g. ``set(basis) == closure`` in exp_closedform).
"""

from __future__ import annotations

from typing import List, Union

from qml_essentials.algebra import (
    PauliWord,
    dim_so2n as dim_g,
    lie_closure_paulis,
    matchgate_basis,
    matchgate_generators,
)

__all__ = [
    "lie_closure",
    "matchgate_generators",
    "matchgate_basis",
    "dim_g",
    "xx_yy_generators",
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
