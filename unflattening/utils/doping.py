"""Magic / non-Gaussian doping of the matchgate ansatz

The matchgate LASA (``Ansaetze.Matchgate``) realises a 2-design on e^g = SO(2n) and
is classically simulable.  We *dope* it with ``t`` two-qubit ZZ rotations
``RZZ(phi) = exp(-i phi/2 Z_a Z_b)``.  Under Jordan-Wigner Z_a Z_b is *quartic*
in Majorana operators, hence outside the quadratic algebra g = so(2n): each RZZ
is a genuine non-Gaussian ("fermionic magic") gate, the canonical non-matchgate
insertion (cf. the SWAP/CZ/CPhase universality-enabling gates of Jozsa-style
extensions).  The doping angles are *parameters* of the ansatz, so the loss
variance below is taken over the full doped ensemble.

This module mirrors ``qml_essentials.trainability.loss_variance`` but threads ``t`` extra RZZ gates,
with their own random angles, into the brickwork at deterministic, spread-out
(layer, bond) positions.  ``doping_generators`` returns the ZZ Pauli strings so
the doped DLA dimension can be read off via ``dla.lie_closure``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from qml_essentials.gates import Gates
from qml_essentials import operations as op
from qml_essentials import jaqsi as js

from qml_essentials.ansaetze import Ansaetze


def doping_positions(n: int, depth: int, t: int) -> list[tuple[int, tuple[int, int]]]:
    """``t`` deterministic insertion points ``(after_layer_d, (a, b))``.

    Spread in time (layers evenly across the brickwork) and space (cycling
    through the nearest-neighbour bonds), so the doping is not concentrated in
    one place.  Returns ``[]`` for ``t == 0`` (the pure matchgate baseline).
    """
    if t <= 0 or n < 2:
        return []
    positions = []
    for i in range(t):
        d = min(depth - 1, (i + 1) * depth // (t + 1))
        a = i % (n - 1)
        positions.append((d, (a, a + 1)))
    return positions


def doping_generators(n: int, positions) -> list[str]:
    """The distinct ZZ Pauli strings injected at ``positions`` (for the DLA
    closure ``dla.lie_closure(matchgate_generators(n) + doping_generators(...))``).
    """
    gens = set()
    for _, (a, b) in positions:
        s = ["I"] * n
        s[a] = "Z"
        s[b] = "Z"
        gens.add("".join(s))
    return sorted(gens)


def doped_loss_variance(
    theta: np.ndarray,
    depth: int,
    t: int,
    n_samples: int,
    key,
    out_qubit: int | None = None,
    shots: int | None = None,
):
    """Empirical Var_W[<Z_i>] for the matchgate brickwork doped with ``t`` RZZ
    gates, sampling W ~ U[0, 2pi) over *all* params (matchgate + doping angles).

    Args mirror ``qml_essentials.trainability.loss_variance``; ``t`` is the number of non-Gaussian
    RZZ insertions.  Returns ``(variance, losses)``.
    """
    theta = jnp.asarray(theta, dtype=float)
    n = int(theta.shape[0])
    i_out = n // 2 if out_qubit is None else out_qubit
    n_mg = Ansaetze.Matchgate.n_params_per_layer(n)
    positions = doping_positions(n, depth, t)
    n_params = depth * n_mg + len(positions)

    def circ(params, th):
        for q in range(n):
            Gates.RY(th[q], wires=q)
        mg = params[: depth * n_mg].reshape(depth, n_mg)
        dop = params[depth * n_mg:]
        di = 0
        for d in range(depth):
            Ansaetze.Matchgate.build(mg[d], n)
            for (dl, (a, b)) in positions:
                if dl == d:
                    Gates.RZZ(dop[di], wires=[a, b])
                    di += 1

    script = js.Script(f=circ, n_qubits=n)
    W = jax.random.uniform(key, (n_samples, n_params), minval=0.0, maxval=2 * np.pi)
    vals = script.execute(
        type="expval",
        obs=[op.PauliZ(wires=i_out)],
        args=(W, theta),
        in_axes=(0, None),
        shots=shots,
        key=key,
    )
    vals = np.asarray(vals).reshape(-1)
    return float(np.var(vals)), vals
