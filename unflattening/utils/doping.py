"""Insert parameterized RZZ gates into matchgate brickwork circuits.

The ZZ generators extend the matchgate Lie algebra; sampled loss variance
includes both matchgate and RZZ parameters."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaqsi.gates import Gates
from jaqsi import gateset as op
import jaqsi as js

from qml_essentials.ansaetze import Ansaetze


def doping_positions(n: int, depth: int, t: int) -> list[tuple[int, tuple[int, int]]]:
    """Place t insertions across layers and nearest-neighbor bonds."""
    if t <= 0 or n < 2:
        return []
    positions = []
    for i in range(t):
        d = min(depth - 1, (i + 1) * depth // (t + 1))
        a = i % (n - 1)
        positions.append((d, (a, a + 1)))
    return positions


def doping_generators(n: int, positions) -> list[str]:
    """Return the distinct ZZ strings inserted at ``positions``."""
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
    """Sample output variance over all matchgate and RZZ parameters.

    Returns ``(variance, losses)`` for ``t`` RZZ insertions.
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
