"""reuploading -- does the input-distribution dichotomy survive data re-uploading?

The single-block analysis (Props. 1-5) covers |psi(Theta)> -> U(W) -> <O>.  A QFM
re-uploads the R_y encoding between trainable blocks,

    |0> -> S(Theta) U(W_1) S(Theta) U(W_2) ... S(Theta) U(W_L) -> <O>,

which the 2-design-on-e^g argument does not formally cover.  Two facts checked here:

  * exact zero at any depth: at clustered angles theta in {0,pi}^n each encoding
    layer S(Theta) is a Pauli string (R_y(pi) = -iY).  Pauli strings normalise any
    Pauli-generated DLA (conjugation maps basis strings to +-themselves) and map
    computational states to computational states, so the whole circuit collapses to
    (one e^g circuit) x (one Pauli) and Prop. 5 applies verbatim: the loss vanishes
    identically for every depth L and every W.
  * empirically, the uniform-prior variance stays at its single-block level as L
    grows (the dichotomy is depth-stable), for the floor-free off-diagonal family,
    while the floored matchgate stays flat under both priors.

The circuit is built with the maintained ``qml_essentials.Model`` in its native
ansatz-first (Schuld ``L+1``) convention, ``U(W_0) S(Theta) ... S(Theta) U(W_L)``,
rather than the encoding-first equation above.  The clustered-angle collapse and
the variance dichotomy are order-agnostic (a Pauli encoder normalises the DLA and
commutes through either way; the exact-zero check below confirms it), so the two
conventions are equivalent for the claims here.
TODO: reconcile with main.tex -- keep the encoding-first equation and add a one-line
remark that the block ordering is immaterial to the collapse/variance claims.

Figure: fig_reupload (Var_W[<O>] vs re-uploading depth L; off-diagonal uniform vs
clustered, matchgate uniform vs clustered).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from unflattening.utils import plotting
from qml_essentials.model import Model
from qml_essentials import operations as op
from unflattening.utils.priors import sample_uniform, sample_clustered
from unflattening.utils.plotting import plt, DATA_DIR, TEAL, ORANGE, ACCENT, NAVY

N_QUBITS = 6              # qubits (statevector)
DEPTHS = [1, 2, 3, 4, 6, 8]
K_TH = 10          # angle configurations per prior
N_SAMPLES = 1000           # random-W draws per configuration
RAW_EPS = 0.03     # spread of the clustered angles around 0 (sample_clustered); the sigma=0 check uses exact {0, pi}^n
SEED = 11


def _dmask(depth: int) -> np.ndarray:
    """Per-layer diagonal data-reupload mask (qubit q reads input feature q), so the
    encoding S(Theta) = prod_q R_y(theta_q) has per-qubit distinct angles."""
    return np.broadcast_to(np.eye(N_QUBITS, dtype=bool), (depth, N_QUBITS, N_QUBITS)).copy()


def var_vs_depth(circuit_type: str, thetas: np.ndarray, obs, key) -> np.ndarray:
    """Var_W[<O>] for each theta row and depth in DEPTHS, via qml_essentials.Model.

    ``Model`` builds the ansatz-first re-uploading circuit with ``n_layers=depth``
    (data_reupload -> depth+1 ansatz layers); ``obs`` is summed to the scalar loss
    <sum_i O_i>.  Returns ``(len(DEPTHS), K_TH)``.
    """
    out = np.zeros((len(DEPTHS), thetas.shape[0]))
    for di, depth in enumerate(DEPTHS):
        model = Model(
            n_qubits=N_QUBITS, n_layers=depth, circuit_type=circuit_type,
            data_reupload=_dmask(depth), encoding=["RY"] * N_QUBITS,
            observables=obs,
        )
        key, sub = jax.random.split(key)
        W = jax.random.uniform(sub, (N_SAMPLES, *model._params_shape), minval=0.0, maxval=2 * np.pi)
        for ti, th in enumerate(thetas):
            vals = np.asarray(model(params=W, inputs=jnp.asarray(th, dtype=float),
                                    execution_type="expval"))
            if vals.ndim > 1:  # multiple observables -> loss = <sum_i O_i>
                vals = vals.sum(axis=-1)
            out[di, ti] = float(np.var(vals))
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    key = jax.random.PRNGKey(SEED)
    i_out = N_QUBITS // 2 - 1

    # ---- off-diagonal family: XX+YY layers, in-algebra readout on a bulk bond ----
    XXYY = [
        op.PauliX(wires=i_out) @ op.PauliX(wires=i_out + 1),
        op.PauliY(wires=i_out) @ op.PauliY(wires=i_out + 1),
    ]
    # ---- matchgate control: RZ + XX layers, in-algebra readout Z_i ----
    Zi = [op.PauliZ(wires=N_QUBITS // 2)]

    th_unif = sample_uniform(rng, K_TH, N_QUBITS)
    th_clus = sample_clustered(rng, K_TH, N_QUBITS, RAW_EPS)
    th_exact = rng.integers(0, 2, size=(K_TH, N_QUBITS)) * np.pi  # sigma = 0

    key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    od_unif = var_vs_depth("XY_Brickwork", th_unif, XXYY, k1)
    od_clus = var_vs_depth("XY_Brickwork", th_clus, XXYY, k2)
    mg_unif = var_vs_depth("Matchgate", th_unif, Zi, k3)
    mg_clus = var_vs_depth("Matchgate", th_clus, Zi, k4)

    # exact zero at sigma = 0, at every depth (Pauli-normaliser collapse + Prop. 5):
    # measure the raw loss values, not the variance, and demand machine zero.
    worst = 0.0
    for depth in (1, DEPTHS[-1]):
        model = Model(
            n_qubits=N_QUBITS, n_layers=depth, circuit_type="XY_Brickwork",
            data_reupload=_dmask(depth), encoding=["RY"] * N_QUBITS,
            observables=XXYY,
        )
        key, sub = jax.random.split(key)
        W = jax.random.uniform(sub, (64, *model._params_shape), minval=0.0, maxval=2 * np.pi)
        vals = np.asarray(model(params=W, inputs=jnp.asarray(th_exact[0], dtype=float),
                                execution_type="expval")).sum(axis=-1)
        worst = max(worst, float(np.abs(vals).max()))
    print(f"exact-clustered reuploading loss: max|<O>| = {worst:.2e} (any depth)")

    # runnable checks: the dichotomy is depth-stable.
    assert worst < 1e-10, "clustered re-uploading loss must vanish identically (Prop. 5 extension)"
    assert od_clus.mean(axis=1)[-1] < 0.05 * od_unif.mean(axis=1)[-1], \
        "off-diagonal clustered variance must stay collapsed at large depth"
    assert od_unif.mean(axis=1)[-1] > 0.2 * od_unif.mean(axis=1)[0], \
        "off-diagonal uniform variance must not collapse with depth"
    assert mg_clus.mean(axis=1)[-1] > 0.2 * mg_unif.mean(axis=1)[-1], \
        "matchgate must stay distribution-insensitive under re-uploading"

    fig, ax = plt.subplots(figsize=(plotting.COL, 2.5))
    for dat, color, ls, label in (
        (od_unif, TEAL, "-", r"Off-diag., uniform"),
        (od_clus, ORANGE, "-", r"Off-diag., clustered"),
        (mg_unif, ACCENT, "--", r"Matchgate, uniform"),
        (mg_clus, NAVY, "--", r"Matchgate, clustered"),
    ):
        mean = dat.mean(axis=1)
        lo, hi = np.quantile(dat, 0.1, axis=1), np.quantile(dat, 0.9, axis=1)
        ax.plot(DEPTHS, mean, "o" + ls, color=color, ms=3.5, lw=1.1, label=label)
        ax.fill_between(DEPTHS, lo, hi, color=color, alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel(r"Re-uploading depth $L$")
    ax.set_ylabel(r"$\mathrm{Var}_{\boldsymbol{\theta}}[\langle \mathcal{M}\rangle]$")
    ax.set_xticks(DEPTHS)
    plotting.top_legend(ax, ncol=2)
    plotting.save(fig, "fig_reupload")
    np.savez(DATA_DIR / "reupload.npz", depths=DEPTHS, od_unif=od_unif, od_clus=od_clus,
             mg_unif=mg_unif, mg_clus=mg_clus, exact_zero_max=worst)
    print("saved fig_reupload")


if __name__ == "__main__":
    main()
