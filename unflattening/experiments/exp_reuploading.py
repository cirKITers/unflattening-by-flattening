"""Compare output variance across data re-uploading depths.

The Model circuit has L encodings and L+1 trainable blocks. Compare
off-diagonal XX/YY and matchgate readouts under uniform and clustered
angles, and check the exact zero of the off-diagonal output at basis-state
angles. The reuploading_depth figure shows finite-depth sampled variance;
product-input purity formulas do not apply to intermediate states."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from unflattening import figures
from qml_essentials.model import Model
from qml_essentials import operations as op
from unflattening.utils.priors import sample_uniform, sample_clustered

N_QUBITS = 6              # qubits (statevector)
DEPTHS = (1, 2, 3, 4, 6, 8)
K_TH = 10          # angle configurations per prior
N_SAMPLES = 1000           # random-W draws per configuration
RAW_EPS = 0.03     # spread of the clustered angles around 0 (sample_clustered); the sigma=0 check uses exact {0, pi}^n
SEED = 11


def _reupload_mask(depth: int, n_qubits: int = N_QUBITS) -> np.ndarray:
    """Return the diagonal feature-to-qubit mask for each encoding layer."""
    return np.broadcast_to(np.eye(n_qubits, dtype=bool), (depth, n_qubits, n_qubits)).copy()


def var_vs_depth(circuit_type: str, thetas: np.ndarray, obs, key,
                 n_qubits: int = N_QUBITS, depths=DEPTHS,
                 n_samples: int = N_SAMPLES) -> np.ndarray:
    """Return sampled output variance for each depth and angle row.

    The result has shape ``(len(depths), len(thetas))``.
    """
    out = np.zeros((len(depths), thetas.shape[0]))
    for di, depth in enumerate(depths):
        model = Model(
            n_qubits=n_qubits, n_layers=depth, circuit_type=circuit_type,
            data_reupload=_reupload_mask(depth, n_qubits), encoding=["RY"] * n_qubits,
            observables=obs,
        )
        key, sub = jax.random.split(key)
        W = jax.random.uniform(sub, (n_samples, *model._params_shape), minval=0.0, maxval=2 * np.pi)
        for ti, th in enumerate(thetas):
            vals = np.asarray(model(params=W, inputs=jnp.asarray(th, dtype=float),
                                    execution_type="expval"))
            if vals.ndim > 1:  # multiple observables -> loss = <sum_i O_i>
                vals = vals.sum(axis=-1)
            out[di, ti] = float(np.var(vals))
    return out


def part_reuploading(rng, key, n_qubits=N_QUBITS, depths=DEPTHS, k_th=K_TH,
                     n_samples=N_SAMPLES, raw_eps=RAW_EPS) -> None:
    """Var_W[<O>] vs re-uploading depth, and the exact zero at clustered angles."""
    i_out = n_qubits // 2 - 1

    # ---- off-diagonal family: XX+YY layers, in-algebra readout on a bulk bond ----
    XXYY = [
        op.PauliX(wires=i_out) @ op.PauliX(wires=i_out + 1),
        op.PauliY(wires=i_out) @ op.PauliY(wires=i_out + 1),
    ]
    # ---- matchgate control: RZ + XX layers, in-algebra readout Z_i ----
    Zi = [op.PauliZ(wires=n_qubits // 2)]

    th_unif = sample_uniform(rng, k_th, n_qubits)
    th_clus = sample_clustered(rng, k_th, n_qubits, raw_eps)
    th_exact = rng.integers(0, 2, size=(k_th, n_qubits)) * np.pi  # sigma = 0

    key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    od_unif = var_vs_depth("XY_Brickwork", th_unif, XXYY, k1, n_qubits, depths, n_samples)
    od_clus = var_vs_depth("XY_Brickwork", th_clus, XXYY, k2, n_qubits, depths, n_samples)
    mg_unif = var_vs_depth("Matchgate", th_unif, Zi, k3, n_qubits, depths, n_samples)
    mg_clus = var_vs_depth("Matchgate", th_clus, Zi, k4, n_qubits, depths, n_samples)

    # exact zero at sigma = 0, at every depth (Pauli-normalizer argument in main.tex):
    # measure the raw output values, not the variance, and demand machine zero.
    worst = 0.0
    for depth in (1, depths[-1]):
        model = Model(
            n_qubits=n_qubits, n_layers=depth, circuit_type="XY_Brickwork",
            data_reupload=_reupload_mask(depth, n_qubits), encoding=["RY"] * n_qubits,
            observables=XXYY,
        )
        key, sub = jax.random.split(key)
        W = jax.random.uniform(sub, (64, *model._params_shape), minval=0.0, maxval=2 * np.pi)
        vals = np.asarray(model(params=W, inputs=jnp.asarray(th_exact[0], dtype=float),
                                execution_type="expval")).sum(axis=-1)
        worst = max(worst, float(np.abs(vals).max()))
    print(f"exact-clustered reuploading output: max|<O>| = {worst:.2e} (checked endpoint depths)")

    # finite-depth checks of the observed input-prior contrast.
    assert worst < 1e-10, "exact-clustered re-uploading output must vanish"
    assert od_clus.mean(axis=1)[-1] < 0.05 * od_unif.mean(axis=1)[-1], \
        "off-diagonal clustered variance must stay collapsed at large depth"
    assert od_unif.mean(axis=1)[-1] > 0.2 * od_unif.mean(axis=1)[0], \
        "off-diagonal uniform variance must not collapse with depth"
    assert mg_clus.mean(axis=1)[-1] > 0.2 * mg_unif.mean(axis=1)[-1], \
        "clustered matchgate output variance must retain a nonzero signal"

    # long format, depth-major: one row per (depth, theta configuration); the figure
    # takes the mean and the 10/90 quantiles over the k_th configurations.
    figures.write_csv("reuploading_depth", dict(
        depth=np.repeat(depths, k_th), theta_idx=np.tile(np.arange(k_th), len(depths)),
        od_unif=od_unif.ravel(), od_clus=od_clus.ravel(),
        mg_unif=mg_unif.ravel(), mg_clus=mg_clus.ravel(), exact_zero_max=worst))
    figures.fig_reuploading_depth()
    print("saved reuploading_depth")


def main() -> None:
    part_reuploading(np.random.default_rng(SEED), jax.random.PRNGKey(SEED))
