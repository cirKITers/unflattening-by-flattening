"""Compare off-diagonal purity for Hamming, binary, and ternary encodings.

Encode one scalar x as per-qubit angles w_k x. Binary and ternary weights
recover the independent uniform-prior mean for uniform x; Hamming weights
follow a different exact mean. Validate those formulas and model statevectors,
then write encoding_landscape and encoding_mean for plotting."""

from __future__ import annotations

from math import comb

import numpy as np

from unflattening import figures
from unflattening.utils.purity import offdiag_closed_form, offdiag_uniform_mean

from qml_essentials.ansaetze import Encoding

N_QUBITS = 6         # qubits for the landscape panel
RANGE_QUBITS = tuple(range(2, 15))  # qubit range for the mean panel
VALIDATE_RANGE = (4, 6, 8, 10)      # qubit counts for the exact-mean validation
N_SAMPLES = 200_000        # Monte-Carlo x draws per point
RAW_EPS = 0.03     # jitter of the clustered scalar input around {0, pi}
SEED = 13
TOL = 1e-9

# Weights and spectrum sizes come from the qml-essentials Encoding strategies.
ENCODINGS = {
    "hamming": Encoding("hamming", ["RY"]),
    "binary": Encoding("binary", ["RY"]),
    "ternary": Encoding("ternary", ["RY"]),
}


def weights(enc: Encoding, n: int) -> np.ndarray:
    """Per-qubit weight vector w (phi_k = w_k x) as a float64 numpy array.

    get_weights returns jax float32; the integer weights are exact there, but we
    cast to float64 so downstream angle arithmetic keeps full precision.
    """
    return np.asarray(enc.get_weights(n), dtype=float)


def hamming_mean(n: int) -> float:
    """E_x[P_god(1 x)] under uniform x: second differences of a_j = C(2j,j)/4^j."""
    a = lambda j: comb(2 * j, j) / 4.0 ** j  # noqa: E731
    return sum((n - d) * (a(d - 1) - 2 * a(d) + a(d + 1)) for d in range(1, n, 2))


def exact_grid_mean(w: np.ndarray) -> float:
    """Exact E_x[P_god(w x)]: uniform grid beyond the trig-polynomial Nyquist rate."""
    m = int(2 ** np.ceil(np.log2(4 * w.sum() + 16)))
    x = np.arange(m) * 2 * np.pi / m
    return float(offdiag_closed_form(x[:, None] * w[None, :]).mean())


def part_crosscheck(n_qubits: int = N_QUBITS, seed: int = SEED) -> None:
    """Compare model statevector purity with the weighted-angle closed form."""
    from qml_essentials.model import Model
    from unflattening.utils.dla import lie_closure_paulis, xx_yy_generators
    from unflattening.utils.purity import g_purity_from_basis

    n = n_qubits
    basis = lie_closure_paulis(xx_yy_generators(n))
    xs = np.random.default_rng(seed).uniform(0, 2 * np.pi, size=64)  # own rng: no draw-order shift
    for kind, enc in ENCODINGS.items():
        w = weights(enc, n)
        model = Model(n_qubits=n, n_layers=1, circuit_type="No_Ansatz",
                      encoding=enc, remove_zero_encoding=False)
        states = np.asarray(model(inputs=xs[:, None], execution_type="state"))
        p_model = np.array([g_purity_from_basis(s, basis) for s in states])
        p_cf = offdiag_closed_form(xs[:, None] * w[None, :])
        assert np.max(np.abs(p_model - p_cf)) < 1e-4, \
            f"{kind}: package Model g-purity must match offdiag_closed_form(w x)"
    print("package cross-check: Model(R_y, No_Ansatz) == offdiag_closed_form(w x)  PASS")


def part_validate(ns=VALIDATE_RANGE, n_qubits: int = N_QUBITS, tol: float = TOL) -> None:
    """Check exact uniform means and binary/ternary zero-purity inputs."""
    hamming, binary, ternary = (ENCODINGS[k] for k in ("hamming", "binary", "ternary"))
    for n in ns:
        iid = offdiag_uniform_mean(n)
        assert abs(exact_grid_mean(weights(binary, n)) - iid) < tol, "binary must equal iid mean"
        assert abs(exact_grid_mean(weights(ternary, n)) - iid) < tol, "ternary must equal iid mean"
        assert abs(exact_grid_mean(weights(hamming, n)) - hamming_mean(n)) < tol, "Hamming mean formula"
    # new exact zeros at benign-looking inputs (all but one qubit clusters):
    assert offdiag_closed_form(weights(binary, n_qubits) * np.pi / 2) < 1e-12
    assert offdiag_closed_form(weights(ternary, n_qubits) * np.pi / 3) < 1e-12


def part_landscape(n_qubits: int = N_QUBITS) -> None:
    """Write scalar-input purity landscapes for all three weight schemes."""
    land = {}
    for kind in ("hamming", "binary", "ternary"):
        enc = ENCODINGS[kind]
        w = weights(enc, n_qubits)
        m = int(2 ** np.ceil(np.log2(32 * w.sum() + 16)))
        x = np.arange(m) * 2 * np.pi / m
        land[kind] = (x, offdiag_closed_form(x[:, None] * w[None, :]))

    # long format: the three landscapes have different Nyquist grid sizes.
    figures.write_csv("encoding_landscape", dict(
        encoding=[k for k, (x, _) in land.items() for _ in x],
        x=np.concatenate([x for x, _ in land.values()]),
        purity=np.concatenate([P for _, P in land.values()]),
        uniform_mean=offdiag_uniform_mean(n_qubits)))
    figures.fig_encoding_landscape()


def part_mean(rng, ns=RANGE_QUBITS, n_samples: int = N_SAMPLES, raw_eps: float = RAW_EPS) -> None:
    """E_x[P_god] vs n under uniform and clustered scalar x, against the iid mean."""
    mean_u = {k: np.zeros(len(ns)) for k in ENCODINGS}
    mean_c = {k: np.zeros(len(ns)) for k in ENCODINGS}
    for i, n in enumerate(ns):
        xu = rng.uniform(0, 2 * np.pi, size=n_samples)
        xc = rng.integers(0, 2, size=n_samples) * np.pi + rng.normal(0.0, raw_eps, size=n_samples)
        for kind, enc in ENCODINGS.items():
            w = weights(enc, n)
            mean_u[kind][i] = float(offdiag_closed_form(xu[:, None] * w[None, :]).mean())
            mean_c[kind][i] = float(offdiag_closed_form(xc[:, None] * w[None, :]).mean())
    iid = np.array([offdiag_uniform_mean(n) for n in ns])

    # runnable checks: uniform x sits on the iid line for dissociated weights; clustered x
    # collapses Hamming while exponential weights partially recover by noise amplification.
    assert np.max(np.abs(mean_u["binary"] - iid) / iid) < 0.05, "binary uniform ~ iid line"
    assert np.max(np.abs(mean_u["ternary"] - iid) / iid) < 0.05, "ternary uniform ~ iid line"
    assert mean_c["hamming"][-1] < 1e-3 * mean_u["hamming"][-1], "Hamming clustered collapses"
    assert mean_c["binary"][-1] > 100 * mean_c["hamming"][-1], \
        "binary weights amplify jitter and partially recover"

    # cheap cross-check of the |Omega| closed forms the figure's right axis draws
    # (get_n_freqs would enumerate a 3^n set at n=14, hence the closed forms there).
    for kind, enc in ENCODINGS.items():
        want = {"hamming": 2 * 6 + 1, "binary": 2 ** 7 - 1, "ternary": 3 ** 6}[kind]
        assert enc.get_n_freqs(np.ones(6, dtype=bool)) == want, f"{kind}: |Omega| closed form vs get_n_freqs"

    figures.write_csv("encoding_mean", dict(n=ns, iid=iid,
                      **{f"u_{k}": mean_u[k] for k in ENCODINGS},
                      **{f"c_{k}": mean_c[k] for k in ENCODINGS}))
    figures.fig_encoding_mean()
    for kind, enc in ENCODINGS.items():
        # get_n_freqs takes the (n_qubits,) reupload mask: one fully-reuploaded layer
        mask = np.ones(ns[-1], dtype=bool)
        print(f"{kind:8s} n={ns[-1]}: uniform {mean_u[kind][-1]:.3f}  "
              f"clustered {mean_c[kind][-1]:.3e}  |Omega|={enc.get_n_freqs(mask)}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    part_validate()
    part_crosscheck()
    part_landscape()
    part_mean(rng)


if __name__ == "__main__":
    main()
