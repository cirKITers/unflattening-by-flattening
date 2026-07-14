"""encoding_weights -- scalar-input weight hierarchies (Hamming / binary / ternary).

For a scalar input x encoded as theta_k = w_k x, the model's frequency spectrum is
Omega = {sum_k s_k w_k, s_k in {-1,0,1}}: Hamming (w_k = 1) gives |Omega| = 2n+1,
binary (w_k = 2^{k-1}) gives 2^{n+1}-1 and ternary (w_k = 3^{k-1}) gives 3^n
(exponential encodings, cf. Shin et al.).  The input purity connects to this
hierarchy through the weights' arithmetic:

  * P_g(rho(w x)) is a trigonometric polynomial in x with max frequency
    2 sum_k w_k, i.e. the purity landscape inherits the model's spectrum.
  * the purity depends on the angle law only through moments E[prod cos(2 theta_k)];
    for DISSOCIATED weights (binary/ternary: no +-1 combination sums to zero) all
    these moments vanish under uniform x, so E_x[P_g] equals the iid uniform-prior
    mean EXACTLY -- a single uniform feature spectrally preconditions the encoding.
  * Hamming weights are not dissociated; uniform x gives the steeper mean
    E_x[P_god] = sum_{d odd} (n-d) [a_{d-1} - 2 a_d + a_{d+1}],  a_j = C(2j,j)/4^j,
    with slope ~0.409 n vs the iid n/3, but a strongly fluctuating landscape.
  * clustered x near {0, pi} collapses every integer-weight encoding (theta_k lands
    in {0, pi}); exponential weights however AMPLIFY any spread in x by w_k, so a
    small jitter already re-uniformises the high-weight qubits (partial recovery),
    while new exact zeros appear at benign-looking inputs (x = pi/2 under binary,
    x = pi/3 under ternary: all but one qubit clusters).

Figures: fig_encodings_landscape (purity landscape P_god(x) at n=6) and
fig_encodings_mean (E_x[P_god] vs n under uniform and clustered x).
"""
from __future__ import annotations

from math import comb

import numpy as np

from unflattening.utils import plotting
from unflattening.utils.plotting import plt, DATA_DIR, GREY_REF
from unflattening.utils.purity import offdiag_closed_form, offdiag_uniform_mean

from qml_essentials.ansaetze import Encoding

N_QUBITS = 6         # qubits for the landscape panel
RANGE_QUBITS = list(range(2, 15))  # qubit range for the mean panel
N_SAMPLES = 200_000        # Monte-Carlo x draws per point
RAW_EPS = 0.03     # jitter of the clustered scalar input around {0, pi}
SEED = 13
TOL = 1e-9

# ordered spectrum sweep equal < binary < ternary -> teal->blue->navy ramp
# (ordinal_colors keeps BAD=orange reserved for the good/bad path).
# Weights and spectrum sizes come from the qml-essentials Encoding strategies:
# get_weights(n) -> hamming w_k=1, binary w_k=2^k, ternary w_k=3^k (phi_k = w_k x).
_ENC_COLORS = plotting.ordinal_colors(3)
ENCODINGS = {
    "hamming": (_ENC_COLORS[0], Encoding("hamming", ["RY"])),
    "binary": (_ENC_COLORS[1], Encoding("binary", ["RY"])),
    "ternary": (_ENC_COLORS[2], Encoding("ternary", ["RY"])),
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


def part_package_crosscheck() -> None:
    """Double-check: the qml-essentials Fourier model reproduces the figure's encoding path.

    A Model with No_Ansatz, R_y gates and strategy ``kind`` prepares the state
    prod_k R_y(w_k x)|0>, so its off-diagonal g-purity must match
    offdiag_closed_form(w x) (to float32 precision, jax's default dtype).
    """
    from qml_essentials.model import Model
    from unflattening.utils.dla import lie_closure_paulis, xx_yy_generators
    from unflattening.utils.purity import g_purity_from_basis

    n = N_QUBITS
    basis = lie_closure_paulis(xx_yy_generators(n))
    xs = np.random.default_rng(SEED).uniform(0, 2 * np.pi, size=64)  # own rng: no draw-order shift
    for kind, (_, enc) in ENCODINGS.items():
        w = weights(enc, n)
        model = Model(n_qubits=n, n_layers=1, circuit_type="No_Ansatz",
                      encoding=enc, remove_zero_encoding=False)
        states = np.asarray(model(inputs=xs[:, None], execution_type="state"))
        p_model = np.array([g_purity_from_basis(s, basis) for s in states])
        p_cf = offdiag_closed_form(xs[:, None] * w[None, :])
        assert np.max(np.abs(p_model - p_cf)) < 1e-4, \
            f"{kind}: package Model g-purity must match offdiag_closed_form(w x)"
    print("package cross-check: Model(R_y, No_Ansatz) == offdiag_closed_form(w x)  PASS")


def main() -> None:
    rng = np.random.default_rng(SEED)

    # ---- validation: dissociated weights reproduce the iid mean exactly ----
    hamming, binary, ternary = (ENCODINGS[k][1] for k in ("hamming", "binary", "ternary"))
    for n in (4, 6, 8, 10):
        iid = offdiag_uniform_mean(n)
        assert abs(exact_grid_mean(weights(binary, n)) - iid) < TOL, "binary must equal iid mean"
        assert abs(exact_grid_mean(weights(ternary, n)) - iid) < TOL, "ternary must equal iid mean"
        assert abs(exact_grid_mean(weights(hamming, n)) - hamming_mean(n)) < TOL, "Hamming mean formula"
    # new exact zeros at benign-looking inputs (all but one qubit clusters):
    assert offdiag_closed_form(weights(binary, N_QUBITS) * np.pi / 2) < 1e-12
    assert offdiag_closed_form(weights(ternary, N_QUBITS) * np.pi / 3) < 1e-12
    part_package_crosscheck()

    # ---- top panel: purity landscape at n = 6 (equal, binary, ternary). Ternary
    # spans 3^n frequencies and renders as a dense band drawn behind the others ----
    land = {}
    for kind in ("hamming", "binary", "ternary"):
        color, enc = ENCODINGS[kind]
        w = weights(enc, N_QUBITS)
        m = int(2 ** np.ceil(np.log2(32 * w.sum() + 16)))
        x = np.arange(m) * 2 * np.pi / m
        land[kind] = (x, offdiag_closed_form(x[:, None] * w[None, :]))

    # ---- bottom panel: means vs n under uniform and clustered scalar x ----
    mean_u = {k: np.zeros(len(RANGE_QUBITS)) for k in ENCODINGS}
    mean_c = {k: np.zeros(len(RANGE_QUBITS)) for k in ENCODINGS}
    for i, n in enumerate(RANGE_QUBITS):
        xu = rng.uniform(0, 2 * np.pi, size=N_SAMPLES)
        xc = rng.integers(0, 2, size=N_SAMPLES) * np.pi + rng.normal(0.0, RAW_EPS, size=N_SAMPLES)
        for kind, (color, enc) in ENCODINGS.items():
            w = weights(enc, n)
            mean_u[kind][i] = float(offdiag_closed_form(xu[:, None] * w[None, :]).mean())
            mean_c[kind][i] = float(offdiag_closed_form(xc[:, None] * w[None, :]).mean())
    iid = np.array([offdiag_uniform_mean(n) for n in RANGE_QUBITS])

    # runnable checks: uniform x sits on the iid line for dissociated weights; clustered x
    # collapses Hamming while exponential weights partially recover by noise amplification.
    assert np.max(np.abs(mean_u["binary"] - iid) / iid) < 0.05, "binary uniform ~ iid line"
    assert np.max(np.abs(mean_u["ternary"] - iid) / iid) < 0.05, "ternary uniform ~ iid line"
    assert mean_c["hamming"][-1] < 1e-3 * mean_u["hamming"][-1], "Hamming clustered collapses"
    assert mean_c["binary"][-1] > 100 * mean_c["hamming"][-1], \
        "binary weights amplify jitter and partially recover"

    display = {"hamming": "equal", "binary": "binary", "ternary": "ternary"}
    lstyle = {  # ternary spans 3^n freqs: thin/faint/behind, equal on top
        "hamming": dict(alpha=0.9, lw=0.9, zorder=3),
        "binary": dict(alpha=0.8, lw=0.8, zorder=2),
        "ternary": dict(alpha=0.45, lw=0.4, zorder=1),
    }

    figA, axA = plt.subplots(figsize=(plotting.COL, 2.2))
    for kind, (x, P) in land.items():
        axA.plot(x, np.maximum(P, 1e-4), "-", color=ENCODINGS[kind][0],
                 label=display[kind], **lstyle[kind])
    axA.axhline(offdiag_uniform_mean(N_QUBITS), color=GREY_REF, ls=":", lw=0.9)
    axA.set_yscale("log")
    axA.set_ylim(1e-4, 12)
    axA.set_xlim(0, 2 * np.pi)
    axA.set_xticks([0, np.pi, 2 * np.pi])
    axA.set_xticklabels(["$0$", r"$\pi$", r"$2\pi$"])
    axA.set_xlabel(r"Scalar input $x$")
    axA.set_ylabel(r"$P_{\mathfrak{g}}(\rho(\boldsymbol{w}x))$")
    plotting.top_legend(axA, ncol=3)
    plotting.save(figA, "fig_encodings_landscape")

    figB, axB = plt.subplots(figsize=(plotting.COL, 2.5))
    for kind, (color, _) in ENCODINGS.items():
        axB.plot(RANGE_QUBITS, mean_u[kind], "o-", color=color, ms=3, lw=1.1, label=display[kind])
        axB.plot(RANGE_QUBITS, np.maximum(mean_c[kind], 1e-7), "s--", color=color, ms=3, lw=1.1)
    axB.plot(RANGE_QUBITS, iid, ":", color=GREY_REF, lw=1.3, zorder=0, label=r"iid mean")
    axB.plot([], [], "o-", color="0.4", ms=3, lw=1.1, label=r"uniform $x$")
    axB.plot([], [], "s--", color="0.4", ms=3, lw=1.1, label=r"clustered $x$")
    axB.set_yscale("log")
    axB.set_xlabel("$n$ Qubits")
    axB.set_ylabel(r"$\mathbb{E}_x[P_{\mathfrak{g}}]$")
    axB.locator_params(axis="x", integer=True)
    plotting.top_legend(axB, ncol=3)
    plotting.save(figB, "fig_encodings_mean")
    np.savez(DATA_DIR / "encoding_weights.npz", ns=RANGE_QUBITS, iid=iid,
             **{f"u_{k}": mean_u[k] for k in ENCODINGS},
             **{f"c_{k}": mean_c[k] for k in ENCODINGS})
    print("saved fig_encodings_landscape + fig_encodings_mean")
    for kind, (_, enc) in ENCODINGS.items():
        # get_n_freqs takes the (n_qubits,) reupload mask: one fully-reuploaded layer
        mask = np.ones(RANGE_QUBITS[-1], dtype=bool)
        print(f"{kind:8s} n={RANGE_QUBITS[-1]}: uniform {mean_u[kind][-1]:.3f}  "
              f"clustered {mean_c[kind][-1]:.3e}  |Omega|={enc.get_n_freqs(mask)}")


if __name__ == "__main__":
    main()
