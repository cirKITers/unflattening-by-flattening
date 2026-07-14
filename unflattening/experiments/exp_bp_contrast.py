"""bp_contrast -- the conditional thesis: a polynomial DLA is *not* sufficient.

Same encoding-as-readout pipeline and the same JAQSI harness as in exp_theorem1, but the
ansatz U(W) is swapped:
  * matchgate LASA (polynomial DLA so(2n))      -> Var_W = Theta(1/n)   (BP-free);
  * Strongly_Entangling (full DLA su(2^n))       -> Var_W ~ exp(-c n)    (barren).

The input state rho(Theta) and observable Z_i are identical; only the circuit's
DLA differs.  This is the numerical face of Eq. (ragone): trainability is set by
the 1/dim(g_j) weighting, exponential for the full-rank ansatz, polynomial for
the matchgate one.

Figures: fig_contrast (single panel: Var vs n with poly/exp reference lines).
"""

from __future__ import annotations

import jax
import numpy as np

from qml_essentials.ansaetze import Ansaetze
from qml_essentials import trainability
from unflattening.utils.purity import analytic_loss_variance
from unflattening.utils import plotting
from unflattening.utils.plotting import plt, DATA_DIR, GREY_REF, TEAL, ORANGE

N_SAMPLES = 2500


def main() -> None:
    rng = np.random.default_rng(3)
    ns = list(range(2, 10))
    ge_layer, ge_npl = trainability.ansatz_layer("Strongly_Entangling")

    var_mg, var_ge, pred_mg = [], [], []
    key = jax.random.PRNGKey(11)
    print("bp_contrast -- matchgate (poly DLA) vs Strongly_Entangling (full DLA)")
    for n in ns:
        theta = rng.uniform(0.0, 2 * np.pi, n)
        depth = max(16, 4 * n)
        key, k1, k2 = jax.random.split(key, 3)
        vmg, _ = trainability.loss_variance(
            Ansaetze.Matchgate.build, Ansaetze.Matchgate.n_params_per_layer(n),
            theta, depth, N_SAMPLES, k1
        )
        vge, _ = trainability.loss_variance(ge_layer, ge_npl(n), theta, depth, N_SAMPLES, k2)
        var_mg.append(vmg)
        var_ge.append(vge)
        pred_mg.append(analytic_loss_variance(theta))
        print(f"  n={n}: matchgate Var={vmg:.3e} (pred {pred_mg[-1]:.3e})  "
              f"Strongly_Entangling Var={vge:.3e}")

    ns = np.array(ns, float)
    var_mg, var_ge = np.array(var_mg), np.array(var_ge)
    # poly slope (log-log) for matchgate; exponential rate (semilog) for generic.
    poly_slope = np.polyfit(np.log(ns), np.log(var_mg), 1)[0]
    exp_rate = np.polyfit(ns, np.log(var_ge), 1)[0]
    print(f"  matchgate log-log slope={poly_slope:.2f} (poly);  "
          f"Strongly_Entangling log-Var vs n slope={exp_rate:.2f} (exp decay)")

    fig, ax = plt.subplots(figsize=(plotting.COL, 2.6))
    # Same input rho(Theta) and observable Z_i; only the ansatz DLA differs.
    # matchgate hugs the 1/n guide (polynomial); the full-DLA ansatz follows the
    # exponential fit -- the conditional reading of Eq. (ragone) in one panel.
    ax.plot(ns, var_mg, "o-", color=TEAL, label="MGA (Poly DLA)")
    ax.plot(ns, var_ge, "s-", color=ORANGE, label="SEA (Full DLA)")
    ax.plot(ns, var_mg[0] * ns[0] / ns, ":", color=GREY_REF, label=r"$\propto 1/n$")
    ax.plot(ns, np.exp(np.polyval([exp_rate, np.log(var_ge[0]) - exp_rate * ns[0]], ns)),
            "--", color=ORANGE, lw=0.9, label=f"$\\propto e^{{{exp_rate:.2f}\\,n}}$")
    ax.set_yscale("log")
    ax.set_xlabel("$n$ Qubits"); ax.set_ylabel(r"$\mathrm{Var}_{\boldsymbol{\theta}}[\langle Z_i\rangle]$")
    ax.locator_params(axis="x", integer=True)
    plotting.top_legend(ax, ncol=2)
    plotting.save(fig, "fig_contrast")
    np.savez(DATA_DIR / "contrast.npz", n=ns, matchgate=var_mg, generic=var_ge,
             pred_matchgate=np.array(pred_mg), poly_slope=poly_slope, exp_rate=exp_rate)
    print("bp_contrast: done")


if __name__ == "__main__":
    main()
