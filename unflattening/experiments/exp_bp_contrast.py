"""bp_contrast -- the conditional thesis: a polynomial DLA is *not* sufficient.

Same encoding-as-readout pipeline and the same JAQSI harness as in exp_theorem1, but the
ansatz U(W) is swapped:
  * matchgate LASA (polynomial DLA so(2n))      -> Var_W = Theta(1/n)   (BP-free);
  * Strongly_Entangling (full DLA su(2^n))       -> Var_W ~ exp(-c n)    (barren).

The input state rho(Theta) and observable Z_i are identical; only the circuit's
DLA differs.  This is the numerical face of Eq. (ragone): trainability is set by
the 1/dim(g_j) weighting, exponential for the full-rank ansatz, polynomial for
the matchgate one.

Figures: dla_regime_contrast (single panel: Var vs n with poly/exp reference lines).
"""

from __future__ import annotations

import jax
import numpy as np

from qml_essentials.ansaetze import Ansaetze
from qml_essentials import trainability
from unflattening import figures
from unflattening.utils.purity import analytic_loss_variance

N_SAMPLES = 2500          # random W draws per n
N_RANGE = tuple(range(2, 10))  # qubit sweep


def part_contrast(rng, key, ns=N_RANGE, n_samples=N_SAMPLES) -> None:
    """Var_W vs n for the matchgate (poly DLA) and Strongly_Entangling (full DLA) ansaetze."""
    ge_layer, ge_npl = trainability.ansatz_layer("Strongly_Entangling")

    var_mg, var_ge, pred_mg = [], [], []
    print("bp_contrast -- matchgate (poly DLA) vs Strongly_Entangling (full DLA)")
    for n in ns:
        theta = rng.uniform(0.0, 2 * np.pi, n)
        depth = max(16, 4 * n)
        key, k1, k2 = jax.random.split(key, 3)
        vmg, _ = trainability.loss_variance(
            Ansaetze.Matchgate.build, Ansaetze.Matchgate.n_params_per_layer(n),
            theta, depth, n_samples, k1
        )
        vge, _ = trainability.loss_variance(ge_layer, ge_npl(n), theta, depth, n_samples, k2)
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

    figures.write_csv("dla_regime_contrast",
                      dict(n=ns, matchgate=var_mg, generic=var_ge,
                           pred_matchgate=np.array(pred_mg), poly_slope=poly_slope,
                           exp_rate=exp_rate))
    figures.fig_dla_regime_contrast()
    print("bp_contrast: done")


def main() -> None:
    rng = np.random.default_rng(3)
    part_contrast(rng, jax.random.PRNGKey(11))


if __name__ == "__main__":
    main()
