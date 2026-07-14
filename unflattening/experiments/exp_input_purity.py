"""input_purity -- the conditional thesis on the *input* axis: a polynomial DLA
is not sufficient when the input g-purity is small.

Same matchgate LASA (polynomial DLA so(2n), dim g = n(2n-1)) and the same readout
Z_i throughout; only the *input state* is swapped:
  * structured product state rho(Theta)  -> P_g >= n-1,  Var_W = Theta(1/n)   (BP-free);
  * Haar-random input |psi>              -> E[P_g] = n(2n-1)/(2^n+1),  Var_W = Theta(2^-n).

With the denominator dim g fixed and polynomial in both cases, the input g-purity
numerator of Eq. (ragone) alone decides trainability.  This is the input-side
companion of exp_bp_contrast, which instead varies dim g.

E[P_g] = n(2n-1)/(2^n+1) is the 2-design average E[<psi|B|psi>^2] = 1/(2^n+1)
over the n(2n-1) matchgate basis elements; with the in-algebra readout P_g(Z_i)=1
this gives E[Var_W] = 1/(2^n+1).

Figures: fig_input_purity (single panel: Var_W vs n, product vs Haar input).
"""

from __future__ import annotations

import jax
import numpy as np

from qml_essentials.ansaetze import Ansaetze
from qml_essentials import trainability
from unflattening.utils.dla import matchgate_basis
from qml_essentials.states import haar_state
from unflattening.utils.purity import product_state, analytic_loss_variance, g_purity_from_basis
from unflattening.utils import plotting
from unflattening.utils.plotting import plt, DATA_DIR, GREY_REF, TEAL, ORANGE

N_SAMPLES = 2000   # random W draws per input
N_INPUTS = 8      # Haar input draws per n


def main() -> None:
    rng = np.random.default_rng(5)
    ns = list(range(2, 10))  # match fig_contrast (exp_bp_contrast) qubit range

    pg_prod, pg_haar = [], []
    var_prod, var_haar = [], []
    pred_prod, pred_haar = [], []
    key = jax.random.PRNGKey(7)
    print("input_purity -- product input (P_g >= n-1) vs Haar input (P_g exp-small)")
    for n in ns:
        basis = matchgate_basis(n)
        theta = rng.uniform(0.0, 2 * np.pi, n)
        depth = max(16, 4 * n)
        haar = [haar_state(n, rng) for _ in range(N_INPUTS)]

        # g-purity: structured product (closed-form >= n-1) vs Haar-random input.
        ppg = float(g_purity_from_basis(product_state(theta), basis))
        hpg = float(np.mean([g_purity_from_basis(psi, basis) for psi in haar]))
        pg_prod.append(ppg)
        pg_haar.append(hpg)

        # loss variance Var_W[<Z_i>] on the *same* matchgate DLA + readout Z_i.
        npl = Ansaetze.Matchgate.n_params_per_layer(n)
        key, kp = jax.random.split(key)
        vprod, _ = trainability.loss_variance(Ansaetze.Matchgate.build, npl, theta, depth, N_SAMPLES, kp)
        vh = []
        for psi in haar:
            key, kh = jax.random.split(key)
            v, _ = trainability.loss_variance(
                Ansaetze.Matchgate.build, npl, None, depth, N_SAMPLES, kh, init_state=psi
            )
            vh.append(v)
        var_prod.append(vprod)
        var_haar.append(float(np.mean(vh)))

        # analytic overlays: product P_g/dim g (closed form); Haar 1/(2^n+1).
        pred_prod.append(analytic_loss_variance(theta))
        pred_haar.append(1.0 / (2**n + 1))
        print(f"  n={n:2d}: P_g prod={ppg:7.3f} (>= {n-1}),  Haar={hpg:.3e} "
              f"(pred {n*(2*n-1)/(2**n+1):.3e});  "
              f"Var prod={vprod:.3e}  Haar={var_haar[-1]:.3e}")

    ns = np.array(ns, float)
    var_prod, var_haar = np.array(var_prod), np.array(var_haar)
    pg_prod, pg_haar = np.array(pg_prod), np.array(pg_haar)
    poly_slope = np.polyfit(np.log(ns), np.log(var_prod), 1)[0]
    exp_rate = np.polyfit(ns, np.log(var_haar), 1)[0]
    print(f"  product log-log slope={poly_slope:.2f} (poly);  "
          f"Haar log-Var vs n slope={exp_rate:.2f} (exp decay, ~ -0.69)")

    # runnable checks: at fixed polynomial dim g, the input g-purity numerator of
    # Eq. (ragone) alone separates trainable (product) from barren (Haar).
    assert np.all(pg_prod >= ns - 1 - 1e-9), "product P_g must respect the n-1 floor"
    ilast = len(ns) - 1
    assert var_haar[ilast] < 0.1 * var_prod[ilast], "Haar Var must collapse below product"

    fig, ax = plt.subplots(figsize=(plotting.COL, 2.6))
    # Same matchgate DLA (dim g = n(2n-1)) and readout Z_i; only the input differs.
    ax.plot(ns, var_prod, "-", color=TEAL, label=r"Product $\rho(\boldsymbol{\phi})$")
    ax.plot(ns, var_haar, "-", color=ORANGE, label="Haar Input")
    ax.plot(ns, var_prod[0] * ns[0] / ns, ":", color=GREY_REF, label=r"$\propto 1/n$")
    ax.plot(ns, 1.0 / (2**ns + 1), "--", color=ORANGE, lw=0.9, label=r"$1/(2^n+1)$")
    ax.set_yscale("log")
    ax.set_xlabel("$n$ Qubits"); ax.set_ylabel(r"$\mathrm{Var}_{\boldsymbol{\theta}}[\langle Z_i\rangle]$")
    ax.locator_params(axis="x", integer=True)
    plotting.top_legend(ax, ncol=2)
    plotting.save(fig, "fig_input_purity")
    np.savez(DATA_DIR / "input_purity.npz", n=ns, var_product=var_prod, var_haar=var_haar,
             pg_product=pg_prod, pg_haar=pg_haar, pred_product=np.array(pred_prod),
             pred_haar=np.array(pred_haar), poly_slope=poly_slope, exp_rate=exp_rate)
    print("input_purity: done")


if __name__ == "__main__":
    main()
