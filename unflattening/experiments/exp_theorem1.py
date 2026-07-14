"""theorem1 -- Theorem 1 (the angle-encoded matchgate pipeline is BP-free).

Builds the pipeline |psi(Theta)> -> U(W) -> <Z_i> with a matchgate LASA U(W)
(RZ + nearest-neighbour RXX brickwork) in JAQSI, samples W ~ U[0,2pi), and
estimates the loss variance Var_W[<Z_i>].

  Panel 1 (convergence): at fixed n, Var_W converges to the analytic value
      P_g(rho(Theta))/dim g as the brickwork depth grows (the 2-design
      hypothesis of Theorem 1 kicking in).
  Panel 2 (scaling): across n, Var_W tracks P_g/dim g = Theta(1/n) (slope -1 on
      log-log), inside the proven range [(n-1)/(n(2n-1)), 1/(2n-1)].

Figures: fig_convergence, fig_scaling.
"""

from __future__ import annotations

import jax
import numpy as np
from matplotlib.ticker import LogFormatterSciNotation

from qml_essentials.ansaetze import Ansaetze
from qml_essentials import trainability
from unflattening.utils.purity import analytic_loss_variance
from unflattening.utils import plotting
from unflattening.utils.plotting import plt, DATA_DIR, GREY_FILL, TEAL, ACCENT, NAVY

N_SAMPLES = 2000


def part_convergence(rng, key):
    ns = [4, 6, 8]
    depths = [1, 2, 4, 8, 16, 24, 32]
    series = (TEAL, ACCENT, NAVY)  # all trainable: cool gradient, no barren gold
    fig, ax = plt.subplots(figsize=(plotting.COL, 2.6))
    print("  convergence Var_W -> P_g/dim_g:")
    for j, n in enumerate(ns):
        theta = rng.uniform(0.0, 2 * np.pi, n)
        pred = analytic_loss_variance(theta)
        vs = []
        for d in depths:
            key, sub = jax.random.split(key)
            v, _ = trainability.loss_variance(
                Ansaetze.Matchgate.build, Ansaetze.Matchgate.n_params_per_layer(n),
                theta, d, N_SAMPLES, sub
            )
            vs.append(v)
        print(f"    n={n}: pred={pred:.5f}  Var(depth={depths[-1]})={vs[-1]:.5f}")
        ax.plot(depths, vs, "o-", color=series[j], label=f"$n={n}$")
        ax.axhline(pred, color=series[j], ls="--", lw=1.0,
                   label=(r"$P_{\mathfrak{g}}/\dim\mathfrak{g}$" if j == 0 else None))
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("MGA Depth")
    ax.set_ylabel(r"$\mathrm{Var}_{\boldsymbol{\theta}}[\langle Z_i\rangle]$")
    # legend with the analytic entry first
    handles, labels = ax.get_legend_handles_labels()
    ai = labels.index(r"$P_{\mathfrak{g}}/\dim\mathfrak{g}$")
    order = [ai] + [i for i in range(len(labels)) if i != ai]
    plotting.top_legend(ax, [handles[i] for i in order], [labels[i] for i in order])
    plotting.save(fig, "fig_convergence")


def part_scaling(rng, key):
    ns = list(range(2, 13))
    emp, pred, lo, hi = [], [], [], []
    print("  scaling Var_W vs n (depth=6n):")
    for n in ns:
        theta = rng.uniform(0.0, 2 * np.pi, n)
        depth = 6 * n
        key, sub = jax.random.split(key)
        v, _ = trainability.loss_variance(
            Ansaetze.Matchgate.build, Ansaetze.Matchgate.n_params_per_layer(n),
            theta, depth, N_SAMPLES, sub
        )
        emp.append(v)
        pred.append(analytic_loss_variance(theta))
        lo.append((n - 1) / (n * (2 * n - 1)))  # P_g=n-1 -> Var
        hi.append(1.0 / (2 * n - 1))            # P_g=n -> Var
        print(f"    n={n:2d}  depth={depth:3d}  Var={v:.5f}  pred={pred[-1]:.5f}  "
              f"ratio={v / pred[-1]:.3f}")
    ns = np.array(ns, float)
    emp, pred = np.array(emp), np.array(pred)
    slope = np.polyfit(np.log(ns), np.log(emp), 1)[0]
    print(f"  log-log slope of empirical Var vs n: {slope:.3f}  (Theorem 1: -1)")

    fig, ax = plt.subplots(figsize=(plotting.COL, 2.6))
    ax.fill_between(ns, lo, hi, color=GREY_FILL, label="Proven Range")
    ax.plot(ns, pred, "-", color=TEAL, label=r"$P_{\mathfrak{g}}/\dim\mathfrak{g}$")
    ax.plot(ns, emp, "o", color=NAVY, label=r"$\mathrm{Var}_{\boldsymbol{\theta}}$", zorder=5)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    # match the y-ticks of fig_convergence: label only {6e-2, 1e-1, 2e-1, 3e-1}
    ax.set_yticks([6e-2, 1e-1, 2e-1, 3e-1])
    ax.set_yticks([], minor=True)
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    ax.set_xlabel("$n$ Qubits")
    ax.set_ylabel(r"$\mathrm{Var}_{\boldsymbol{\theta}}[\langle Z_i\rangle]$")
    plotting.top_legend(ax, ncol=3)
    plotting.save(fig, "fig_scaling")
    np.savez(DATA_DIR / "scaling.npz", n=ns, empirical=emp, analytic=pred, slope=slope)


def main() -> None:
    rng = np.random.default_rng(2)
    key = jax.random.PRNGKey(2)
    print("theorem1 -- matchgate pipeline Var_W = Theta(1/n)")
    part_convergence(rng, key)
    part_scaling(rng, jax.random.PRNGKey(7))
    print("theorem1: done")


if __name__ == "__main__":
    main()
