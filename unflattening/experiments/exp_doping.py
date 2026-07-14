"""doping -- magic / non-Gaussian doping: trading trainability for hardness.

Dopes the matchgate LASA with ``t`` two-qubit RZZ gates (generator Z_a Z_b is
quartic in Majoranas, hence non-Gaussian / outside g = so(2n); the canonical
non-matchgate insertion).  We measure how the loss variance Var_W[<Z_i>] decays
in n as the doping budget t grows, and how the doped DLA dimension explodes.

Mechanism shown (not the asymptotic threshold -- see caveat):
  * t = 0 reproduces Theorem 1: Var_W = P_g/dim g = Theta(1/n) (log-log slope ~ -1);
  * each non-Gaussian gate steepens the decay (the gradient-variance cost of
    escaping Gaussianity), interpolating matchgate Omega(1/n) -> barren e^{-Theta(n)};
  * one RZZ already blows the Lie closure from n(2n-1) to exp-size.

The classical simulation cost grows as poly(n, 2^t), efficient only for t = O(log n)
(literature), so this variance decay is the trainability price of escaping Gaussianity.

CAVEAT: exact statevector caps n <~ 12, so this probes the *mechanism* (monotone
steepening + DLA blow-up), not an n -> infinity threshold.

Figures: fig_doping (left: Var vs n per t; right: fixed-n decay Var ~ c^{-t}).
"""

from __future__ import annotations

import jax
import numpy as np

from unflattening.utils import doping, dla
from unflattening.utils.purity import analytic_loss_variance
from unflattening.utils import plotting
from unflattening.utils.plotting import plt, DATA_DIR, GREY_REF

N_SAMPLES = 600
TS = [0, 1, 2, 3, 4]


def main() -> None:
    rng = np.random.default_rng(7)
    ns = list(range(3, 10))  # exact statevector; n<=9 keeps runtime modest
    key = jax.random.PRNGKey(23)

    var = {t: [] for t in TS}
    pred0 = []  # t=0 analytic P_g/dim g (Theorem 1 check)
    print("doping -- non-Gaussian (RZZ) doping of the matchgate LASA")
    for n in ns:
        theta = rng.uniform(0.0, 2 * np.pi, n)
        depth = max(16, 4 * n)
        pred0.append(analytic_loss_variance(theta))
        row = []
        for t in TS:
            key, k = jax.random.split(key)
            v, _ = doping.doped_loss_variance(theta, depth, t, N_SAMPLES, k)
            var[t].append(v)
            row.append(v)
        print(f"  n={n:2d}: " + "  ".join(f"t={t}:{v:.3e}" for t, v in zip(TS, row))
              + f"   (t=0 pred {pred0[-1]:.3e})", flush=True)

    ns = np.array(ns, float)
    for t in TS:
        var[t] = np.array(var[t])
    # log-log decay slope per fixed t (more negative = closer to barren).
    slope = {t: float(np.polyfit(np.log(ns), np.log(var[t]), 1)[0]) for t in TS}
    print("  log-log slopes: " + ", ".join(f"t={t}:{slope[t]:.2f}" for t in TS))

    # DLA closure blow-up (one representative n), reported in the caption/inset.
    n_rep = 6
    pos1 = doping.doping_positions(n_rep, max(16, 4 * n_rep), 1)
    dim_mg = dla.dim_g(n_rep)
    dim_doped = len(dla.lie_closure(
        dla.matchgate_generators(n_rep) + doping.doping_generators(n_rep, pos1)))
    print(f"  DLA dim at n={n_rep}: matchgate {dim_mg} -> doped {dim_doped}")

    # per-fixed-n exponential-in-t rate: Var ~ c^{-t} (Eq. doping-decay).
    rate = np.array([np.polyfit(TS, np.log([var[t][i] for t in TS]), 1)[0]
                     for i in range(len(ns))])
    c_fit = float(np.exp(-rate.mean()))
    print(f"  fixed-n decay Var ~ c^-t: c ~ {c_fit:.3f} (mean over n)", flush=True)

    # save raw data BEFORE plotting so a backend hiccup cannot lose the run.
    np.savez(DATA_DIR / "doping.npz", n=ns, pred0=np.array(pred0),
             **{f"var_t{t}": var[t] for t in TS},
             slopes=np.array([slope[t] for t in TS]), ts=np.array(TS),
             c_fit=c_fit, dim_mg=dim_mg, dim_doped=dim_doped)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(plotting.WIDE, 2.7))
    # (a) Var vs n per fixed t; t=0 is the matchgate baseline ~ 1/n.
    # sequential cool ramp keyed to the ordinal doping count t (avoids the
    # BAD=orange bleed and the colour reuse of a wrapping categorical cycle).
    tcolors = plotting.ordinal_colors(len(TS))
    for j, t in enumerate(TS):
        a0.plot(ns, var[t], "o-", color=tcolors[j],
                label=f"$t={t}$" + (" (mg)" if t == 0 else ""))
    a0.plot(ns, var[0][0] * ns[0] / ns, ":", color=GREY_REF, label=r"$\propto 1/n$")
    a0.set_xscale("log"); a0.set_yscale("log")
    a0.set_xlabel("$n$ Qubits")
    a0.set_ylabel(r"$\mathrm{Var}_{\boldsymbol{\theta}}[\langle Z_i\rangle]$")
    a0.legend(loc="lower left", ncol=2, fontsize=8)
    # (b) fixed-n decay in t: Var ~ c^{-t} (Eq. doping-decay), rate ~ n-independent.
    idx = list(ns.astype(int))
    reps = [k for k in (6, 8, 10) if k in idx] or [idx[-1]]
    for k in reps:
        i = idx.index(k)
        a1.semilogy(TS, [var[t][i] for t in TS], "o-", label=f"$n={k}$")
    a1.semilogy(TS, var[0][idx.index(reps[-1])] * c_fit ** (-np.array(TS, float)),
                "k--", lw=1.0, label=rf"$\propto c^{{-t}}$ ($c\approx{c_fit:.2f}$)")
    a1.set_xlabel("non-Gaussian gates $t$")
    a1.set_ylabel(r"$\mathrm{Var}_{\boldsymbol{\theta}}[\langle Z_i\rangle]$")
    a1.set_xticks(TS)
    a1.set_title(f"DLA dim {dim_mg}" + r"$\,\to\,$" + f"{dim_doped} (1 ZZ gate)",
                 fontsize=8.5)
    a1.legend(loc="lower left", fontsize=8)
    plotting.save(fig, "fig_doping")
    print("doping: done", flush=True)


if __name__ == "__main__":
    main()
