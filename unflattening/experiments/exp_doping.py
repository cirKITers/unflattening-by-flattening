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

Figures: doping_variance (left: Var vs n per t; right: fixed-n decay Var ~ c^{-t}).
"""

from __future__ import annotations

import jax
import numpy as np

from unflattening import figures
from unflattening.utils import doping, dla
from unflattening.utils.purity import analytic_loss_variance

N_SAMPLES = 600
TS = (0, 1, 2, 3, 4)           # non-Gaussian gate budgets
N_RANGE = tuple(range(3, 10))  # exact statevector; n<=9 keeps runtime modest
N_REP = 6                      # representative n for the DLA closure blow-up


def part_doping(rng, key, ns=N_RANGE, ts=TS, n_samples=N_SAMPLES, n_rep=N_REP) -> None:
    """Var_W vs n per doping budget t, and the fixed-n decay Var ~ c^{-t}."""
    var = {t: [] for t in ts}
    pred0 = []  # t=0 analytic P_g/dim g (Theorem 1 check)
    print("doping -- non-Gaussian (RZZ) doping of the matchgate LASA")
    for n in ns:
        theta = rng.uniform(0.0, 2 * np.pi, n)
        depth = max(16, 4 * n)
        pred0.append(analytic_loss_variance(theta))
        row = []
        for t in ts:
            key, k = jax.random.split(key)
            v, _ = doping.doped_loss_variance(theta, depth, t, n_samples, k)
            var[t].append(v)
            row.append(v)
        print(f"  n={n:2d}: " + "  ".join(f"t={t}:{v:.3e}" for t, v in zip(ts, row))
              + f"   (t=0 pred {pred0[-1]:.3e})", flush=True)

    ns = np.array(ns, float)
    for t in ts:
        var[t] = np.array(var[t])
    # log-log decay slope per fixed t (more negative = closer to barren).
    slope = {t: float(np.polyfit(np.log(ns), np.log(var[t]), 1)[0]) for t in ts}
    print("  log-log slopes: " + ", ".join(f"t={t}:{slope[t]:.2f}" for t in ts))

    # DLA closure blow-up (one representative n), reported in the caption/inset.
    pos1 = doping.doping_positions(n_rep, max(16, 4 * n_rep), 1)
    dim_mg = dla.dim_g(n_rep)
    dim_doped = len(dla.lie_closure(
        dla.matchgate_generators(n_rep) + doping.doping_generators(n_rep, pos1)))
    print(f"  DLA dim at n={n_rep}: matchgate {dim_mg} -> doped {dim_doped}")

    # per-fixed-n exponential-in-t rate: Var ~ c^{-t} (Eq. doping-decay).
    rate = np.array([np.polyfit(ts, np.log([var[t][i] for t in ts]), 1)[0]
                     for i in range(len(ns))])
    c_fit = float(np.exp(-rate.mean()))
    print(f"  fixed-n decay Var ~ c^-t: c ~ {c_fit:.3f} (mean over n)", flush=True)

    # save raw data BEFORE plotting so a backend hiccup cannot lose the run.
    figures.write_csv("doping_variance", dict(n=ns, pred0=np.array(pred0),
                      **{f"var_t{t}": var[t] for t in ts},
                      c_fit=c_fit, dim_mg=dim_mg, dim_doped=dim_doped))
    figures.write_csv("doping_variance_fits",
                      dict(t=np.array(ts), slope=np.array([slope[t] for t in ts])))

    figures.fig_doping_variance()
    print("doping: done", flush=True)


def main() -> None:
    rng = np.random.default_rng(7)
    part_doping(rng, jax.random.PRNGKey(23))


if __name__ == "__main__":
    main()
