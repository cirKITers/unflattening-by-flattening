"""Test matchgate output-variance convergence and qubit-count scaling.

Sample random matchgate parameters for RY product inputs. At fixed qubit
count, test convergence toward Var_W[<Z_i>] = P_g/dim g as depth grows.
Across qubit counts, test the predicted 1/n scaling. The two sweeps produce
matchgate_convergence and matchgate_scaling."""

from __future__ import annotations

import jax
import numpy as np

from qml_essentials.ansaetze import Ansaetze
from qml_essentials import trainability
from unflattening import figures
from unflattening.utils.purity import analytic_loss_variance

N_SAMPLES = 2000
CONV_RANGE = (4, 6, 8)                        # qubit counts for the convergence panel
CONV_DEPTHS = (1, 2, 4, 8, 16, 24, 32)        # brickwork depths swept at fixed n
SCALING_RANGE = tuple(range(2, 13))           # qubit sweep for the scaling panel
SCALING_DEPTH_FACTOR = 6                      # depth = SCALING_DEPTH_FACTOR * n


def part_convergence(rng, key, ns=CONV_RANGE, depths=CONV_DEPTHS, n_samples=N_SAMPLES):
    """Var_W -> P_g/dim g as the brickwork depth grows, at fixed n."""
    col_n, col_depth, col_var, col_pred = [], [], [], []
    print("  convergence Var_W -> P_g/dim_g:")
    for n in ns:
        theta = rng.uniform(0.0, 2 * np.pi, n)
        pred = analytic_loss_variance(theta)
        vs = []
        for d in depths:
            key, sub = jax.random.split(key)
            v, _ = trainability.loss_variance(
                Ansaetze.Matchgate.build, Ansaetze.Matchgate.n_params_per_layer(n),
                theta, d, n_samples, sub
            )
            vs.append(v)
        print(f"    n={n}: pred={pred:.5f}  Var(depth={depths[-1]})={vs[-1]:.5f}")
        col_n += [n] * len(depths)          # long format, n-major: one group per n
        col_depth += list(depths)
        col_var += vs
        col_pred += [pred] * len(depths)    # the analytic level is per n

    figures.write_csv("matchgate_convergence",
                      dict(depth=col_depth, n=col_n, var=col_var, pred=col_pred))
    figures.fig_matchgate_convergence()


def part_scaling(rng, key, ns=SCALING_RANGE, n_samples=N_SAMPLES,
                 depth_factor=SCALING_DEPTH_FACTOR):
    """Var_W vs n at converged depth: tracks P_g/dim g = Theta(1/n)."""
    emp, pred, lo, hi = [], [], [], []
    print(f"  scaling Var_W vs n (depth={depth_factor}n):")
    for n in ns:
        theta = rng.uniform(0.0, 2 * np.pi, n)
        depth = depth_factor * n
        key, sub = jax.random.split(key)
        v, _ = trainability.loss_variance(
            Ansaetze.Matchgate.build, Ansaetze.Matchgate.n_params_per_layer(n),
            theta, depth, n_samples, sub
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

    figures.write_csv("matchgate_scaling", dict(n=ns, empirical=emp, analytic=pred,
                                               lo=lo, hi=hi, slope=slope))
    figures.fig_matchgate_scaling()


def main() -> None:
    rng = np.random.default_rng(2)
    key = jax.random.PRNGKey(2)
    print("theorem1 -- matchgate pipeline Var_W = Theta(1/n)")
    part_convergence(rng, key)
    part_scaling(rng, jax.random.PRNGKey(7))
    print("theorem1: done")


if __name__ == "__main__":
    main()
