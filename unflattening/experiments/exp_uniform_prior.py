"""uniform_prior -- Lemmas 1 and 2 (deterministic bound + uniform-prior purity).

  (1) Mean: E_Theta[P_g] under the uniform prior matches n-1+2^{-n} (Lemma 2),
      for n up to 100 -- the "cheap sanity check" the manuscript proposes.
  (2) Deterministic range: every sampled (and adversarial) Theta obeys
      n-1 <= P_g <= n (Lemma 1); empirical infimum ~ n-1 (theta_k=pi/2).
  (3) Polar-encoding worked example: isotropic (random-rotation) preconditioning
      turns anisotropic data into near-uniform polar angles -- the prior that
      makes Lemma 2 / Prop. 2 apply.

Figures: uniform_prior_mean, preconditioning_effect.
"""

from __future__ import annotations

import numpy as np

from unflattening.utils import priors
from unflattening.utils.purity import g_purity_closed_form
from unflattening.utils import plotting
from unflattening.utils.plotting import plt, DATA_DIR, GREY_FILL, TEAL, ORANGE, NAVY

N_RANGE = (2, 4, 8, 16, 32, 64, 100)  # qubit counts for the mean check
N_SAMPLES = 4000                      # angle configurations per point
N_POLAR = 32                          # qubits for the polar worked example
ANISO_RHO = 0.97                      # coordinate-pair correlation of the raw data


def analytic_mean(n: int) -> float:
    return n - 1 + 2.0 ** (-n)


def part_mean(rng, ns=N_RANGE, m=N_SAMPLES):
    """E_Theta[P_g] vs n against the Lemma-2 mean n-1+2^{-n}, inside the Lemma-1 band."""
    emp, ana, lo, hi = [], [], [], []
    for n in ns:
        th = priors.sample_uniform(rng, m, n)
        pg = g_purity_closed_form(th)
        emp.append(pg.mean())
        ana.append(analytic_mean(n))
        lo.append(n - 1)  # Lemma 1 proven lower bound
        hi.append(n)      # Lemma 1 upper bound
    ns = np.array(ns, float)
    emp, ana = np.array(emp), np.array(ana)
    print("  mean check (n: empirical vs n-1+2^-n):")
    for n, e, a in zip(ns, emp, ana):
        print(f"    n={int(n):4d}  emp={e:9.4f}  analytic={a:9.4f}  rel.err={abs(e-a)/a:.2e}")

    fig, ax = plt.subplots(figsize=(plotting.COL, 1.9))
    ax.fill_between(ns, lo, hi, color=GREY_FILL, label="Proven Range")
    ax.plot(ns, ana, "-", color=TEAL, label=r"$n-1+2^{-n}$ (analytic mean)")
    ax.plot(ns, emp, "o", color=NAVY, label="Empirical Mean", zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$n$ Qubits"); ax.set_ylabel(r"$P_{\mathfrak{g}}(\rho(\boldsymbol{\phi}))$")
    plotting.top_legend(ax, ncol=2)
    plotting.save(fig, "uniform_prior_mean")
    np.savez(DATA_DIR / "uniform_prior_mean.npz", n=ns, empirical=emp, analytic=ana)


def part_preconditioning(rng, n=N_POLAR, m=N_SAMPLES, rho=ANISO_RHO):
    """Isotropic preconditioning turns anisotropic data into near-uniform polar angles."""
    d = 2 * n
    X = priors.anisotropic_data(rng, m, d, rho=rho)
    raw = priors.polar_angles(X)
    pre = priors.isotropic_precondition(rng, X)
    unif = priors.sample_uniform(rng, m, n)
    pg_raw = g_purity_closed_form(raw).mean()
    pg_pre = g_purity_closed_form(pre).mean()
    pg_uni = g_purity_closed_form(unif).mean()
    print(f"  polar (n={n}): E[P_g] raw={pg_raw:.3f}  preconditioned={pg_pre:.3f}  "
          f"uniform={pg_uni:.3f}  (analytic {analytic_mean(n):.3f})")

    fig, ax = plt.subplots(figsize=(plotting.COL, 1.9))
    bins = np.linspace(0, 2 * np.pi, 40)
    ax.hist(raw.ravel(), bins=bins, density=True, histtype="step", color=ORANGE,
            label="Raw")
    ax.hist(pre.ravel(), bins=bins, density=True, histtype="step", color=TEAL,
            label="Preconditioned")
    ax.set_xlabel(r"Polar Angle $\phi$"); ax.set_ylabel("Density")
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax.set_xticklabels(["$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_ylim(bottom=0)
    plotting.top_legend(ax, ncol=2)
    plotting.save(fig, "preconditioning_effect")


def main() -> None:
    rng = np.random.default_rng(1)
    print("uniform_prior -- Lemmas 1 & 2")
    part_mean(rng)
    part_preconditioning(rng)
    print("uniform_prior: done")


if __name__ == "__main__":
    main()
