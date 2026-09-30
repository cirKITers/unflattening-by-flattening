"""Check uniform-prior matchgate purity and polar preconditioning.

Check the deterministic band n-1 <= P_g <= n and the uniform-prior mean
n-1+2^-n. Then rotate anisotropic data once and measure the resulting
polar-angle distribution. The outputs are uniform_prior_mean and
preconditioning_effect."""

from __future__ import annotations

import numpy as np

from unflattening import figures
from unflattening.utils import priors
from unflattening.utils.purity import g_purity_closed_form

N_RANGE = tuple(range(2, 19))         # qubit counts for the mean check (matches the other n sweeps)
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

    figures.write_csv("uniform_prior_mean", dict(n=ns, empirical=emp, analytic=ana,
                                                lo=lo, hi=hi))
    figures.fig_uniform_prior_mean()


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

    # persist the binned densities (what the figure's step outline actually draws)
    # rather than the 2 x m*n raw angles.
    bins = np.linspace(0, 2 * np.pi, 40)
    dens_raw, _ = np.histogram(raw.ravel(), bins=bins, density=True)
    dens_pre, _ = np.histogram(pre.ravel(), bins=bins, density=True)
    figures.write_csv("preconditioning_effect",
                      dict(bin_left=bins[:-1], bin_right=bins[1:],
                           density_raw=dens_raw, density_pre=dens_pre))
    figures.fig_preconditioning_effect()


def main() -> None:
    rng = np.random.default_rng(1)
    print("uniform_prior -- Lemmas 1 & 2")
    part_mean(rng)
    part_preconditioning(rng)
    print("uniform_prior: done")


if __name__ == "__main__":
    main()
