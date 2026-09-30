"""Estimate acceptance of the fixed-Q input-purity test.

For each qubit count, hold an anisotropic dataset fixed and sample Haar
rotations Q. Accept a rotation when its dataset-averaged off-diagonal purity
reaches half the uniform-prior mean. Compare the observed rate with the
Markov bound in data/acceptance_rate.csv; this experiment has no figure."""

from __future__ import annotations

import numpy as np
from scipy.stats import special_ortho_group

from unflattening.utils.priors import anisotropic_data, polar_angles
from unflattening.utils.purity import offdiag_closed_form, offdiag_uniform_mean
from unflattening import figures

N_RANGE = (6, 10, 14)   # qubits (feature dimension D = 2n)
N_DATA = 256            # dataset rows m
N_DRAWS = 1000          # Haar draws of Q per n
ANISO_RHO = 0.97        # coordinate-pair correlation of the raw data (cf. fig:polar)


def flat_data(rng: np.random.Generator, m: int, d: int, eps: float = 0.03) -> np.ndarray:
    """Raw features whose coordinate pairs have a negligible second component, so
    the polar angles cluster near {0, pi} -- the barren case of the manuscript.
    (``anisotropic_data`` instead clusters near pi/4, where the purity is fine.)"""
    x = rng.normal(size=(m, d))
    x[:, 1::2] *= eps
    return x


def part_acceptance(rng, ns=N_RANGE, m=N_DATA, draws=N_DRAWS, rho=ANISO_RHO) -> None:
    """Fraction of Haar draws of Q passing the hat(P) >= mu_n/2 acceptance test."""
    rates, raws, flats, means, mus = [], [], [], [], []
    print("acceptance_rate -- fixed-Q test, Markov bound (mu_n/2)/(n-mu_n/2) -> 1/5")
    for n in ns:
        mu = offdiag_uniform_mean(n)
        X = anisotropic_data(rng, m, 2 * n, rho)
        # the raw angles of fig:polar (clustered near pi/4) and the barren
        # ones clustered near {0, pi} that the test has to reject
        pg_raw = float(offdiag_closed_form(polar_angles(X)).mean())
        pg_flat = float(offdiag_closed_form(polar_angles(flat_data(rng, m, 2 * n))).mean())
        # one dataset-averaged purity per Haar draw of Q
        pg = np.array([
            offdiag_closed_form(polar_angles(X @ special_ortho_group.rvs(
                dim=2 * n, random_state=rng).T)).mean()
            for _ in range(draws)
        ])
        rate = float((pg >= mu / 2).mean())
        bound = (mu / 2) / (n - mu / 2)
        rates.append(rate)
        raws.append(pg_raw)
        flats.append(pg_flat)
        means.append(float(pg.mean()))
        mus.append(mu)
        print(f"  n={n:2d}: mu_n={mu:6.3f}  raw E_x[P]={pg_raw:6.3f}  "
              f"clustered={pg_flat:.2e}  E_Q[hat(P)]={pg.mean():6.3f} "
              f"(min {pg.min():.3f})  accept={rate:.3f}  (Markov bound {bound:.3f})")

    ns = np.array(ns, float)
    rates, raws, flats, means, mus = map(np.array, (rates, raws, flats, means, mus))
    # runnable checks: the test passes its own guarantee, and it is not vacuous
    # (angles clustered near {0, pi} sit below the acceptance threshold).
    assert np.all(rates >= mus / 2 / (ns - mus / 2)), "empirical rate must respect Markov"
    assert np.all(flats < mus / 2), "angles clustered near {0, pi} must fail the test"
    assert np.allclose(means, mus, rtol=0.05), "E_Q[hat(P)] must match mu_n (eq:fixedq-mean)"

    figures.write_csv("acceptance_rate",
                      dict(n=ns, accept_rate=rates, pg_raw=raws, pg_clustered=flats,
                           pg_mean=means, mu_n=mus,
                           markov_bound=mus / 2 / (ns - mus / 2), m=m, draws=draws))
    print(f"acceptance_rate: min rate {rates.min():.3f} over n={list(ns.astype(int))}, done")


def main() -> None:
    part_acceptance(np.random.default_rng(11))


if __name__ == "__main__":
    main()
