"""offdiag_closedform

The off-diagonal DLA g = <{X_k X_{k+1}, Y_k Y_{k+1}}> ~= so(n) (+) so(n) has no
single-qubit Z, hence no deterministic purity floor.  For an R_y product state the
input g-purity has the closed form

    P_g(rho(Theta)) = sum_{j<k, k-j odd} sin^2 t_j sin^2 t_k prod_{j<l<k} cos^2 t_l ,

i.e. the matchgate cross-sum (Prop. 1) restricted to opposite-parity endpoints, with
the on-site sum cos^2 floor term removed.  Three checks:
  * closed form == brute-force g_purity_from_basis over lie_closure_paulis(...)
  * uniform prior keeps E[P_g] = Omega(n) (-> n/3) while a prior clustered near {0,pi}
    drives P_g -> 0 (barren)
  * the two-ideal readout factor Var_theta = 2 P_g/dim g (sec:proofs) against deep
    random e^{g_od} statevector circuits at n = 6 (part_variance).

Figures: offdiag_purity (mean P_g vs n, uniform vs clustered, with the closed-form
mean overlaid and brute-force-basis markers) and purity_regime_contrast (matchgate
floor band vs off-diagonal collapse as the angles cluster).
"""
from __future__ import annotations

import numpy as np

from unflattening import figures
from unflattening.utils.dla import lie_closure_paulis, matchgate_basis, xx_yy_generators
from unflattening.utils.purity import (product_state, g_purity_from_basis,
                                       g_purity_closed_form, offdiag_closed_form,
                                       offdiag_uniform_mean)
from unflattening.utils.priors import sample_uniform, sample_raw

N_VALID = 6        # closed-form vs basis brute force up to here
N_FIG = 18         # closed-form mean curve up to here
N_BRUTE = 10       # brute-force basis markers up to here
N_RANDOM = 50      # random configs per n in the validation
N_SAMPLES_PRIOR = 4000      # samples for the prior means
N_SAMPLES_BRUTE = 4000      # samples for the brute-force markers
RAW_EPS = 0.03     # spread of the clustered angles around {0, pi}
TOL = 1e-10
N_REGIME = 10      # qubits for the regime contrast
SIGMA_POINTS = 24  # angle-spread grid points in the regime contrast
N_VARIANCE = 6     # qubits for the statevector variance check
M_VARIANCE = 2000  # random circuits in the statevector variance check
DEPTH_VARIANCE = 2000  # rotations per random circuit


def part_validate(rng: np.random.Generator, n_max: int = N_VALID,
                  n_random: int = N_RANDOM, tol: float = TOL) -> None:
    """Closed form == brute-force lie-closure basis, and dim g = n(n-1)."""
    ok = True
    for n in range(2, n_max + 1):
        basis = lie_closure_paulis(xx_yy_generators(n))
        dim_ok = len(basis) == n * (n - 1)
        max_diff = 0.0
        for _ in range(n_random):
            theta = rng.uniform(0.0, 2 * np.pi, n)
            p_closed = float(offdiag_closed_form(theta))
            p_basis = float(g_purity_from_basis(product_state(theta), basis))
            max_diff = max(max_diff, abs(p_closed - p_basis))
        passed = dim_ok and max_diff < tol
        ok = ok and passed
        print(f"  n={n}: dim g={len(basis)} (=n(n-1)? {dim_ok})  "
              f"max|closed-basis|={max_diff:.2e}  {'PASS' if passed else 'FAIL'}")
    assert ok, "off-diagonal closed form must equal the brute-force DLA-basis purity, and dim g = n(n-1)"


def diagonal_count(basis, axis: str = "Z") -> int:
    """Number of Pauli strings diagonal in the axis-eigenbasis (letters in {I, axis}) = d_A(g).
    A Pauli string commutes with every A_k iff each factor is I or A, so this counts d_A."""
    return sum(1 for B in basis
               if set(B if isinstance(B, str) else B.to_pauli_string()) <= {"I", axis})


def part_criterion(n_max: int = N_VALID) -> None:
    """Covariant floor criterion P_g(clustered) = d_A(g), for the clustering observable A.
    A=Z (R_y, |0...0>): off-diag d_Z=0 (floor-free), matchgate d_Z=n.
    A=X (phase enc, |+...+>): off-diag d_X=n-1 (FLOORED) -- floor-freeness is a joint property
    of the encoding and the algebra, not the algebra alone."""
    ok = True
    for n in range(2, n_max + 1):
        offdiag, mg = lie_closure_paulis(xx_yy_generators(n)), matchgate_basis(n)
        zero = product_state(np.zeros(n))            # |0...0>, A=Z clustering (R_y poles)
        plus = product_state(np.full(n, np.pi / 2))  # |+...+>, A=X clustering (phase-encoding poles)
        cases = [(offdiag, zero, "Z", 0), (offdiag, plus, "X", n - 1),
                 (mg, zero, "Z", n), (mg, plus, "X", n - 1)]
        passed = True
        for basis, state, ax, expect in cases:
            dA = diagonal_count(basis, ax)
            pg = float(g_purity_from_basis(state, basis))
            passed = passed and dA == expect and abs(pg - expect) < 1e-9
        ok = ok and passed
        print(f"  n={n}: off-diag d_Z={diagonal_count(offdiag,'Z')} (floor-free) "
              f"d_X={diagonal_count(offdiag,'X')} (floored) | "
              f"matchgate d_Z={diagonal_count(mg,'Z')} d_X={diagonal_count(mg,'X')}  "
              f"{'PASS' if passed else 'FAIL'}")
    assert ok, "covariant floor criterion: P_g(clustered)=d_A(g) for A in {Z (|0>), X (|+>)}"


def part_purity(rng: np.random.Generator, n_max: int = N_FIG, n_brute: int = N_BRUTE,
                n_samples: int = N_SAMPLES_PRIOR, n_samples_brute: int = N_SAMPLES_BRUTE,
                raw_eps: float = RAW_EPS) -> None:
    """Mean P_g vs n: uniform (Omega(n)) vs clustered (-> 0); closed form vs basis."""
    ns = np.arange(2, n_max + 1)
    mean_unif, mean_clus, ana = [], [], []
    for n in ns:
        mean_unif.append(float(offdiag_closed_form(sample_uniform(rng, n_samples, n)).mean()))
        mean_clus.append(float(offdiag_closed_form(sample_raw(rng, n_samples, n, raw_eps)).mean()))
        ana.append(offdiag_uniform_mean(int(n)))
    mean_unif, mean_clus, ana = np.array(mean_unif), np.array(mean_clus), np.array(ana)

    # brute-force-basis means at small n, to show the closed form == the actual DLA basis
    ns_b = np.arange(2, n_brute + 1)
    mean_brute = []
    for n in ns_b:
        basis = lie_closure_paulis(xx_yy_generators(int(n)))
        thetas = sample_uniform(rng, n_samples_brute, int(n))
        mean_brute.append(float(np.mean([g_purity_from_basis(product_state(t), basis) for t in thetas])))
    mean_brute = np.array(mean_brute)

    dim = ns * (ns - 1)                                    # dim g_od = n(n-1)
    # Var_theta = 2 P_g/dim g: eq (ragone) sums over the two n(n-1)/2-dim ideals
    # of so(n)+so(n) and the XX+YY readout puts one basis string in each ideal, so
    # Var = P_A/(dim/2) + P_B/(dim/2) = 2 P_g/dim g regardless of the P split
    # (validated against statevector simulation in part_variance).
    var_unif, var_clus = 2 * mean_unif / dim, 2 * mean_clus / dim

    # runnable checks: closed form tracks analytic mean + brute-force basis; uniform purity grows
    # Omega(n) yet its variance FALLS as Theta(1/n) (purity up, variance down); clustered collapses.
    assert np.max(np.abs(mean_unif - ana)) < 0.05 * ana[-1], "closed-form mean must track E[P_g]"
    assert np.max(np.abs(mean_brute - ana[: len(ns_b)])) < 0.1 * ana[len(ns_b) - 1], \
        "brute-force basis mean must match the closed-form mean"
    assert mean_unif[-1] > mean_unif[0] * 1.5, "uniform E[P_g] must grow (Omega(n))"
    assert mean_clus[-1] < 0.01 * mean_unif[-1], "clustered prior must collapse far below uniform"
    assert var_unif[-1] < var_unif[0], "uniform Var_W must fall (purity up, variance down ~ Theta(1/n))"

    # ns_b is the leading prefix of ns, so the brute-force column is simply shorter
    # than the rest of the table (tail-padded in the CSV).
    figures.write_csv("offdiag_purity",
                      dict(n=ns, uniform=mean_unif, clustered=mean_clus, analytic=ana,
                           var_uniform=var_unif, var_clustered=var_clus, brute=mean_brute))
    figures.fig_offdiag_purity()


def part_regime(rng: np.random.Generator, n: int = N_REGIME,
                n_samples: int = N_SAMPLES_PRIOR, sigma_points: int = SIGMA_POINTS) -> None:
    """Regime contrast supporting the floor criterion: as the R_y angles cluster at {0,pi},
    the matchgate purity stays in its floor band [n-1, n] (distribution-insensitive, d_Z=n)
    while the off-diagonal purity collapses toward 0 (distribution-decides, d_Z=0)."""
    sigmas = np.logspace(np.log10(0.03), np.log10(np.pi), sigma_points)  # clustered -> ~uniform
    mg = np.array([float(g_purity_closed_form(sample_raw(rng, n_samples, n, s)).mean()) for s in sigmas])
    od = np.array([float(offdiag_closed_form(sample_raw(rng, n_samples, n, s)).mean()) for s in sigmas])
    # Var_theta = 2 P_g/dim g for the in-algebra XX+YY readout (same as part_purity /
    # fig:offdiag): so(2n) and so(n)(+)so(n) each carry two readout strings, so both use 2P/dim.
    dim_mg, dim_od = n * (2 * n - 1), n * (n - 1)
    var_mg, var_od = 2 * mg / dim_mg, 2 * od / dim_od
    # runnable checks: matchgate stays in the proven floor band [n-1, n] across every spread
    # (distribution-insensitive), while the off-diagonal collapses when the angles cluster;
    # the readout variance inherits the same behaviour (matchgate flat/floored, off-diag collapses).
    assert mg.min() > n - 1 - 1e-6 and mg.max() < n + 1e-6, "matchgate purity must stay in [n-1, n]"
    assert od[0] < 0.05 * od[-1], "off-diagonal purity must collapse under clustering (floor-free)"
    assert var_mg.min() > 0.5 * var_mg.max(), "matchgate variance stays flat (floored)"
    assert var_od[0] < 0.05 * var_od[-1], "off-diagonal variance collapses (floor-free)"

    figures.write_csv("purity_regime_contrast",
                      dict(sigma=sigmas, matchgate=mg, offdiag=od,
                           var_matchgate=var_mg, var_offdiag=var_od, n=n))
    figures.fig_purity_regime_contrast()


def part_variance(rng: np.random.Generator, n: int = N_VARIANCE, M: int = M_VARIANCE,
                  depth: int = DEPTH_VARIANCE) -> None:
    """Statevector check of the two-ideal readout factor (sec:proofs): deep random
    e^{g_od} circuits with the in-algebra readout O = X_i X_{i+1} + Y_i Y_{i+1}
    give Var_theta = 2 P_g/dim g, i.e. ~2x the naive single-ideal P_g/dim g."""
    from unflattening.utils.dla import pauli_to_bitmasks, word_matrix, random_dla_variance

    i = n // 2 - 1  # bulk bond (i, i+1), as in exp_reuploading/exp_precondition
    gen_mats = [word_matrix(*pauli_to_bitmasks(g), n) for g in xx_yy_generators(n)]
    O = word_matrix(*pauli_to_bitmasks("I" * i + "XX" + "I" * (n - i - 2)), n) \
        + word_matrix(*pauli_to_bitmasks("I" * i + "YY" + "I" * (n - i - 2)), n)
    dim = n * (n - 1)
    rows = []
    for label in ("uniform-1", "uniform-2"):
        theta = rng.uniform(0.0, 2 * np.pi, n)
        pg = float(offdiag_closed_form(theta))
        psi0 = product_state(theta)
        emp = random_dla_variance(psi0, gen_mats, O, depth, M, rng)
        rows.append((label, 2 * pg / dim, pg / dim, emp))
        print(f"    {label:10s} 2P/dim={2 * pg / dim:.3e}  P/dim={pg / dim:.3e}  "
              f"empirical={emp:.3e}  ratio(2P/dim)={emp / (2 * pg / dim):.2f}  "
              f"ratio(P/dim)={emp / (pg / dim):.2f}", flush=True)
    assert all(0.8 < e / p2 < 1.2 for _, p2, _, e in rows), \
        "empirical Var_theta must match the two-ideal 2 P_g/dim g"
    figures.write_csv("offdiag_variance",
                      dict(label=[r[0] for r in rows], two_ideal=[r[1] for r in rows],
                           single_ideal=[r[2] for r in rows], empirical=[r[3] for r in rows],
                           n=n, M=M, depth=depth))


def main() -> None:
    rng = np.random.default_rng(0)
    print("offdiag_closedform -- off-diagonal P_g: closed form vs basis, uniform vs clustered")
    part_validate(rng)
    part_criterion()
    part_regime(rng)   # before part_purity: shared rng, draw order is load-bearing
    part_purity(rng)
    print(f"  two-ideal variance factor vs statevector, g_od at n={N_VARIANCE}:")
    part_variance(np.random.default_rng(1))  # own rng: keeps the figure data unchanged
    print("offdiag_closedform: done")


if __name__ == "__main__":
    main()
