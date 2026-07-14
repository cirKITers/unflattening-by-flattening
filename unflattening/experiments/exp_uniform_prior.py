"""uniform_prior -- Lemmas 1 and 2 (deterministic bound + uniform-prior purity).

  (1) Mean: E_Theta[P_g] under the uniform prior matches n-1+2^{-n} (Lemma 2),
      for n up to 100 -- the "cheap sanity check" the manuscript proposes.
  (2) Deterministic range: every sampled (and adversarial) Theta obeys
      n-1 <= P_g <= n (Lemma 1); empirical infimum ~ n-1 (theta_k=pi/2).
  (3) Concentration: P_g sits in a narrow band near n-1, inside the proven
      [n-1, n] range; the upper tail obeys Markov, Pr[P_g>=n-1+tau]<=2^{-n}/tau (Lemma 2).
  (4) Polar-encoding worked example: isotropic (random-rotation) preconditioning
      turns anisotropic data into near-uniform polar angles -- the prior that
      makes Lemma 2 / Prop. 2 apply.

Figures: fig_mean, fig_concentration, fig_polar.
"""

from __future__ import annotations

import numpy as np

from unflattening.utils import priors
from unflattening.utils.purity import g_purity_closed_form
from unflattening.utils import plotting
from unflattening.utils.plotting import plt, DATA_DIR, GREY_FILL, GREY_REF, TEAL, ORANGE, ACCENT, NAVY


def analytic_mean(n: int) -> float:
    return n - 1 + 2.0 ** (-n)


def part_mean(rng):
    ns = [2, 4, 8, 16, 32, 64, 100]
    m = 4000
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
    plotting.save(fig, "fig_mean")
    np.savez(DATA_DIR / "mean.npz", n=ns, empirical=emp, analytic=ana)


def part_concentration(rng):
    # The uniform prior pins P_g essentially at the floor n-1 with a tiny spread
    # (Lemma 2): the deviation D = P_g - (n-1) = prod_k cos^2(theta_k) lies in
    # [0,1] with mean 2^{-n}, so the distribution at small n sits in a narrow band
    # inside the proven [n-1, n], with an exponentially small Markov upper tail.
    n_hist, m_hist = 6, 60000
    pg = g_purity_closed_form(priors.sample_uniform(rng, m_hist, n_hist))
    lb_det = n_hist - 1
    # Markov upper tail on the deviation D = P_g - (n-1): E[D] = 2^{-n}, hence
    # Pr[P_g >= n-1 + tau] <= 2^{-n}/tau (Lemma 2).
    tau = 2.0 ** (-n_hist / 2)
    emp_tail = float(np.mean(pg - lb_det >= tau))
    markov = 2.0 ** (-n_hist) / tau
    print(f"  concentration n={n_hist}: mean={pg.mean():.4f} std={pg.std():.4f} "
          f"min={pg.min():.4f} max={pg.max():.4f}")
    print(f"    proven range [n-1, n]=[{lb_det:.1f},{n_hist}]; "
          f"all in range: {bool(pg.min() >= lb_det and pg.max() <= n_hist)}; "
          f"empirical inf ~ n-1={n_hist - 1}: min>={n_hist - 1}? {bool(pg.min() >= n_hist - 1 - 1e-9)}; "
          f"Markov Pr[P_g>=n-1+{tau:.3g}]={emp_tail:.2e}<={markov:.2e}")

    # Broken x-axis: zoom on the squeezed distribution near the floor n-1 (left)
    # and show the distant upper bound n (right), eliding the empty middle with a
    # // break -- this emphasises how tightly P_g concentrates inside [n-1, n].
    hi_left = (n_hist - 1) + 0.10  # zoom tightly on the concentrated bulk
    fig, (axL, axR) = plt.subplots(
        1, 2, sharey=True, figsize=(plotting.COL, 1.9),
        gridspec_kw={"width_ratios": [6, 1], "wspace": 0.07},
    )
    axL.hist(pg, bins=np.linspace(n_hist - 1, n_hist, 200), density=True,
             color=TEAL, alpha=0.85)
    axL.axvline(analytic_mean(n_hist), color=NAVY, lw=1.3, label=r"$n-1+2^{-n}$")
    axL.axvline(n_hist - 1, color=ACCENT, ls="--", lw=1.1, label=r"$n-1$ (Emp. Inf)")
    axR.axvline(n_hist, color=GREY_REF, ls=":", lw=1.3, label=r"$n$ (UB)")
    axL.set_xlim(n_hist - 1 - 0.008, hi_left)  # a little headroom so the n-1 line clears the spine
    axR.set_xlim(n_hist - 0.03, n_hist + 0.01)
    axR.set_xticks([n_hist])
    # hide the facing spines and stitch the gap with diagonal break markers
    axL.spines["right"].set_visible(False)
    axR.spines["left"].set_visible(False)
    axR.tick_params(axis="y", length=0)
    d = 0.5
    mk = dict(marker=[(-1, -d), (1, d)], markersize=8, linestyle="none",
              color="k", mec="k", mew=1, clip_on=False)
    axL.plot([1, 1], [0, 1], transform=axL.transAxes, **mk)
    axR.plot([0, 0], [0, 1], transform=axR.transAxes, **mk)
    axL.set_ylabel("Density")
    fig.supxlabel(r"$P_{\mathfrak{g}}(\rho(\boldsymbol{\phi}))$, $n=6$", fontsize=9)
    # broken-axis panel: keep the legend inside (no room on top); the global
    # rcParams already make it frameless.
    hL, lL = axL.get_legend_handles_labels()
    hR, lR = axR.get_legend_handles_labels()
    axL.legend(hL + hR, lL + lR, loc="upper right", fontsize=8)
    plotting.save(fig, "fig_concentration")


def part_polar(rng):
    n, m = 32, 4000
    d = 2 * n
    X = priors.anisotropic_data(rng, m, d, rho=0.97)
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
    plotting.save(fig, "fig_polar")


def main() -> None:
    rng = np.random.default_rng(1)
    print("uniform_prior -- Lemmas 1 & 2")
    part_mean(rng)
    part_concentration(rng)
    part_polar(rng)
    print("uniform_prior: done")


if __name__ == "__main__":
    main()
