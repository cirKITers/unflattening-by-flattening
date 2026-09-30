"""Render the experiment CSV files as PGF figures and PNG previews.

Experiments write numeric results to data/. Run this module to redraw figures
from those CSV files without rerunning the experiments. CSV columns may contain
repeated scalars or trailing empty cells for shorter series."""

from __future__ import annotations

import csv
import sys

import numpy as np
from matplotlib.ticker import LogFormatterSciNotation

from unflattening.utils import plotting
from unflattening.utils.plotting import (plt, DATA_DIR, GREY_FILL, GREY_REF,
                                         TEAL, ORANGE, ACCENT, NAVY)

# Figure-only styling for the encoding landscapes and means.
ENC_KINDS = ("hamming", "binary", "ternary")
ENC_COLOR = {"hamming": ORANGE, "binary": TEAL, "ternary": ACCENT}
_LSTYLE = {  # ternary spans 3^n freqs: thin/faint/behind, hamming on top
    "hamming": dict(alpha=0.9, lw=0.9, zorder=3),
    "binary": dict(alpha=0.8, lw=0.8, zorder=2),
    "ternary": dict(alpha=0.45, lw=0.4, zorder=1),
}


# ---- CSV I/O ----------------------------------------------------------------

def _fmt(v) -> str:
    """One cell.  ``repr`` of the native Python type round-trips floats exactly."""
    if isinstance(v, str):
        return v
    if isinstance(v, (bool, np.bool_)):
        return repr(bool(v))
    if isinstance(v, (int, np.integer)):
        return repr(int(v))
    return repr(float(v))


def write_csv(stem: str, cols: dict) -> None:
    """Write columns to data/, repeating scalars and padding short columns."""
    cells, n_rows = {}, 0
    for key, val in cols.items():
        if isinstance(val, np.ndarray) and val.ndim == 0:
            val = val.item()
        if isinstance(val, np.ndarray):
            val = val.tolist()
        if isinstance(val, (list, tuple)):
            cells[key] = [_fmt(v) for v in val]
            n_rows = max(n_rows, len(cells[key]))
        else:
            cells[key] = _fmt(val)  # scalar -> broadcast below
    with open(DATA_DIR / f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cells.keys())
        for i in range(n_rows):
            w.writerow([c if isinstance(c, str) else (c[i] if i < len(c) else "")
                        for c in cells.values()])
    print(f"  wrote data/{stem}.csv")


def read_csv(stem: str) -> dict:
    """Read data/<stem>.csv, dropping trailing empty cells in each column."""
    with open(DATA_DIR / f"{stem}.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    out = {}
    for j, name in enumerate(header):
        col = [r[j] for r in body]
        while col and col[-1] == "":
            col.pop()
        try:
            out[name] = np.array([float(v) for v in col])
        except ValueError:
            out[name] = np.array(col, dtype=object)
    return out


# ---- figures ----------------------------------------------------------------

def fig_dla_regime_contrast() -> None:
    d = read_csv("dla_regime_contrast")
    ns, var_mg, var_ge = d["n"], d["matchgate"], d["generic"]
    exp_rate = float(d["exp_rate"][0])

    fig, ax = plt.subplots(figsize=(plotting.COL, 2.6))
    # Same input rho(Theta) and observable Z_i; only the ansatz DLA differs.
    # matchgate hugs the 1/n guide (polynomial); the full-DLA ansatz follows the
    # exponential fit -- the conditional reading of Eq. (ragone) in one panel.
    ax.plot(ns, var_mg, "o-", color=TEAL, label="MGA (Poly DLA)")
    ax.plot(ns, var_ge, "s-", color=ORANGE, label="SEA (Full DLA)")
    ax.plot(ns, var_mg[0] * ns[0] / ns, ":", color=GREY_REF, label=r"$\propto 1/n$")
    ax.plot(ns, np.exp(np.polyval([exp_rate, np.log(var_ge[0]) - exp_rate * ns[0]], ns)),
            "--", color=ORANGE, lw=0.9, label=f"$\\propto e^{{{exp_rate:.2f}\\,n}}$")
    ax.set_yscale("log")
    ax.set_xlabel("$n$ Qubits"); ax.set_ylabel(r"$\mathrm{Var}_{\boldsymbol{\theta}}[\langle Z_i\rangle]$")
    ax.locator_params(axis="x", integer=True)
    plotting.top_legend(ax, ncol=2)
    plotting.save(fig, "dla_regime_contrast")


def fig_matchgate_convergence() -> None:
    d = read_csv("matchgate_convergence")
    series = (TEAL, ACCENT, NAVY)  # all trainable: cool gradient, no barren gold
    fig, ax = plt.subplots(figsize=(plotting.COL, 2.6))
    for j, n in enumerate(dict.fromkeys(d["n"].tolist())):  # groups in file order
        m = d["n"] == n
        ax.plot(d["depth"][m], d["var"][m], "o-", color=series[j], label=f"$n={int(n)}$")
        ax.axhline(float(d["pred"][m][0]), color=series[j], ls="--", lw=1.0,
                   label=(r"$\mathcal{P}_{\mathfrak{g}}/\dim\mathfrak{g}$" if j == 0 else None))
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("MGA Depth")
    ax.set_ylabel(r"$\mathrm{Var}_{\boldsymbol{\theta}}[\langle Z_i\rangle]$")
    # legend with the analytic entry first
    handles, labels = ax.get_legend_handles_labels()
    ai = labels.index(r"$\mathcal{P}_{\mathfrak{g}}/\dim\mathfrak{g}$")
    order = [ai] + [i for i in range(len(labels)) if i != ai]
    plotting.top_legend(ax, [handles[i] for i in order], [labels[i] for i in order])
    plotting.save(fig, "matchgate_convergence")


def fig_matchgate_scaling() -> None:
    d = read_csv("matchgate_scaling")
    ns, emp, pred, lo, hi = d["n"], d["empirical"], d["analytic"], d["lo"], d["hi"]

    fig, ax = plt.subplots(figsize=(plotting.COL, 2.6))
    ax.fill_between(ns, lo, hi, color=GREY_FILL, label="Proven Range")
    ax.plot(ns, pred, "-", color=TEAL, label=r"$\mathcal{P}_{\mathfrak{g}}/\dim\mathfrak{g}$")
    ax.plot(ns, emp, "o", color=NAVY, label=r"$\mathrm{Var}_{\boldsymbol{\theta}}$", zorder=5)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    # match the y-ticks of matchgate_convergence: label only {6e-2, 1e-1, 2e-1, 3e-1}
    ax.set_yticks([6e-2, 1e-1, 2e-1, 3e-1])
    ax.set_yticks([], minor=True)
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    ax.set_xlabel("$n$ Qubits")
    ax.set_ylabel(r"$\mathrm{Var}_{\boldsymbol{\theta}}[\langle Z_i\rangle]$")
    plotting.top_legend(ax, ncol=3)
    plotting.save(fig, "matchgate_scaling")


def fig_uniform_prior_mean() -> None:
    d = read_csv("uniform_prior_mean")
    ns, emp, ana, lo, hi = d["n"], d["empirical"], d["analytic"], d["lo"], d["hi"]

    fig, ax = plt.subplots(figsize=(plotting.COL, 1.9))
    ax.fill_between(ns, lo, hi, color=GREY_FILL, label="Proven Range")
    ax.plot(ns, ana, "-", color=TEAL, label=r"$n-1+2^{-n}$ (analytic mean)")
    ax.plot(ns, emp, "o", color=NAVY, label="Empirical Mean", zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("$n$ Qubits"); ax.set_ylabel(r"$\mathcal{P}_{\mathfrak{g}}(\rho(\boldsymbol{\phi}))$")
    plotting.top_legend(ax, ncol=2)
    plotting.save(fig, "uniform_prior_mean")


def fig_preconditioning_effect() -> None:
    d = read_csv("preconditioning_effect")
    # replay the stored densities as bin weights: same bin edges, same heights,
    # hence the same step outline as the original density histogram.
    bins = np.append(d["bin_left"], d["bin_right"][-1])

    fig, ax = plt.subplots(figsize=(plotting.COL, 1.9))
    ax.hist(d["bin_left"], bins=bins, weights=d["density_raw"], histtype="step",
            color=ORANGE, label="Raw")
    ax.hist(d["bin_left"], bins=bins, weights=d["density_pre"], histtype="step",
            color=TEAL, label="Preconditioned")
    ax.set_xlabel(r"Polar Angle $\phi$"); ax.set_ylabel("Density")
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax.set_xticklabels(["$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_ylim(bottom=0)
    plotting.top_legend(ax, ncol=2)
    plotting.save(fig, "preconditioning_effect")


def fig_input_purity_scaling() -> None:
    d = read_csv("input_purity_scaling")
    ns, var_prod, var_haar = d["n"], d["var_product"], d["var_haar"]

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
    plotting.save(fig, "input_purity_scaling")


def fig_doping_variance() -> None:
    d = read_csv("doping_variance")
    ns = d["n"]
    ts = [int(k[len("var_t"):]) for k in d if k.startswith("var_t")]
    var = {t: d[f"var_t{t}"] for t in ts}
    c_fit = float(d["c_fit"][0])
    dim_mg, dim_doped = int(d["dim_mg"][0]), int(d["dim_doped"][0])

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(plotting.WIDE, 2.7))
    # (a) Var vs n per fixed t; t=0 is the matchgate baseline ~ 1/n.
    # sequential cool ramp keyed to the ordinal doping count t (avoids the
    # BAD=orange bleed and the colour reuse of a wrapping categorical cycle).
    tcolors = plotting.ordinal_colors(len(ts))
    for j, t in enumerate(ts):
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
        a1.semilogy(ts, [var[t][i] for t in ts], "o-", label=f"$n={k}$")
    a1.semilogy(ts, var[0][idx.index(reps[-1])] * c_fit ** (-np.array(ts, float)),
                "k--", lw=1.0, label=rf"$\propto c^{{-t}}$ ($c\approx{c_fit:.2f}$)")
    a1.set_xlabel("non-Gaussian gates $t$")
    a1.set_ylabel(r"$\mathrm{Var}_{\boldsymbol{\theta}}[\langle Z_i\rangle]$")
    a1.set_xticks(ts)
    a1.set_title(f"DLA dim {dim_mg}" + r"$\,\to\,$" + f"{dim_doped} (1 ZZ gate)",
                 fontsize=8.5)
    a1.legend(loc="lower left", fontsize=8)
    plotting.save(fig, "doping_variance")


def fig_precondition_training() -> None:
    d = read_csv("precondition_training")
    epoch_axis = d["epoch"]
    rel_pre, rel_pre_sd = d["rel_pre"], d["rel_pre_sd"]
    rel_raw, rel_raw_sd = d["rel_raw"], d["rel_raw_sd"]
    gv_pre, gv_pre_lo, gv_pre_hi = d["gradvar_pre"], d["gradvar_pre_lo"], d["gradvar_pre_hi"]
    gv_raw, gv_raw_lo, gv_raw_hi = d["gradvar_raw"], d["gradvar_raw_lo"], d["gradvar_raw_hi"]

    fig, ax = plt.subplots(figsize=(plotting.COL, 2.6))
    ax.plot(epoch_axis, rel_pre, "-", color=TEAL)
    ax.fill_between(epoch_axis, np.clip(rel_pre - rel_pre_sd, 0, None), rel_pre + rel_pre_sd,
                    color=TEAL, alpha=0.2, lw=0)
    ax.plot(epoch_axis, rel_raw, "-", color=ORANGE)
    ax.fill_between(epoch_axis, np.clip(rel_raw - rel_raw_sd, 0, None), rel_raw + rel_raw_sd,
                    color=ORANGE, alpha=0.2, lw=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\mathcal{L}/\mathcal{L}_0$")

    # Coordinate-wise gradient dispersion at each iterate, not an ensemble
    # gradient variance. Bands are exp(mean(log v) +/- std(log v)) across seeds.
    # Legacy CSV suffixes pre/raw denote the uniform/clustered input priors.
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.plot(epoch_axis, gv_pre, "--", color=TEAL, lw=0.9)
    ax2.fill_between(epoch_axis, gv_pre_lo, gv_pre_hi, color=TEAL, alpha=0.15, lw=0)
    ax2.plot(epoch_axis, gv_raw, "--", color=ORANGE, lw=0.9)
    ax2.fill_between(epoch_axis, gv_raw_lo, gv_raw_hi, color=ORANGE, alpha=0.15, lw=0)
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$\mathrm{Var}_j[\partial_{\theta_j}\mathcal{L}]$")

    # legend outside on top (as in fig8/fig9): colour = input law, style = quantity.
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=TEAL, ls="-", label="Uniform prior"),
               Line2D([], [], color=ORANGE, ls="-", label="Clustered prior"),
               Line2D([], [], color="0.4", ls="-", label=r"Loss $\mathcal{L}/\mathcal{L}_0$"),
               Line2D([], [], color="0.4", ls="--", label="Gradient dispersion")]
    plotting.top_legend(ax, handles=handles,
                        labels=[h.get_label() for h in handles], ncol=2)
    plotting.save(fig, "precondition_training")


def fig_reuploading_depth() -> None:
    d = read_csv("reuploading_depth")
    depths = np.array(list(dict.fromkeys(d["depth"].tolist())), dtype=int)
    k_th = len(d["depth"]) // len(depths)

    fig, ax = plt.subplots(figsize=(plotting.COL, 2.5))
    for key, color, ls, label in (
        ("od_unif", TEAL, "-", r"Off-diag., uniform"),
        ("od_clus", ORANGE, "-", r"Off-diag., clustered"),
        ("mg_unif", ACCENT, "--", r"Matchgate, uniform"),
        ("mg_clus", NAVY, "--", r"Matchgate, clustered"),
    ):
        dat = d[key].reshape(len(depths), k_th)
        mean = dat.mean(axis=1)
        lo, hi = np.quantile(dat, 0.1, axis=1), np.quantile(dat, 0.9, axis=1)
        ax.plot(depths, mean, "o" + ls, color=color, ms=3.5, lw=1.1, label=label)
        ax.fill_between(depths, lo, hi, color=color, alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel(r"Re-uploading depth $L$")
    ax.set_ylabel(r"$\mathrm{Var}_{\boldsymbol{\theta}}[\langle O\rangle]$")
    ax.set_xticks(depths)
    plotting.unify_grid(ax)                                # major decade grid, no minor lines
    plotting.top_legend(ax, ncol=2)
    plotting.save(fig, "reuploading_depth")


# latent_drift / latent_drift_hist styling (mirrors ARMS in exp_latent_drift, and
# the colour assignment of fig_reuploading_depth: teal/orange = the off-diagonal
# uniform/raw pair, blue/navy = the floored matchgate control).
_DRIFT_SERIES = (
    ("od_unif", TEAL, r"Off-diag., uniform"),
    ("od_raw", ORANGE, r"Off-diag., raw"),
    ("mg_unif", ACCENT, r"Matchgate, uniform"),
    ("mg_raw", NAVY, r"Matchgate, raw"),
)


def _angle_axis(ax) -> None:
    """x axis over one period in units of pi/2 (shared by the latent histograms)."""
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax.set_xticklabels(["$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])


def fig_latent_drift() -> None:
    """Training of the MLP-preconditioned model: loss, the gradient variance split by
    parameter group, and the dataset-averaged latent purity, all vs epoch."""
    d = read_csv("latent_drift")
    epoch = d["epoch"]
    n, mu_n = int(d["n"][0]), float(d["mu_n"][0])

    fig, (a0, a1, a2) = plt.subplots(3, 1, sharex=True, figsize=(plotting.COL, 5.0))
    for key, color, label in _DRIFT_SERIES:
        a0.plot(epoch, d[f"{key}_rel"], "-", color=color, label=label)
        a0.fill_between(epoch, d[f"{key}_rel_lo"], d[f"{key}_rel_hi"],
                        color=color, alpha=0.18, lw=0)
        # solid = circuit parameters, dashed = the MLP, i.e. the signal that actually
        # reaches the classical front end (exactly zero at clustered latents).
        a1.plot(epoch, d[f"{key}_gvW"], "-", color=color, lw=1.0)
        a1.fill_between(epoch, d[f"{key}_gvW_lo"], d[f"{key}_gvW_hi"],
                        color=color, alpha=0.13, lw=0)
        a1.plot(epoch, d[f"{key}_gvM"], "--", color=color, lw=0.9)
        a2.plot(epoch, d[f"{key}_pur"], "-", color=color)
        a2.fill_between(epoch, np.clip(d[f"{key}_pur"] - d[f"{key}_pur_sd"], 1e-30, None),
                        d[f"{key}_pur"] + d[f"{key}_pur_sd"], color=color, alpha=0.18, lw=0)
    # log: the arms span three decades of relative loss, and an early matchgate
    # transient overshoots L_0 by more than an order of magnitude.
    a0.set_yscale("log")
    a0.set_ylabel(r"$\mathcal{L}/\mathcal{L}_0$")

    a1.set_yscale("log")
    a1.set_ylabel(r"$\mathrm{Var}[\partial \mathcal{L}]$")

    # guides: the uniform-prior mean mu_n and the attainable maximum n-1 (all latents
    # at pi/2).  A purity-seeking drift would run to the upper guide.
    a2.axhline(mu_n, color=GREY_REF, ls=":", lw=0.9)
    a2.axhline(n - 1, color=GREY_REF, ls=":", lw=0.9)
    a2.text(epoch[0], mu_n, r"$\mu_n$", fontsize=7.5, color=GREY_REF,
            ha="left", va="bottom")
    a2.text(epoch[0], n - 1, "$n-1$", fontsize=7.5, color=GREY_REF,
            ha="left", va="bottom")
    a2.set_yscale("log")
    a2.set_xlabel("Epoch")
    a2.set_ylabel(r"$\hat{\mathcal{P}}(\boldsymbol{\phi})$")
    # log epoch axis (shared): essentially all of the latent motion happens inside the
    # first ~20 epochs, which a linear axis compresses into the left edge.
    a2.set_xscale("log")
    plotting.unify_grid(a0)
    plotting.unify_grid(a1)
    plotting.unify_grid(a2)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, ls="-", label=lab) for _, c, lab in _DRIFT_SERIES]
    handles += [Line2D([], [], color="0.4", ls="-", label="Circuit"),
                Line2D([], [], color="0.4", ls="--", label="MLP")]
    plotting.top_legend(a0, handles=handles, labels=[h.get_label() for h in handles],
                        ncol=2, fontsize=7.5)
    plotting.save(fig, "latent_drift")


def fig_latent_drift_hist() -> None:
    """The discriminator: per-site latent densities at three training snapshots.
    Flat = uniform, a peak at pi/2 (or 3pi/2) = the X-eigenstate purity maximum,
    anything else = task structure."""
    d = read_csv("latent_drift_hist")
    snaps = np.array(sorted(set(d["snap"].tolist())), dtype=int)
    rows = [(k, c, lab) for k, c, lab in _DRIFT_SERIES]

    fig, axes = plt.subplots(len(rows), len(snaps), sharex=True, sharey="row",
                             figsize=(plotting.WIDE, 1.35 * len(rows) + 0.9))
    for i, (key, color, label) in enumerate(rows):
        for j, t in enumerate(snaps):
            ax = axes[i, j]
            m = (d["arm"] == key) & (d["snap"] == t)
            for site in sorted(set(d["site"][m].tolist())):
                ms = m & (d["site"] == site)
                pooled = site < 0
                ax.plot(d["bin_center"][ms], d["density"][ms], "-", color=color,
                        lw=1.3 if pooled else 0.5, alpha=1.0 if pooled else 0.35,
                        zorder=3 if pooled else 2)
            # the purity-maximising latent positions (X eigenstates)
            for g in (np.pi / 2, 3 * np.pi / 2):
                ax.axvline(g, color=GREY_REF, ls=":", lw=0.8, zorder=1)
            _angle_axis(ax)
            ax.set_ylim(bottom=0)
            if i == 0:
                ax.set_title(f"Epoch {t}", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"{label}", fontsize=7.5)
            if i == len(rows) - 1:
                ax.set_xlabel(r"Latent angle $\phi$")
    plotting.save(fig, "latent_drift_hist")


# channel_scaling styling (mirrors FAMILIES in exp_channel_scaling).  Colours follow
# _HOLLOW_SERIES: the off-diagonal witness in blue, the hollow doped family in navy,
# the floored matchgate control in teal.
_CHANNEL_SERIES = (
    ("g_od", ACCENT, r"Off-diag. ($d_Z{=}0$)"),
    ("+XIY", NAVY, r"Hollow doped ($d_Z{=}0$)"),
    ("mg", TEAL, r"Matchgate ($d_Z{=}n$)"),
)


def fig_channel_scaling() -> None:
    """The circuit and encoder gradient channels side by side.

    (a) both vs the angle spread sigma: the circuit channel tracks the purity at
    sigma^4, the encoder channel recovers at sigma^2, i.e. as its square root.
    (b) the suppression each channel suffers from clustered inputs, vs n, against the
    square-root prediction -- does the separation survive growing n?
    """
    d = read_csv("channel_scaling")
    dn = read_csv("channel_scaling_n")
    sig = d["sigma"]
    n_sig, sigma_fixed = int(d["n_sigma"][0]), float(d["sigma_fixed"][0])
    pos = sig > 0  # sigma = 0 is an exact zero, undrawable on a log axis

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(plotting.WIDE, 2.8))
    for key, color, label in _CHANNEL_SERIES:
        a0.plot(sig[pos], d[f"{key}_circ"][pos], "o-", color=color, ms=3, label=label)
        a0.plot(sig[pos], d[f"{key}_enc"][pos], "s--", color=color, ms=3, lw=1.1)
    # guides over the fitted range only, anchored to the off-diagonal curves
    fit = pos & (sig < 0.12)
    s0 = sig[fit][0]
    for q, key in ((4.0, "g_od_circ"), (2.0, "g_od_enc")):
        y0 = d[key][fit][0]
        a0.plot(sig[fit], y0 * (sig[fit] / s0) ** q, ":", color=GREY_REF, lw=0.9)
    a0.text(0.97, 0.05, r"$\propto\sigma^4$ / $\propto\sigma^2$", fontsize=7.5,
            color=GREY_REF, ha="right", va="bottom", transform=a0.transAxes)
    a0.text(0.03, 0.96, f"(a) $n={n_sig}$", fontsize=8.5, ha="left", va="top",
            transform=a0.transAxes)
    a0.set_xscale("log")
    a0.set_yscale("log")
    a0.set_xlabel(r"Angle spread $\sigma$ (clustered $\to$ uniform)")
    a0.set_ylabel(r"$\mathrm{Var}_{\boldsymbol{\theta}}$")
    plotting.unify_grid(a0)

    for key, color, label in _CHANNEL_SERIES:
        m = dn["family"] == key
        ns = dn["n"][m]
        s_circ = dn["circ_unif"][m] / dn["circ_clus"][m]
        s_enc = dn["enc_unif"][m] / dn["enc_clus"][m]
        a1.plot(ns, s_circ, "o-", color=color, ms=3, label=label)
        a1.plot(ns, s_enc, "s--", color=color, ms=3, lw=1.1)
        # square-root prediction for the encoder, from the measured circuit channel
        a1.plot(ns, np.sqrt(s_circ), ":", color=color, lw=0.9, alpha=0.7)
    a1.set_yscale("log")
    a1.set_xlabel("$n$ Qubits")
    a1.set_ylabel(r"Suppression $\mathrm{Var}^{\mathrm{unif}}/\mathrm{Var}^{\mathrm{clus}}$")
    a1.text(0.03, 0.96, rf"(b) $\sigma={sigma_fixed}$", fontsize=8.5, ha="left",
            va="top", transform=a1.transAxes)
    a1.locator_params(axis="x", integer=True)
    plotting.unify_grid(a1)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, ls="-", marker="o", ms=3, label=lab)
               for _, c, lab in _CHANNEL_SERIES]
    handles += [Line2D([], [], color="0.4", ls="-", marker="o", ms=3, label="Circuit"),
                Line2D([], [], color="0.4", ls="--", marker="s", ms=3, label="Encoder"),
                Line2D([], [], color="0.4", ls=":", lw=0.9, label=r"$\sqrt{\;\cdot\;}$ / guide")]
    plotting.top_legend(a0, handles=handles, labels=[h.get_label() for h in handles],
                        ncol=3, fontsize=7.5)
    plotting.save(fig, "channel_scaling")


def fig_encoding_landscape() -> None:
    d = read_csv("encoding_landscape")
    uniform_mean = float(d["uniform_mean"][0])

    figA, axA = plt.subplots(figsize=(plotting.COL, 2.2))
    for kind in ENC_KINDS:
        m = d["encoding"] == kind
        axA.plot(d["x"][m], np.maximum(d["purity"][m], 1e-4), "-", color=ENC_COLOR[kind],
                 label=kind.capitalize(), **_LSTYLE[kind])
    axA.axhline(uniform_mean, color=GREY_REF, ls=":", lw=0.9)
    axA.set_yscale("log")
    axA.set_ylim(1e-4, 12)
    axA.set_xlim(0, 2 * np.pi)
    axA.set_xticks([0, np.pi, 2 * np.pi])
    axA.set_xticklabels(["$0$", r"$\pi$", r"$2\pi$"])
    axA.set_xlabel(r"Scalar input $x$")
    axA.set_ylabel(r"$\mathcal{P}_{\mathfrak{g}}(\rho(\boldsymbol{w}x))$")
    plotting.top_legend(axA, ncol=3)
    plotting.save(figA, "encoding_landscape")


def fig_encoding_mean() -> None:
    d = read_csv("encoding_mean")
    ns, iid = d["n"], d["iid"]
    mean_u = {k: d[f"u_{k}"] for k in ENC_KINDS}
    mean_c = {k: d[f"c_{k}"] for k in ENC_KINDS}

    figB, axB = plt.subplots(figsize=(plotting.COL, 2.5))
    for kind in ENC_KINDS:
        color = ENC_COLOR[kind]
        axB.plot(ns, mean_u[kind], "o-", color=color, ms=3, lw=1.1, label=kind.capitalize())
        axB.plot(ns, np.maximum(mean_c[kind], 1e-7), "s--", color=color, ms=3, lw=1.1)
    axB.plot(ns, iid, ":", color=GREY_REF, lw=1.3, zorder=0, label=r"IID mean")
    axB.plot([], [], "o-", color="0.4", ms=3, lw=1.1, label=r"Uniform $x$")
    axB.plot([], [], "s--", color="0.4", ms=3, lw=1.1, label=r"Clustered $x$")
    axB.set_yscale("log")
    axB.set_xlabel("$n$ Qubits")
    axB.set_ylabel(r"$\mathbb{E}_x[\mathcal{P}_{\mathfrak{g}}]$")
    axB.locator_params(axis="x", integer=True)

    # spectrum size |Omega|(n) on the right axis (dotted): hamming 2n+1, binary 2^{n+1}-1,
    # ternary 3^n (main_condensed eq:spectrum) -- ties the purity recovery to the encoding's
    # frequency count.  Analytic closed forms (get_n_freqs would enumerate a 3^n set at n=14).
    nsa = ns.astype(int)                                   # integer powers: exact
    n_freqs = {"hamming": 2 * nsa + 1, "binary": 2 ** (nsa + 1) - 1, "ternary": 3 ** nsa}
    axf = axB.twinx()
    for kind in ENC_KINDS:
        axf.plot(ns, n_freqs[kind], ":", color=ENC_COLOR[kind], lw=0.9, alpha=0.8)
    axf.set_yscale("log")
    axf.set_ylabel(r"number of frequencies $|\Omega|$")

    plotting.unify_grid(axB, axf)                          # major decade grid on the primary axis only
    plotting.top_legend(axB, ncol=3)
    plotting.save(figB, "encoding_mean")


def fig_offdiag_purity() -> None:
    d = read_csv("offdiag_purity")
    ns, mean_unif, mean_clus = d["n"], d["uniform"], d["clustered"]
    var_unif, var_clus = d["var_uniform"], d["var_clustered"]
    mean_brute = d["brute"]
    ns_b = ns[:len(mean_brute)]                            # brute force runs to a smaller n

    # Broken y-axis: the uniform band and the clustered band lie ~4 decades apart with
    # empty middle decades on both the left (P_g) and right (Var) axes.  Split into an
    # upper (uniform) and lower (clustered) panel sharing x, clip each curve to its band,
    # and drop the middle so the two traces sit close across the break.
    fig, (axhi, axlo) = plt.subplots(
        2, 1, sharex=True, figsize=(plotting.COL, 2.6),
        gridspec_kw={"height_ratios": [1, 1]})
    fig.get_layout_engine().set(hspace=0.0, h_pad=0.02)    # close the broken-axis gap (todo: less whitespace)
    for a in (axhi, axlo):
        lbl = a is axhi                                    # legend handles from the upper panel only
        a.plot(ns_b, mean_brute, "+", color=NAVY, ms=7, mew=1.4, zorder=5,
               label="Brute-force basis" if lbl else None)
        a.plot(ns, mean_unif, "-", color=TEAL, label="Uniform prior" if lbl else None)
        a.plot(ns, mean_clus, "-", color=ORANGE, label="Clustered prior" if lbl else None)
        a.set_yscale("log")
    axhi.set_ylim(0.18, 8.0)                               # uniform band
    axlo.set_ylim(5e-7, 1.2e-4)                            # clustered band

    # proxy for the dashed variance curves; on axhi so top_legend(axhi) actually picks it up
    axhi.plot([], [], "--", color=NAVY, lw=1.1, label=r"$\mathrm{Var}_{\boldsymbol{\theta}}$")
    axhi_v, axlo_v = axhi.twinx(), axlo.twinx()            # variance on the right axis (dashed)
    nonabelian = ns >= 3  # n=2 has zero variance, omitted on logarithmic axes
    for av in (axhi_v, axlo_v):
        av.plot(ns[nonabelian], var_unif[nonabelian], "--", color=TEAL, lw=1.1)
        av.plot(ns[nonabelian], var_clus[nonabelian], "--", color=ORANGE, lw=1.1)
        av.set_yscale("log")
    axhi_v.set_ylim(0.028, 0.32)                           # uniform var band
    axlo_v.set_ylim(3.3e-7, 1.0e-6)                        # clustered var band
    plotting.unify_grid(axhi, axhi_v)                      # major decade grid on the primary panels only
    plotting.unify_grid(axlo, axlo_v)

    for a in (axhi, axhi_v):                               # hide the inner (facing) spines and top-panel x ticks
        a.spines["bottom"].set_visible(False)
    for a in (axlo, axlo_v):
        a.spines["top"].set_visible(False)
    axhi.tick_params(axis="x", which="both", bottom=False)
    d_ = 0.5                                                # diagonal break marks at the cut
    brk = dict(marker=[(-1, -d_), (1, d_)], markersize=7, linestyle="none",
               color="k", mec="k", mew=1, clip_on=False)
    axhi.plot([0, 1], [0, 0], transform=axhi.transAxes, **brk)
    axlo.plot([0, 1], [1, 1], transform=axlo.transAxes, **brk)

    axlo.set_xlabel("$n$ Qubits")
    axlo.locator_params(axis="x", integer=True)
    axlo.set_ylabel(r"$\mathbb{E}_{\boldsymbol{\phi}}[\mathcal{P}_{\mathfrak{g}}]$")
    axlo.yaxis.set_label_coords(-0.19, 1.0)                # centre the shared label across the break, clear of the ticks
    axlo_v.set_ylabel(r"$\mathbb{E}_{\boldsymbol{\phi}}[\mathrm{Var}_{\boldsymbol{\theta}} f]$")
    axlo_v.yaxis.set_label_coords(1.16, 1.0)

    plotting.top_legend(axhi, ncol=2)
    plotting.save(fig, "offdiag_purity")


def fig_purity_regime_contrast() -> None:
    d = read_csv("purity_regime_contrast")
    sigmas, mg, od = d["sigma"], d["matchgate"], d["offdiag"]
    var_mg, var_od = d["var_matchgate"], d["var_offdiag"]
    n = int(d["n"][0])

    fig, ax = plt.subplots(figsize=(plotting.COL, 1.9))    # flatter plot box, matches fig:hollow aspect
    ax.axhline(n, color=GREY_REF, ls=":", lw=0.9)          # d_Z = n, matchgate clustered limit
    ax.plot(sigmas, mg, "o-", color=TEAL, ms=3.5, label=r"Matchgate ($d_Z{=}n$)")
    ax.plot(sigmas, od, "o-", color=ORANGE, ms=3.5, label=r"Off-diagonal ($d_Z{=}0$)")
    ax.plot([], [], "--", color="0.35", lw=1.1, label=r"$\mathrm{Var}_{\boldsymbol{\theta}}$")  # dashed = variance
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Angle spread $\sigma$ (clustered $\to$ uniform)")
    ax.set_ylabel(r"$\mathbb{E}_{\boldsymbol{\phi}}[\mathcal{P}_{\mathfrak{g}}]$")

    axv = ax.twinx()                                       # loss variance on the right axis (dashed)
    axv.plot(sigmas, var_mg, "--", color=TEAL, lw=1.1)
    axv.plot(sigmas, var_od, "--", color=ORANGE, lw=1.1)
    axv.set_yscale("log")
    axv.set_ylabel(r"$\mathbb{E}_{\boldsymbol{\phi}}[\mathrm{Var}_{\boldsymbol{\theta}} f]$")

    plotting.unify_grid(ax, axv)                            # major decade grid on the primary axis only
    ax.legend(loc="lower right", fontsize=7.5)
    plotting.save(fig, "purity_regime_contrast")


# hollow_dimension / hollow_sweep share this series definition: label, colour,
# marker, hollow (filled marker) flag, and the family key used in both CSVs.
_HOLLOW_SERIES = (
    (r"Off-diagonal ($d_Z{=}0$)", ACCENT, "o", True, "g_od", "hollow_scan"),
    (r"+ Bipartite ($d_Z{=}0$)", TEAL, "o", True, "chord(1,4)", "hollow_graphs"),
    (r"+ Doped chain ($d_Z{=}0$)", NAVY, "o", True, "+XIY", "hollow_scan"),
    (r"+ Odd cycle ($d_Z{>}0$)", ORANGE, "o", False, "chord(0,2)", "hollow_graphs"),
)


def fig_hollow_dimension() -> None:
    """DLA growth of the hollow families (appendix panel of fig:hollow)."""
    src = {name: read_csv(name) for name in ("hollow_scan", "hollow_graphs")}

    figd, a0 = plt.subplots(figsize=(plotting.COL, 2.6))
    for label, col, mk, hollow, key, table in _HOLLOW_SERIES:
        fill = dict() if hollow else dict(markerfacecolor="white")
        d = src[table]
        m = d["family"] == key
        a0.semilogy(d["n"][m], d["dim"][m], mk + "-", color=col, label=label, **fill)
    a0.set_xlabel("$n$ Qubits")
    a0.set_ylabel(r"$\dim\mathfrak{g}$")
    a0.legend(loc="upper left", fontsize=7.5)
    plotting.save(figd, "hollow_dimension")


def fig_hollow_sweep() -> None:
    """Regime sweep at n=8 (cf. purity_regime_contrast): the floored family levels off
    at its diagonal count d_Z, the hollow families collapse to exactly 0."""
    d = read_csv("hollow_sweep")
    sigmas = d["sigma"]
    dZ = int(d["d_z"][0])

    figs, a1 = plt.subplots(figsize=(plotting.COL, 2.4))
    for label, col, mk, hollow, key, _ in _HOLLOW_SERIES:
        fill = dict() if hollow else dict(markerfacecolor="white")
        a1.plot(sigmas, d[key], mk + "-", color=col, label=label, ms=3.5, **fill)
    a1.axhline(dZ, color=GREY_REF, ls=":", lw=0.9)
    a1.text(0.04, dZ * 1.6, rf"$d_Z={dZ}$", fontsize=7.5, color=GREY_REF)
    a1.plot([], [], "--", color="0.35", lw=1.1, label=r"$\mathrm{Var}_{\boldsymbol{\theta}}$")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_ylim(top=dZ * 6)
    a1.set_xlabel(r"Angle spread $\sigma$ (clustered $\to$ uniform)")
    a1.set_ylabel(r"$\mathbb{E}_{\boldsymbol{\phi}}[\mathcal{P}_{\mathfrak{g}}]$")

    # At the plotted n=8, XX+YY has equal HS projection weight in each of
    # the equal-dimensional simple ideals, giving Var = 2 P_g / dim g.
    # The graph families have four ideals, the chain and doped chain two.
    axv = a1.twinx()
    for _, col, _, _, key, _ in _HOLLOW_SERIES:
        axv.plot(sigmas, 2 * d[key] / d[f"dim_{key}"][0], "--", color=col, lw=1.1)
    axv.set_yscale("log")
    axv.set_ylabel(r"$\mathbb{E}_{\boldsymbol{\phi}}[\mathrm{Var}_{\boldsymbol{\theta}} f]$")
    plotting.unify_grid(a1, axv)                            # major decade grid on the primary axis only

    plotting.top_legend(a1, ncol=2, fontsize=7.5, handlelength=1.2,
                        handletextpad=0.4, labelspacing=0.3, columnspacing=1.0)
    plotting.save(figs, "hollow_sweep")


FIGURES = {
    "matchgate_convergence": fig_matchgate_convergence,
    "matchgate_scaling": fig_matchgate_scaling,
    "uniform_prior_mean": fig_uniform_prior_mean,
    "preconditioning_effect": fig_preconditioning_effect,
    "dla_regime_contrast": fig_dla_regime_contrast,
    "input_purity_scaling": fig_input_purity_scaling,
    "offdiag_purity": fig_offdiag_purity,
    "purity_regime_contrast": fig_purity_regime_contrast,
    "precondition_training": fig_precondition_training,
    "latent_drift": fig_latent_drift,
    "latent_drift_hist": fig_latent_drift_hist,
    "channel_scaling": fig_channel_scaling,
    "doping_variance": fig_doping_variance,
    "hollow_dimension": fig_hollow_dimension,
    "hollow_sweep": fig_hollow_sweep,
    "reuploading_depth": fig_reuploading_depth,
    "encoding_landscape": fig_encoding_landscape,
    "encoding_mean": fig_encoding_mean,
}


def main(argv=None) -> None:
    """Redraw figures from ``data/*.csv``: all of them, or the named ones."""
    names = list(argv if argv is not None else sys.argv[1:]) or list(FIGURES)
    for name in names:
        FIGURES[name]()


if __name__ == "__main__":
    main()
