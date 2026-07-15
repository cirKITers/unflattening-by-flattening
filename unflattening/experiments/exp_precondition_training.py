"""precondition_training -- the Outlook claim in actual gradient descent: for a
floor-free off-diagonal family, the input distribution decides trainability.

A quantum Fourier model (R_y angle encoding reuploaded with a trainable XX+YY
brickwork, generators {X_k X_{k+1}, Y_k Y_{k+1}}; in-algebra readout
O = X_i X_{i+1} + Y_i Y_{i+1}) is trained on the *same* Fourier-series target, from
the *same* initial parameters, under two input angle distributions:
  * preconditioned theta ~ U[0,2pi)^n  -> P_g^{XX+YY} = Omega(n) (Lemma isotropic
    realises this uniform prior), gradients are healthy, the loss descends;
  * raw, clustered near {0,pi}          -> P_g^{XX+YY} ~ 0, the W-landscape is flat
    (Var_W collapses), the loss is frozen near its start (barren).

Unlike the matchgate so(2n), this off-diagonal DLA has no single-qubit Z and hence
no deterministic purity floor (cf. Lemma deterministic): the input g-purity is not
bounded below, so the input distribution alone separates trainable from barren.
The readout is an in-algebra XY observable, not the matchgate Z_i, precisely
because Z_i sits on the floor-protected diagonal and cannot expose this effect.

The circuit is built with the maintained qml_essentials.Model in its ansatz-first
(Schuld L+1) convention (observables=[XX, YY]); per-qubit encoding angles use a
diagonal data-reupload mask.  The input-distribution dichotomy is order-agnostic
(see exp_reuploading).  TODO: reconcile the ordering remark with main.tex.

Figure: precondition_training (training loss vs epoch, preconditioned vs raw).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from qml_essentials import operations as op
from qml_essentials.model import Model
from qml_essentials.coefficients import Coefficients

from unflattening.utils import plotting
from unflattening.utils.dla import lie_closure_paulis, xx_yy_generators
from unflattening.utils.purity import product_state, g_purity_from_basis
from unflattening.utils.priors import sample_uniform, sample_raw
from unflattening.utils.plotting import plt, DATA_DIR, TEAL, ORANGE

N_QUBITS = 6            # qubits
DEPTH = 10       # reuploading XX+YY layers
EPOCHS = 800
LR = 0.05
N_TRAIN = 256
M_VAR = 2000     # random-W bank for the Var_W[<O>] barren-plateau diagnostic
RAW_EPS = 0.03   # spread of the raw angles around {0, pi}
MAX_ORDER = 2    # target keeps Fourier terms with at most this many active features
N_SEEDS = 8      # independent W0 initialisations for the mean +/- std error bands
N_REF = 4000     # uniform reference sample standardising the target function
PG_SAMPLES = 128  # dataset rows averaged for the input g-purity diagnostic
VAR_SAMPLES = 16  # dataset rows averaged for the Var_W[<O>] diagnostic


def build_model(n_qubits: int = N_QUBITS, depth: int = DEPTH):
    """The XY_Brickwork Fourier model, its <O> predictor, and its DLA basis.

    R_y encoding (per-qubit angles via a diagonal data-reupload mask) re-uploaded
    with trainable XX+YY layers, built by the maintained qml_essentials.Model in
    its ansatz-first (Schuld L+1) convention; observables=[XX, YY] gives the
    off-diagonal readout.  Trainable params W have shape ``model._params_shape``.
    """
    i_out = n_qubits // 2 - 1  # a bulk bond (i_out, i_out+1)
    # in-algebra readout O = X_i X_{i+1} + Y_i Y_{i+1} on a bulk bond.
    XX = op.PauliX(wires=i_out) @ op.PauliX(wires=i_out + 1)
    YY = op.PauliY(wires=i_out) @ op.PauliY(wires=i_out + 1)

    dmask = np.broadcast_to(np.eye(n_qubits, dtype=bool), (depth, n_qubits, n_qubits)).copy()
    model = Model(n_qubits=n_qubits, n_layers=depth, circuit_type="XY_Brickwork",
                  data_reupload=dmask, encoding=["RY"] * n_qubits,
                  observables=[XX, YY])

    def predict(W, Theta):  # <O> per input row (Model batches over params or inputs)
        return model(params=W, inputs=Theta, execution_type="expval").sum(axis=-1)

    basis = lie_closure_paulis(xx_yy_generators(n_qubits))
    return model, predict, basis


def build_target(rng, key, n_qubits: int = N_QUBITS, max_order: int = MAX_ORDER,
                 n_ref: int = N_REF):
    """Fixed low-order real Fourier-series target (R_y degree-1 support), standardised
    on a uniform reference so both runs share the same target *function*."""
    key, kr, ki = jax.random.split(key, 3)
    freq_axes = [jnp.array([-1, 0, 1])] * n_qubits
    C = jax.random.normal(kr, (3,) * n_qubits) + 1j * jax.random.normal(ki, (3,) * n_qubits)
    # keep only terms with <= max_order active features (index 1 == frequency 0),
    # so the local XX+YY student can reach the target on the uniform support.
    grid = np.indices((3,) * n_qubits)
    active = sum((grid[k] != 1) for k in range(n_qubits))
    C = C * jnp.asarray(active <= max_order)

    def target(Theta):
        return np.asarray(
            Coefficients.evaluate_Fourier_series(C, freq_axes, jnp.asarray(Theta))
        )

    ref = sample_uniform(rng, n_ref, n_qubits)
    yref = target(ref)
    mu, sigma = float(yref.mean()), float(yref.std())
    return lambda T: (target(T) - mu) / sigma


def part_diagnostics(model, predict, basis, th_pre, th_raw, th_pre_j, th_raw_j,
                     m_var: int = M_VAR, pg_samples: int = PG_SAMPLES,
                     var_samples: int = VAR_SAMPLES):
    """Input g-purity and the Var_W[<O>] barren-plateau diagnostic, per input law."""
    pg_pre = float(np.mean([g_purity_from_basis(product_state(t), basis) for t in th_pre[:pg_samples]]))
    pg_raw = float(np.mean([g_purity_from_basis(product_state(t), basis) for t in th_raw[:pg_samples]]))

    Wbank = jax.random.uniform(
        jax.random.PRNGKey(7), (m_var, *model._params_shape), minval=0.0, maxval=2 * np.pi
    )

    def varW(thetas):
        vs = []
        for theta1 in thetas:
            vals = predict(Wbank, theta1)  # batched over W, single input row
            vs.append(float(np.var(np.asarray(vals))))
        return float(np.mean(vs))

    return pg_pre, pg_raw, varW(th_pre_j[:var_samples]), varW(th_raw_j[:var_samples])


def part_training(model, predict, th_pre_j, y_pre, th_raw_j, y_raw,
                  epochs: int = EPOCHS, lr: float = LR, n_seeds: int = N_SEEDS):
    """Train both input laws from the same W0, over ``n_seeds`` independent draws.

    The same W0 is shared by the two input laws within a seed (a controlled
    comparison), and each loss is normalised by its own L_0 before stacking so the
    bands reflect training *progress*.  Stacks are (n_seeds, epochs).
    """
    def train(Theta, y, W0):
        opt = optax.adam(lr)
        W, state = W0, opt.init(W0)

        @jax.jit
        def step(W, state):
            l, g = jax.value_and_grad(lambda W: jnp.mean((predict(W, Theta) - y) ** 2))(W)
            updates, state = opt.update(g, state)
            return optax.apply_updates(W, updates), state, l, jnp.var(g)

        losses, gradvars = [], []
        for _ in range(epochs):
            W, state, l, gv = step(W, state)
            losses.append(float(l))
            gradvars.append(float(gv))
        return np.array(losses), np.array(gradvars)

    rel_pre_s, rel_raw_s, gv_pre_s, gv_raw_s = [], [], [], []
    for s in range(n_seeds):
        W0 = jax.random.uniform(jax.random.PRNGKey(100 + s), model._params_shape,
                                minval=0.0, maxval=2 * np.pi)
        lp, gp = train(th_pre_j, y_pre, W0)
        lr_, gr = train(th_raw_j, y_raw, W0)
        rel_pre_s.append(lp / lp[0]); rel_raw_s.append(lr_ / lr_[0])
        gv_pre_s.append(gp); gv_raw_s.append(gr)
    return (np.array(rel_pre_s), np.array(rel_raw_s),
            np.array(gv_pre_s), np.array(gv_raw_s))


def part_figure(stacks, diagnostics, targets, dim_g: int, epochs: int = EPOCHS,
                n_seeds: int = N_SEEDS) -> None:
    """Training loss vs epoch (preconditioned vs raw), with the gradient-variance
    diagnostic on a secondary axis."""
    rel_pre_s, rel_raw_s, gv_pre_s, gv_raw_s = stacks
    pg_pre, pg_raw, vw_pre, vw_raw = diagnostics
    y_pre, y_raw = targets

    # loss: linear mean/std (linear axis).  grad variance: geometric mean/std
    # (log axis), so band edges stay positive.
    rel_pre, rel_pre_sd = rel_pre_s.mean(0), rel_pre_s.std(0)
    rel_raw, rel_raw_sd = rel_raw_s.mean(0), rel_raw_s.std(0)
    lgp, lgr = np.log(np.maximum(gv_pre_s, 1e-300)), np.log(np.maximum(gv_raw_s, 1e-300))
    gv_pre, gv_pre_lo, gv_pre_hi = np.exp(lgp.mean(0)), np.exp(lgp.mean(0) - lgp.std(0)), np.exp(lgp.mean(0) + lgp.std(0))
    gv_raw, gv_raw_lo, gv_raw_hi = np.exp(lgr.mean(0)), np.exp(lgr.mean(0) - lgr.std(0)), np.exp(lgr.mean(0) + lgr.std(0))
    print(f"  loss preconditioned (mean): {rel_pre[0]:.3f} -> {rel_pre[-1]:.3f}")
    print(f"  loss raw (mean):            {rel_raw[0]:.3f} -> {rel_raw[-1]:.3f}")
    print(f"  grad var (init): preconditioned={gv_pre[0]:.2e}  raw={gv_raw[0]:.2e}  ratio={gv_pre[0]/gv_raw[0]:.1e}")

    # runnable checks: at fixed circuit, readout and target, the input distribution
    # alone separates a descending (preconditioned) from a frozen (raw) loss, on
    # average over the n_seeds initialisations.
    assert pg_pre > 50 * pg_raw, "preconditioned input must have far higher g-purity"
    assert vw_pre > 5 * vw_raw, "preconditioned loss landscape must be far less flat"
    assert y_pre.std() > 0.5 and y_raw.std() > 0.5, "target must be non-trivial on both"
    assert rel_pre[-1] < 0.75, "preconditioned training must descend (mean)"
    assert rel_raw[-1] > 0.85, "raw training must stall (frozen, barren; mean)"
    assert rel_pre[-1] < 0.7 * rel_raw[-1], "preconditioned must progress far more than raw"
    assert gv_pre[0] > 50 * gv_raw[0], "preconditioned must start with far larger gradient variance"

    epoch_axis = np.arange(1, epochs + 1)
    fig, ax = plt.subplots(figsize=(plotting.COL, 2.6))
    ax.plot(epoch_axis, rel_pre, "-", color=TEAL)
    ax.fill_between(epoch_axis, np.clip(rel_pre - rel_pre_sd, 0, None), rel_pre + rel_pre_sd,
                    color=TEAL, alpha=0.2, lw=0)
    ax.plot(epoch_axis, rel_raw, "-", color=ORANGE)
    ax.fill_between(epoch_axis, np.clip(rel_raw - rel_raw_sd, 0, None), rel_raw + rel_raw_sd,
                    color=ORANGE, alpha=0.2, lw=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\mathcal{L}/\mathcal{L}_0$")

    # secondary axis: gradient variance Var[d_W L], the barren-plateau diagnostic
    # (dashed).  Vanishing floor for raw throughout; for preconditioned it starts
    # orders of magnitude higher and decays only as the loss converges.  Bands are
    # geometric mean +/- std over the initialisations.
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.plot(epoch_axis, gv_pre, "--", color=TEAL, lw=0.9)
    ax2.fill_between(epoch_axis, gv_pre_lo, gv_pre_hi, color=TEAL, alpha=0.15, lw=0)
    ax2.plot(epoch_axis, gv_raw, "--", color=ORANGE, lw=0.9)
    ax2.fill_between(epoch_axis, gv_raw_lo, gv_raw_hi, color=ORANGE, alpha=0.15, lw=0)
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$\mathrm{Var}[\partial_{\boldsymbol{\theta}} \mathcal{L}]$")

    # legend outside on top (as in fig8/fig9): colour = input law, style = quantity.
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=TEAL, ls="-", label="Preconditioned"),
               Line2D([], [], color=ORANGE, ls="-", label="Raw"),
               Line2D([], [], color="0.4", ls="-", label=r"Loss $\mathcal{L}/\mathcal{L}_0$"),
               Line2D([], [], color="0.4", ls="--", label="Grad. Var.")]
    plotting.top_legend(ax, handles=handles,
                        labels=[h.get_label() for h in handles], ncol=2)
    plotting.save(fig, "precondition_training")
    np.savez(DATA_DIR / "precondition_training.npz", epochs=epoch_axis,
             rel_pre=rel_pre, rel_raw=rel_raw, rel_pre_sd=rel_pre_sd, rel_raw_sd=rel_raw_sd,
             gradvar_pre=gv_pre, gradvar_raw=gv_raw,
             gradvar_pre_lo=gv_pre_lo, gradvar_pre_hi=gv_pre_hi,
             gradvar_raw_lo=gv_raw_lo, gradvar_raw_hi=gv_raw_hi,
             rel_pre_seeds=rel_pre_s, rel_raw_seeds=rel_raw_s,
             gradvar_pre_seeds=gv_pre_s, gradvar_raw_seeds=gv_raw_s,
             pg_pre=pg_pre, pg_raw=pg_raw, varW_pre=vw_pre, varW_raw=vw_raw,
             dim_g=dim_g, n_seeds=n_seeds)
    print("precondition_training: done")


def main() -> None:
    rng = np.random.default_rng(0)
    model, predict, basis = build_model()
    std_target = build_target(rng, jax.random.PRNGKey(1))

    # ---- two input datasets, same target ----
    th_pre = sample_uniform(rng, N_TRAIN, N_QUBITS)
    th_raw = sample_raw(rng, N_TRAIN, N_QUBITS, RAW_EPS)
    y_pre = jnp.asarray(std_target(th_pre))
    y_raw = jnp.asarray(std_target(th_raw))
    th_pre_j, th_raw_j = jnp.asarray(th_pre), jnp.asarray(th_raw)

    diagnostics = part_diagnostics(model, predict, basis, th_pre, th_raw, th_pre_j, th_raw_j)
    pg_pre, pg_raw, vw_pre, vw_raw = diagnostics
    print(f"precondition_training -- off-diagonal XX+YY DLA, n={N_QUBITS}, depth={DEPTH}, dim g={len(basis)}")
    print(f"  P_g^XXYY:  preconditioned={pg_pre:.3f}   raw={pg_raw:.3e}   ratio={pg_pre/pg_raw:.1e}")
    print(f"  Var_W[<O>]: preconditioned={vw_pre:.3e}  raw={vw_raw:.3e}  ratio={vw_pre/max(vw_raw,1e-30):.1e}")

    stacks = part_training(model, predict, th_pre_j, y_pre, th_raw_j, y_raw)
    part_figure(stacks, diagnostics, (y_pre, y_raw), len(basis))


if __name__ == "__main__":
    main()
