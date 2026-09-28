"""Training comparison between uniform and clustered angle priors.

The priors supply separate datasets labeled by the same Fourier target function.
Initial parameters are paired across priors. No rotation is applied to a shared
labeled dataset. The ansatz-first circuit has 10 R_y encoding blocks and 11
trainable XX/YY blocks, with an in-algebra XX+YY readout.

The product-state purity is a reference diagnostic, not the actual state purity
at the first trainable block of this circuit. Saved gradvar fields measure
coordinate-wise gradient dispersion at one iterate, not ensemble gradient
variance. Legacy pre/raw names mean uniform/clustered and are kept for CSV
compatibility. These finite-size results do not establish BP scaling.

Figure: precondition_training (relative loss and gradient dispersion vs epoch).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from qml_essentials import operations as op
from qml_essentials.model import Model
from qml_essentials.coefficients import Coefficients

from unflattening import figures
from unflattening.utils.dla import lie_closure_paulis, xx_yy_generators
from unflattening.utils.purity import product_state, g_purity_from_basis
from unflattening.utils.priors import sample_uniform, sample_raw

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
    """Product-state reference purity and sampled output variance per input law."""
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
            # Dispersion across parameter coordinates at this iterate.
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
    """Training loss vs epoch (uniform vs clustered), with the gradient-dispersion
    diagnostic on a secondary axis."""
    rel_pre_s, rel_raw_s, gv_pre_s, gv_raw_s = stacks
    pg_pre, pg_raw, vw_pre, vw_raw = diagnostics
    y_pre, y_raw = targets

    # loss: linear mean/std (linear axis).  gradient dispersion: summaries in log space
    # (log axis), so band edges stay positive.
    rel_pre, rel_pre_sd = rel_pre_s.mean(0), rel_pre_s.std(0)
    rel_raw, rel_raw_sd = rel_raw_s.mean(0), rel_raw_s.std(0)
    lgp, lgr = np.log(np.maximum(gv_pre_s, 1e-300)), np.log(np.maximum(gv_raw_s, 1e-300))
    gv_pre, gv_pre_lo, gv_pre_hi = np.exp(lgp.mean(0)), np.exp(lgp.mean(0) - lgp.std(0)), np.exp(lgp.mean(0) + lgp.std(0))
    gv_raw, gv_raw_lo, gv_raw_hi = np.exp(lgr.mean(0)), np.exp(lgr.mean(0) - lgr.std(0)), np.exp(lgr.mean(0) + lgr.std(0))
    print(f"  loss uniform (mean): {rel_pre[0]:.3f} -> {rel_pre[-1]:.3f}")
    print(f"  loss clustered (mean):            {rel_raw[0]:.3f} -> {rel_raw[-1]:.3f}")
    print(f"  gradient dispersion (init): uniform={gv_pre[0]:.2e}  clustered={gv_raw[0]:.2e}  ratio={gv_pre[0]/gv_raw[0]:.1e}")

    # Finite-experiment checks of the saved contrast between the two priors.
    # They are not an asymptotic BP or preprocessing guarantee.
    assert pg_pre > 50 * pg_raw, "uniform input must have far higher g-purity"
    assert vw_pre > 5 * vw_raw, "uniform output variance must be much larger"
    assert y_pre.std() > 0.5 and y_raw.std() > 0.5, "target must be non-trivial on both"
    assert rel_pre[-1] < 0.75, "uniform training must descend (mean)"
    assert rel_raw[-1] > 0.85, "clustered training has limited progress (mean)"
    assert rel_pre[-1] < 0.7 * rel_raw[-1], "uniform must progress far more than clustered"
    assert gv_pre[0] > 50 * gv_raw[0], "uniform must start with far larger gradient dispersion"

    epoch_axis = np.arange(1, epochs + 1)
    figures.write_csv("precondition_training", dict(
        epoch=epoch_axis,
        rel_pre=rel_pre, rel_pre_sd=rel_pre_sd, rel_raw=rel_raw, rel_raw_sd=rel_raw_sd,
        gradvar_pre=gv_pre, gradvar_pre_lo=gv_pre_lo, gradvar_pre_hi=gv_pre_hi,
        gradvar_raw=gv_raw, gradvar_raw_lo=gv_raw_lo, gradvar_raw_hi=gv_raw_hi,
        pg_pre=pg_pre, pg_raw=pg_raw, varW_pre=vw_pre, varW_raw=vw_raw,
        dim_g=dim_g, n_seeds=n_seeds))
    # the per-seed curves behind the mean/band (kept as a record, not plotted)
    figures.write_csv("precondition_training_seeds", dict(
        seed=np.repeat(np.arange(n_seeds), epochs), epoch=np.tile(epoch_axis, n_seeds),
        rel_pre=rel_pre_s.ravel(), rel_raw=rel_raw_s.ravel(),
        gradvar_pre=gv_pre_s.ravel(), gradvar_raw=gv_raw_s.ravel()))
    figures.fig_precondition_training()
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
    print(f"  P_g^XXYY:  uniform={pg_pre:.3f}   clustered={pg_raw:.3e}   ratio={pg_pre/pg_raw:.1e}")
    print(f"  Var_W[<O>]: uniform={vw_pre:.3e}  clustered={vw_raw:.3e}  ratio={vw_pre/max(vw_raw,1e-30):.1e}")

    stacks = part_training(model, predict, th_pre_j, y_pre, th_raw_j, y_raw)
    part_figure(stacks, diagnostics, (y_pre, y_raw), len(basis))


if __name__ == "__main__":
    main()
