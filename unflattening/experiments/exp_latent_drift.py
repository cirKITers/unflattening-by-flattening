"""Track latent angles during joint classical and quantum training.

A per-site residual MLP starts as the identity map before a re-uploading
quantum head. Across paired seeds, compare uniform and clustered inputs with
off-diagonal and matchgate heads. Record loss, parameter-group gradient
dispersion, latent purity, and per-site angle histograms. The clustered
off-diagonal arm recovers under Adam. Compare loss within an arm only:
clustered and uniform datasets have different task difficulty."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from qml_essentials import operations as op
from qml_essentials.model import Model

from unflattening import figures
from unflattening.utils.dla import dim_g, lie_closure_paulis, xx_yy_generators
from unflattening.utils.priors import sample_raw, sample_uniform
from unflattening.utils.purity import (g_purity_closed_form, offdiag_closed_form,
                                       offdiag_uniform_mean)
from unflattening.experiments.exp_precondition_training import build_target

N_QUBITS = 6
DEPTH = 10        # re-uploading layers
EPOCHS = 500
LR = 0.05         # one Adam for the joint (circuit, MLP) parameter tree
N_TRAIN = 256
RAW_EPS = 0.03    # spread of the raw angles around {0, pi}
N_SEEDS = 8       # independent (W0, MLP0) draws -> mean +/- std bands
HIDDEN = 16       # tanh units per site of the elementwise residual MLP
SNAPSHOTS = (0, 1, 2, 100)  # epochs at which the latent sample is stored
HIST_BINS = 48    # bins on [0, 2pi) for the latent histograms

# (arm key, quantum head, input law); the head fixes the DLA and the readout, the
# law fixes the *initial* latent distribution (the MLP starts at the identity).
ARMS = (
    ("od_unif", "od", "unif"),
    ("od_raw", "od", "raw"),
    ("mg_unif", "mg", "unif"),
    ("mg_raw", "mg", "raw"),
)
W0_KEY = {"od": 100, "mg": 200}  # od matches exp_precondition_training's W0 stream


def build_head(kind: str, n_qubits: int = N_QUBITS, depth: int = DEPTH):
    """The quantum head: model, its <O> predictor, and the matching purity formula.

    ``od`` is the floor-free off-diagonal family (XX+YY brickwork, d_Z = 0, in-algebra
    readout X_i X_{i+1} + Y_i Y_{i+1} on a bulk bond) -- the only regime in which the
    input law can move the gradient variance at all.  ``mg`` is the floored matchgate
    control (readout Z_i, purity pinned to [n-1, n] by eq:productform).
    """
    i_out = n_qubits // 2 - 1  # a bulk bond (i_out, i_out+1)
    if kind == "od":
        obs = [op.PauliX(wires=i_out) @ op.PauliX(wires=i_out + 1),
               op.PauliY(wires=i_out) @ op.PauliY(wires=i_out + 1)]
        circuit_type, purity_fn = "XY_Brickwork", offdiag_closed_form
    else:
        obs = [op.PauliZ(wires=n_qubits // 2)]
        circuit_type, purity_fn = "Matchgate", g_purity_closed_form

    dmask = np.broadcast_to(np.eye(n_qubits, dtype=bool), (depth, n_qubits, n_qubits)).copy()
    model = Model(n_qubits=n_qubits, n_layers=depth, circuit_type=circuit_type,
                  data_reupload=dmask, encoding=["RY"] * n_qubits, observables=obs)

    def predict(W, Phi):  # <O> per row (Phi is the *latent* angle batch)
        vals = model(params=W, inputs=Phi, execution_type="expval")
        # several observables come back as (batch, n_obs) and must be summed over the
        # observable axis; a *single* observable comes back as (batch,), where the same
        # sum would instead collapse the batch into one scalar prediction for the whole
        # dataset (cf. the same guard in exp_reuploading.var_vs_depth).
        return vals.sum(axis=-1) if vals.ndim > 1 else vals

    probe = predict(np.full(model._params_shape, 0.5), np.full((3, n_qubits), 0.5))
    assert np.shape(probe) == (3,), \
        f"{kind}: predictor must return one value per input row, got {np.shape(probe)}"
    return model, predict, purity_fn


# ---- the classical front end ------------------------------------------------

def init_mlp(key, n_qubits: int = N_QUBITS, hidden: int = HIDDEN) -> dict:
    """Per-site residual MLP parameters, output layer at zero so phi = x at epoch 0.

    The zero output layer makes the initial latent law *exactly* the input law, so the
    epoch-0 histogram is the input histogram and every later change is attributable to
    training.  It also means w1/b1 see no gradient on the very first step; Adam picks
    them up as soon as w2 leaves zero.
    """
    k1, k2 = jax.random.split(key)
    return {"w1": jax.random.normal(k1, (n_qubits, hidden)),
            "b1": jax.random.normal(k2, (n_qubits, hidden)),
            "w2": jnp.zeros((n_qubits, hidden)),
            "b2": jnp.zeros((n_qubits,))}


def mlp_forward(p: dict, X: jnp.ndarray) -> jnp.ndarray:
    """x (m, n) -> latent angles phi (m, n), elementwise in the feature index.

    Note: inputs are angles in [0, 2pi) by construction, so the map is not made
    2pi-periodic; it only ever sees that window.
    """
    h = jnp.tanh((X[..., None] - jnp.pi) * p["w1"] + p["b1"])   # (m, n, hidden)
    return X + jnp.sum(h * p["w2"], axis=-1) + p["b2"]


MLP_KEYS = ("w1", "b1", "w2", "b2")


# ---- training ---------------------------------------------------------------

def train_arm(predict, purity_fn, X: np.ndarray, y, W0, mlp0: dict,
              epochs: int = EPOCHS, lr: float = LR):
    """Joint (circuit, MLP) training; per-epoch loss, gradient variances and latent law.

    Tracked per epoch, all evaluated at the *pre-update* parameters that produced the
    loss: the loss; the gradient variance split by parameter group (circuit W vs MLP,
    the latter being the signal that actually reaches the classical front end); the
    dataset-averaged purity P^ of the latent angles; and the mean latent displacement
    |phi - x|, which guards the histograms against a runaway-drift artefact (a latent
    law wrapped many times around the circle looks uniform for trivial reasons).
    """
    Xj = jnp.asarray(X)
    params = {"W": W0, **mlp0}
    opt = optax.adam(lr)
    state = opt.init(params)

    def loss_fn(p):
        return jnp.mean((predict(p["W"], mlp_forward(p, Xj)) - y) ** 2)

    @jax.jit
    def step(params, state):
        phi = mlp_forward(params, Xj)
        loss, g = jax.value_and_grad(loss_fn)(params)
        updates, state = opt.update(g, state)
        gm = jnp.concatenate([jnp.ravel(g[k]) for k in MLP_KEYS])
        return (optax.apply_updates(params, updates), state, loss,
                jnp.var(g["W"]), jnp.var(gm), phi)

    losses, gvW, gvM, pur, drift, snaps = [], [], [], [], [], {}
    for t in range(epochs):
        params, state, loss, gw, gm, phi = step(params, state)
        phi = np.asarray(phi)
        if t in SNAPSHOTS:
            snaps[t] = phi
        losses.append(float(loss))
        gvW.append(float(gw))
        gvM.append(float(gm))
        pur.append(float(purity_fn(phi).mean()))
        drift.append(float(np.abs(phi - X).mean()))
    if epochs in SNAPSHOTS:  # the state left by the final update
        snaps[epochs] = np.asarray(mlp_forward(params, Xj))
    return (np.array(losses), np.array(gvW), np.array(gvM), np.array(pur),
            np.array(drift), snaps)


def part_training(heads, X, y, epochs: int = EPOCHS, lr: float = LR,
                  n_seeds: int = N_SEEDS) -> dict:
    """Every arm over ``n_seeds`` seeds.  Within a seed all arms share the MLP
    initialisation, and each head shares its W0 across the two input laws, so an arm
    pair differs only in the latent law it starts from."""
    out = {}
    for key, head, law in ARMS:
        model, predict, purity_fn = heads[head]
        runs = []
        for s in range(n_seeds):
            W0 = jax.random.uniform(jax.random.PRNGKey(W0_KEY[head] + s),
                                    model._params_shape, minval=0.0, maxval=2 * np.pi)
            runs.append(train_arm(predict, purity_fn, X[law], y[law], W0,
                                  init_mlp(jax.random.PRNGKey(300 + s)), epochs, lr))
        losses, gvW, gvM, pur, drift, snaps = zip(*runs)
        losses = np.array(losses)
        out[key] = dict(
            rel=losses / losses[:, :1], gvW=np.array(gvW), gvM=np.array(gvM),
            pur=np.array(pur), drift=np.array(drift),
            # snapshots pooled over seeds: (n_seeds * m, n) latent samples per epoch
            snaps={t: np.concatenate([s[t] for s in snaps]) for t in SNAPSHOTS})
        print(f"  {key}: L/L0 {out[key]['rel'][:, 0].mean():.3f} -> {out[key]['rel'][:, -1].mean():.3f}"
              f"   P^ {out[key]['pur'][:, 0].mean():.3e} -> {out[key]['pur'][:, -1].mean():.3e}"
              f"   |phi-x| -> {out[key]['drift'][:, -1].mean():.2f}")
    return out


def _geo(stack):
    """Geometric mean and +/- 1 std band of a (n_seeds, epochs) stack of positive
    quantities, so the band edges stay positive on the log axis."""
    lg = np.log(np.maximum(stack, 1e-300))
    return np.exp(lg.mean(0)), np.exp(lg.mean(0) - lg.std(0)), np.exp(lg.mean(0) + lg.std(0))


def part_figure(res: dict, X: dict, n_qubits: int = N_QUBITS, epochs: int = EPOCHS,
                n_seeds: int = N_SEEDS) -> None:
    """Runnable checks, CSVs and the two figures."""
    mu_n = offdiag_uniform_mean(n_qubits)

    # -- hard checks: the instrument behaves as designed, and the floored control
    # really is floored.
    for key, _, law in ARMS:
        s0 = res[key]["snaps"][0]
        assert np.abs(s0 - np.tile(X[law], (n_seeds, 1))).max() < 1e-5, \
            f"{key}: the MLP must start at the identity (phi = x at epoch 0)"
    for key in ("mg_unif", "mg_raw"):
        p = res[key]["pur"]
        assert p.min() > n_qubits - 1 - 1e-6 and p.max() < n_qubits + 1e-6, \
            f"{key}: matchgate purity must stay on its floor [n-1, n] (eq:productform)"
    assert res["od_unif"]["rel"].mean(0)[-1] < 0.75, "off-diagonal uniform arm must descend"

    # -- the experiment's outcomes: printed, not asserted.  These are what is being
    # measured, so pinning them with a threshold up front would only bake in the
    # prediction.
    pr, pu = res["od_raw"]["pur"].mean(0), res["od_unif"]["pur"].mean(0)
    print(f"  [pred 1 chain-rule trap] od_raw: P^ {pr[0]:.2e} -> {pr[-1]:.3f} "
          f"({pr[-1] / pr[0]:.1e}x), L/L0 -> {res['od_raw']['rel'].mean(0)[-1]:.3f}"
          f"   -> {'frozen (trap holds)' if res['od_raw']['rel'].mean(0)[-1] > 0.85 else 'RESCUED (trap broken)'}")
    print(f"  [pred 2 selection] od_unif: P^ min={pu.min():.3f}, end={pu[-1]:.3f}; "
          f"guides mu_n/2={mu_n / 2:.3f}, mu_n={mu_n:.3f}, max n-1={n_qubits - 1}"
          f"   -> {'holds' if pu.min() >= mu_n / 2 else 'VIOLATED'}")
    gm_u, gm_r = res["od_unif"]["gvM"].mean(0), res["od_raw"]["gvM"].mean(0)
    print(f"  [MLP gradient signal] Var[d_MLP L] at epoch 1: unif={gm_u[0]:.2e} "
          f"raw={gm_r[0]:.2e}  ratio={gm_u[0] / max(gm_r[0], 1e-300):.1e}")
    for key, _, _ in ARMS:  # runaway-wrapping guard for the histogram reading
        d = res[key]["drift"].mean(0)[-1]
        print(f"  [drift guard] {key}: mean |phi-x| = {d:.2f} rad"
              + ("   WARNING: exceeds 2pi, histograms may be wrap-dominated" if d > 2 * np.pi else ""))

    # -- main CSV: per-arm curves vs epoch
    cols = dict(epoch=np.arange(1, epochs + 1))
    for key, _, _ in ARMS:
        r = res[key]
        # the relative loss spans decades and is plotted on a log axis, so its band is
        # geometric like the gradient variances (a linear +/- std band would reach
        # below zero wherever the seed spread exceeds the mean).
        rl, rl_lo, rl_hi = _geo(r["rel"])
        gw, gw_lo, gw_hi = _geo(r["gvW"])
        gm, gm_lo, gm_hi = _geo(r["gvM"])
        cols.update({
            f"{key}_rel": rl, f"{key}_rel_lo": rl_lo, f"{key}_rel_hi": rl_hi,
            f"{key}_gvW": gw, f"{key}_gvW_lo": gw_lo, f"{key}_gvW_hi": gw_hi,
            f"{key}_gvM": gm, f"{key}_gvM_lo": gm_lo, f"{key}_gvM_hi": gm_hi,
            f"{key}_pur": r["pur"].mean(0), f"{key}_pur_sd": r["pur"].std(0),
            f"{key}_drift": r["drift"].mean(0)})
    cols.update(mu_n=mu_n, n=n_qubits, n_seeds=n_seeds, n_train=N_TRAIN, depth=DEPTH,
                dim_g_od=len(lie_closure_paulis(xx_yy_generators(n_qubits))),
                dim_g_mg=dim_g(n_qubits))
    figures.write_csv("latent_drift", cols)

    # -- per-seed record behind the bands (not plotted)
    figures.write_csv("latent_drift_seeds", dict(
        seed=np.repeat(np.arange(n_seeds), epochs),
        epoch=np.tile(np.arange(1, epochs + 1), n_seeds),
        **{f"{k}_{q}": res[k][q].ravel() for k, _, _ in ARMS for q in ("rel", "pur")}))

    # -- histogram CSV: latent density per arm, snapshot and site (site -1 = pooled)
    edges = np.linspace(0.0, 2 * np.pi, HIST_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    arm_c, snap_c, site_c, bin_c, dens_c = [], [], [], [], []
    for key, _, _ in ARMS:
        for t in SNAPSHOTS:
            phi = np.mod(res[key]["snaps"][t], 2 * np.pi)
            for site in list(range(n_qubits)) + [-1]:
                v = phi.ravel() if site < 0 else phi[:, site]
                dens, _ = np.histogram(v, bins=edges, density=True)
                arm_c += [key] * HIST_BINS
                snap_c.append(np.full(HIST_BINS, t))
                site_c.append(np.full(HIST_BINS, site))
                bin_c.append(centers)
                dens_c.append(dens)
    figures.write_csv("latent_drift_hist", dict(
        arm=arm_c, snap=np.concatenate(snap_c), site=np.concatenate(site_c),
        bin_center=np.concatenate(bin_c), density=np.concatenate(dens_c)))

    figures.fig_latent_drift()
    figures.fig_latent_drift_hist()
    print("latent_drift: done")


def main() -> None:
    # the RNG call order of exp_precondition_training, so the datasets and the target
    # are bit-identical to fig:precondition_training (build_head consumes no rng).
    rng = np.random.default_rng(0)
    heads = {"od": build_head("od"), "mg": build_head("mg")}
    std_target = build_target(rng, jax.random.PRNGKey(1))
    X = {"unif": sample_uniform(rng, N_TRAIN, N_QUBITS),
         "raw": sample_raw(rng, N_TRAIN, N_QUBITS, RAW_EPS)}
    y = {k: jnp.asarray(std_target(v)) for k, v in X.items()}
    assert all(float(v.std()) > 0.5 for v in y.values()), "target must be non-trivial on both laws"

    print(f"latent_drift -- n={N_QUBITS}, depth={DEPTH}, {N_SEEDS} seeds, "
          f"MLP 1->{HIDDEN}->1 per site, mu_n={offdiag_uniform_mean(N_QUBITS):.3f}")
    part_figure(part_training(heads, X, y), X)


if __name__ == "__main__":
    main()
