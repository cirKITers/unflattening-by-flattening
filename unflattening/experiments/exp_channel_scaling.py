"""channel_scaling -- a barren plateau in the circuit is not one in the front end.

The purity bound of eq:variance governs Var_W[<O>] for an *in-algebra* readout, and
with it the gradient the circuit parameters see.  A classical map that produces the
encoding angles is fed by a different quantity,

    d<O>/dphi_k = (i/2) <psi(phi)| [Y_k, U^dag O U] |psi(phi)>          (R_y encoding)

and Y_k is not in g for any of the families here (none contains a single-qubit term),
so the commutator lies *outside* i g, where eq:variance does not apply.  The two
channels are therefore not bound to each other, and this experiment measures how far
apart they actually are.

Write the input angles as phi = a + sigma xi with a in {0, pi}^n a computational-basis
configuration.  Both channels vanish identically at sigma = 0 (eq:exact-zero for the
circuit; for the encoder because every [Y_k, B] with B in the basis of i g is again
non-diagonal, so its basis-state expectation is zero too).  They leave that point at
different orders, which is the whole content of the finding:

  * P_g and Var_W[<O>] need *two* sites displaced off the basis -- each term of
    eq:offdiag-closedform carries sin^2 phi_j sin^2 phi_k -- hence sigma^4;
  * the Y_k insertion in the commutator supplies one displacement for free, leaving
    one site to be displaced, hence amplitude sigma and variance sigma^2.

So the encoder channel is parametrically the *square root* of the circuit channel.
That is a real separation and it is what lets a trainable front end escape a region
where the circuit alone is stuck (exp_latent_drift), but it is not a free lunch:
halving an exponent does not remove it.  Whether the separation survives growing n is
the open question this experiment exists to answer, and it is why panel (b) sweeps n
rather than reporting the n = 6 numbers alone.

Method.  Random e^g circuits built directly from the Pauli generators (the 2-design
setting eq:variance assumes), rather than a fixed hardware ansatz, so that all three
families are treated identically -- the hollow doped family has no Model ansatz.
Applying the same circuit to |psi> and to Y_k|psi> gives both channels from one draw:

    <O>          = <chi|O|chi>,      d<O>/dphi_k = -Im <eta|O|chi>
    chi = U|psi>,                    eta = U Y_k |psi>

Families, all with the split-independent in-algebra readout O = X_1X_2 + Y_1Y_2
(one basis string in each of the two ideals, cf. exp_hollow_doping):

  * g_od   -- off-diagonal {XX, YY}, d_Z = 0, dim n(n-1); the floor-free witness.
  * mg     -- matchgate, d_Z = n, dim n(2n-1); floored, so its purity never collapses.
  * +XIY   -- g_od doped with X_k Y_{k+2}, d_Z = 0 but dim 4^{n-1} - 2^{n-1}; the
              exponential floor-free family, where the suppression is exponential in n
              rather than driven by sigma.

k is taken at a readout site, which is the best case for the encoder channel and the
one that governs an escape (a front end escapes through whichever site has the largest
gradient); a site far from the readout would additionally decay through the lightcone
and confound the n sweep.

Outcome (300 circuits per point).  The sigma laws hold, and the encoder channel obeys
the *same* law in every family regardless of its floor structure -- measured slopes
d log Var / d log sigma at n = 6:

    family   circuit   encoder
    g_od       3.96      2.01
    +XIY       3.55      1.91
    mg        -0.00      1.80

The matchgate row is the surprise, and it corrects the expectation this experiment was
written with.  The floor protects the *circuit* channel completely (slope 0, the purity
cannot fall below n-1), but it does not protect the encoder channel at all, which still
decays as sigma^2.  The floored family therefore shows the *reverse* asymmetry: at
clustered inputs its circuit trains happily while a classical front end is the part that
freezes.  So the two channels are not merely separated on floor-free families, they are
decoupled in general -- the floor structure governs one and says nothing about the other.
Neither is a special case of the other; d_Z decides which of the two is the dead one.

Suppression at sigma = 0.03, over n = 4..12 (4..8 for +XIY):

    family   circuit          encoder
    g_od     1.2e5 .. 7.8e5   2.5e1 .. 2.6e2
    +XIY     3.9e4 .. 4.5e5   1.2e2 .. 4.0e2
    mg       0.7 .. 0.87      1.6e2 .. 4.3e2

Both separations are flat in n: they neither close nor widen over the range, so this is
a parametric statement about sigma that holds uniformly in n, not an asymptotic claim.

SCOPE, and the most important limitation.  On the exponential family the separation is
absent in the n direction: at *uniform* inputs both channels decay at the same rate,
Var ~ 2^{-1.05 n} (circuit) and 2^{-1.06 n} (encoder) over n = 4..8.  The encoder channel
is therefore not protected against the DLA-dimension barren plateau, only against the
input-clustering one.  A classical front end can rescue a model whose gradients died
because its inputs sit on the computational basis; it cannot rescue one whose gradients
died because its algebra is exponentially large.

Figure: channel_scaling -- (a) both channels vs sigma at fixed n with the sigma^4 and
sigma^2 guides, (b) the clustered-input suppression of each channel vs n, against the
square-root prediction.

TODO: n <= 12 (8 for the exponential family), one readout, one encoding.  TODO: the
sigma^2 and sigma^4 laws are read off numerically; a closed form for the encoder channel
(a purity of rho against the coset Y_k g rather than against g) would prove them.
"""

from __future__ import annotations

import numpy as np

from unflattening import figures
from unflattening.utils.dla import (apply_word, dim_g, matchgate_generators,
                                    pauli_to_bitmasks, word_matrix)
from unflattening.utils.purity import (g_purity_closed_form, offdiag_closed_form,
                                       product_state)
from unflattening.experiments.exp_hollow_doping import (capped_closure, offdiag_basis,
                                                        translate_pattern)

N_SIGMA = 6            # qubits for the sigma sweep, panel (a)
SIGMAS = np.concatenate([[0.0], np.logspace(-2.5, np.log10(np.pi / 2), 11)])
SIGMA_FIXED = 0.03     # the manuscript's raw-input spread, used for panel (b)
M_CIRC = 300           # random e^g circuits per point
K_TH = 6               # angle configurations averaged per point
DEPTH_POLY = 400       # rotations per circuit, polynomial DLAs (dim <= ~300)
DEPTH_EXP = 2000       # ... exponential DLA (matches exp_hollow_doping)
SEED = 17

# (key, label, n range, circuit depth).  The hollow family stops at n = 8: its closure
# is 4^{n-1} - 2^{n-1}, so the analytic cross-check and the 2-design assumption both
# become impractical beyond that.
FAMILIES = (
    ("g_od", r"Off-diagonal ($d_Z{=}0$)", tuple(range(4, 13)), DEPTH_POLY),
    ("mg", r"Matchgate ($d_Z{=}n$)", tuple(range(4, 13)), DEPTH_POLY),
    ("+XIY", r"Hollow doped ($d_Z{=}0$)", (4, 5, 6, 7, 8), DEPTH_EXP),
)


DEPTHS = {key: depth for key, _, _, depth in FAMILIES}


def generators(key: str, n: int):
    """(x, z) bitmask generators of the family's DLA."""
    if key == "g_od":
        return offdiag_basis(n)
    if key == "mg":
        return [pauli_to_bitmasks(s) for s in matchgate_generators(n)]
    return offdiag_basis(n) + translate_pattern("XIY", n)


def readout(n: int):
    """O = X_1X_2 + Y_1Y_2 as (x, z) words, and the encoder site k on that bond."""
    return [pauli_to_bitmasks(c + c + "I" * (n - 2)) for c in "XY"], 0


def channels(key: str, n: int, thetas: np.ndarray, rng: np.random.Generator,
             m_circ: int = M_CIRC, depth: int = DEPTH_POLY) -> tuple[float, float]:
    """Var over random e^g circuits of <O> and of d<O>/dphi_k, averaged over ``thetas``.

    Both channels come from the same circuit draw: the rotation sequence is applied to
    |psi> and to Y_k|psi> in lockstep, so the pair (chi, eta) is exact and the two
    variances are measured on an identical ensemble.
    """
    gens = generators(key, n)
    O_words, k = readout(n)
    Yk = pauli_to_bitmasks("I" * k + "Y" + "I" * (n - k - 1))
    v_circ, v_enc = [], []
    for th in thetas:
        psi0 = product_state(th)
        eta0 = apply_word(psi0, *Yk, n)
        vals, grads = np.empty(m_circ), np.empty(m_circ)
        for m in range(m_circ):
            chi, eta = psi0.astype(complex), eta0.astype(complex)
            gi = rng.integers(0, len(gens), depth)
            ts = rng.uniform(0, 2 * np.pi, depth)
            for g, t in zip(gi, ts):  # exp(-i t G / 2) applied to both vectors
                x, z = gens[g]
                c, s = np.cos(t / 2), -1j * np.sin(t / 2)
                chi = c * chi + s * apply_word(chi, x, z, n)
                eta = c * eta + s * apply_word(eta, x, z, n)
            Ochi = sum(apply_word(chi, x, z, n) for x, z in O_words)
            vals[m] = np.real(np.vdot(chi, Ochi))
            grads[m] = -np.imag(np.vdot(eta, Ochi))
        v_circ.append(vals.var())
        v_enc.append(grads.var())
    return float(np.mean(v_circ)), float(np.mean(v_enc))


def analytic_circuit_var(key: str, n: int, theta: np.ndarray) -> float:
    """2 P_g / dim g, the eq:variance prediction for the circuit channel.

    Closed form for the two polynomial families; for the hollow family the purity is
    summed over the explicit closure, which is only affordable at small n.
    """
    if key == "g_od":
        return 2 * float(offdiag_closed_form(theta)) / (n * (n - 1))
    if key == "mg":
        return 2 * float(g_purity_closed_form(theta)) / dim_g(n)
    psi = product_state(theta)
    X, Z = capped_closure(generators(key, n))
    P = sum(float(np.real(np.vdot(psi, word_matrix(int(a), int(b), n) @ psi))) ** 2
            for a, b in zip(X, Z))
    return 2 * P / len(X)


def part_checks(rng) -> None:
    """apply_word against word_matrix, and the encoder identity against a finite
    difference of <O> in phi_k."""
    n = 4
    for s in ("XIYZ", "YYXI", "ZZII"):
        x, z = pauli_to_bitmasks(s)
        psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        assert np.allclose(apply_word(psi, x, z, n), word_matrix(x, z, n) @ psi), s
    print("  apply_word == word_matrix: ok")

    # d<O>/dphi_k = -Im <eta|O|chi> against a central difference, at one random circuit
    gens, (O_words, k) = generators("g_od", n), readout(n)
    Yk = pauli_to_bitmasks("I" * k + "Y" + "I" * (n - k - 1))
    gi, ts = rng.integers(0, len(gens), 60), rng.uniform(0, 2 * np.pi, 60)

    def evolve(v):
        for g, t in zip(gi, ts):
            x, z = gens[g]
            v = np.cos(t / 2) * v - 1j * np.sin(t / 2) * apply_word(v, x, z, n)
        return v

    def expval(th):
        chi = evolve(product_state(th))
        return sum(np.real(np.vdot(chi, apply_word(chi, x, z, n))) for x, z in O_words)

    th = rng.uniform(0, 2 * np.pi, n)
    chi, eta = evolve(product_state(th)), evolve(apply_word(product_state(th), *Yk, n))
    ana = -np.imag(np.vdot(eta, sum(apply_word(chi, x, z, n) for x, z in O_words)))
    eps = 1e-5
    fd = (expval(th + eps * np.eye(n)[k]) - expval(th - eps * np.eye(n)[k])) / (2 * eps)
    assert abs(ana - fd) < 1e-5, (ana, fd)
    print(f"  encoder identity vs finite difference: {ana:.6f} vs {fd:.6f}: ok")


def part_sigma(rng, n: int = N_SIGMA) -> dict:
    """(a) both channels vs the angle spread sigma, at fixed n."""
    print(f"\n(a) sigma sweep at n={n}:")
    out = {}
    for key, label, _, depth in FAMILIES:
        circ, enc, pur = [], [], []
        for sg in SIGMAS:
            b = rng.integers(0, 2, size=(K_TH, n)) * np.pi
            th = b + rng.normal(0.0, sg, size=(K_TH, n)) if sg > 0 else b.astype(float)
            c, e = channels(key, n, th, rng, depth=depth)
            circ.append(c)
            enc.append(e)
            pur.append(float(np.mean(offdiag_closed_form(th))))
        out[key] = (np.array(circ), np.array(enc), np.array(pur))
        sl = _slopes(out[key])
        print(f"  {key:6s} slopes over sigma in [0.003, 0.1]:  circuit={sl[0]:5.2f}  "
              f"encoder={sl[1]:5.2f}")
    return out


def _slopes(triple) -> tuple[float, float]:
    """d log(Var) / d log(sigma) on the small-sigma part of the sweep."""
    circ, enc, _ = triple
    m = (SIGMAS > 0.002) & (SIGMAS < 0.12)
    ls = np.log(SIGMAS[m])
    return (float(np.polyfit(ls, np.log(np.maximum(circ[m], 1e-300)), 1)[0]),
            float(np.polyfit(ls, np.log(np.maximum(enc[m], 1e-300)), 1)[0]))


def part_scaling(rng) -> dict:
    """(b) how much clustering suppresses each channel, as a function of n."""
    print(f"\n(b) n sweep at sigma={SIGMA_FIXED} (suppression = uniform / clustered):")
    out = {}
    for key, label, ns, depth in FAMILIES:
        rows = []
        for n in ns:
            th_u = rng.uniform(0, 2 * np.pi, size=(K_TH, n))
            b = rng.integers(0, 2, size=(K_TH, n)) * np.pi
            th_c = b + rng.normal(0.0, SIGMA_FIXED, size=(K_TH, n))
            cu, eu = channels(key, n, th_u, rng, depth=depth)
            cc, ec = channels(key, n, th_c, rng, depth=depth)
            rows.append((n, cu, eu, cc, ec))
            print(f"  {key:6s} n={n:2d}  circuit {cu:.2e}/{cc:.2e} = {cu / cc:8.2e}"
                  f"   encoder {eu:.2e}/{ec:.2e} = {eu / ec:8.2e}", flush=True)
        out[key] = np.array(rows, dtype=float)
    return out


def part_figure(sig: dict, scal: dict, rng) -> None:
    # CSVs first: the sweeps above are expensive, so nothing below may be able to
    # discard them.
    cols = dict(sigma=SIGMAS)
    for key, _, _, _ in FAMILIES:
        circ, enc, pur = sig[key]
        cols.update({f"{key}_circ": circ, f"{key}_enc": enc, f"{key}_pur": pur})
    cols.update(n_sigma=N_SIGMA, sigma_fixed=SIGMA_FIXED, m_circ=M_CIRC)
    figures.write_csv("channel_scaling", cols)

    rows = []
    for key, _, _, _ in FAMILIES:
        for n, cu, eu, cc, ec in scal[key]:
            rows.append((key, n, cu, eu, cc, ec))
    figures.write_csv("channel_scaling_n", dict(
        family=[r[0] for r in rows], n=np.array([r[1] for r in rows]),
        circ_unif=np.array([r[2] for r in rows]), enc_unif=np.array([r[3] for r in rows]),
        circ_clus=np.array([r[4] for r in rows]), enc_clus=np.array([r[5] for r in rows])))

    figures.fig_channel_scaling()

    # the eq:variance prediction must reproduce the measured circuit channel, which is
    # simultaneously a 2-design-convergence check on the chosen depths.
    for key, _, _, _ in FAMILIES:
        th = rng.uniform(0, 2 * np.pi, N_SIGMA)
        emp, _ = channels(key, N_SIGMA, th[None, :], rng, depth=DEPTHS[key])
        ana = analytic_circuit_var(key, N_SIGMA, th)
        print(f"  eq:variance check {key:6s} analytic={ana:.3e} empirical={emp:.3e} "
              f"ratio={emp / ana:.2f}")
        assert 0.7 < emp / ana < 1.4, f"{key}: circuit channel off eq:variance ({emp / ana:.2f})"

    # the finding itself: the encoder channel decays as sigma^2 in *every* family, while
    # the circuit channel is sigma^4 when d_Z = 0 and flat when floored.  So d_Z decides
    # which channel dies, and neither one bounds the other.
    for key, _, _, _ in FAMILIES:
        c_slope, e_slope = _slopes(sig[key])
        assert 1.6 < e_slope < 2.4, f"{key}: encoder channel must decay as sigma^2 ({e_slope:.2f})"
        print(f"  slopes {key:6s} circuit={c_slope:5.2f} encoder={e_slope:5.2f}"
              f"   -> {'square-root separation' if c_slope > 3 else 'reversed: floor protects the circuit only'}")
    assert _slopes(sig["mg"])[0] < 0.5, "the floored family's circuit channel must not decay"
    for key in ("g_od", "+XIY"):
        assert _slopes(sig[key])[0] > 3.0, f"{key}: floor-free circuit channel must decay as sigma^4"
    print("channel_scaling: done")


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"channel_scaling -- circuit vs encoder gradient channel, {M_CIRC} random "
          f"e^g circuits per point")
    part_checks(rng)
    sig = part_sigma(rng)
    scal = part_scaling(rng)
    part_figure(sig, scal, rng)


if __name__ == "__main__":
    main()
