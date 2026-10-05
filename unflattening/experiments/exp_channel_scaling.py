"""Compare circuit and encoder variance channels near clustered inputs.

Random Pauli-generator circuits test off-diagonal, matchgate, and XIY-doped
families. Near computational-basis inputs, the off-diagonal circuit variance
falls as sigma^4 while its encoder variance falls as sigma^2. Matchgate
purity retains a floor, yet its encoder channel still weakens. The qubit
sweep checks this separation at finite sizes; both channels of the
exponential XIY family decay with qubit count."""

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
