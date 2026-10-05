"""Search for large off-diagonal algebras with no Z-diagonal purity floor.

Scan translation-invariant dopants of the XX/YY chain and XY interaction
graphs. XIY and YIX dopants yield large algebras with zero Z-diagonal count
in the tested sizes. Compare purity floors and sampled variance with
statevector predictions, then write the hollow_dimension and hollow_sweep
figures. The finite-size scans do not prove an all-qubit result."""

from __future__ import annotations

import numpy as np

from unflattening import figures
from unflattening.utils.dla import random_dla_variance, pauli_to_bitmasks, word_matrix
from unflattening.utils.purity import product_state

POP = np.bitwise_count

CLOSURE_CAP = 250_000     # Lie-closure size cap (chain scan)
GRAPH_CLOSURE_CAP = 80_000  # tighter cap for the graph families
N_RANGE = tuple(range(4, 9))   # qubit range for the dopant scan
GRAPH_RANGE = (5, 6, 7, 8)     # qubit range for the graph scan
DIM_CUTOFF = 20_000       # stop a dopant's scan once the closure exceeds this
N_PROFILE = 8             # qubits whose basis profile drives the exact sweep
N_VARIANCE = 6            # qubits for the statevector variance validation
M_VARIANCE = 600          # random circuits in the variance validation
M_SWEEP_VARIANCE = 800    # random circuits per family in the fig:hollow axis check
DEPTH_VARIANCE = 2000     # rotations per random circuit
SIGMA_POINTS = 24         # angle-spread grid points in the regime sweep
# all sixteen TI dopants of range <= 3: 4 two-site NNN, 8 contiguous three-site,
# 4 Z-containing two-site; XXXX is an extra range-4 probe.
TI_DOPANTS = ("XIX", "XIY", "YIX", "YIY",
              "XXX", "XXY", "XYX", "XYY", "YXX", "YXY", "YYX", "YYY",
              "XZ", "ZX", "YZ", "ZY")
RANGE4_PROBE = "XXXX"


# ---- fast symplectic Lie closure (bitmask Pauli words) ----------------------

def capped_closure(gens, cap=CLOSURE_CAP):
    """Lie closure of (x, z) words; products of anticommuting pairs."""
    seen = set(gens)
    X = np.array([g[0] for g in seen], dtype=np.uint64)
    Z = np.array([g[1] for g in seen], dtype=np.uint64)
    frontier = np.arange(len(X))
    while len(frontier) and len(X) <= cap:
        new = {}
        for i in frontier:
            ax, az = X[i], Z[i]
            anti = (POP(ax & Z) + POP(az & X)) % 2 == 1
            for c in zip((X[anti] ^ ax).tolist(), (Z[anti] ^ az).tolist()):
                if c not in seen and c not in new:
                    new[c] = True
        if not new:
            break
        seen.update(new)
        frontier = np.arange(len(X), len(X) + len(new))
        X = np.concatenate([X, np.fromiter((c[0] for c in new), np.uint64)])
        Z = np.concatenate([Z, np.fromiter((c[1] for c in new), np.uint64)])
    return X, Z


def translate_pattern(pattern: str, n: int):
    """Translation-invariant placements of `pattern` on the open n-chain."""
    w = len(pattern)
    return [pauli_to_bitmasks("I" * k + pattern + "I" * (n - k - w)) for k in range(n - w + 1)]


def offdiag_basis(n: int):
    return translate_pattern("XX", n) + translate_pattern("YY", n)


def is_jw_bilinear(x: int, z: int, n: int) -> bool:
    """sigma Z..Z sigma' (contiguous, endpoints in {X,Y}) or a single Z."""
    if x == 0:
        return bin(z).count("1") == 1
    xs = [i for i in range(n) if (x >> i) & 1]
    if len(xs) != 2:
        return False
    j, k = xs
    interior = ((1 << k) - 1) & ~((1 << (j + 1)) - 1)
    return (z & interior) == interior and (z & ~interior & ~(1 << j) & ~(1 << k)
                                           & ((1 << n) - 1)) == 0


def bilinear_stats(X, Z, n):
    """(dim, d_Z, #non-bilinear, mean uniform-prior purity)."""
    dZ = int(np.sum(X == 0))
    nonbil = sum(not is_jw_bilinear(int(a), int(b), n) for a, b in zip(X, Z))
    supp = POP(X | Z)
    meanP = float(np.sum(np.where((X & Z) == 0, 0.5 ** supp.astype(float), 0.0)))
    return len(X), dZ, nonbil, meanP


# ---- (C) statevector variance validation ------------------------------------

def xy_generators(edges, n):
    """{XX_e, YY_e} generators for the XY interaction on the given edges."""
    gens = []
    for (i, j) in edges:
        for c in ("X", "Y"):
            s = ["I"] * n
            s[i] = s[j] = c
            gens.append(pauli_to_bitmasks("".join(s)))
    return gens


def weight_profile(X, Z):
    """(nx, nz) site counts of the XZ-only basis strings (Y-strings have zero
    R_y product-state expectation and drop out of every purity mean)."""
    m = (X & Z) == 0
    return POP(X[m]).astype(int), POP(Z[m]).astype(int)


def sweep_purity(nx, nz, sigmas):
    """Exact E[P_g] under the clustered prior of fig:criterion (angles at
    {0,pi} + N(0,sigma^2); both branches share E[cos^2]=(1+e^{-2s^2})/2,
    E[sin^2]=(1-e^{-2s^2})/2, sites iid)."""
    e = np.exp(-2.0 * np.asarray(sigmas) ** 2)
    s, c = (1 - e) / 2, (1 + e) / 2
    return np.array([float((si ** nx * ci ** nz).sum()) for si, ci in zip(s, c)])


def validate_variance(n: int, rng: np.random.Generator, M: int = M_VARIANCE,
                      depth: int = DEPTH_VARIANCE):
    X, Z = capped_closure(offdiag_basis(n) + translate_pattern("XIY", n))
    d = 2 ** n
    Gam = word_matrix(0, (1 << n) - 1, n)  # fermion parity Z^n
    bases, dims = [], []
    for s in (+1, -1):
        Pj = (np.eye(d) + s * Gam) / 2
        V = np.array([(word_matrix(int(a), int(b), n) @ Pj).flatten()
                      for a, b in zip(X, Z)])
        _, sv, Vt = np.linalg.svd(V, full_matrices=False)
        bases.append(Vt[sv > 1e-8].conj())
        dims.append(int((sv > 1e-8).sum()))
    O = word_matrix(*pauli_to_bitmasks("XX" + "I" * (n - 2)), n)
    PO = [float(np.sum(np.abs(B @ O.flatten()) ** 2)) for B in bases]
    gen_mats = [word_matrix(a, b, n) for a, b in offdiag_basis(n) + translate_pattern("XIY", n)]

    def predict(theta):
        psi = product_state(theta)
        rho = np.outer(psi, psi.conj()).flatten()
        return sum(float(np.sum(np.abs(B @ rho) ** 2)) * po / dj
                   for B, po, dj in zip(bases, PO, dims)), psi

    rows = []
    for label, theta in [("uniform-1", rng.uniform(0, 2 * np.pi, n)),
                         ("uniform-2", rng.uniform(0, 2 * np.pi, n)),
                         ("clustered eps=0.15", rng.normal(0, 0.15, n)),
                         ("exact theta=0", np.zeros(n))]:
        pred, psi0 = predict(theta)
        emp = random_dla_variance(psi0, gen_mats, O, depth, M, rng)
        rows.append((label, pred, emp))
        print(f"    {label:20s} analytic={pred:.3e}  empirical={emp:.3e}"
              + (f"  ratio={emp / pred:.2f}" if pred > 1e-20 else "  (identically 0)"),
              flush=True)
    return rows, dims


def part_hollow_sweeps(prof, sigma_points: int = SIGMA_POINTS) -> tuple[np.ndarray, dict, int]:
    """Exact regime sweep E[P_g](sigma) per family, for fig:hollow's results panel.

    prof: family -> (nx, nz) profile of the n = 8 basis.  Also returns d_Z of the
    floored family, the level the sweep flattens out at (annotated in the figure).
    """
    sigmas = np.logspace(np.log10(0.03), np.log10(np.pi), sigma_points)
    sweeps = {key: sweep_purity(*prof[key], sigmas)
              for key in ("g_od", "chord(1,4)", "+XIY", "chord(0,2)")}
    return sigmas, sweeps, int(np.sum(prof["chord(0,2)"][0] == 0))


def matchgate_gens(n):
    """Matchgate generators {Z_k} u {X_k X_{k+1}} as (x, z) bitmask words."""
    return ([pauli_to_bitmasks("I" * k + "Z" + "I" * (n - k - 1)) for k in range(n)]
            + translate_pattern("XX", n))


def part_selfcheck() -> None:
    """Closure dimensions against known values: matchgate, RZZ-doped, off-diagonal."""
    X, Z = capped_closure(matchgate_gens(6))
    assert len(X) == 66 and int(np.sum(X == 0)) == 6
    X, Z = capped_closure(matchgate_gens(6) + [pauli_to_bitmasks("IZZIII")])
    assert len(X) == 2046  # the paper's RZZ doping control
    X, Z = capped_closure(offdiag_basis(6))
    assert len(X) == 30 and int(np.sum(X == 0)) == 0


def part_dopant_scan(ns=N_RANGE, dopants=TI_DOPANTS, range4_probe=RANGE4_PROBE,
                     dim_cutoff: int = DIM_CUTOFF) -> dict:
    """(A) TI hollow dopants added to {XX,YY}: dim / d_Z / non-bilinear growth."""
    print("hollow_doping -- (A) TI hollow dopants added to {XX,YY}:")
    rec = {}
    for d in [None] + list(dopants) + [range4_probe]:
        name = "g_od" if d is None else f"+{d}"
        rows = []
        for n in ns:
            gens = offdiag_basis(n) + (translate_pattern(d, n) if d else [])
            dim, dZ, nonbil, meanP = bilinear_stats(*capped_closure(gens), n)
            rows.append((n, dim, dZ, nonbil, meanP))
            if dim > dim_cutoff:
                break
        rec[name] = rows
        print(f"  {name:7s} " + "  ".join(
            f"n={n}:dim={dim},dZ={dZ},nB={nb}" for n, dim, dZ, nb, _ in rows),
            flush=True)
    hollow16 = {f"+{d}" for d in dopants if all(r[2] == 0 for r in rec[f"+{d}"])}
    assert hollow16 == {"+XIY", "+YIX"}, \
        f"exactly the two XY-type NNN dopants must be hollow, got {hollow16}"
    print("  headline: among the sixteen range<=3 dopants exactly +XIY/+YIX have"
          " d_Z=0 with dim = 4^{n-1}-2^{n-1} = dim so(2^{n-1})+so(2^{n-1});"
          " every other one stays Gaussian or restores d_Z>0;"
          " +XXXX (range 4) is hollow at dim Theta(2^n).")

    # single-gate doping
    X, Z = capped_closure(offdiag_basis(8) + [pauli_to_bitmasks("XIY" + "I" * 5)])
    print(f"  single X_1Y_3 gate at n=8: dim {8*7} -> {len(X)}, "
          f"d_Z={int(np.sum(X == 0))}  (one hollow non-Gaussian gate suffices)")

    # dedicated n = 9 closure for the doped chain: the paper claims d_Z = 0 for
    # n <= 9; dim 4^8 - 2^8 = 65280 stays under the closure cap.
    print("  +XIY n=9 closure (largest case) ...", flush=True)
    dim9, dZ9, _, _ = bilinear_stats(*capped_closure(offdiag_basis(9) + translate_pattern("XIY", 9)), 9)
    assert dim9 == 4**8 - 2**8 and dZ9 == 0, (dim9, dZ9)
    print(f"  +XIY n=9: dim={dim9} (= 4^8-2^8), d_Z={dZ9}", flush=True)
    return rec


def part_graph_scan(ns=GRAPH_RANGE, cap: int = GRAPH_CLOSURE_CAP) -> tuple[list, dict]:
    """(B) XY interaction {XX_e, YY_e} on graphs: bipartiteness dichotomy for the floor."""
    print("\n(B) XY interaction {XX_e,YY_e} on graphs, d_Z of closure:")
    graph_rows = []
    graph_fam = {"chord(1,4)": [], "star": [], "chord(0,2)": []}
    for n in ns:
        for fam, name, edges in [
            ("chord(1,4)", f"path+chord(1,4) n={n}",
             [(k, k + 1) for k in range(n - 1)] + [(1, 4)]),
            ("star", f"star K(1,{n-1}) n={n}", [(0, k) for k in range(1, n)]),
            ("chord(0,2)", f"path+chord(0,2) n={n}",
             [(k, k + 1) for k in range(n - 1)] + [(0, 2)]),
        ]:
            dim, dZ, _, meanP = bilinear_stats(*capped_closure(xy_generators(edges, n), cap=cap), n)
            graph_rows.append((name, dim, dZ, meanP))
            graph_fam[fam].append((n, dim, dZ, meanP))
            print(f"    {name:22s} dim={dim:6d} d_Z={dZ:4d} meanP={meanP:8.3f}")
    print("  bipartite deg>2 (chord(1,4), star): exponential AND d_Z=0;"
          " odd chord (non-bipartite): d_Z>0.")
    return graph_rows, graph_fam


def validate_sweep_variance(n: int, rng: np.random.Generator, M: int = M_SWEEP_VARIANCE,
                            depth: int = DEPTH_VARIANCE) -> list:
    """Var = 2 P_g / dim g for the readout plotted on the right axis of fig:hollow.

    At the checked sizes, XX+YY has equal HS weight in equal-dimensional
    simple ideals. The number of ideals depends on the family and n.
    The chain has two at n>=5, while the n=8 graph families have four.
    This check compares the aggregate Haar prediction to finite random circuits.
    A single XX readout need not weight the chain ideals equally.
    """
    path = [(k, k + 1) for k in range(n - 1)]
    fams = {"g_od": offdiag_basis(n),
            "chord(1,4)": xy_generators(path + [(1, 4)], n),
            "+XIY": offdiag_basis(n) + translate_pattern("XIY", n),
            "chord(0,2)": xy_generators(path + [(0, 2)], n)}
    O = sum(word_matrix(*pauli_to_bitmasks(c + c + "I" * (n - 2)), n) for c in "XY")
    psi = product_state(rng.uniform(0, 2 * np.pi, n))
    rows = []
    for name, gens in fams.items():
        X, Z = capped_closure(gens)
        P = sum(float(np.real(psi.conj() @ (word_matrix(int(a), int(b), n) @ psi))) ** 2
                for a, b in zip(X, Z))
        emp = random_dla_variance(psi, [word_matrix(a, b, n) for a, b in gens], O, depth, M, rng)
        pred = 2 * P / len(X)
        rows.append((name, pred, emp))
        print(f"    {name:12s} dim={len(X):6d}  analytic={pred:.3e}  empirical={emp:.3e}"
              f"  ratio={emp / pred:.2f}", flush=True)
    ratios = [e / p for _, p, e in rows]
    assert all(0.85 < r < 1.25 for r in ratios), ratios   # 2-design + sampling tolerance
    return rows


def part_variance(rng, n: int = N_VARIANCE) -> list:
    """(C) LASA two-ideal variance formula vs deep random statevector circuits."""
    print(f"\n(C) LASA two-ideal variance vs statevector, g_h at n={n}:")
    var_rows, ideal_dims = validate_variance(n, rng)
    print(f"    ideal dims {ideal_dims} (= so(2^{n - 1}) twice)")
    # own seed: the manuscript quotes these ratios, so they must not depend on how
    # many draws validate_variance happened to consume first.
    print(f"  fig:hollow right axis, Var = 2 P_g/dim g at n={n}:")
    validate_sweep_variance(n, np.random.default_rng(3))
    return var_rows


def basis_profiles(n: int = N_PROFILE) -> tuple[dict, dict]:
    """(nx, nz) profile and dim g of each family's n-qubit basis, for the exact
    regime sweep.  The dims carry the sweep's second axis Var = 2 P_g / dim g."""
    path = [(k, k + 1) for k in range(n - 1)]
    closures = {
        "g_od": capped_closure(offdiag_basis(n)),
        "+XIY": capped_closure(offdiag_basis(n) + translate_pattern("XIY", n)),
        "chord(1,4)": capped_closure(xy_generators(path + [(1, 4)], n)),
        "chord(0,2)": capped_closure(xy_generators(path + [(0, 2)], n)),
    }
    return ({k: weight_profile(*XZ) for k, XZ in closures.items()},
            {k: len(XZ[0]) for k, XZ in closures.items()})


def main() -> None:
    part_selfcheck()
    rec = part_dopant_scan()
    _, graph_fam = part_graph_scan()   # the per-family rows carry the same data, keyed
    var_rows = part_variance(np.random.default_rng(11))

    # (D) figures: DLA growth + regime sweep (cf. purity_regime_contrast).
    prof, dims = basis_profiles()
    sigmas, sweeps, d_z = part_hollow_sweeps(prof)

    # long format throughout: one row per (family, n) for the two scans, one row per
    # sigma for the sweep.  fig_hollow_dimension reads the two scans, fig_hollow_sweep
    # the sweep.
    scan = [(k, *r) for k, rows in rec.items() for r in rows]
    figures.write_csv("hollow_scan", dict(
        family=[r[0] for r in scan], n=[r[1] for r in scan], dim=[r[2] for r in scan],
        d_z=[r[3] for r in scan], nonbil=[r[4] for r in scan], mean_p=[r[5] for r in scan]))
    graphs = [(fam, *r) for fam, rows in graph_fam.items() for r in rows]
    figures.write_csv("hollow_graphs", dict(
        family=[r[0] for r in graphs], n=[r[1] for r in graphs], dim=[r[2] for r in graphs],
        d_z=[r[3] for r in graphs], mean_p=[r[4] for r in graphs]))
    figures.write_csv("hollow_variance", dict(
        label=[r[0] for r in var_rows], analytic=[r[1] for r in var_rows],
        empirical=[r[2] for r in var_rows],
        n=N_VARIANCE, M=M_VARIANCE, depth=DEPTH_VARIANCE))
    figures.write_csv("hollow_sweep", dict(
        sigma=sigmas, **sweeps, d_z=d_z,
        **{f"dim_{k}": v for k, v in dims.items()}))

    figures.fig_hollow_dimension()
    figures.fig_hollow_sweep()
    print("hollow_doping: done", flush=True)
