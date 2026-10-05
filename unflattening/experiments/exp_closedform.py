"""Validate the matchgate g-purity closed form against its Lie-closure basis.

For n=2..8, compare the closed form and its telescoping product form with
the squared Pauli expectations over the matchgate basis. Also check the
product-state implementation against JAQSI statevectors. Write numerical
errors and basis dimensions to closedform.csv; there is no figure."""

from __future__ import annotations

import csv

import jax.numpy as jnp
import numpy as np

from qml_essentials import jaqsi as js
from qml_essentials.gates import Gates

from unflattening.utils import dla
from unflattening.utils.purity import product_state, g_purity_closed_form, g_purity_from_basis
from unflattening.utils import plotting

N_MAX = 8
N_RANDOM = 50
TOL = 1e-10


def jaqsi_state(theta: np.ndarray) -> np.ndarray:
    n = len(theta)

    def prod(th):
        for q in range(n):
            Gates.RY(th[q], wires=q)

    return np.asarray(
        js.Script(f=prod, n_qubits=n).execute(
            type="state", args=(jnp.asarray(theta),), in_axes=None
        )
    )


def part_validate(rng, n_max=N_MAX, n_random=N_RANDOM, tol=TOL) -> bool:
    """Basis == Lie closure, and the closed / product forms == the basis sum (Prop. 1)."""
    rows = []
    ok = True
    for n in range(2, n_max + 1):
        basis = dla.matchgate_basis(n)
        closure = dla.lie_closure(dla.matchgate_generators(n))
        dim_ok = len(basis) == dla.dim_g(n) == len(closure)
        set_ok = set(basis) == closure

        max_diff = 0.0
        max_product_diff = 0.0
        max_state_err = 0.0
        for _ in range(n_random):
            theta = rng.uniform(0.0, 2 * np.pi, n)
            psi = product_state(theta)
            p_closed = float(g_purity_closed_form(theta))
            p_basis = g_purity_from_basis(psi, basis)
            max_diff = max(max_diff, abs(p_closed - p_basis))
            # Telescoping collapse Eq. (productform): (n-1) + prod_k cos^2(theta_k).
            p_product = (n - 1) + float(np.prod(np.cos(theta) ** 2))
            max_product_diff = max(max_product_diff, abs(p_product - p_basis))
            # JAQSI statevector cross-check (qubit-0-leftmost convention).
            max_state_err = max(max_state_err, float(np.max(np.abs(jaqsi_state(theta) - psi))))

        passed = dim_ok and set_ok and max_diff < tol and max_product_diff < tol and max_state_err < 1e-6
        ok = ok and passed
        print(
            f"n={n}: dim={len(basis)} (=n(2n-1)={dla.dim_g(n)}) "
            f"closure==basis:{set_ok} | max|closed-basis|={max_diff:.2e} "
            f"max|product-basis|={max_product_diff:.2e} "
            f"max|jaqsi-state|={max_state_err:.2e} -> {'PASS' if passed else 'FAIL'}"
        )
        rows.append(
            dict(
                n=n,
                dim_basis=len(basis),
                dim_g=dla.dim_g(n),
                dim_closure=len(closure),
                closure_eq_basis=set_ok,
                max_abs_diff=max_diff,
                max_product_diff=max_product_diff,
                max_state_err=max_state_err,
            )
        )

    out = plotting.DATA_DIR / "closedform.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plotting.WRITTEN.append(out)
    print(f"\nwrote {out}")
    return ok


def main() -> None:
    ok = part_validate(np.random.default_rng(0))
    print("closedform:", "PASS" if ok else "FAIL")
