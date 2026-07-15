"""closedform -- Proposition 1 (closed form for the g-purity).

Validates, for n = 2..8 and random angle configurations:
  (1) the matchgate Pauli-string basis Eq. (strings) has dim n(2n-1) and equals
      the Lie closure of the generators {Z_k} u {X_k X_{k+1}};
  (2) the closed form Eq. (closedform) equals the direct sum of squared Pauli
      expectations Eq. (gpurity-pauli) over that basis (Prop. 1) to ~1e-12;
  (3) the single-term form (n-1) + prod_k cos^2(theta_k) Eq. (productform) equals
      that basis sum (the telescoping collapse of Prop. 1);
  (4) JAQSI's statevector of the R_y product encoding reproduces product_state()
      and the single-site moments <Z_k> = cos theta_k.

Extends the n<=5 symbolic/numerical check reported in the manuscript appendix.
Writes data/closedform.csv.  No figure.
"""

from __future__ import annotations

import csv

import jax.numpy as jnp
import numpy as np

from qml_essentials import jaqsi as js
from qml_essentials.gates import Gates

from unflattening.utils import dla
from unflattening.utils.purity import product_state, g_purity_closed_form, g_purity_from_basis
from unflattening.utils.plotting import DATA_DIR

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

    out = DATA_DIR / "closedform.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
    return ok


def main() -> None:
    ok = part_validate(np.random.default_rng(0))
    print("closedform:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
