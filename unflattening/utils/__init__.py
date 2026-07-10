"""unflattening -- numerical validation of the g-purity / barren-plateau manuscript.

Core quantum primitives (Pauli algebra, Lie closure, matchgate DLA, g-purity,
certified input states) now live in ``qml-essentials``
(:mod:`qml_essentials.algebra`, :mod:`qml_essentials.states`,
:class:`qml_essentials.operations.PauliWord`).  The modules below either
re-export them under the experiment-facing API (``dla``, the
``g_purity_from_basis`` re-export in ``purity``) or hold the paper-specific
pieces that are not general-purpose library primitives.

Modules:
  dla        -- adapter: Lie closure, matchgate + off-diagonal generators/basis,
                dim g = n(2n-1).
  purity     -- g-purity closed form (Eq. closedform); ``g_purity_from_basis``
                re-exported from qml_essentials.algebra.
  priors     -- uniform / clustered / raw angle priors and isotropic preconditioning.
  plotting   -- PGF/PNG output helpers.

The matchgate / XY-brickwork ansatz layers and the loss-variance harness now live
in ``qml_essentials.ansaetze`` (``Ansaetze.Matchgate`` / ``Ansaetze.XY_Brickwork``)
and ``qml_essentials.trainability`` (``loss_variance``).
"""
