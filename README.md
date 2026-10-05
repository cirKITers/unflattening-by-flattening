# Unflattening by Flattening

This project contains the numerical experiments and figure generator for
*Unflattening by Flattening: How Input Distributions Shape Output Variance in
Angle-Encoded Circuits*. The paper studies how input distributions and circuit
algebras affect the trainability of quantum machine learning circuits.

Technology:

- [qml-essentials](https://github.com/cirKITers/qml-essentials): quantum Fourier models and Lie-algebra utilities
- [jaqsi](https://github.com/cirKITers/jaqsi): simulator in JAX
- JAX: array computation and automatic differentiation
- Optax: optimization and training
- NumPy: sampling and numerical analysis

## Layout

```text
unflattening/     experiments and utilities shared by all studies
├── cli.py        experiment registry and command-line runner
├── experiments/  one module per study, with parameters and a main() entry point
├── figures.py    CSV I/O and figure rendering
└── utils/
    ├── dla.py    Pauli-string Lie algebras and statevector validation
    ├── doping.py non-Gaussian RZZ gates in matchgate circuits
    ├── priors.py angle priors, polar encoding, and input preconditioning
    ├── purity.py product-state g-purity and analytic variance
    └── plotting.py shared figure style and PGF/PNG export
data/             numeric CSV results
figures/          PGF figures and PNG previews
```

## Getting started

Install Python 3.12 or later and [uv](https://docs.astral.sh/uv/), then run
`uv sync --locked`. The default JAX CPU installation is sufficient. PGF export
requires a working LaTeX installation with `pdflatex`, `amsmath`, and `amssymb`;
PNG previews are written even when PGF export fails.

**1. List the experiments.** The command shows the available names and a short
description of each study:

```sh
uv run --no-sync unflatten --list
```

**2. Run experiments.** Pass one or more names, or use `all` to run every
experiment in the order shown by `--list`:

```sh
uv run --no-sync unflatten closedform uniform_prior theorem1
uv run --no-sync unflatten latent_drift channel_scaling
uv run --no-sync unflatten all
```

Each experiment can also run as a module, for example
`uv run --no-sync python -m unflattening.experiments.exp_closedform`. Set random
seeds, sample counts, qubit ranges, and training epochs through the constants
near the top of each module. The command-line interface uses these defaults;
there is no separate configuration file or external dataset. Results are
stochastic, so small numerical differences across machines or JAX backends are
possible.

The experiment names and their outputs are listed below. CSV files use the
`.csv` extension; figures use `.pgf` and `.png`:

| Experiment | CSV output in `data/` | Figure names in `figures/` |
| --- | --- | --- |
| `closedform` | `closedform` | none |
| `uniform_prior` | `uniform_prior_mean`, `preconditioning_effect` | same names |
| `theorem1` | `matchgate_convergence`, `matchgate_scaling` | same names |
| `bp_contrast` | `dla_regime_contrast` | same name |
| `input_purity` | `input_purity_scaling` | same name |
| `precondition_training` | `precondition_training`, `precondition_training_seeds` | `precondition_training` |
| `offdiag` | `offdiag_purity`, `purity_regime_contrast`, `offdiag_variance` | first two names |
| `latent_drift` | `latent_drift`, `latent_drift_seeds`, `latent_drift_hist` | `latent_drift`, `latent_drift_hist` |
| `channel_scaling` | `channel_scaling`, `channel_scaling_n` | `channel_scaling` |
| `doping` | `doping_variance`, `doping_variance_fits` | `doping_variance` |
| `hollow_doping` | `hollow_scan`, `hollow_graphs`, `hollow_variance`, `hollow_sweep` | `hollow_dimension`, `hollow_sweep` |
| `reuploading` | `reuploading_depth` | same name |
| `encoding_weights` | `encoding_landscape`, `encoding_mean` | same names |
| `acceptance_rate` | `acceptance_rate` | none |

The training curves for `precondition_training` and `latent_drift` compare
progress within each input law. Their uniform and clustered datasets have
different task difficulty, so relative losses across laws are not a direct
measure of which prior trains better. The re-uploading and hollow-algebra
studies are finite-size numerical checks, not asymptotic proofs.

**3. Redraw figures.** Render every figure from existing CSV files:

```sh
uv run --no-sync python -m unflattening.figures
```

To redraw selected figures, append their names:

```sh
uv run --no-sync python -m unflattening.figures offdiag_purity
```

## Architecture

The experiments compare input g-purity and output variance across input
distributions and circuit algebras. Matchgate and off-diagonal XX/YY circuits
provide the main comparisons. Further studies cover non-Gaussian doping, data
re-uploading, encoding weights, and classical preconditioning. The training
studies fit a Fourier target and track loss, gradient dispersion, and latent
input distributions.

The project separates numerical experiments from figure rendering. Each
experiment generates its own inputs, writes numeric CSV files to `data/`, and
renders its figures in `figures/`. The figure module can redraw those figures
from existing CSV files without rerunning the experiments.
