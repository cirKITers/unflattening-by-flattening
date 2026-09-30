# Numerical experiments for *Unflattening by flattening*

This repository contains the numerical experiments and figure generator for the paper. The experiments study how input distributions and circuit algebras affect the trainability of quantum machine learning circuits.

## Getting Started

We require Python 3.12 and [uv](https://docs.astral.sh/uv/) installed.
Numerical experiments use JAX; the default CPU installation is sufficient.

```bash
uv sync --locked
uv run --no-sync unflatten --list
uv run --no-sync unflatten all
```

`all` runs the experiments in the order shown by `--list`. 
To run selected experiments, pass one or more names:

```bash
uv run --no-sync unflatten closedform uniform_prior theorem1
uv run --no-sync unflatten latent_drift channel_scaling
```

Each experiment can also run as a module, for example `uv run --no-sync python -m unflattening.experiments.exp_closedform`.
Experiment parameters, including random seeds, sample counts, qubit ranges, and training epochs, are constants near the top of each module.
The command-line interface uses these defaults; there is no separate configuration file or external dataset.
Results are stochastic, so small numerical differences across machines or JAX backends are possible.

Experiments write numeric CSV files to `data/` and render PGF figures with PNG previews in `figures/`. 
To redraw every figure from existing CSV files without rerunning the experiments, use:

```bash
uv run --no-sync python -m unflattening.figures
```

To redraw one figure, append its name, for example `uv run --no-sync python -m unflattening.figures offdiag_purity`.
PGF export requires a working LaTeX installation with `pdflatex`, `amsmath`, and `amssymb`.
PNG exports are additionally available and run without LaTeX.

## Experiments

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

The training curves for `precondition_training` and `latent_drift` compare progress within each input law.
Their uniform and clustered datasets have different task difficulty, so relative losses across laws are not a direct measure of which prior trains better.
The re-uploading and hollow-algebra studies are finite-size numerical checks, not asymptotic proofs.
