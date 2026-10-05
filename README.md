# Unflattening by Flattening

This project contains the numerical experiments and figure generator for
*Unflattening by Flattening: How Input Distributions Shape Output Variance in
Angle-Encoded Circuits*. The paper studies how input distributions and circuit
algebras affect the trainability of quantum machine learning circuits.

Technology:

- [qml-essentials](https://github.com/cirKITers/qml-essentials): quantum Fourier models and Lie-algebra utilities
- [jaqsi](https://github.com/cirKITers/jaqsi): simulator in JAX
- JAX: array computation and automatic differentiation
- Fluksio: data pipeline and experiment tracking
- Optax: optimization and training
- NumPy: sampling and numerical analysis

## Layout

```text
unflattening/     experiments and utilities shared by all studies
├── experiments/  one module per experiment, with parameters and a main() entry point
├── pipeline.py   one Fluksio flow per study, one node per experiment
├── figures.py    CSV I/O and figure rendering
└── utils/
    ├── dla.py    Pauli-string Lie algebras and statevector validation
    ├── doping.py non-Gaussian RZZ gates in matchgate circuits
    ├── priors.py angle priors, polar encoding, and input preconditioning
    ├── purity.py product-state g-purity and analytic variance
    └── plotting.py shared figure style and PGF/PNG export
dev/              one folder per study, plus the engine script
├── serve.sh          the engine, using ./.fluksio
├── s1-closed-forms/  closed-form g-purity: matchgate, uniform prior, off-diagonal
├── s2-dla-scaling/   variance scaling with the DLA and the input g-purity
├── s3-training/      preconditioned training, latent drift, gradient channels
├── s4-hardness/      non-Gaussian doping and hard floor-free algebras
└── s5-encoding/      re-uploading, encoding weights, fixed-Q acceptance rate
                      each study has its own data/ (CSV) and figures/ (PGF, PNG)
```

## Getting started

Install Python 3.12 or later and [uv](https://docs.astral.sh/uv/), then run
`uv sync --locked`. The default JAX CPU installation is sufficient. PGF export
requires a working LaTeX installation with `pdflatex`, `amsmath`, and `amssymb`;
PNG previews are written even when PGF export fails.

**1. Start an engine.** Run `dev/serve.sh` in a separate terminal. It gives
every worker all cores for BLAS, as a standalone run has. Fluksio's default
per-worker thread cap, which also applies to `--local`, changes the last digits
of BLAS-heavy results such as `hollow_variance`.

**2. Run a study.** Each study is one flow, named like its folder with
underscores. Pass `--sync unflattening` so the engine runs the current code:

```sh
uv run --no-sync fluksio run s1_closed_forms --sync unflattening --wait
```

To run every study:

```sh
for s in s1_closed_forms s2_dla_scaling s3_training s4_hardness s5_encoding; do
  uv run --no-sync fluksio run $s --sync unflattening --wait
done
```

Each experiment is one node. It writes its CSV files to `dev/<study>/data/` and
its figures to `dev/<study>/figures/`, and the run keeps those files as
artifacts, stamped with the commit they ran at. Set random seeds, sample
counts, qubit ranges, and training epochs through the constants near the top
of each experiment module; there is no separate configuration file or
external dataset. Results are stochastic, so small numerical differences
across machines or JAX backends are possible.

The studies, their experiments, and their outputs are listed below. CSV files
use the `.csv` extension; figures use `.pgf` and `.png`:

| Study | Experiment | CSV output in `data/` | Figure names in `figures/` |
| --- | --- | --- | --- |
| `s1-closed-forms` | `closedform` | `closedform` | none |
| | `uniform_prior` | `uniform_prior_mean`, `preconditioning_effect` | same names |
| | `offdiag_closedform` | `offdiag_purity`, `purity_regime_contrast`, `offdiag_variance` | first two names |
| `s2-dla-scaling` | `theorem1` | `matchgate_convergence`, `matchgate_scaling` | same names |
| | `bp_contrast` | `dla_regime_contrast` | same name |
| | `input_purity` | `input_purity_scaling` | same name |
| `s3-training` | `precondition_training` | `precondition_training`, `precondition_training_seeds` | `precondition_training` |
| | `latent_drift` | `latent_drift`, `latent_drift_seeds`, `latent_drift_hist` | `latent_drift`, `latent_drift_hist` |
| | `channel_scaling` | `channel_scaling`, `channel_scaling_n` | `channel_scaling` |
| `s4-hardness` | `doping` | `doping_variance`, `doping_variance_fits` | `doping_variance` |
| | `hollow_doping` | `hollow_scan`, `hollow_graphs`, `hollow_variance`, `hollow_sweep` | `hollow_dimension`, `hollow_sweep` |
| `s5-encoding` | `reuploading` | `reuploading_depth` | same name |
| | `encoding_weights` | `encoding_landscape`, `encoding_mean` | same names |
| | `acceptance_rate` | `acceptance_rate` | none |

The training curves for `precondition_training` and `latent_drift` compare
progress within each input law. Their uniform and clustered datasets have
different task difficulty, so relative losses across laws are not a direct
measure of which prior trains better. The re-uploading and hollow-algebra
studies are finite-size numerical checks, not asymptotic proofs.

**3. Redraw figures.** Render every figure from the existing CSV files, without
an engine:

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
experiment generates its own inputs, writes numeric CSV files to its study's
`data/`, and renders its figures in the study's `figures/`. The figure module
can redraw those figures from existing CSV files without rerunning the
experiments.

The experiments are organized in five Fluksio flows, one per study, declared in
`unflattening/pipeline.py`. Each node runs an experiment's `main()` unchanged.
Nodes use `cache=False`, because a node's fingerprint covers only the wrapper,
not the experiment code it calls.
