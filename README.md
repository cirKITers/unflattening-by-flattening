# Numerical experiments for the `Unflattening by flattening` paper

In this work we study how the input distribution affects the trainability of a quantum machine learning circuits.

Documentation is still WIP.

## Usage

```bash
uv sync                                 # installs the requirements and the unflattening package
uv run unflatten --list                 # list the experiments
uv run unflatten uniform_prior          # run one experiment by name
uv run unflatten closedform theorem1    # run several, in order
uv run unflatten all                    # run every experiment
```

Each experiment also runs standalone as a module, e.g.
`uv run python -m unflattening.experiments.exp_closedform`.
