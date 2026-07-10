# Numerical experiments for the `Unflattening by flattening` paper

Quantum circuits are simulated with the **JAQSI** simulator shipped in the [`qml-essentials`](https://cirkiters.github.io/qml-essentials/) package, which also provides the $\mathfrak{so}(2n)$ DLA basis, the $\mathfrak{g}$-purity, the Lie-closure helpers and the certified input states (`qml_essentials.algebra`, `qml_essentials.states`, `qml_essentials.operations.PauliWord`).  
This is accompanied by a small `utils` subpackage (`unflattening.utils`) which holds adapters for the given experiments.

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
