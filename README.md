# Numerical experiments for the `Unflattening by flattening` paper

Quantum circuits are simulated with the **JAQSI** simulator shipped in the [`qml-essentials`](https://cirkiters.github.io/qml-essentials/) package, which also provides the $\mathfrak{so}(2n)$ DLA basis, the $\mathfrak{g}$-purity, the Lie-closure helpers and the certified input states (`qml_essentials.algebra`, `qml_essentials.states`, `qml_essentials.operations.PauliWord`).  
This is accompanied by a small `utils` package which holds adapters for the given experiments.

## Usage

```bash
uv sync                                    # installs requirements
uv run python main.py --list               # list the experiments
uv run python main.py uniform_prior        # run one experiment by name
uv run python main.py closedform theorem1  # run several, in order
uv run python main.py all                  # run every experiment
```

Each script also runs standalone, e.g.
`uv run python experiments/exp_closedform.py`.