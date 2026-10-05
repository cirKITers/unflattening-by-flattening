# s2: Variance scaling with the algebra and the input

## Question

How does output variance scale with qubit count when the circuit algebra or
the input g-purity changes?

## Method

| Experiment | Paper | Check |
| --- | --- | --- |
| `theorem1` | Theorem 1 | Random matchgate circuits on RY product inputs: convergence of Var[<Z_i>] to P_g/dim g with depth, and 1/n scaling across qubit counts |
| `bp_contrast` | Conditional thesis | Same input and Z readout, matchgate (polynomial algebra) against Strongly_Entangling (full algebra) |
| `input_purity` | Conditional thesis | Product against Haar inputs at fixed matchgate algebra and Z readout; Haar variance predicted as 1/(2^n+1) |

## Reproduce

```sh
dev/serve.sh
uv run --no-sync fluksio run s2_dla_scaling --sync unflattening --wait
uv run --no-sync python -m unflattening.figures matchgate_convergence matchgate_scaling dla_regime_contrast input_purity_scaling
```

`data/` and `figures/` are gitignored.
