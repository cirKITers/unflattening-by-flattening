# s1: Closed-form g-purity

## Question

Do the closed forms of the product-state g-purity match their Lie-closure
bases, and how does the input prior move them?

## Method

| Experiment | Paper | Check |
| --- | --- | --- |
| `closedform` | Prop. 1 | Matchgate closed form and its telescoping product form against the basis sum for n = 2..8; product states against JAQSI statevectors |
| `uniform_prior` | Lemmas 1, 2 | Band n-1 <= P_g <= n and uniform-prior mean n-1+2^-n; polar-angle distribution after one rotation of anisotropic data |
| `offdiag_closedform` | Prop. 2 | Off-diagonal XX/YY closed form against the basis; uniform purity grows with n while clustered angles collapse it; 2 P_g/dim g against deep statevector circuits |

## Reproduce

```sh
dev/serve.sh
uv run --no-sync fluksio run s1_closed_forms --sync unflattening --wait
uv run --no-sync python -m unflattening.figures uniform_prior_mean preconditioning_effect offdiag_purity purity_regime_contrast
```

`data/` and `figures/` are gitignored.
