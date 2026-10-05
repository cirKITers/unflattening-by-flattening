# s4: Doping and hard floor-free algebras

## Question

What does non-Gaussian doping cost in trainability, and do large algebras
without a Z-diagonal purity floor exist?

## Method

| Experiment | Paper | Check |
| --- | --- | --- |
| `doping` | Appendix | t RZZ gates at fixed positions in matchgate brickwork; sampled variance against the t = 0 purity prediction, and Lie-algebra growth over n and t |
| `hollow_doping` | Outlook | Translation-invariant dopants of the XX/YY chain and XY interaction graphs; purity floors and sampled variance against statevector predictions |

Both are finite-size scans, not asymptotic results.

## Reproduce

```sh
dev/serve.sh
uv run --no-sync fluksio run s4_hardness --sync unflattening --wait
uv run --no-sync python -m unflattening.figures doping_variance hollow_dimension hollow_sweep
```

`data/` and `figures/` are gitignored.
