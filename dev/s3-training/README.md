# s3: Training under preconditioning

## Question

Does the input distribution decide trainability once training and a classical
preconditioner enter, and through which gradient channel?

## Method

| Experiment | Paper | Check |
| --- | --- | --- |
| `precondition_training` | Outlook | XX/YY re-uploading model trained on one Fourier target under uniform and clustered input laws, initial parameters paired across laws; relative loss and gradient dispersion over epochs |
| `latent_drift` | | Identity-initialised residual MLP before a re-uploading head, off-diagonal and matchgate heads, paired seeds; loss, gradient dispersion per parameter group, latent purity, and per-site angle histograms |
| `channel_scaling` | | Circuit against encoder variance near clustered inputs for off-diagonal, matchgate, and XIY-doped families: sigma^4 against sigma^2, and the qubit sweep |

Compare losses within an input law only: the uniform and clustered datasets
have different task difficulty.

## Reproduce

```sh
dev/serve.sh
uv run --no-sync fluksio run s3_training --sync unflattening --wait
uv run --no-sync python -m unflattening.figures precondition_training latent_drift latent_drift_hist channel_scaling
```

`data/` and `figures/` are gitignored.
