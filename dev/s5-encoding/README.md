# s5: Re-uploading and encoding

## Question

Does the regime distinction survive data re-uploading and other encoding
weights, and how often does the fixed-Q input-purity test accept?

## Method

| Experiment | Paper | Check |
| --- | --- | --- |
| `reuploading` | Appendix | Sampled variance over re-uploading depth for off-diagonal and matchgate readouts under uniform and clustered angles; exact zero of the off-diagonal output at basis-state angles |
| `encoding_weights` | Appendix | Off-diagonal purity of one scalar under Hamming, binary, and ternary weights; exact means against model statevectors |
| `acceptance_rate` | Appendix | Acceptance of Haar rotations Q on a fixed anisotropic dataset against the Markov bound |

## Reproduce

```sh
dev/serve.sh
uv run --no-sync fluksio run s5_encoding --sync unflattening --wait
uv run --no-sync python -m unflattening.figures reuploading_depth encoding_landscape encoding_mean
```

`data/` and `figures/` are gitignored.
