#!/usr/bin/env bash
# The engine. Run from the repo root, so its store is the default ./.fluksio and
# there is no data-dir to pass. RUNS bounds the runs, cascades and worker
# processes in flight at once.
#
# The BLAS thread count changes float rounding (e.g. the SVD in hollow_doping),
# so every worker gets all cores, as a standalone run does. Fluksio would
# otherwise cap each worker at cores/RUNS, but only for vars left unset here.
set -u
cd "$(dirname "$0")/.."
for var in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS VECLIB_MAXIMUM_THREADS NUMEXPR_NUM_THREADS; do
  export "$var=${!var:-$(nproc)}"
done
exec uv run fluksio serve --port "${PORT:-8765}" \
  --max-runs "${RUNS:-5}" --max-cascades "${RUNS:-5}" --max-workers "${RUNS:-5}" "$@"
