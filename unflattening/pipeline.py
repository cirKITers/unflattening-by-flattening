"""Declare one Fluksio flow per study, with one node per experiment.

Each node runs an experiment's unchanged ``main()``, which writes its CSVs and
figures to ``dev/<study>/``, and returns those files as artifacts of the run.
"""

import importlib
import mimetypes

import fluksio
from fluksio import Flow, Port, node, use

from unflattening.utils import plotting


# The digest covers only this wrapper, not the experiment it runs, so a cached
# result could be stale.
@node(provides=Port("files", "list"), cache=False)
def experiment(*, module: str = "", study: str = "") -> list[dict]:
    """Run ``unflattening.experiments.<module>.main()`` with outputs in ``dev/<study>/``."""
    plotting.set_study(study)
    importlib.import_module(f"unflattening.experiments.{module}").main()
    return [
        fluksio.save_artifact(path, media_type=mimetypes.guess_type(path)[0] or "text/plain")
        for path in plotting.WRITTEN
    ]


def _study(name: str, title: str, *experiments: str) -> Flow:
    """A flow named after study ``name``, whose outputs are the experiments' file lists."""
    return Flow(
        name.replace("-", "_"),
        title=title,
        nodes=[
            use(experiment, id=e, wire={"files": e}, module=f"exp_{e}", study=name)
            for e in experiments
        ],
        outputs=list(experiments),
    )


s1_closed_forms = _study(
    "s1-closed-forms",
    "Closed-form g-purity: matchgate, uniform prior, off-diagonal",
    "closedform", "uniform_prior", "offdiag_closedform",
)
s2_dla_scaling = _study(
    "s2-dla-scaling",
    "Variance scaling with the DLA and the input g-purity",
    "theorem1", "bp_contrast", "input_purity",
)
s3_training = _study(
    "s3-training",
    "Trainability under preconditioning: loss, latent drift, gradient channels",
    "precondition_training", "latent_drift", "channel_scaling",
)
s4_hardness = _study(
    "s4-hardness",
    "Non-Gaussian doping and hard floor-free algebras",
    "doping", "hollow_doping",
)
s5_encoding = _study(
    "s5-encoding",
    "Re-uploading, encoding weights, and the fixed-Q acceptance rate",
    "reuploading", "encoding_weights", "acceptance_rate",
)
