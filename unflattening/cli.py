from __future__ import annotations

import argparse
import importlib

EXPERIMENTS = {
    "closedform": ("exp_closedform", "Prop. 1: closed-form g-purity"),
    "uniform_prior": ("exp_uniform_prior", "Lemmas 1 & 2: uniform-prior purity"),
    "theorem1": ("exp_theorem1", "Theorem 1: matchgate Var = Theta(1/n)  [~5 min]"),
    "bp_contrast": ("exp_bp_contrast", "Conditional thesis: poly vs full DLA"),
    "input_purity": ("exp_input_purity", "Conditional thesis: input g-purity at fixed poly DLA  [~8 min]"),
    "precondition_training": ("exp_precondition_training", "Outlook: input distribution decides trainability (off-diagonal XX+YY DLA)"),
    "offdiag": ("exp_offdiag_closedform", "Prop 2: off-diagonal so(n)(+)so(n) closed form + prior contrast"),
    "doping": ("exp_doping", "Appendix: non-Gaussian doping trades trainability for hardness (dim BP)  [~min]"),
    "hollow_doping": ("exp_hollow_doping", "Outlook open question 1: hard floor-free (d_Z=0) families exist  [~10 min]"),
    "reuploading": ("exp_reuploading", "Appendix: dichotomy survives data re-uploading (exact zero at any depth)"),
    "encoding_weights": ("exp_encoding_weights", "Appendix: Hamming/binary/ternary weights, spectrum vs input distribution"),
    "acceptance_rate": ("exp_acceptance_rate", "Appendix: empirical acceptance rate of the fixed-Q test"),
}


def run(key: str) -> None:
    module_name, desc = EXPERIMENTS[key]
    print(f"=== {key} -- {desc} ===")
    importlib.import_module(f"unflattening.experiments.{module_name}").main()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run g-purity / barren-plateau validation experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "experiments",
        nargs="*",
        choices=list(EXPERIMENTS) + ["all"],
        metavar="<experiment>",
        help="experiment name(s) to run (see --list), or 'all' to run every one",
    )
    parser.add_argument("--list", action="store_true", help="list experiments and exit")
    args = parser.parse_args()

    if args.list or not args.experiments:
        for key, (_, desc) in EXPERIMENTS.items():
            print(f"  {key}  {desc}")
        if not args.list:
            print("\nPass one or more keys (or 'all'); see --help.")
        return

    keys = list(EXPERIMENTS) if "all" in args.experiments else args.experiments
    for key in keys:
        run(key)


if __name__ == "__main__":
    main()
