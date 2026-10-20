"""Run registered models through the shared evaluation and log the results.

Examples (from the project root):
    python -m scripts.run_experiment --model global_median lightgbm_default
    python -m scripts.run_experiment --model lightgbm_default --features ALL --target-mode index
    python -m scripts.run_experiment --model lightgbm_default --split random5
"""

from __future__ import annotations

import argparse

from src.evaluation import SPLITS, TARGET_MODES, run_experiment
from src.features import load_modeling_data
from src.models import MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", required=True, choices=sorted(MODELS))
    parser.add_argument("--features", nargs="+", default=["G0+G1"], help='e.g. G0, "G0+G1", "G0+G1+G2", ALL')
    parser.add_argument("--split", nargs="+", default=["time"], choices=SPLITS)
    parser.add_argument("--target-mode", nargs="+", default=["raw"], choices=TARGET_MODES)
    parser.add_argument("--member", default="", help="Your member letter, e.g. A")
    parser.add_argument("--notes", default="")
    parser.add_argument("--no-log", action="store_true", help="Print results without writing them")
    args = parser.parse_args()

    data = load_modeling_data()
    for model in args.model:
        for features in args.features:
            for split in args.split:
                for mode in args.target_mode:
                    run_experiment(model, MODELS[model], feature_set=features, split=split,
                                   target_mode=mode, member=args.member, notes=args.notes,
                                   data=data, log=not args.no_log)


if __name__ == "__main__":
    main()
