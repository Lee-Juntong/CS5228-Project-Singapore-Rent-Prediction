"""Score a teammate's holdout predictions, aligning rows strictly by ROW_ID."""

import argparse
from pathlib import Path
import pandas as pd

from src.evaluation import time_holdout_masks
from src.evaluation_artifacts import evaluate_predictions
from src.features import load_modeling_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train, _ = load_modeling_data()
    _, valid = time_holdout_masks(train)
    prediction = pd.read_parquet(args.predictions) if args.predictions.suffix == ".parquet" else pd.read_csv(args.predictions)
    print(evaluate_predictions(train.loc[valid], prediction, args.output))


if __name__ == "__main__":
    main()
