"""Load a saved preprocessing/model bundle and predict the local test data."""

import argparse
from pathlib import Path
import joblib
import pandas as pd

from src.features import load_modeling_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}; choose a new output file")
    _, test = load_modeling_data()
    fitted = joblib.load(args.model)
    prediction = fitted.predict(test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ROW_ID": test.ROW_ID.to_numpy(), "y_pred": prediction}).to_csv(args.output, index=False)
    print(f"Saved {len(prediction):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
