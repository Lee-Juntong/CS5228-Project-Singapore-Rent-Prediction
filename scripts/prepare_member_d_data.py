"""Prepare local CSV copies and invoke the unchanged shared preprocessing."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

from src.config import DATA_DIR, PROJECT_ROOT
from src.data_preprocessing import DATASET_FILES, run_pipeline
from src.member_d_report import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, default=PROJECT_ROOT / "tmp/member_d_inputs")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        parser.error("Use the project's Python 3.10 virtual environment")
    staging = args.staging.resolve()
    if not staging.is_relative_to((PROJECT_ROOT / "tmp").resolve()) or staging.exists():
        parser.error("Choose a fresh staging directory under tmp/")
    sources = [DATA_DIR / name for name in DATASET_FILES.values()]
    for path in sources:
        if not path.is_file():
            parser.error(f"Missing required input: {path}")
    preparation = {"member": "D", "command": subprocess.list2cmdline(sys.argv), "inputs": []}
    for source in sources:
        relative = source.relative_to(DATA_DIR)
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        columns = pd.read_csv(source, nrows=0).columns
        padded = any(column != column.strip() for column in columns)
        if padded:
            # Read strings before stripping so identifiers retain leading zeroes.
            frame = pd.read_csv(source, dtype="string", keep_default_na=False)
            frame.columns = frame.columns.str.strip()
            if frame.columns.duplicated().any():
                raise ValueError(f"Trimming produces duplicate column names: {source}")
            for column in frame:
                frame[column] = frame[column].str.strip()
            frame.to_csv(destination, index=False)
        else:
            shutil.copyfile(source, destination)
        preparation["inputs"].append({"source": source.relative_to(PROJECT_ROOT).as_posix(),
            "source_sha256": file_hash(source), "staged_sha256": file_hash(destination),
            "formatting": "Trim header and field padding; retain all rows/order" if padded else "Unchanged byte copy"})
    audit_dir = PROJECT_ROOT / "results/member_d/preprocessing"
    result = run_pipeline(data_dir=staging, audit_dir=audit_dir)
    write_json(audit_dir / "input_preparation.json", preparation)
    print(result["audit"]["cleaning"])
    print(f"Processed train/test: {len(result['train']):,}/{len(result['test']):,}")


if __name__ == "__main__":
    main()
