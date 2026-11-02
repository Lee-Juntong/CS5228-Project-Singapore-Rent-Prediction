"""Check that input preparation preserves source records and saved audits."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts import prepare_analysis_data as preparation
from src.config import PROJECT_ROOT
from src.analysis_report import file_hash


class PreparationChecks(unittest.TestCase):
    def test_staging_preserves_identifiers_order_and_archived_evidence(self):
        (PROJECT_ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as folder:
            root = Path(folder)
            data = root / "data"
            for name in preparation.DATASET_FILES.values():
                source = data / name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("ROW_ID,BLOCK\n9,007\n4,001\n", encoding="utf-8")
            padded = data / preparation.DATASET_FILES["hdb"]
            padded.write_text(" BLOCK , POSTAL_CODE \n 007 , 012345 \n 001 , 001234 \n", encoding="utf-8")
            before = {p: p.read_bytes() for p in data.rglob("*.csv")}
            archive = root / "results/analysis/reference/verification.json"
            archive.parent.mkdir(parents=True)
            archive.write_text('{"saved": true}', encoding="utf-8")
            processed = data / "processed"
            staging = root / "tmp/staged"

            def fake_pipeline(*, data_dir, processed_dir, audit_dir):
                self.assertEqual(data_dir, staging)
                self.assertEqual(processed_dir, processed)
                self.assertEqual(audit_dir, processed / "analysis_preparation")
                audit_dir.mkdir(parents=True)
                for part in ("train", "test"):
                    pd.read_csv(data_dir / f"{part}.csv", dtype="string").to_parquet(
                        processed_dir / f"{part}_features.parquet", index=False)
                (audit_dir / "audit_summary.json").write_text("{}", encoding="utf-8")
                return {"audit": {"cleaning": {}}, "train": [9, 4], "test": [9, 4]}

            with ExitStack() as stack:
                stack.enter_context(patch.object(preparation, "PROJECT_ROOT", root))
                stack.enter_context(patch.object(preparation, "DATA_DIR", data))
                stack.enter_context(patch.object(preparation, "PROCESSED_DIR", processed))
                stack.enter_context(patch.object(preparation, "run_pipeline", side_effect=fake_pipeline))
                stack.enter_context(redirect_stdout(io.StringIO()))
                preparation.main(["--staging", str(staging)])
            self.assertEqual(archive.read_text(encoding="utf-8"), '{"saved": true}')
            for source, original in before.items():
                self.assertEqual(source.read_bytes(), original)
            self.assertEqual((staging / "train.csv").read_bytes(), before[data / "train.csv"])
            staged = pd.read_csv(staging / preparation.DATASET_FILES["hdb"], dtype="string")
            self.assertEqual(staged.BLOCK.tolist(), ["007", "001"])
            self.assertEqual(staged.POSTAL_CODE.tolist(), ["012345", "001234"])
            record = json.loads((processed / "analysis_preparation/input_preparation.json").read_text(encoding="utf-8"))
            for path, digest in record["processed_sha256"].items():
                self.assertEqual(file_hash(root / path), digest)

    def test_missing_sources_do_not_create_staging_or_outputs(self):
        (PROJECT_ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as folder:
            root = Path(folder)
            staging = root / "tmp/missing"
            with patch.object(preparation, "PROJECT_ROOT", root), \
                    patch.object(preparation, "DATA_DIR", root / "data"), \
                    patch.object(preparation, "run_pipeline") as pipeline, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    preparation.main(["--staging", str(staging)])
            self.assertFalse(staging.exists())
            pipeline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
