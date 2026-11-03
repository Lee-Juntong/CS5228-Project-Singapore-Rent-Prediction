"""Regression checks for whitespace normalization before CSV inference."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.data_preprocessing import DATASET_FILES, load_raw_data


class RawDataLoadingChecks(unittest.TestCase):
    def make_sources(self, root):
        for relative in DATASET_FILES.values():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("VALUE\n1\n", encoding="utf-8")

    def test_padded_headers_preserve_identifiers_parsing_order_and_source_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_sources(root)
            (root / DATASET_FILES["train"]).write_text(
                ' ROW_ID , BLOCK , MONTHLY_RENT \n9, 007 , 2500 \n4, 001 , 1800 \n', encoding="utf-8")
            (root / DATASET_FILES["hdb"]).write_text(
                ' BLOCK , POSTAL_CODE , LATITUDE , NOTE \n 007 , 012345 , 1.3 ," NA "\n', encoding="utf-8")
            (root / DATASET_FILES["schools"]).write_text(
                ' POSTAL_CODE , NAME \n 001234 ," first, school "\n', encoding="utf-8")
            before = {path: path.read_bytes() for path in root.rglob("*.csv")}
            datasets = load_raw_data(root)
            self.assertEqual(datasets["train"].ROW_ID.tolist(), [9, 4])
            self.assertEqual(datasets["train"].BLOCK.tolist(), ["007", "001"])
            self.assertEqual(datasets["train"].MONTHLY_RENT.tolist(), [2500, 1800])
            self.assertEqual(datasets["hdb"].POSTAL_CODE.tolist(), ["012345"])
            self.assertEqual(datasets["hdb"].LATITUDE.tolist(), [1.3])
            self.assertTrue(pd.isna(datasets["hdb"].NOTE.iloc[0]))
            self.assertEqual(datasets["schools"].POSTAL_CODE.tolist(), ["001234"])
            self.assertEqual(datasets["schools"].NAME.tolist(), ["first, school"])
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_unpadded_headers_keep_existing_field_parsing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_sources(root)
            (root / DATASET_FILES["train"]).write_text(
                'BLOCK,NOTE\n" 007 "," unchanged "\n', encoding="utf-8")
            train = load_raw_data(root)["train"]
            self.assertEqual(train.BLOCK.tolist(), [" 007 "])
            self.assertEqual(train.NOTE.tolist(), [" unchanged "])

    def test_duplicate_normalized_headers_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_sources(root)
            for header in ("BLOCK, BLOCK ", "BLOCK,BLOCK"):
                with self.subTest(header=header):
                    (root / DATASET_FILES["train"]).write_text(header + "\n007,001\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "duplicate column names"):
                        load_raw_data(root)


if __name__ == "__main__":
    unittest.main()
