# Data

Large and Kaggle supplied files are not in git. Put them here before running anything.

| Path | Source | In git |
|---|---|---|
| `train.csv`, `test.csv`, `example-submission.csv` | Kaggle competition page | No |
| `auxiliary/sg-hdb-block.csv`, `sg-mrt-stations.csv`, `sg-schools.csv`, `sg-shopping-malls.csv`, `sg-coe-prices.csv`, `sg-stock-prices.csv` | Kaggle competition page (auxiliary data) | No |
| `auxiliary/sg-rail-lines.geojson` (24 MB) | Team added, rail line geometry. Ask Juntong for the file. Only needed to rebuild `derived/block_extra_features.csv`. | No |
| `auxiliary/sg-cpi.csv` | Team added, SingStat CPI by month | Yes |
| `auxiliary/sg-tertiary-institutions.csv` | Team added, geocoded university and polytechnic campuses | Yes |
| `derived/block_extra_features.csv` | Built by `src/external_features.py`: rail line and tertiary distances per block | Yes |
| `processed/` | Built by the preprocessing step below | No |

Data sources are frozen (decision 4 of the plan). Do not add new external datasets.

## Build the processed files

From the project root:

```bash
python -c "from src.data_preprocessing import run_pipeline; run_pipeline()"
```

This writes `processed/train_features.parquet` and `processed/test_features.parquet`.
