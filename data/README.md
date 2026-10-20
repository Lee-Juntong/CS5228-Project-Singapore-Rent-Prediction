# Data

Large and Kaggle supplied files are not in git. Put them here before running anything.

| Path | Source | In git |
|---|---|---|
| `train.csv`, `test.csv`, `example-submission.csv` | [Kaggle competition page](https://www.kaggle.com/competitions/cs-5228-2610-project), Data tab | No |
| `auxiliary/sg-hdb-block.csv`, `sg-mrt-stations.csv`, `sg-schools.csv`, `sg-shopping-malls.csv`, `sg-coe-prices.csv`, `sg-stock-prices.csv` | [Kaggle competition page](https://www.kaggle.com/competitions/cs-5228-2610-project), Data tab (auxiliary folder) | No |
| `auxiliary/sg-rail-lines.geojson` (24 MB) | Team added. URA "Master Plan 2025 Rail Line Layer" on [data.gov.sg](https://data.gov.sg/datasets/d_1c6365f0cad13a77bd79bdbb499131bf/view), GeoJSON. Only needed to rebuild `derived/block_extra_features.csv`. | No |
| `auxiliary/sg-cpi.csv` | Team added. SingStat table [M213751](https://tablebuilder.singstat.gov.sg/table/TS/M213751), "Consumer Price Index (CPI), 2024 As Base Year, Monthly". Series used: All Items, Accommodation, Actual Rentals For Housing. | Yes |
| `auxiliary/sg-tertiary-institutions.csv` | Team added. 13 university, polytechnic and arts college campuses. See the note below for how the coordinates were checked. | Yes |
| `derived/block_extra_features.csv` | Built by `src/external_features.py`: rail line and tertiary distances per block | Yes |
| `processed/` | Built by the preprocessing step below | No |

Data sources are frozen (decision 4 of the plan). Do not add new external datasets.

## Source notes

All sources were checked on 20 Oct 2026.

* **Kaggle files.** You must be signed in to Kaggle and have joined the competition to download them.
* **Rail lines.** The file's internal name is `G_MP25_RAIL_LINE`, which matches the data.gov.sg layer above (last updated by URA on 30 Jan 2026). The features use the `GRND_LEVEL` field to find above ground track.
* **CPI.** `sg-cpi.csv` holds the three series for Jan 2021 onwards, with `MONTH` as `YYYY-MM`. The values match the SingStat table exactly (for example All Items for Jan 2021 is 86.334). The same data is available from the [Table Builder API](https://tablebuilder.singstat.gov.sg/api/table/tabledata/M213751).
* **Tertiary campuses.** There is no recorded geocoding source for these coordinates. We checked them against the [OneMap search API](https://www.onemap.gov.sg/apidocs/) (Singapore Land Authority). Every campus is within 0.5 km of a OneMap result for its name, and 11 of 13 are within 0.2 km. The largest gaps are LASALLE (0.47 km) and Singapore Polytechnic (0.26 km). That is small next to the 2 km radius the count features use.

## Build the processed files

From the project root:

```bash
python -c "from src.data_preprocessing import run_pipeline; run_pipeline()"
```

This writes `processed/train_features.parquet` and `processed/test_features.parquet`.
