# Feature handling table (draft)

One row per column in the Kaggle `train.csv`. This is the table the feedback asked for in the Preprocessing section. "Group" is the fixed feature group from `src/features.py` that the column or its derived features belong to.

| Column | Decision | What we do | Why | Group |
|---|---|---|---|---|
| `RENT_APPROVAL_DATE` | Transformed | Parsed to the first day of the month. Used to build `MONTH_INDEX` (months since Jan 2021) and the cyclic `MONTH_SIN` and `MONTH_COS`. Also defines the time holdout and joins the monthly COE, CPI and stock values. | Time has the strongest monotonic relation with rent (Spearman ρ = 0.56). A numeric month index lets models use the trend. The test set lies in later months than the training set. | G0 |
| `TOWN` | Kept as is | Lowercased and trimmed. Used as a categorical feature (26 levels). | Low cardinality and a strong location signal that every model can use. | G0 |
| `BLOCK` | Used only for joining | Part of `ADDRESS_KEY` (town, block and street), which joins each row to the HDB block table. Not a model input. | Block alone has 2,720 levels and no meaning without the street. After the join, latitude, longitude, subzone and block attributes carry its information. | (join) |
| `STREET` | Used only for joining | Part of `ADDRESS_KEY`, as for `BLOCK`. Not a model input. | 1,146 levels make it sparse for one hot or lookup models. Location is already captured by coordinates, subzone and town. After address normalization the join matches 100% of rows. | (join) |
| `FLAT_TYPE` | Transformed | Variant spellings ("4 room", "4-room") mapped to one label. Kept as a categorical and also as `FLAT_TYPE_ORDINAL` (1 room to 5 room as 1 to 5, executive as 6). | Flat type sets the number of rooms, which drives rent. The ordinal form helps linear and distance based models. | G0 |
| `FLAT_MODEL` | Kept as is | Lowercased and used as a categorical feature (20 levels). | It records layout and build era (for example "premium apartment", "dbss") that flat type does not capture. | G0 |
| `FLOOR_AREA_SQM` | Kept and transformed | Kept as is. Also `LOG_FLOOR_AREA_SQM` and `FLOOR_AREA_PER_ROOM_PROXY` (area divided by the flat type ordinal). | Area is closely tied to flat type. The log and per room forms give linear models a better shape, and the slides noted that linear models should keep only one of area and flat type. | G0 |
| `FURNISHED` | Dropped | Removed after a schema check. | Always "yes" in both train and test, so it carries no information. The check fails loudly if this ever changes. | none |
| `LEASE_COMMENCE_DATE` | Kept and transformed | Kept as a year. Also `FLAT_AGE_YEARS` (approval year minus lease year) and `INVALID_LEASE_DATE_FLAG`. | Older flats rent for less at equal size. One training row has a lease year after its approval year. We flag it and leave its age missing. | G0 |
| `FEE` | Dropped | Removed after a schema check. | Always 0.0 in both train and test. | none |
| `MONTHLY_RENT` | Target | Predicted in SGD. Not clipped or transformed in the saved data. The trend experiment can divide it by a monthly rent index during training only. | It is the Kaggle target and RMSE is computed on this scale. | target |

## Rows

* 425 exact duplicate training rows are removed, keeping the first. 149,575 training rows remain.
* No test rows are removed.
* About 50 suspicious rents are kept for now. Phase 2 trains with and without them.

## Derived groups beyond the core columns

* G1 location: block latitude, longitude and subzone from `sg-hdb-block.csv`, and distances and counts for MRT stations, schools, malls, rail lines and tertiary campuses.
* G2 block: maximum floor, completion year and block age from the block table, with missing flags for the 159 rows whose block lacks them.
* G3 market: COE, CPI and stock values for the approval month. These would not be known when predicting a future month, so they are a limitation (decision 6).
