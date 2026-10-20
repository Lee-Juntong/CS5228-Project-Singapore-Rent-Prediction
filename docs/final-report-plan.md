# CS5228 Team 55: Plan for the final report and code submission

Based on the progress report (14 slides) and the instructor's feedback, 15 Oct 2026.
The final deadline is still "Nov XX", so all timing below counts back from the deadline (D). Fill in real dates once it is announced.

## 1. Where the team stands

From the progress report, we have:

* Cleaned data: 149,575 training rows (425 duplicates removed) and 50,000 test rows kept. Address normalization raised the HDB block join from about 79% to 100%.
* EDA on the rent distribution, rent over time (median SGD 2,100 in 2021 to 3,100 in 2024), rent by flat type and town, Spearman correlations, and outliers (about 50 suspicious rows, kept for now).
* 35 predictors (30 numeric, 5 categorical): flat type order, log area, area per type index, age at approval, distances and counts for MRT, malls and schools, rail line proximity, COE and CPI by approval month, block height and subzone.
* Extra data from SingStat CPI, URA rail geometry and geocoded campuses.
* Three candidate models: CatBoost, LightGBM and TabM.

We still have no validation result, no simple baseline, no written decision for each core column (street, block) and no roadmap.

Training covers Jan 2021 to Mar 2025 and the test set covers Mar 2025 to Jul 2026, so the task is forecasting future rents. Rent rose about 50% over the training period, and tree models cannot extrapolate a trend past the last value they saw. This is likely our largest source of error, and it gives the report its main story.

## 2. Response to each feedback point

| Feedback | What we do | Where it goes in the report |
|---|---|---|
| "No single feature is a strong predictor" is too strong. Correlation only captures linear relations (monotonic for Spearman), and 0.56 is high. | Rewrite the claim as: time has a strong monotonic relation with rent (ρ = 0.56). Add one nonlinear view, either mutual information or permutation importance, plus partial dependence plots for area, age and MRT distance. | EDA section, one sentence plus a small table or figure |
| State what happened to every core feature, e.g. street, and assume the reader does not know the dataset. | Build a feature handling table with one row per core column: kept as is, transformed, used only for joining, or dropped, with the reason. Street and block are probably used to build the address key for the block join and are not model inputs. If so, explain why (high cardinality, and latitude, longitude and subzone already capture location). | Preprocessing section, one table |
| Add simpler models and compare all models in one simple setup. Do we need boosting at all? | Run a baseline ladder on the same split and feature set (Phase 1 below). | Methods and Evaluation sections, one results table |
| The progress report had no roadmap. | This plan covers it. | Not needed in the report |
| Look beyond RMSE and explain why the best RMSE is not lower. | Error analysis and a noise floor estimate (Phase 3). | Evaluation and interpretation section |
| Extra data is fine, but marginal RMSE gains are not worth chasing. | A feature group ablation shows what each data group adds. We freeze the data sources now and add no new external datasets after Phase 0. | Evaluation section, one ablation table |
| Start the report early and do not over-invest in Kaggle rank. | The report skeleton starts in Phase 1, tuning gets a fixed time budget, and Kaggle submissions are capped. | Not applicable |

## 3. Decisions to make now

1. Validation split. Use a time based holdout that mimics the test gap: train on Jan 2021 to Sep 2024 and validate on Oct 2024 to Mar 2025. Report random 5 fold CV once as well, to show how much a random split overstates performance.
2. Metrics. RMSE is the primary metric because Kaggle uses it. Also report MAE and MAPE, and RMSE per flat type and per validation month.
3. Feature groups. Use the same four groups in every experiment:
   * G0 core: flat type, floor area, lease commence and age, town, approval month index
   * G1 location: latitude, longitude, subzone, and distances and counts for MRT, malls, schools and rail lines
   * G2 block: block height and completion year
   * G3 market: COE, CPI and stock prices if used
4. Data freeze. We add no new external sources from here on. The ideas on the "possible data to explore" slide (segment rents, work pass counts, BTO schedules, travel times, unit storey) go into the limitations and future work paragraph.
5. Time trend. Test at least two ways to handle the trend: (a) the raw month index, and (b) predicting rent divided by a monthly rent index, then multiplying back. The index can be the training median per month, extended linearly into the test period. Compare both on the time holdout. This experiment shows directly why tree models struggle.
6. Leakage. Date matched CPI and COE use values that would not be known at prediction time. We keep them, list this as a limitation, and use the ablation to check whether they help at all.

## 4. Phases

This assumes D is about four weeks away. Shift the phases proportionally if it is earlier or later.

### Phase 0: Common setup (D minus 4 weeks, 2 to 3 days)

* Write one script or notebook that loads the cleaned data, builds the feature groups and runs the time holdout. Every model goes through the same function and writes its results to one CSV (model, features, RMSE, MAE, MAPE, train time).
* Fix random seeds and pin package versions in `requirements.txt`.
* Draft the feature handling table for all 9 core columns.
* One person owns this. The others start Phase 1 models against the shared interface.

### Phase 1: Baseline ladder (D minus 4 to 3 weeks)

Run all models on the same split with G0 and G1 features unless stated otherwise:

1. Global median, as a sanity floor
2. Group median lookup: median rent per town, flat type and approval quarter, falling back to coarser groups
3. Linear regression and Ridge, with one hot town and flat type. Keep only one of area and flat type, as the slides noted.
4. KNN on scaled latitude, longitude, area and month
5. Single decision tree
6. Random forest
7. LightGBM and CatBoost with default settings
8. TabM, only if time allows. It should not delay the report.

The output is one results table and a short paragraph on the gaps. We want to know how close the lookup and linear models get, and how much boosting adds over random forest.

### Phase 2: Feature and design experiments (D minus 3 weeks, about 1 week)

* Run the feature group ablation with the best two models: G0, G0+G1, G0+G1+G2 and all groups.
* Test the trend handling from decision 5.
* Train with and without the ~50 suspicious rows and report the difference in one line.
* Tune with a small fixed budget, for example 30 Optuna trials per boosting model on the time holdout, and stop when it runs out.

### Phase 3: Error analysis and insight (D minus 2 weeks, overlaps Phase 2)

The instructor asked for this most directly.

* Estimate the noise floor. Group rows with identical features (same block, flat type, area and month) and compute the spread of rent within each group. This approximates the error no model can remove, because the data does not record the unit's floor, furnishing, condition, or whether the whole flat or only rooms are rented. Compare it with our best RMSE.
* Break residuals down by town, flat type, rent band, flat age and validation month. Check whether errors grow with months ahead (trend drift) and whether large flats and central towns carry most of the error.
* Inspect the 20 largest errors and group them by cause: data error, missing attribute or rare segment.
* Compute permutation importance or SHAP for the best model, and partial dependence for area, age, MRT distance and month. Compare these with the Spearman ranking from EDA, which also answers the correlation feedback.
* Describe what the boosting model learns that Ridge misses, such as the town and flat type interaction and nonlinear distance effects.

### Phase 4: Final model and Kaggle (D minus 10 days, 1 to 2 days)

* Retrain the chosen model on all training data and submit.
* Submit to Kaggle about twice a week at most until D. Log each submission with its validation score so the public leaderboard does not drive decisions.

### Phase 5: Report writing (starts in Phase 1, main push from D minus 2 weeks to D minus 3 days)

* Set up the course LaTeX template in a git repo now. Each owner writes their section as results come in.
* Finish a complete draft by D minus 5 days. Review it together and cut it to 8 pages by D minus 2 days.
* Make our own figures. The CatBoost slide reused a diagram from an unrelated example ("Song length", "Listener age"). The report does not need to explain the models, so cite the papers and leave such diagrams out.

### Phase 6: Code packaging (D minus 1 week, alongside Phase 5)

* Organize the repo as `data/` (download instructions, no large raw data), `src/` (preprocessing, features, models, evaluation), `notebooks/` (EDA only), `results/` and `README.md`.
* The README explains how to set up the environment, run preprocessing, reproduce the results table and produce the submission file, with one command per step.
* By D minus 3 days, a teammate who did not write the code runs it from scratch in a clean environment.

### Buffer (D minus 2 days to D)

Proofread, check the page count, check names, student IDs, team name and the workload table, then zip and submit.

## 5. Report structure within 8 pages

| Section | Pages | Content |
|---|---|---|
| 1 Introduction and motivation | 0.75 | Stakeholders (renters, landlords, government). The task framed as forecasting future rents. Research questions (below). |
| 2 Data and EDA | 1.5 | Dataset description for a reader who has not seen it. Rent over time, by flat type and by town. Corrected correlation statement. Train and test time ranges. |
| 3 Preprocessing and feature engineering | 1.25 | Feature handling table for all core columns. Cleaning, joins, derived features, external data and its limits. |
| 4 Methods | 1.25 | Validation design and the reason for it. Baseline ladder. Trend handling. Tuning budget. References for CatBoost, LightGBM and TabM. |
| 5 Evaluation | 1.5 | Main results table, ablation table, time holdout compared with random CV. |
| 6 Error analysis and interpretation | 1.25 | Noise floor, residuals by segment, importance and partial dependence, and why the RMSE is not lower. |
| 7 Limitations and future work | 0.5 | Missing unit attributes, market data leakage, data we did not integrate. |
| Appendix | | Workload table (required), extra EDA plots, full hyperparameters. |

Possible research questions for the introduction:

1. How well can we predict rent for future months from flat, location and market attributes?
2. Do complex models (boosting, neural networks) clearly beat simple ones (lookup, linear) on this task?
3. Which attributes drive rent, and what limits prediction accuracy?

## 6. Possible split for four members

Adjust this to who did what so far.

| Member | Main responsibility |
|---|---|
| A | Phase 0 setup, validation, trend handling, Methods section |
| B | Simple baselines and ablation, Evaluation section |
| C | Boosting models and TabM, tuning budget, Kaggle log |
| D | Error analysis and interpretation, EDA section rewrite |
| All | Introduction, limitations, code review, final proofread |

## 7. Checklist before submission

* [ ] Every core column appears in the feature handling table
* [ ] Correlation wording fixed
* [ ] Simple and complex models compared on one split
* [ ] The report explains why the best RMSE is not lower and gives the noise floor number
* [ ] At most 8 pages excluding references and appendix
* [ ] Names, student IDs, team name and workload table included
* [ ] Code runs from the README in a clean environment
