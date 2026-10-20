# CS5228 Final Project Plan
## Predicting Singapore HDB Monthly Rent

**Plan status:** Revised on 22 September 2026 after an independent critique and a direct audit of the supplied CSV files.

This project will build and evaluate a reproducible regression pipeline for predicting monthly HDB rent. The work will prioritize sound validation, clear ablations, interpretable results, and a concise report. Advanced models will be attempted only when the core pipeline is complete and a defined validation gate supports the extra complexity.

---

## 1. Project objective and questions

### 1.1 Prediction task

Given an HDB rental record and the supplied auxiliary data, predict `MONTHLY_RENT` in SGD.

The primary metric will be RMSE because the current project materials and plan identify it as the competition metric. The team must confirm the metric on the Kaggle competition page when it becomes available. MAE, signed mean error, fold-level RMSE, forecast-horizon RMSE, and segment RMSE will support diagnosis. MAPE will remain optional because the low-rent tail can make percentage errors unstable.

### 1.2 Research questions

1. How much do structural, block, geospatial, and macroeconomic feature groups improve out-of-time rent prediction?
2. Which model class gives the best balance of accuracy, interpretability, runtime, and reproducibility?
3. How does performance change across forecast horizons, towns, flat types, and rent bands?
4. Do contemporaneous macroeconomic variables improve the retrospective Kaggle task beyond lagged variables, and how much of that gain survives time-aware validation?

### 1.3 Success criteria

- Beat a recency-aware group-median baseline on every development fold.
- Produce a complete, leakage-safe training and evaluation pipeline.
- Support each retained feature family with an ablation result.
- Explain the main error patterns and limitations.
- Regenerate the tables, figures, validation results, and Kaggle submission from documented commands.

Kaggle rank will be reported as a secondary outcome. Methodological quality and reproducibility drive the project decisions.

---

## 2. Requirements and current evidence

### 2.1 Deliverable checklist

| Requirement | Planned artifact | Completion evidence |
|---|---|---|
| Progress report | PDF slide deck with 10–15 slides | Final 1–2 slides show next steps; deck includes progress and obstacles |
| Progress deadline | 4 October 2026 | Submission receipt or timestamp |
| Final report | Scientific-paper PDF, at most 8 pages excluding references and appendix | Page-count check |
| Team details | Names, student IDs, and team name in both reports | Report checklist |
| Workload allocation | Owner/task table in the appendix | Completed contribution table |
| Source submission | Zipped source or GitHub link | Clean reproduction from documented setup |
| Reproducibility | Environment lock, configs, saved splits, tests, and output manifest | Fresh-run verification |
| Final deadline | Pending release in the project brief | Update this plan once announced |

### 2.2 Verified dataset facts

These statements were reproduced from the supplied files:

- `train.csv`: 150,000 rows, January 2021–March 2025, 51 months.
- `test.csv`: 50,000 rows, March 2025–July 2026, 17 months.
- March 2025 appears in both sets: 3,032 training rows and 380 test rows. The competition test therefore contains a small same-month component followed by 16 future months.
- Training rent ranges from SGD 300 to SGD 7,600; mean SGD 2,719.04; median SGD 2,700; standard deviation SGD 749.73.
- `FURNISHED` is always `yes`; `FEE` is always `0.0`. Both can be removed after a schema assertion confirms this condition.
- The data contains no missing values in the core train and test files.
- `FLAT_TYPE` uses both spaced and hyphenated labels in training. Test labels use the hyphenated form.
- Training contains 20 `FLAT_MODEL` values and test contains 19. `improved maisonette` occurs only in training.
- The auxiliary files contain 9,464 HDB blocks, 243 MRT records, 337 schools, and 113 shopping malls.
- There are 374 training rents below SGD 1,000. Their cause is unknown from the supplied attributes.
- Grouping by month, address, flat type, flat model, floor area, and lease commencement date yields 13,434 repeated-profile groups containing 28,981 rows. Their mean within-group standard deviation is SGD 358.33, median SGD 282.84, and 75th percentile SGD 494.97.

### 2.3 Evidence labels

Every report claim and planning table will use one of these labels:

- **Verified:** reproduced by code from supplied data.
- **Supported externally:** backed by a cited authoritative source.
- **Hypothesis:** plausible explanation awaiting analysis.
- **Planned:** method or experiment that has not run.
- **Result:** output from the frozen evaluation pipeline.

The progress report must distinguish current results from planned work.

---

## 3. Data preparation and leakage controls

### 3.1 Pipeline invariant

Every transformation that uses `MONTHLY_RENT` must be fitted only on the training rows of the current split. This rule covers group medians, location premiums, target encodings, residual targets, calibration models, anchor weights, ensemble weights, and any target-based outlier rule. Validation and test targets may never influence preprocessing, feature construction, tuning, or model selection.

Static features derived without rent values, such as Haversine distances from supplied coordinates, may be cached for all rows after join and schema tests pass.

### 3.2 Core cleaning

1. Parse `RENT_APPROVAL_DATE` as a monthly period.
2. Normalize text with lowercase conversion, whitespace trimming, and a versioned abbreviation map.
3. Normalize flat types to `1-room`, `2-room`, `3-room`, `4-room`, `5-room`, and `executive`.
4. Preserve raw and normalized address fields until join validation is complete.
5. Assert core schema, data types, ranges, uniqueness expectations, and missing-value counts.
6. Drop `FURNISHED` and `FEE` only after train/test constant-value tests pass.
7. Remove additional exact duplicates from the training data across all original columns, keeping the first occurrence in original order and retaining its original `ROW_ID`. The current audit identifies 425 rows for removal.
8. Preserve every test row because the submission requires one prediction per original test record.
9. Record unmatched and duplicate auxiliary joins. Do not silently discard rows during enrichment.

### 3.3 Low-rent observations

The 374 rows below SGD 1,000 will be called **low-rent observations**. The supplied data does not identify their program or tenancy type.

The default experiment will keep them because RMSE evaluates all targets. A focused diagnostic will compare:

- standard squared-error training;
- Huber-loss training;
- segment performance for rents below SGD 1,000, SGD 1,000–3,500, and above SGD 3,500.

Target downweighting or clipping will be adopted only if development-fold RMSE improves under the predeclared decision rule. A hurdle classifier is outside the core scope because the underlying class is unobserved at inference.

### 3.4 Repeated-profile diagnostic

Within-profile rent dispersion will be presented as evidence of unobserved heterogeneity among records that share the available attributes. It is a descriptive noise proxy for a selected subset. It does not identify the Bayes error or a universal RMSE floor.

The analysis will report:

- group definition and size distribution;
- within-group standard deviations;
- RMSE from predicting each repeated row with its group mean, computed without self-inclusion where applicable;
- examples showing how omitted unit-level attributes may contribute to dispersion;
- limits caused by selected duplicate groups, small group sizes, and potentially different units within a block.

### 3.5 March 2025 matching

The earlier count of 98 exact matches could not be reproduced consistently. Raw and normalized matching produce different counts. The revised workflow will:

1. define the exact normalization map and join keys;
2. report matched rows and unique matched profiles separately;
3. aggregate duplicate reference rents using a training-only median;
4. estimate any blend weight through historical boundary simulations;
5. compare the anchor against an ordinary model feature and against no anchor.

No fixed blend weight will be selected from the competition test rows.

---

## 4. Feature groups

Feature groups enter the model sequentially so their contribution can be measured.

### F0: Core structural and temporal features

- Town, block, street, flat type, and flat model.
- Floor area and estimated room count.
- Lease commencement year and flat age at approval.
- Residual lease proxy, with invalid values flagged.
- Approval year, month, linear month index, quarter, and annual sine/cosine terms.
- Selected interactions such as floor area by flat type.

### F1: HDB block enrichment

- Latitude, longitude, maximum floor, completion year, subzone, planning area, and region.
- Distance to the CBD and selected regional centres using versioned coordinates.
- A shrunk location premium estimated from each fold's training targets. The estimate will use global fallback and minimum-count smoothing. The same fitted mapping will be applied to validation and test rows.

The block join must pass coverage, duplicate-key, coordinate-range, and one-row-per-transaction tests.

### F2: Public-transport features

- Distance to the nearest supplied MRT station.
- Counts of supplied stations within 500 m, 1 km, and 2 km.
- Distance to the nearest supplied multi-code station as an interchange proxy, after validating how station codes are represented.

The supplied MRT file does not include historical opening or construction dates. The core model will treat proximity as a static feature. Claims about historical availability or construction disutility require an authoritative dated source and a separate sensitivity analysis.

### F3: School and retail features

- Distance to nearest primary or mixed-level school after inspecting `MAINLEVEL_CODE`.
- Counts of qualifying schools within 1 km and 2 km.
- Distance to nearest shopping mall and mall counts within 1 km and 2 km.

Subjective school rankings will be excluded from the core model. A prestige feature may be added only with an authoritative, versioned source and a clear definition that could be reproduced.

### F4: Macroeconomic features

Two availability regimes will be evaluated:

1. **Deployment-style track:** variables lagged by at least one month, plus rolling summaries computed using data available by the prediction date.
2. **Competition track:** contemporaneous COE and stock values supplied through July 2026.

Candidate variables will be kept low-dimensional: COE Category A/B monthly summaries and a small, predeclared stock basket or composite. Correlations with rent will be treated as associations. Shared time trends will be checked through first differences, detrending, and incremental out-of-time performance. The report will state that contemporaneous future auxiliary values suit the retrospective competition setting and may be unavailable in a live forecasting system.

---

## 5. Evaluation design

### 5.1 Development folds

Use non-overlapping validation windows for ordinary model and feature decisions:

| Fold | Training period | Validation period | Purpose |
|---|---|---|---|
| D1 | 2021-01 to 2022-12 | 2023-01 to 2023-06 | Early regime |
| D2 | 2021-01 to 2023-06 | 2023-07 to 2023-12 | Mid regime |
| D3 | 2021-01 to 2023-12 | 2024-01 to 2024-06 | Recent development regime |

All preprocessing and tuning run separately inside each fold. Hyperparameters will be selected by mean development-fold RMSE with fold dispersion reported. Search budgets will be fixed before execution.

### 5.2 Locked temporal gate

- **Training:** 2021-01 to 2024-06.
- **Validation:** 2024-07 to 2024-12.
- **Use:** one final check after the model family, feature recipe, tuning budget, and decision thresholds are frozen.

The gate result will be reported even if it is weaker than the development estimate. Repeated use turns it into another development fold, so the run log must record its single authorized evaluation.

### 5.3 Long-horizon stress test

- **Training:** 2021-01 to 2023-10.
- **Validation:** 2023-11 to 2025-03, 17 months.

This diagnostic matches the test-set horizon length. It will be run after the main recipe is frozen and will report RMSE, signed bias, and RMSE by forecast month. Because it overlaps other calendar periods, it will not be pooled with development predictions or used to fit stacking weights.

### 5.4 Same-month boundary simulation

The competition includes March 2025 in both train and test. Historical cutoffs will simulate this structure:

- split the cutoff month deterministically into reference and query rows;
- fit target-derived features and anchors on the reference rows plus earlier months;
- score the held-out same-month query rows separately from subsequent-month rows;
- repeat at several development cutoffs;
- keep the split seed and row IDs in versioned files.

This experiment will determine whether a repeated-profile anchor helps and how its benefit changes between same-month and future-month cases.

### 5.5 Metrics and reporting

For every candidate, save:

- RMSE as the primary metric;
- MAE and signed mean error;
- RMSE by fold and forecast horizon;
- RMSE by flat type, region/town, rent band, and seen/unseen profile status;
- runtime and peak memory where practical;
- uncertainty through fold dispersion and a month/profile-cluster bootstrap on the locked gate. Any transaction-level bootstrap will be labeled descriptive because observations share temporal and profile dependence.

Random shuffled cross-validation may appear only as a leakage demonstration. It will not guide final model selection.

---

## 6. Model ladder and decision gates

### 6.1 Core models

| ID | Model | Feature set | Role |
|---|---|---|---|
| B0 | Recent training median by town × flat type with smoothed fallback | Core groups | Naive, reproducible baseline |
| B1 | Ridge regression with one-hot categoricals and scaled numeric features | F0 | Transparent linear baseline |
| M1 | CatBoost or LightGBM, selected after a small pilot | F0 | Primary nonlinear baseline |
| M2 | Primary GBDT | F0 + F1 | Block and location contribution |
| M3 | Primary GBDT | F0 + F1 + F2 + F3 | POI contribution |
| M4 | Primary GBDT | M3 + lagged F4 | Deployment-style macro contribution |
| M5 | Primary GBDT | M4 + contemporaneous F4 | Incremental competition-only macro contribution |
| M6 | Complementary GBDT | Best retained feature recipe | Model-family robustness |

One primary GBDT will receive the main tuning budget. The second receives a smaller fixed budget after M1–M5 are complete.

### 6.2 Optional extensions

The following are stretch experiments:

- low-dimensional trend plus hedonic residual GBDT;
- non-negative weighted blend of the two strongest models;
- XGBoost as a third tree family;
- an entity-embedding neural network.

The trend model must remain low-dimensional because only 51 training months exist. It should use a time spline or linear trend, annual seasonality, and a small set of lagged macro variables. Training it on transaction-level targets or composition-adjusted monthly effects will reduce changes caused by monthly flat mix. Every stage must be re-fitted within each fold.

### 6.3 Decision rules

- Retain a feature group when it improves mean development RMSE by at least 0.5%, or when it materially improves a predeclared stakeholder segment with no more than 0.5% degradation in overall RMSE.
- Require improvement in at least two of the three development folds.
- Attempt an optional extension only after B0–M5, leakage tests, and the progress report are complete.
- Retain an optional extension when it improves mean development RMSE by at least 1% over the strongest simpler model and does not create a severe horizon-specific bias.
- Prefer the simpler model when results fall inside these thresholds.
- Use Kaggle public scores as a final external signal. Do not tune repeatedly to the public leaderboard.

### 6.4 Tuning budget

- Pilot CatBoost and LightGBM with the same small set of defaults on D1–D3.
- Select one primary family using mean RMSE, runtime, and categorical handling.
- Limit the primary search to a predeclared number of trials and wall-clock budget.
- Tune only on D1–D3.
- Freeze the selected configuration before the locked temporal gate.
- Store every trial configuration and result.

---

## 7. Experiment registry

Each row will be updated with `planned`, `running`, `complete`, or `rejected`.

| ID | Hypothesis | Comparison | Training/validation | Primary rule | Cost | Status |
|---|---|---|---|---|---|---|
| E0 | A recency-aware grouped median is competitive with a global median | B0 variants | D1–D3 | Lowest mean RMSE | Low | Planned |
| E1 | Nonlinear interactions improve on Ridge | B1 vs M1 | D1–D3 | ≥0.5% RMSE gain in 2/3 folds | Medium | Planned |
| E2 | Block attributes add useful local information | M1 vs M2 | D1–D3 | Feature-group rule | Medium | Planned |
| E3 | MRT, school, and mall proximity add incremental value | M2 vs M3 and sub-ablations | D1–D3 | Feature-group rule | Medium | Planned |
| E4 | Lagged macro variables improve future prediction | M3 vs M4 | D1–D3 | Feature-group rule plus horizon bias | Medium | Planned |
| E5 | Contemporaneous supplied macro variables add competition value | M4 vs M5 | D1–D3 | Report gain and availability boundary | Medium | Planned |
| E6 | Robust loss improves overall or low-rent performance | Squared loss vs Huber | D1–D3 | Global rule plus rent-band RMSE | Medium | Planned |
| E7 | A same-month profile anchor improves boundary rows | No anchor vs tuned anchor | Boundary simulations | Mean boundary RMSE; future RMSE also reported | Medium | Planned |
| E8 | A second GBDT reduces model-specific error | Primary vs complementary GBDT | D1–D3 | Optional-extension rule | Medium | Planned |
| E9 | A simple blend reduces variance | Best single vs non-negative blend | Non-overlapping D1–D3 OOF predictions | Optional-extension rule | Medium | Conditional |
| E10 | A trend-residual design improves development-fold accuracy and bias | Best single vs two-stage | D1–D3 | Optional-extension rule | High | Conditional |

For each completed experiment, save its exact feature list, configuration, split IDs, seed, predictions, metrics, runtime, and artifact paths.

---

## 8. Interpretation and error analysis

### 8.1 Global interpretation

- Permutation importance on the locked gate after the score is recorded. It may explain the frozen model and cannot trigger feature or model changes.
- SHAP summary for a fixed validation sample when supported by the selected tree model.
- Accumulated local effects or partial dependence for a small predeclared set: floor area, flat age, CBD distance, MRT distance, and time.
- Dependence plots will be described as model behavior and associations.

### 8.2 Failure analysis

- Residual distribution and calibration plot.
- Error and signed bias by time horizon.
- Error by town/region, flat type, rent band, and profile familiarity.
- Review of the largest overpredictions and underpredictions using available fields.
- Separate discussion of low-rent observations and repeated-profile dispersion.

Potential explanations such as renovation quality, storey, facing, furnishing condition, and negotiation circumstances will be presented as plausible omitted variables unless supported by additional evidence.

### 8.3 Limitations

- Unit-level attributes are missing.
- Static POI files may not reflect historical openings.
- Contemporaneous future macro data limits live-deployment interpretation.
- The same-month test component differs from the future-month component.
- Location and macro associations do not establish causal effects.
- The hidden Kaggle labels prevent a complete test-horizon diagnosis.

---

## 9. Report and presentation plan

### 9.1 Progress report: 10–15 slides

1. Title, team, task, and stakeholder motivation.
2. Dataset and verified time split.
3. Target, categorical, and geographic EDA.
4. Data-quality findings and join audit.
5. Main methodological risk: temporal drift and target leakage.
6. Validation design, locked gate, and boundary simulation.
7. Completed baseline results.
8. Completed feature-ablation results.
9. Current error analysis or representative failures.
10. Obstacles, decisions, and open risks.
11. Next steps with owners and dates.
12. Next-step decision gates and expected final artifacts.

Slides will show only completed results available by the deck freeze date. Planned methods will use a clear `planned` label.

### 9.2 Final report: at most 8 pages

Suggested allocation:

- Introduction, motivation, and questions: 0.75 page.
- Data, EDA, and preprocessing: 1.25 pages.
- Validation and methods: 1.5 pages.
- Results and ablations: 2.0 pages.
- Interpretation and error analysis: 1.25 pages.
- Limitations and conclusion: 0.75 page.
- Layout allowance: 0.5 page.

The main paper should contain one principal ablation table and three or four essential figures. Supplemental plots belong in a short appendix. Related work will be integrated where it motivates a design choice. Advanced methods will receive appropriate references.

---

## 10. Reproducibility design

### 10.1 Planned repository structure

```text
Project/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── example-submission.csv
│   └── auxiliary/
├── configs/
│   ├── baseline.yaml
│   └── final.yaml
├── splits/
│   ├── development.json
│   ├── locked_gate.json
│   └── boundary_simulation.json
├── src/
│   ├── data.py
│   ├── features.py
│   ├── validation.py
│   ├── models.py
│   ├── evaluate.py
│   └── reporting.py
├── scripts/
│   ├── audit_data.py
│   ├── run_experiment.py
│   ├── build_report_assets.py
│   └── make_submission.py
├── tests/
│   ├── test_schema.py
│   ├── test_joins.py
│   ├── test_split_leakage.py
│   └── test_submission.py
├── artifacts/
│   ├── metrics/
│   ├── predictions/
│   ├── figures/
│   └── manifest.json
├── reports/
├── environment.lock
└── README.md
```

### 10.2 Reproduction requirements

- Exact dependency lock and Python version.
- Input file names, sizes, hashes, and schema checks.
- Versioned split definitions and feature configs.
- Fixed random seeds with documented GPU nondeterminism where relevant.
- Cached engineered features keyed by input and configuration hashes.
- Tests for row preservation, one-to-one joins, train/validation separation, target-derived feature fitting, and submission order/schema.
- A command that regenerates metrics, figures, tables, and the submission.
- An output manifest linking every reported number to a configuration and result file.
- A fresh-environment reproduction check before submission.

The README will give setup, data placement, execution, expected outputs, and approximate runtime.

---

## 11. Timeline, dependencies, and ownership

Replace `Owner A/B/C` with team names as soon as roles are agreed.

| Dates | Work | Dependency | Owner | Exit criterion |
|---|---|---|---|---|
| 22–23 Sep | Freeze schema audit, joins, split files, and evidence labels | Supplied data | Owner A | Audit script and tests pass |
| 24–25 Sep | Implement B0, B1, and primary-GBDT pilot | Split files | Owner B | D1–D3 baseline table complete |
| 26–28 Sep | Add F1 block and F2/F3 POI features | Join tests | Owner A | E2–E3 complete |
| 29–30 Sep | Add lagged and contemporaneous F4 tracks | Monthly alignment tests | Owner C | E4–E5 complete |
| 1 Oct | Robust-loss and boundary-anchor diagnostics | Core models stable | Owner B | E6–E7 complete or rejected |
| 2 Oct | Freeze progress-report results and figures | Completed experiments | All | Numbers trace to artifacts |
| 3 Oct | Draft and review progress deck | Result freeze | Owner C | 10–15 slides; checklist passes |
| 4 Oct | Submit progress report | Final deck | All | Submission confirmed |
| 5–8 Oct | Tune primary GBDT under fixed budget; test second GBDT | Progress report complete | Owner B | M6 decision recorded |
| 9–11 Oct | Conditional blend or trend-residual experiment on D1–D3 | Core pipeline complete | Owner B | E9/E10 retain/reject decision recorded |
| 12 Oct | Freeze the final feature and model recipe | All retention decisions complete | All | Signed configuration and manifest saved |
| 13 Oct | Run locked temporal gate once | Frozen final recipe | Owner A | Gate score recorded before interpretation |
| 14–15 Oct | Run long-horizon stress test and error analysis | Locked-gate score recorded; recipe remains frozen | Owner A/C | Descriptive diagnostics complete |
| After 15 Oct | Final report, code cleanup, fresh reproduction, Kaggle submission | Final deadline pending | All | Submission checklist complete |

When the final deadline is announced, the last row will be expanded into dated writing, review, reproduction, and submission milestones with at least two days of buffer.

---

## 12. Stop conditions and risk controls

- Stop adding feature families when two consecutive groups fail the retention rule.
- If the locked gate reveals material degradation, report it and investigate leakage, drift, and calibration diagnostically while keeping the submitted recipe frozen. Any later model revision requires a newly declared evaluation protocol and cannot reuse the gate for selection.
- Drop the neural model if the core experiments or report assets are incomplete by 12 October.
- Drop stacking when valid non-overlapping out-of-fold predictions are unavailable.
- Exclude any optional model without a completed retention decision before the 12 October recipe freeze from the final submission model.
- Treat locked-gate and long-horizon results as final evaluation evidence; neither may reopen model or feature selection.
- Drop historical MRT-status claims unless opening dates are sourced and tested.
- Report negative and null results because they justify the final design.
- Keep the final report readable without relying on the appendix.

This staged plan delivers a strong baseline, clear feature evidence, honest limitations, and reproducible outputs before optional complexity is considered.
