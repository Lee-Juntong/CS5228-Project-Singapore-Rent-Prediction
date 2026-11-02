# Verified interpretation findings

Source: selected `lightgbm_tuned`, G0+G1, index target, time holdout, seed 5228; archived reference run `20261025-210800_lightgbm_tuned_G0+G1_time_index`. These findings use the locally reproduced model and its matched 39-input, index-target Ridge comparator. The original archived prediction file was unavailable, so agreement with the archived run is established by aggregate metrics only. The retained local models agree with their local validation predictions by ROW_ID.

Generate the underlying results with `.venv/Scripts/python.exe -m scripts.run_member_d --member D --output results/member_d/<fresh-directory>`. Input, code and configuration hashes are in the parent manifest. The fixed interpretation sample contains 2,000 of 17,037 holdout rows; its baseline RMSE is SGD 435.99, versus SGD 445.78 on the full holdout.

The largest individual permutation effects are ordinal flat type (+166.19 SGD RMSE, repeat SD 3.77), subzone (+111.65, SD 4.38), and floor area (+25.61, SD 1.75). These describe deterioration after independently shuffling one input, with five permutations on the same sample. SD describes permutation variation, not a confidence interval. Correlated and deterministically related inputs can share or mask importance, and shuffling can create implausible feature combinations.

Time has the strongest absolute numeric Spearman association with rent across all 149,575 cleaned rows (rho 0.5640), while its individual permutation effect over the six holdout months is only +0.057 SGD (SD 0.055). This is consistent with the different populations, time ranges and index-target model. All holdout month indices (45–50) exceed the fitted maximum (44), and the boosting model's normalized mean prediction is identical at both endpoints. Its small SGD month response comes from the retained training-fitted index: the fitted tail slope is SGD 2.4476 per month. This does not imply that historical time trends are unimportant. Subzone is categorical and has no numeric Spearman entry; assigning arbitrary category codes would not provide a valid equivalent ranking.

The following standard PDP changes are differences between averaged full-SGD predictions at the grid endpoints, holding every other column fixed. They describe the fitted functions, not observed premiums or causal effects.

| Input range | Boosting change (SGD) | Ridge change (SGD) |
| --- | ---: | ---: |
| Floor area: 59 to 131 sq m | +226.75 | +314.49 |
| Flat age: 9 to 52 years | -86.04 | +1197.01 |
| Nearest open MRT: 0.1796 to 1.1499 km | -78.72 | -84.93 |
| Month index: 45 to 50 | +12.16 | -12.43 |

The age comparison demonstrates the danger of changing an engineered input independently. Updating lease commencement together with age changes the age response to -209.57 SGD for boosting and -581.23 SGD for Ridge; Ridge's direction reverses. Updating log-area and area-per-room alongside area gives endpoint changes of +217.15 and +263.13 SGD, respectively. Ridge's coherent area curve peaks at 104 sq m and then falls toward 131 sq m; this is model behavior under a hypothetical combination, not evidence that larger real flats rent for less.

The boosting MRT curve is nonlinear: predicted rent falls SGD 45.75 from 0.1796 to 0.4022 km, then SGD 32.97 from 0.4022 to 1.1499 km, with small local reversals. Ridge's standard MRT curve is linear in distance. This demonstrates a response shape the fitted boosting model can represent; it does not by itself attribute the matched holdout RMSE improvement (467.13 to 445.78 SGD) to that one feature or to town/type interactions.

Both PNGs were visually inspected and have readable axes, marks and legends without observed clipping. Independent numerical checks verified the sampled ROW_IDs, training-only monthly medians and tail-line fit for both models, full SGD restoration, local prediction agreement, every stored permutation mean and population SD, all PDP grids, 36 selected PDP ordinates across all 12 curves, and all five permutations for four consequential inputs. Details are in `audit.json`; endpoint summaries are in `pdp_curve_summary.csv`.

Interpretation remains limited by repeated model selection on this holdout, correlated inputs, implausible synthetic combinations, and the much longer test horizon. The coherent area/age sensitivities preserve exact derivative relationships but still hold other correlated flat and location attributes fixed. Month curves hold seasonal encodings and age fixed.
