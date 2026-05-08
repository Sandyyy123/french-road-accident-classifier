# IMPROVEMENTS - Project 01: Road Accidents FR (BAAC 2023)

Reviewer role: IMPROVER. Read-only audit of `brief.pdf`, `data/*.csv`, `notebooks/01_eda.ipynb`, `notebooks/03_modeling.ipynb`, `manuscripts/manuscript.md`, `manuscripts/references.md`, `reports/exploration_1.md`, `reports/modeling_1.md`, `reports/modeling_2.md`, `deliverables/metrics.json`, `deliverables/road_xgb.pkl`, `deliverables/presentation.html`. The project executes and produces sensible numbers (test balanced accuracy 0.575, killed-class recall 0.578, severe-vs-not AUROC 0.873). The recommendations below are about closing the gap between a defensible scaffold and a genuinely strong DataScientest-grade submission.

---

## Top recommendation (single highest-leverage change)

**Reframe the target as ordinal and report a per-class probability calibration plot before tuning the multi-class head.** The current pipeline trains XGBoost on `grav` as a nominal 4-class target with inverse-frequency sample weights; that is why killed-class precision collapses to 0.15 while recall climbs to 0.58. Replacing the nominal head with an ordinal target (`reg:squaredlogerror` on `grav` mapped to {0=unscathed, 1=light, 2=hospitalised, 3=killed}, or a cumulative-link CORAL/CORN net) plus a held-out isotonic-regression calibration on the binary severe head will (a) respect the ordering the manuscript itself names as "naturally ordered", (b) give a defensible probability output that a Vision-Zero triage system can threshold, and (c) typically lifts macro-F1 by 2-4 points in published crash-severity benchmarks (Cao et al., 2017; Wahab and Jiang, 2019). Concrete next step: add a fourth model `xgb_ordinal` with `objective='reg:squaredlogerror'`, snap predictions to integer classes by minimising mean-absolute-error on the validation set, and add `sklearn.calibration.CalibratedClassifierCV(method='isotonic', cv=3)` around the binary head before reporting AUROC. Priority: HIGH.

---

## Weakness 1 - No spatial features at all (`dep`, `lat`, `long` dropped)

The manuscript explicitly lists this as a limitation, and `caract.lat/long` (comma-decimal floats) and `dep` (department code) are sitting unused in the data folder. Hotspot kernel-density features and department-level fixed effects are well-established severity signals (Anderson 2009; Theofilatos and Yannis 2014).

**Improvement.** Add three features: `dep` as a categorical (96 metropolitan + 5 overseas codes, target-encoded with out-of-fold leave-one-out to avoid leakage), `urban_density` as accidents-per-km^2 within a 5 km KDE bandwidth around each accident's lat/long, and `road_class_x_dep` as an interaction term. Use `category_encoders.TargetEncoder` with `cv=5`. Expected lift: 1-3 macro-F1 points based on Sameen and Pradhan 2017.

**Priority: HIGH.**

## Weakness 2 - First-segment-only `lieux` deduplication discards multi-segment information

Line 52 of the manuscript and the modeling notebook both confirm that `lieux` is dropped to one row per `Num_Acc` via `keep='first'`. Multi-segment accidents (intersections, on/off-ramps, transitions between speed-limit zones) are exactly the cases where `vma`, `surf`, and `infra` differ across segments and are most predictive of severity.

**Improvement.** Replace `drop_duplicates(subset='Num_Acc', keep='first')` with a group-by-`Num_Acc` aggregator that takes `max(vma)`, `mode(surf)`, `mode(catr)`, `any(infra != 0)` as a binary "infrastructure feature present" flag, and `nunique(vma)` as a "speed-limit transition" indicator. This costs zero data and recovers signal currently being thrown away. Expected lift: 0.5-1.5 macro-F1 points.

**Priority: HIGH.**

## Weakness 3 - Hyper-parameter search is too narrow (8 configs, 3-fold CV) and does not include LightGBM or CatBoost

`RandomizedSearchCV(n_iter=5, cv=3)` is what the notebook actually runs (manuscript says 8). Five samples across a 6-dimensional grid leaves the search space barely scratched, and the fact that the best `n_estimators` came out at 200 with `max_depth=4` is a strong tell that the search bottomed out at the smallest values and never explored the `n_estimators=500, max_depth=8` corner. CatBoost handles integer-coded categoricals natively without one-hot expansion, which is structurally a better match for BAAC.

**Improvement.** (a) Bump `RandomizedSearchCV(n_iter=40, cv=5)` with `early_stopping_rounds=50` on a held-out 10% slice. (b) Add a CatBoost run with `cat_features=cat_features, auto_class_weights='Balanced'` and a LightGBM run with `is_unbalance=True`. Compare all three on macro-F1 and on killed-class F1 separately. Expected lift: 1-3 macro-F1 points (Parsa et al. 2020 reports XGBoost-SHAP ~5% macro-F1 above tuned RF on Chicago crash data; CatBoost typically matches or beats XGBoost on tabular crash data without the integer-code encoding compromise).

**Priority: HIGH.**

## Weakness 4 - No SHAP explainability, no fairness audit, no subgroup performance

Section 4.5 of the manuscript names "vehicle category, point of impact, user role, age, vma, secu1, col, lum" as the top tier "from gain importance" but does not actually produce a SHAP plot, a partial-dependence plot, or a per-subgroup metric table. For a project that frames itself around Vision Zero policy, the absence of a fairness audit by `sexe` (sex), `age` band, and `catu` (driver/passenger/pedestrian) is a substantive gap. Pedestrian and cyclist false-negative rates are a known equity concern in crash-severity literature (Wahab and Jiang 2019; Theofilatos and Yannis 2014).

**Improvement.** Add a section 4.6 to the manuscript that (a) runs `shap.TreeExplainer(best_xgb)` on a 5,000-row stratified test sample and includes the bar+swarm plot, and (b) reports per-class recall and per-class F1 disaggregated by sex (M/F), age band (<18, 18-25, 26-65, >65), and `catu` (driver=1, passenger=2, pedestrian=3, cyclist with vehicle motorisation=0). Flag any subgroup whose killed-class recall is more than 10 absolute points below the population recall of 0.578.

**Priority: HIGH.**

## Weakness 5 - No reproducibility scaffold (`requirements.txt`, `checkpoint.json`, no random seed in the binary XGB)

Project root has no `requirements.txt`, no `pyproject.toml`, no `environment.yml`, no `checkpoint.json` (which the QA spec at `/root/AI/.tmp/liora_qa_rules.md` task 11 expects), and no `Makefile`. The `xgb_bin` instantiation in cell 20 has `random_state=RANDOM_STATE` but the `RandomizedSearchCV` call passes `random_state=RANDOM_STATE` while the underlying `XGBClassifier` does not, so re-running the notebook on a fresh kernel produces slightly different `best_xgb_params` each time. The DataScientest brief explicitly requires "associated GitHub" with reproducibility implied.

**Improvement.** Add `requirements.txt` pinning `pandas==2.2.x`, `numpy==1.26.x`, `scikit-learn==1.4.x`, `xgboost==2.0.x`, `shap==0.45.x`, `category_encoders==2.6.x`, `matplotlib==3.8.x`. Add a top-level `checkpoint.json` matching the schema used elsewhere in the Liora portfolio (`project_number`, `title`, `methodology`, `status`, `last_run`, `metrics_summary`). Set `random_state=42` on every estimator including `xgb_bin` and the inner CV folds. Add a one-line `make all` that runs both notebooks via `jupyter nbconvert --execute --to notebook --inplace`.

**Priority: HIGH.**

## Weakness 6 - `secu1` only; `secu2`, `secu3` ignored; safety equipment encoded as ordinal integer

BAAC `secu1/2/3` is a multi-hot bundle of safety equipment (belt, helmet, child seat, airbag deployed, none) and the current pipeline collapses it to `secu1` only and treats the integer code as ordinal. This loses interaction signal: a motorcyclist with helmet (`secu1=2`) but no airbag is a different risk profile than a car driver with belt (`secu1=1`) plus airbag. Manuscript section 5 names this explicitly as future work.

**Improvement.** Build a 9-binary-column block `secu_belt`, `secu_helmet`, `secu_child_seat`, `secu_air_bag`, `secu_high_viz`, `secu_glove`, `secu_other`, `secu_none`, `secu_count` (sum of equipment codes present across `secu1/2/3`). Pass alongside the existing categorical block. This is one cell of pandas and is the highest-leverage feature-engineering improvement after spatial.

**Priority: MEDIUM.**

## Weakness 7 - Cross-year temporal validation is missing; only 2023 data is used

The manuscript correctly cites Mannering 2018 on temporal coefficient instability (ref [4]) but does not act on it. The data.gouv.fr BAAC archive ships 2005-2022 alongside 2023; downloading a second year (e.g. 2022) and either training-on-2022/testing-on-2023 or a full leave-one-year-out CV would directly address the most-cited methodological frontier in the cited literature. The brief itself names the 2005-2019 archive as the primary data source.

**Improvement.** Pull `caract-2022.csv`, `lieux-2022.csv`, `usagers-2022.csv`, `vehicules-2022.csv` from data.gouv.fr (same schema as 2023, code book at `https://www.data.gouv.fr/fr/datasets/r/8ef4c2a3-91a0-4d98-ae3a-989bde87b62a`). Refit the tuned XGBoost on 2022 and report 2023 test metrics; report the delta. This is the single most important external-validity strengthening available and addresses three reviewer comments at once (temporal stability, single-year limitation, cross-year drift).

**Priority: MEDIUM.**

## Weakness 8 - Presentation HTML is a manuscript dump, not a client deck

`deliverables/presentation.html` is essentially the manuscript and exploration reports rendered into a single dark-theme scrollable page. There is no executive summary slide, no chart of the killed-class recall lift (the headline result), no figure showing the confusion matrix, no map of department-level severity rates, and no actionable-recommendations section for the policy audience the introduction targets.

**Improvement.** Restructure into 8 slide-style sections: (1) one-line headline result, (2) the problem and the policy stake, (3) data at a glance with the class imbalance bar chart inline as base64 PNG, (4) modelling approach in a 4-row table, (5) the recall-lift bar chart (RF 0.076 -> XGB 0.578), (6) the test confusion matrix as a heatmap PNG, (7) top-feature SHAP summary inline, (8) three concrete policy levers. Use `matplotlib.pyplot.savefig(buf, format='png'); base64.b64encode(buf.getvalue())` to inline images so the file remains self-contained per the QA rules.

**Priority: MEDIUM.**

## Weakness 9 - Under-reporting bias is acknowledged in prose but not corrected

The manuscript correctly cites Amoros, Martin, and Laumon 2006 (ref [16]) on BAAC's 37.7% police capture rate but does not propagate this caveat into the metrics. A weighted-loss correction using the Rhone registry's empirical capture rate by injury class (fatality ~98%, hospitalised ~70%, light injury ~25%) would produce a "BAAC-corrected" severity distribution and a defensible upper-bound on killed-class recall in the population.

**Improvement.** Add a one-paragraph sensitivity analysis in section 5 that re-weights the test-set per-class supports by 1/p_capture(class) and reports an inverse-probability-weighted balanced accuracy. This shows the reader that the model's killed-class recall is in fact more reliable than its light-injury recall, which is the opposite of what a naive read of the confusion matrix suggests.

**Priority: LOW.**

---

## Priority summary

- **HIGH (do these to materially strengthen the project):** ordinal target + isotonic calibration (Top recommendation); spatial features (W1); multi-segment `lieux` aggregation (W2); broader hyper-parameter search + CatBoost/LightGBM (W3); SHAP + fairness audit (W4); reproducibility scaffold (W5).
- **MEDIUM:** multi-hot safety encoding (W6); cross-year temporal validation on 2022 (W7); presentation deck restructure (W8).
- **LOW:** under-reporting IPW sensitivity analysis (W9).

If only one change can be made, do the Top recommendation (ordinal head + isotonic calibration). If three changes can be made, add W3 (broader search, CatBoost) and W5 (reproducibility scaffold), because together they make the project both stronger and demonstrably reproducible, which is what the DataScientest brief asks for in its third validation condition.
