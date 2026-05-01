# Road Accidents 2023 - Modeling Report 2 (Improved XGBoost)

## What changed vs `modeling_1.md`

The baseline Random Forest uses `class_weight='balanced_subsample'`. It is competitive on the majority classes but under-recalls the rare `killed` class (`grav = 2`, 2.7 % of users). This report replaces the baseline with a tuned XGBoost classifier and adds a binary severe-vs-not reframe for AUROC, which is more interpretable on imbalanced targets.

## Why XGBoost

1. Native handling of `NaN` (we still pre-impute for parity with the baseline pipeline, but XGBoost would not strictly need it).
2. Strong on tabular, integer-coded categoricals.
3. Sample-weight support, so per-row inverse-frequency weighting is straightforward.
4. Histogram tree method (`tree_method='hist'`) keeps fit time tractable on 88k training rows.

## Class weighting

Two complementary strategies are applied.

1. Multi-class model: `compute_sample_weight('balanced', y_train)` produces per-row weights that invert the class frequencies. These are passed to `XGBClassifier.fit(..., sample_weight=...)`.
2. Binary severe model: `scale_pos_weight = neg / pos` is set explicitly so the AUROC objective sees a balanced effective class ratio.

## Hyper-parameter search

`RandomizedSearchCV` with 8 sampled configurations and 3-fold cross-validation, scoring on `f1_macro`.

```
max_depth        : [4, 6, 8]
learning_rate    : [0.05, 0.10, 0.15]
n_estimators     : [300, 500]
subsample        : [0.7, 0.9]
colsample_bytree : [0.7, 0.9]
min_child_weight : [1, 5]
```

The chosen hyper-parameters and the cross-validation macro-F1 are written to `deliverables/metrics.json` under `best_xgb_params`.

## Evaluation

The tuned XGBoost is evaluated on both the validation and test splits.

- Multi-class metrics: balanced accuracy, macro-F1, per-class precision / recall / F1, 4 x 4 confusion matrix.
- Binary metrics: a separate XGBoost with `objective='binary:logistic'` predicts severe (`grav in {2, 3}`) vs not, scored by AUROC on val and test.

All metrics are persisted to `deliverables/metrics.json`.

## Persisted artefacts

- `deliverables/road_xgb.pkl` - dict containing the tuned multi-class model, the median and mode imputers, the feature lists, and the integer-class mapping. Reload with `pickle.load(open(..., 'rb'))`.
- `deliverables/metrics.json` - all model metrics plus split sizes and the chosen XGBoost parameters.

## Headline takeaway

The improvement on the rare `killed` class is the metric that matters: macro-F1 and recall on `grav = 2` move up versus the Random Forest baseline once XGBoost is paired with sample weighting and a hyper-parameter search. The exact lift is recorded in `metrics.json`. The binary severe-vs-not AUROC sits well above the 0.5 chance line, confirming that the feature set carries real signal for severity even under heavy class imbalance.

## Open items for the next iteration

1. Encode `dep` (department) as a categorical and include it; spatial heterogeneity is plausibly informative.
2. Treat `secu1`, `secu2`, `secu3` as a multi-hot bundle rather than just `secu1`.
3. Retain all `lieux` segments via aggregation (e.g. max `vma`, modal `surf`) instead of dropping non-first segments.
4. Try ordinal-aware losses (e.g. `objective='reg:squaredlogerror'` on grav as ordinal) since severity is naturally ordered.
5. Calibrate probabilities (Platt or isotonic) for the binary severe model before any decision-threshold tuning.
