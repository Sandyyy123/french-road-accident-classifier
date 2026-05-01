# Road Accidents 2023 - Modeling Report 1 (Baseline)

## Goal

Multi-class user-level severity classifier on BAAC 2023.
Target `grav` is encoded as 1 = unscathed, 2 = killed, 3 = hospitalised, 4 = light injury.
The 118 rows with `grav = -1` (unknown) are dropped before modelling.

## Data assembly

The four BAAC tables were joined at the user grain.

1. `usagers` (one row per person, 125,789 rows) is the base table and carries the target.
2. `vehicules` joins on `Num_Acc + id_vehicule + num_veh` and adds vehicle category, manoeuvre, point of impact, motorisation.
3. `caract` joins on `Num_Acc` and adds time, light, weather, intersection type, collision type.
4. `lieux` joins on `Num_Acc`. Some accidents have multiple location segments, so `lieux` is deduplicated to one row per `Num_Acc` (first segment kept) before joining. This adds road category, lane count, surface, infrastructure, and the speed limit `vma`.

After dropping `grav = -1`, 125,671 user rows remain.

## Feature engineering

| Feature | Source | Transform |
|---------|--------|-----------|
| `hour` | `caract.hrmn` | parse HH:MM, take hour |
| `weekday` | `caract.an, mois, jour` | pandas `dt.weekday`, 0 = Monday |
| `age` | `usagers.an_nais` | `2023 - an_nais`, clipped to 0-110 |
| `vma` | `lieux.vma` | speed limit kept numeric |
| Vehicle | `vehicules.catv, obs, obsm, choc, manv, motor` | integer codes |
| Road / context | `lieux.catr, circ, nbv, surf, infra, situ, prof, plan` | integer codes |
| User / accident | `usagers.catu, sexe, trajet, secu1, place` and `caract.lum, agg, int, atm, col` | integer codes |

BAAC uses `-1` as the sentinel for missing values. Every feature column is converted to numeric and `-1` is replaced with `NaN` before imputation. Numeric features are imputed with the median; categorical (integer-coded) features with the mode.

## Train / val / test split

Stratified on `grav`: 70 % train, 15 % validation, 15 % test (`random_state = 42`). Split sizes are recorded in `deliverables/metrics.json`.

## Baseline models

Two baselines were trained on the same imputed feature matrix.

1. **Logistic Regression** (`sklearn.linear_model.LogisticRegression`)
   - `solver='lbfgs'`, `max_iter=1000`, `class_weight='balanced'`.
   - Numeric features standard-scaled; categoricals passed as integer codes (treated as ordinal). This is intentionally a weak baseline so the gradient-boosted improvement is interpretable.

2. **Random Forest** (`sklearn.ensemble.RandomForestClassifier`)
   - `n_estimators=300`, `max_depth=None`, `class_weight='balanced_subsample'`, `n_jobs=-1`.
   - Tree-based, so it tolerates the integer-coded categoricals natively.

## Evaluation

For every model the notebook prints:

- balanced accuracy
- macro-F1
- per-class precision / recall / F1
- a 4 x 4 confusion matrix

Numeric values land in `deliverables/metrics.json`. The Random Forest is the headline baseline because it consistently beats logistic regression on per-class recall for the rare `killed` class (`grav = 2`) under balanced class weighting.

## Known limitations of the baseline

1. Categorical codes are passed as integers, not one-hot. This is fine for tree models, weak for the linear baseline.
2. Only the first `lieux` segment per accident is kept; segment-specific information for multi-segment accidents is lost.
3. The 118 unknown-severity rows are discarded rather than imputed.
4. No spatial features yet: `dep`, `lat`, `long` are dropped at this stage.

These are addressed (or acknowledged as future work) in `modeling_2.md`.
