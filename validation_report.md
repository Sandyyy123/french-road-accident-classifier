# VALIDATOR Report: Project #01 Road Accidents FR (BAAC 2023)

## Compact summary

**Overall: PASS-WITH-WARNINGS.**

The project is technically sound. Manuscript is within target word count, fully IMRaD, all 24 reference entries are present, 5 random CrossRef lookups all returned HTTP 200 with title match. Both notebooks parse as valid JSON. Methods named in the manuscript Methods section all map to code in `notebooks/03_modeling.ipynb` (substituting for the empty `src/`). Em-dash count across artefacts is 0; no AI-tell phrases present. Saved model artefacts (`road_xgb.pkl`, `metrics.json`) are present.

Warnings (not failures):
1. `src/` is empty; model code lives in `notebooks/03_modeling.ipynb`. Acceptable per rules for #01-#08.
2. `checkpoint.json` is absent at the project root.
3. Reference [21] (LightGBM, Ke et al. 2017) is declared but never cited inline.
4. `brief.md` is absent (only `brief.pdf` present).
5. `deliverables/presentation.html` contains 24 `href="http..."` hits, but all are DOI/arXiv citation links inside the references section, not external CSS/JS dependencies.

---

## Task results

### 1. Notebook validity (JSON parse)
- [PASS] `notebooks/01_eda.ipynb` parses as JSON (12 cells, kernel python3).
- [PASS] `notebooks/03_modeling.ipynb` parses as JSON (23 cells).
- [WARN] No notebook named `01_EDA.ipynb` exactly; project uses lowercase `01_eda.ipynb`. Both files exist and parse cleanly. Substitution: `01_eda.ipynb` is the EDA notebook; modeling notebook is `03_modeling.ipynb`.

### 2. Python script syntax
- [WARN] `src/` is empty. No `model_baseline.py` or `model_advanced.py` exist on disk.
  - Per rules: "Projects #1-#8 may have a saved model under `deliverables/`; src/ may be empty (model code lives in notebooks/)." This is the documented case. Substitution check: `notebooks/03_modeling.ipynb` was loaded via `json.load`, all 23 code cells were concatenated and inspected; no `SyntaxError` occurred when scanning.

### 3. Manuscript word count
- [PASS] `manuscripts/manuscript.md`: **4,408 words** (target 4,000-5,000). Within bounds.

### 4. Self-contained HTML
- [PASS] `deliverables/presentation.html` contains zero external CSS/JS dependencies (`<link href="http..."` and `<script src="http..."` both 0).
- [WARN] 24 `href="http..."` hits exist, but all are DOI / arXiv citation links inside the References slide (e.g. `https://doi.org/10.1016/j.aap.2011.03.025`, `https://arxiv.org/abs/1603.02754`). These are intentional reference targets, not external resources the page depends on to render. No external fonts, stylesheets, or scripts are loaded. The HTML is functionally self-contained for offline rendering.

### 5. IMRaD completeness
Manuscript section roll:
- [PASS] Title (line 1)
- [PASS] Abstract
- [PASS] Introduction (`## 1. Introduction`)
- [PASS] Data (`## 2. Data`)
- [PASS] Methods (`## 3. Methods`) with subsections 3.1-3.4
- [PASS] Results (`## 4. Results`) with subsections 4.1-4.5
- [PASS] Discussion (`## 5. Discussion`)
- [PASS] Conclusion (`## 6. Conclusion`)
- [PASS] References (`## References`)

All eight required sections present.

### 6. Method drift (Methods section vs code)
Methods section names: logistic regression (LBFGS, `class_weight='balanced'`), random forest (`balanced_subsample`, `n_estimators=300`), tuned XGBoost (`tree_method='hist'`, `compute_sample_weight('balanced')`, `RandomizedSearchCV`, binary reframe with `scale_pos_weight = neg/pos`), stratified train/val/test split, median/mode imputation.

Substituting `notebooks/03_modeling.ipynb` for the missing `src/*.py`:
- [PASS] `LogisticRegression` present
- [PASS] `LBFGS` / `lbfgs` present
- [PASS] `class_weight` / `balanced_subsample` present
- [PASS] `RandomForestClassifier` present
- [PASS] `XGBClassifier` present
- [PASS] `RandomizedSearchCV` present
- [PASS] `compute_sample_weight` present
- [PASS] `scale_pos_weight` present
- [PASS] `tree_method` / `hist` present
- [PASS] `train_test_split` with `stratify` present

No method drift detected.

### 7. Citation drift
- [PASS] Inline numeric citations used in body (after stripping the confusion-matrix code block): {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24}.
- [PASS] Reference entries declared in `## References`: {1...24}.
- No orphan citations (every inline number maps to an entry).
- [WARN] Reference [21] (Ke et al. 2017, LightGBM) is declared but never cited inline. Recommend either adding an inline citation in Methods (where histogram-based GBDT is discussed) or removing entry 21 from the reference list.
- [PASS] `manuscripts/references.md` exists as a parallel annotated bibliography with 24 entries matching the manuscript reference list 1:1.

### 8. CrossRef live re-verification (5 random entries)

| # | DOI | HTTP | Title from CrossRef | Match |
|---|-----|------|---------------------|-------|
| 1 | 10.1016/j.aap.2011.03.025 | 200 | "The statistical analysis of highway crash-injury severities: A review and assessment of methodological alternatives" | YES |
| 8 | 10.1016/j.aap.2017.08.008 | 200 | "Comparison of four statistical and machine learning methods for crash severity prediction" | YES |
| 9 | 10.1371/journal.pone.0214966 | 200 | "A comparative study on machine learning based algorithms for prediction of motorcycle crash severity" | YES |
| 20 | 10.1145/2939672.2939785 | 200 | "XGBoost: A Scalable Tree Boosting System" | YES |
| 23 | 10.1613/jair.953 | 200 | "SMOTE: Synthetic Minority Over-sampling Technique" | YES |

[PASS] All 5 sampled DOIs resolve via `api.crossref.org/works/{doi}` with HTTP 200 and title match against the manuscript reference text.

### 9. Em-dash scan
- [PASS] Total em-dash count across `notebooks/01_eda.ipynb`, `notebooks/03_modeling.ipynb`, `manuscripts/references.md`, `manuscripts/manuscript.md`, `deliverables/presentation.html`, and `reports/*.md`: **0**.

### 10. AI-tell scan
- [PASS] `grep -riE 'verified by [0-9]+ agents|AI-verified|cross-checked by Claude'` over the project folder returned exit code 1 (no matches). No AI-tell phrases present.

### 11. Checkpoint schema
- [WARN] `checkpoint.json` does NOT exist at the project root. Cannot verify required fields `project_number`, `title`, `methodology`, `status`. Recommend creating a checkpoint file capturing the four required fields plus any state needed for reproducibility.

### Bonus: deliverables artefacts (rule-specified for #01-#08)
- [PASS] `deliverables/road_xgb.pkl` present (1.36 MB), pickled tuned XGBoost plus imputers and feature lists.
- [PASS] `deliverables/metrics.json` present, keys: `baseline_logreg_val`, `baseline_rf_val`, `improved_xgb_val`, `improved_xgb_test`, `binary_severe_auroc`, `best_xgb_params`, `n_train`, `n_val`, `n_test`.
- [PASS] `deliverables/presentation.html` present (33 KB).

---

## Top 3 findings (validator)
1. Reference [21] (LightGBM) declared but never cited inline. Either cite it in Methods 3.3 (where histogram tree boosting is discussed) or drop it from the bibliography.
2. `checkpoint.json` is missing at the project root. Recommend adding one with the four required fields.
3. Project structure deviates from the spec on `src/*.py` and `brief.md`: `src/` is empty (acceptable per rules for #01-#08 since model code lives in the notebook) and only `brief.pdf` exists; consider adding a thin `brief.md` that mirrors the PDF text for grep-friendly downstream tooling.

## Blockers
None. All required artefacts readable. CrossRef API responded for all 5 sampled DOIs.

**Role A (VALIDATOR) complete.**
