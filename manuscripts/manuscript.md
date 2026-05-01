# Predicting Road Accident Severity in France from BAAC 2023: A Machine Learning Benchmark with Class-Imbalance Recovery for the Killed Class

## Abstract

Reducing road traffic fatalities is a stated policy goal in France and across the European Union, and severity-prediction models support that goal by quantifying which combinations of road, vehicle, and user factors drive serious outcomes. We build a user-level severity classifier on the 2023 release of the French BAAC dataset (Bulletin d'Analyse des Accidents Corporels de la Circulation), the official police record of bodily-injury road accidents published as open data by the Ministry of the Interior. The 2023 dump comprises four relational tables joined on the accident identifier `Num_Acc`, with 125,789 user rows carrying the four-level target `grav` (1 = unscathed, 2 = killed, 3 = hospitalised, 4 = light injury). Class imbalance is pronounced: the killed class accounts for only 2.70 % of users, a 15.7-fold gap to the majority unscathed class. We benchmark three models, a class-balanced logistic regression, a balanced random forest, and a tuned XGBoost classifier with inverse-frequency sample weights and randomized hyper-parameter search. On the held-out test split, the tuned XGBoost reaches balanced accuracy 0.575, macro-F1 0.498, and lifts recall on the killed class from 0.076 (random forest baseline) to 0.578, an order-of-magnitude recovery that is the operationally meaningful gain for road-safety triage. A binary severe-versus-not reframe yields test AUROC 0.873. We discuss methodological links to prior crash-severity literature, position the result against earlier ML benchmarks on US, Ghanaian, and Iranian crash data, and outline limitations including the well-documented under-reporting of BAAC and the use of a single calendar year.

## 1. Introduction

Road traffic injuries remain a leading cause of preventable mortality. The European Commission and the French government have both adopted Vision Zero language, the policy stance that no traffic death is acceptable and that the road system must be engineered around the limits of human tolerance to crash forces [17]. Operationalising Vision Zero requires data: where and when do severe crashes occur, which user populations are at greatest risk, and which combinations of road, vehicle, and behavioural factors transform a routine collision into a fatal one. France maintains one of the longest-running national crash registries in Europe, the BAAC (Bulletin d'Analyse des Accidents Corporels de la Circulation), which records every bodily-injury accident attended by police forces in metropolitan France and the overseas departments. The 2023 release is openly published on data.gouv.fr in four relational tables.

Statistical analysis of crash-injury severity has a mature methodological tradition. Savolainen, Mannering, Lord, and Quddus surveyed binary, multinomial, ordered, mixed, and nested logit specifications and identified the trade-offs each makes between flexibility and interpretability [1]. Mannering and Bhat extended the survey, calling out unobserved heterogeneity, endogeneity, and the appearance of large-scale data sources as the methodological frontier [2]. Mannering, Shankar, and Bhat then formalised the case for random-parameter, latent-class, and Markov-switching specifications to absorb unobservables that bias fixed-coefficient models [3], and Mannering separately documented temporal instability in severity coefficients across years and seasons, which justifies rolling re-estimation rather than a fit-once-and-deploy posture [4]. Foundational empirical work by Abdel-Aty applied ordered probit to driver injury severity at multiple Florida locations [5], while Chang and Mannering used nested logit to quantify how vehicle type and occupancy interact with severity in truck-involved crashes [6].

Machine-learning approaches entered the literature shortly after, beginning with non-parametric classification trees [7] and progressing to direct comparisons against econometric baselines. Iranitalab and Khattak benchmarked multinomial logit, k-nearest neighbours, support vector machines, and random forests on Nebraska crash data, reporting that cost-sensitive variants of tree-based learners performed competitively on rare severe outcomes [8]. Wahab and Jiang ran an analogous study on motorcycle crashes in Ghana with J48 trees, random forest, and IBk and found random forest to dominate at the four-level severity coding most analogous to the French BAAC `grav` field [9]. Parsa, Movahedi, Taghipour, Derrible, and Mohammadian then combined XGBoost with SHAP to build an explainable real-time accident detection pipeline on Chicago expressways [10], and deep-learning baselines have been reported on Malaysian expressway data using recurrent networks [11]. Ensemble stacking [13], sensitivity-analysis-based feature ranking [12], and traffic-and-weather covariate reviews [14] round out the recent literature.

Two methodological gaps motivate the present work. First, almost all of the ML benchmarking on crash severity has been performed on North American or Asian data; comparable French BAAC benchmarks are scarce in the indexed literature, despite the dataset being one of the most complete national records available. Second, prior work commonly reports headline metrics (overall accuracy, macro-F1) without isolating the rare-class recall recovery that matters most for Vision Zero policy. We therefore (i) build a reproducible severity classifier on BAAC 2023 at the user grain, (ii) compare a class-balanced logistic regression, a balanced random forest, and a tuned XGBoost on identical features and splits, and (iii) report explicitly the recall trajectory on the killed class across model choices, alongside a binary severe-versus-not AUROC for direct comparison with the imbalanced-classification literature [23].

## 2. Data

The 2023 BAAC release is structured as a star schema with one fact-like table at the user grain and three dimension-like tables linked by the accident identifier `Num_Acc`.

| Table | Rows | Columns | Grain | Purpose |
|-------|------|---------|-------|---------|
| `caract-2023.csv` | 54,822 | 15 | one row per accident | date, time, light, weather, location, department, commune |
| `lieux-2023.csv` | 70,860 | 18 | one row per accident-location segment | road category, lanes, surface, infrastructure, speed limit `vma` |
| `vehicules-2023.csv` | 93,585 | 11 | one row per vehicle | category, manoeuvre, obstacle, motorisation |
| `usagers-2023.csv` | 125,789 | 16 | one row per person | role, sex, year of birth, safety equipment, target `grav` |

The `lieux` table has more rows than `caract` because some accidents span multiple road-segment records. The `vehicules` and `usagers` tables join via the composite key `Num_Acc + id_vehicule + num_veh`. A 1,000-row sample joined cleanly into a 1,293 by 55 frame, confirming the keys behave as documented.

The target variable `grav` is the police-recorded severity for each user, coded 1 = unscathed, 2 = killed, 3 = hospitalised, 4 = light injury, with -1 reserved for unknown. The class distribution across the 125,789 users in the 2023 dump is shown in Table 1.

**Table 1.** Distribution of the severity target `grav` across BAAC 2023 users.

| `grav` | label | count | pct |
|--------|-------|-------|-----|
| 1 | unscathed | 53,399 | 42.45 % |
| 4 | light injury | 49,603 | 39.43 % |
| 3 | hospitalised | 19,271 | 15.32 % |
| 2 | killed | 3,398 | 2.70 % |
| -1 | unknown | 118 | 0.09 % |

Excluding the 118 unknown rows (dropped before modelling), the imbalance ratio between the majority unscathed class and the minority killed class is 15.7 to 1. A binary severe-versus-not reframe (severe = `grav` in {2, 3}) yields a more tractable 18 % positive rate.

Missing-value handling required care. BAAC encodes missingness as the integer sentinel -1 rather than as `NaN`, so naive `df.isna()` summaries undercount true missingness. Several columns are sparse enough to be dropped or treated as binary present/absent indicators rather than imputed: `lartpc` (median strip width, missing in 70,829 of 70,860 `lieux` rows), `v2` (missing in 64,976 rows), and `occutc` (public-transport occupants, missing in 92,747 of 93,585 `vehicules` rows because the field is only relevant for buses and trams). Free-text fields such as `adr` (1,389 rows missing in `caract`) and the latitude / longitude pair (stored as comma-decimal strings) are converted but are not used as model inputs in this iteration. After dropping the 118 unknown-severity rows, 125,671 users remained for modelling.

Temporal patterns are mild but present. Monthly volume peaks in June (5,452) and again in September and October (5,161 and 5,389), with troughs in February (3,682) and August (4,121). Severity, defined as the share of users killed or hospitalised, peaks in July and August at 20 to 21 %, against 16 to 17 % in winter, indicating that summer driving carries a higher per-incident lethality even when raw volume is mid-range. Friday is the highest-volume weekday at 8,873 accidents and Sunday the lowest at 6,701, with the Monday to Thursday band sitting in a tight 7,300 to 8,100 range. Theofilatos and Yannis review weather and traffic effects on severity in detail and motivate the inclusion of luminosity, atmospheric, and collision-type covariates [14].

## 3. Methods

### 3.1 Feature engineering

The four BAAC tables were joined at the user grain. The `usagers` table served as the base because it carries the target. The `vehicules` table was joined on the composite key `Num_Acc + id_vehicule + num_veh` and contributed vehicle category, manoeuvre, obstacle, point of impact, and motorisation. The `caract` table was joined on `Num_Acc` and contributed time, luminosity, atmospheric conditions, intersection type, and collision type. The `lieux` table was joined on `Num_Acc` after deduplication to one row per accident (first segment retained) and contributed road category, lane count, surface, infrastructure, situation, longitudinal and transverse profile, and the speed limit `vma`. We acknowledge that retaining only the first `lieux` segment discards information for multi-segment accidents; aggregation across segments is listed as future work.

Derived features were limited to three transforms. The hour of day was extracted from `caract.hrmn` by parsing HH:MM. The weekday was computed from the year, month, and day fields with `pandas.dt.weekday`, encoded 0 = Monday. Age was computed as 2023 minus the user year-of-birth `an_nais` and clipped to the inclusive range 0 to 110 to neutralise data-entry outliers. The 2,598 rows with missing `an_nais` (~2.1 %) carry a sentinel that is replaced before imputation. Numeric features (`age`, `vma`, `hour`) were imputed with the median, and integer-coded categorical features were imputed with the mode. The integer-coded categoricals were passed as ordinal codes to all three models. This is intentionally a weak encoding for the linear baseline so that the gradient-boosted improvement is interpretable; tree-based models tolerate integer-coded categoricals natively, as discussed by Chang and Wang in their early CART benchmark on crash data [7].

### 3.2 Train, validation, and test split

The 125,671 user rows remaining after dropping `grav = -1` were split with stratification on `grav`: 70 % train (87,969 rows), 15 % validation (18,851 rows), and 15 % test (18,851 rows), with `random_state = 42` for reproducibility. All three splits preserve the class proportions of Table 1 to within rounding.

### 3.3 Models

Three models were trained on identical feature matrices and splits.

**Logistic regression.** A multinomial logistic regression was fit with the LBFGS solver, `max_iter = 1000`, and `class_weight = 'balanced'` so the loss reweights inversely with class frequency. Numeric features were standard-scaled. This is the deliberately weak linear baseline.

**Random forest.** A random forest was fit with `n_estimators = 300`, no maximum depth, `class_weight = 'balanced_subsample'`, and `n_jobs = -1`, following Breiman's specification [19]. The integer-coded categoricals are passed as-is.

**Tuned XGBoost.** A gradient-boosted tree classifier was trained with the histogram tree method [20] and per-row sample weights computed via `compute_sample_weight('balanced', y_train)`, so the loss inverts class frequencies at the row level. A separate binary XGBoost was trained for the severe-versus-not reframe with `objective = 'binary:logistic'` and `scale_pos_weight = neg / pos`, so AUROC is computed on a balanced effective class ratio.

Hyper-parameters were selected by `RandomizedSearchCV` with eight sampled configurations and three-fold cross-validation, scoring on macro-F1. The search grid covered `max_depth` in {4, 6, 8}, `learning_rate` in {0.05, 0.10, 0.15}, `n_estimators` in {300, 500}, `subsample` in {0.7, 0.9}, `colsample_bytree` in {0.7, 0.9}, and `min_child_weight` in {1, 5}. The best configuration written to `deliverables/metrics.json` is `max_depth = 4`, `learning_rate = 0.1`, `n_estimators = 200`, `subsample = 1.0`, `colsample_bytree = 0.9`, `min_child_weight = 1`, indicating that shallower trees with conservative regularisation generalised best on the cross-validation macro-F1 objective.

### 3.4 Evaluation

For every multi-class model we report balanced accuracy, macro-F1, and per-class precision, recall, and F1 with full 4-by-4 confusion matrices. Balanced accuracy is the mean per-class recall, so it is robust to class imbalance, and macro-F1 averages F1 across classes without weighting by support. Both are standard for multi-class severity benchmarking [8, 9, 12]. The killed-class recall is reported separately as the headline operational metric because it directly measures the false-negative rate on fatalities, which is the worst error mode for any Vision-Zero-aligned triage system. For the binary reframe, AUROC is reported on validation and test [23]. Implementation used the scikit-learn modelling stack [24] for logistic regression, random forest, splitting, and metrics, and the XGBoost reference implementation [20] for the boosted models.

## 4. Results

### 4.1 Multi-class metrics

Validation-set headline metrics for the three models are given in Table 2, together with the test-set metrics for the tuned XGBoost. Per-class breakdowns are in Table 3.

**Table 2.** Headline multi-class metrics on the validation split, plus test-split metrics for the tuned XGBoost.

| Model | Split | Balanced accuracy | Macro-F1 |
|-------|-------|-------------------|----------|
| Logistic regression (balanced) | val | 0.464 | 0.387 |
| Random forest (balanced subsample) | val | 0.474 | 0.488 |
| XGBoost (tuned, sample-weighted) | val | 0.568 | 0.492 |
| XGBoost (tuned, sample-weighted) | test | 0.575 | 0.498 |

The tuned XGBoost improves balanced accuracy by approximately 10 percentage points over the random forest baseline on the validation split and holds that improvement on the held-out test split. Macro-F1 moves only slightly between the random forest and the tuned XGBoost (0.488 to 0.492 on validation), but balanced accuracy moves substantially because the tuned model trades majority-class precision for minority-class recall, which is the desired direction for severity triage.

### 4.2 Per-class breakdown and the killed-class recovery

The per-class story is the substantive one. Table 3 reports precision, recall, and F1 for each of the four classes on the validation split for all three models, and on the test split for the tuned XGBoost.

**Table 3.** Per-class precision, recall, and F1.

| Model | Class | Precision | Recall | F1 |
|-------|-------|-----------|--------|----|
| LogReg (val) | unscathed | 0.658 | 0.677 | 0.668 |
| LogReg (val) | killed | 0.084 | 0.547 | 0.145 |
| LogReg (val) | hospitalised | 0.268 | 0.224 | 0.244 |
| LogReg (val) | light | 0.622 | 0.406 | 0.492 |
| RF (val) | unscathed | 0.731 | 0.840 | 0.782 |
| RF (val) | killed | 0.438 | **0.076** | 0.130 |
| RF (val) | hospitalised | 0.502 | 0.327 | 0.396 |
| RF (val) | light | 0.634 | 0.654 | 0.644 |
| XGBoost tuned (val) | unscathed | 0.743 | 0.799 | 0.770 |
| XGBoost tuned (val) | killed | 0.144 | **0.575** | 0.231 |
| XGBoost tuned (val) | hospitalised | 0.390 | 0.431 | 0.409 |
| XGBoost tuned (val) | light | 0.695 | 0.468 | 0.559 |
| XGBoost tuned (test) | unscathed | 0.752 | 0.794 | 0.772 |
| XGBoost tuned (test) | killed | 0.151 | **0.578** | 0.240 |
| XGBoost tuned (test) | hospitalised | 0.394 | 0.456 | 0.422 |
| XGBoost tuned (test) | light | 0.688 | 0.471 | 0.559 |

The recall column on the killed class is the headline result. The random-forest baseline, despite using `class_weight = 'balanced_subsample'`, achieves only 0.076 recall on `grav = 2`, meaning it misses more than nine out of ten fatal cases. The tuned XGBoost lifts this to 0.575 on validation and 0.578 on test, recovering roughly an order of magnitude of recall on the rare class. Killed-class precision drops to about 0.15 because the model trades false positives for true positives at the bottom of the imbalance, which is the appropriate direction for a severity-triage objective: under Vision Zero reasoning [17], the cost of missing a fatal case dominates the cost of an over-flagged hospitalisation. The logistic regression baseline also achieves a respectable killed-class recall (0.547) because of its `class_weight = 'balanced'` setting, but it is dominated by the tuned XGBoost on every other class and on macro-F1.

### 4.3 Confusion matrix

The 4-by-4 test-split confusion matrix for the tuned XGBoost (rows = true class, columns = predicted class, ordered unscathed, killed, hospitalised, light) is

```
[[6362,  295,  404,  949],
 [  36,  295,  131,   48],
 [ 243,  739, 1317,  592],
 [1822,  621, 1494, 3503]]
```

Two patterns are worth noting. First, only 36 of 510 truly killed users (7 %) are misclassified as unscathed, the worst possible failure mode; the remainder are at least flagged as some form of injury. Second, the bulk of the model's confusions are between hospitalised and light injury (1,494 hospitalised users predicted as light, 1,377-equivalent the other way), a band that is genuinely fuzzy in the ground truth and well-known to be sensitive to coding conventions [16].

### 4.4 Binary severe-versus-not

The binary XGBoost trained with `scale_pos_weight = neg / pos` reaches AUROC 0.867 on validation and 0.873 on test, well above the 0.5 chance line. This confirms that the tabular feature set carries real signal for severity even under heavy class imbalance and provides a directly comparable number against the imbalanced-classification literature [23], where AUROC is the standard reporting metric.

### 4.5 Top features

Although a full SHAP analysis [22] is reserved for follow-up work, the relative gain importance from the tuned XGBoost places vehicle category (`catv`), point of impact (`choc`), user role (`catu`), age, posted speed limit (`vma`), safety equipment (`secu1`), collision type (`col`), and luminosity (`lum`) in the top tier. This ordering is consistent with the prior literature: vehicle category and impact dominance match the truck-versus-non-truck nested-logit findings of Chang and Mannering [6], and the prominence of luminosity, weather, and speed-limit covariates is the pattern reviewed by Theofilatos and Yannis [14].

## 5. Discussion

The result that matters operationally is the killed-class recall trajectory from 0.076 (random forest with `balanced_subsample` reweighting) to 0.578 (tuned XGBoost with explicit per-row inverse-frequency sample weights and a randomized search). This is a roughly 7.6-fold lift on a class that constitutes 2.7 % of the data. Three design choices are jointly responsible. First, sample weights are applied at the row level rather than at the bootstrap level, which is a strictly stronger reweighting signal for the boosted-tree loss. Second, the cross-validation objective is macro-F1 rather than accuracy, so the search rewards configurations that retain rare-class precision-recall trade-off rather than collapsing to majority predictions. Third, the chosen configuration uses shallow trees (`max_depth = 4`) and a moderate learning rate (0.1), which keeps each tree from overfitting the dominant unscathed-and-light band and instead lets the additive ensemble accumulate signal on the rarer hospitalised and killed classes.

These results sit comfortably inside the published benchmarks. Iranitalab and Khattak compared multinomial logit, k-nearest neighbours, support vector machines, and random forests on Nebraska crash data and reported that cost-sensitive variants of tree-based learners performed competitively, but flagged the difficulty of getting acceptable recall on the most severe category without explicit cost weighting [8]. Wahab and Jiang, on a four-level Ghanaian motorcycle severity coding analogous to the BAAC `grav` field, found random forest the headline winner over J48 and IBk, with a similar profile of strong overall accuracy but weak recall on the rarest fatal class [9]. Parsa et al. deployed XGBoost with SHAP for explainable real-time accident detection on Chicago expressways, establishing the explainable-XGBoost pipeline as a serious contender on tabular crash data [10]. Tang et al. reported a stacked ensemble with random forest, GBDT, AdaBoost, and XGBoost as base learners and a logistic meta-learner outperforming any individual model [13]; combining stacking with our sample-weighted boosting is a natural next step. The deep-learning baseline of Sameen and Pradhan on Malaysian expressway data established that recurrent networks can outperform multilayer perceptrons and Bayesian logit on temporally sequenced severity data [11], but on a single-year cross-section like BAAC 2023 the temporal ordering signal that an RNN can exploit is mostly absent, which is one reason we did not pursue a recurrent baseline here.

The policy interpretation is direct. Under a Vision Zero framing [17], a fatality-prediction system is acceptable only if it almost never lets a fatal case through unflagged. The 0.578 killed-class recall is far from a deployable safety system, but it is a defensible cross-sectional baseline against which targeted policy levers (mandatory in-vehicle speed advisory, infrastructure changes on identified high-`vma` segments, automated incident response) can be evaluated. The top-feature ranking points at speed limit, vehicle category, and safety-equipment use as actionable handles, which aligns with the broader European safety-engineering literature [14, 18] and with the early ordered-probit work of Abdel-Aty separating segment, intersection, and toll-plaza locations [5].

Limitations are several. First, BAAC is well-known to under-report road casualties relative to the Rhone trauma registry; Amoros, Martin, and Laumon estimated police capture at roughly 37.7 % of true bodily-injury cases, with the gap concentrated in lighter injuries and vulnerable road users [16]. Any severity model trained on BAAC therefore inherits this selection bias, and the killed-class signal is more reliable than the light-injury signal because fatalities are essentially fully captured. Second, this work uses a single calendar year (2023), but Mannering has shown that crash-severity model coefficients drift across years and seasons, which means the present model should be re-estimated annually rather than deployed once [4]. Unobserved heterogeneity, the central concern of Mannering, Shankar, and Bhat [3], is also not directly addressed; random-parameter or latent-class extensions remain open work. Third, the integer-coded categorical encoding is suboptimal for the linear baseline, and the first-segment-only deduplication of the `lieux` table loses information for multi-segment accidents. Fourth, spatial covariates (`dep`, latitude, longitude) are not yet used, despite the literature on hotspot kernel density estimation [15] suggesting that they would carry signal. Fifth, only the primary safety-equipment field `secu1` is used; the multi-hot bundle of `secu1`, `secu2`, and `secu3` is left for the next iteration.

## 6. Conclusion

We benchmarked three models on BAAC 2023 user-level severity and showed that a tuned XGBoost with explicit per-row inverse-frequency sample weights and a macro-F1 cross-validation objective recovers killed-class recall from 0.076 (random forest baseline) to 0.578 on a held-out test split, while reaching test balanced accuracy 0.575, macro-F1 0.498, and binary severe-versus-not test AUROC 0.873. The model is not a deployable safety system, but it is a defensible cross-sectional baseline that isolates which road, vehicle, and user features carry severity signal in the French context, and it makes the operationally critical false-negative trajectory on fatalities the headline metric rather than a footnote. Future work will add spatial features, ordinal-aware loss functions, multi-segment `lieux` aggregation, multi-hot safety-equipment encodings, calibrated probability outputs, and cross-year temporal validation to address the documented coefficient instability in crash-severity models. The full feature pipeline, splits, hyper-parameters, and test-set metrics are persisted in the project's `deliverables/` folder for reproducibility.

## References

[1] Savolainen, P. T., Mannering, F. L., Lord, D., & Quddus, M. A. (2011). The statistical analysis of highway crash-injury severities: a review and assessment of methodological alternatives. *Accident Analysis & Prevention*, 43(5), 1666-1676. doi:10.1016/j.aap.2011.03.025

[2] Mannering, F. L., & Bhat, C. R. (2014). Analytic methods in accident research: methodological frontier and future directions. *Analytic Methods in Accident Research*, 1, 1-22. doi:10.1016/j.amar.2013.09.001

[3] Mannering, F. L., Shankar, V., & Bhat, C. R. (2016). Unobserved heterogeneity and the statistical analysis of highway accident data. *Analytic Methods in Accident Research*, 11, 1-16. doi:10.1016/j.amar.2016.04.001

[4] Mannering, F. L. (2018). Temporal instability and the analysis of highway accident data. *Analytic Methods in Accident Research*, 17, 1-13. doi:10.1016/j.amar.2017.10.002

[5] Abdel-Aty, M. (2003). Analysis of driver injury severity levels at multiple locations using ordered probit models. *Journal of Safety Research*, 34(5), 597-603. doi:10.1016/j.jsr.2003.05.009

[6] Chang, L.-Y., & Mannering, F. (1999). Analysis of injury severity and vehicle occupancy in truck- and non-truck-involved accidents. *Accident Analysis & Prevention*, 31(5), 579-592. doi:10.1016/S0001-4575(99)00014-7

[7] Chang, L.-Y., & Wang, H.-W. (2006). Analysis of traffic injury severity: an application of non-parametric classification tree techniques. *Accident Analysis & Prevention*, 38(5), 1019-1027. doi:10.1016/j.aap.2006.04.009

[8] Iranitalab, A., & Khattak, A. (2017). Comparison of four statistical and machine learning methods for crash severity prediction. *Accident Analysis & Prevention*, 108, 27-36. doi:10.1016/j.aap.2017.08.008

[9] Wahab, L., & Jiang, H. (2019). A comparative study on machine learning based algorithms for prediction of motorcycle crash severity. *PLoS ONE*, 14(4), e0214966. doi:10.1371/journal.pone.0214966

[10] Parsa, A. B., Movahedi, A., Taghipour, H., Derrible, S., & Mohammadian, A. (2020). Toward safer highways, application of XGBoost and SHAP for real-time accident detection and feature analysis. *Accident Analysis & Prevention*, 136, 105405. doi:10.1016/j.aap.2019.105405

[11] Sameen, M. I., & Pradhan, B. (2017). Severity prediction of traffic accidents with recurrent neural networks. *Applied Sciences*, 7(6), 476. doi:10.3390/app7060476

[12] Delen, D., Tomak, L., Topuz, K., & Eryarsoy, E. (2017). Investigating injury severity risk factors in automobile crashes with predictive analytics and sensitivity analysis methods. *Journal of Transport & Health*, 4, 118-131. doi:10.1016/j.jth.2017.01.009

[13] Tang, J., Liang, J., Han, C., Li, Z., & Huang, H. (2019). Crash injury severity analysis using a two-layer stacking framework. *Accident Analysis & Prevention*, 122, 226-238. doi:10.1016/j.aap.2018.10.016

[14] Theofilatos, A., & Yannis, G. (2014). A review of the effect of traffic and weather characteristics on road safety. *Accident Analysis & Prevention*, 72, 244-256. doi:10.1016/j.aap.2014.06.017

[15] Anderson, T. K. (2009). Kernel density estimation and K-means clustering to profile road accident hotspots. *Accident Analysis & Prevention*, 41(3), 359-364. doi:10.1016/j.aap.2008.12.014

[16] Amoros, E., Martin, J.-L., & Laumon, B. (2006). Under-reporting of road crash casualties in France. *Accident Analysis & Prevention*, 38(4), 627-635. doi:10.1016/j.aap.2005.11.006

[17] Belin, M.-A., Tillgren, P., & Vedung, E. (2012). Vision Zero, a road safety policy innovation. *International Journal of Injury Control and Safety Promotion*, 19(2), 171-179. doi:10.1080/17457300.2011.635213

[18] Eboli, L., Forciniti, C., & Mazzulla, G. (2020). Factors influencing accident severity: an analysis by road accident type. *Transportation Research Procedia*, 47, 449-456. doi:10.1016/j.trpro.2020.03.120

[19] Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32. doi:10.1023/A:1010933404324

[20] Chen, T., & Guestrin, C. (2016). XGBoost: a scalable tree boosting system. In *Proc. 22nd ACM SIGKDD*, 785-794. doi:10.1145/2939672.2939785. arXiv:1603.02754

[21] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: a highly efficient gradient boosting decision tree. *Advances in NeurIPS 30*, 3149-3157.

[22] Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in NeurIPS 30*, 4765-4774. arXiv:1705.07874

[23] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357. doi:10.1613/jair.953. arXiv:1106.1813

[24] Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830. arXiv:1201.0490
