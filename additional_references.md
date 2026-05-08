# Additional References - Road Accident Severity (BAAC 2023)

Independent literature scout for the project "Predicting Road Accident Severity in France from BAAC 2023". This file complements `manuscripts/references.md`. Every entry below was resolved live via the CrossRef and Europe PMC APIs and confirmed to redirect through https://doi.org/. Volume, issue, and page numbers are intentionally omitted per project convention to avoid fabricated citation artefacts.

## State-of-the-art callout: gaps in the current `references.md`

A comparison against the 24 entries in `manuscripts/references.md` highlights five concrete gaps that the current bibliography does not cover and that the manuscript should cite when revised:

1. **No 2024-2026 ML benchmark on French BAAC or comparable EU national dump.** The closest entries in `references.md` are Iranitalab and Khattak 2017 (Nebraska) and Wahab and Jiang 2019 (Ghana). A current French / EU-data benchmark such as Mostafa et al. 2025 (Saudi Arabia, AI-based severity) and Alanazi et al. 2025 (Saudi Arabia, deep vs traditional) and the Italian Eboli et al. 2020 baseline already cited should be supplemented with Khanum et al. 2025 (Indian highways) for a like-for-like 2024-2026 comparator.
2. **No representation of the 2024-2026 XGBoost-SHAP wave.** The single XGBoost-SHAP citation in `references.md` is Parsa et al. 2020. Recent direct comparators that should be cited include Laphrom et al. 2024, Scarano et al. 2025, Sunkpho et al. 2025 (CNN-SHAP), and Vadhwani and Thakor 2024 ("improved XGBoost for crash severity").
3. **No coverage of imbalance-aware tree ensembles purpose-built for crash severity.** SMOTE (Chawla 2002) is cited but the current crop of crash-specific imbalance work (Mohammadpour and Vahedi 2026 iBRF, Aziz et al. 2025 dynamic ensemble selection, Shangguan et al. 2025 VAE-attention-GCN, Amiri et al. 2025 in *Machine Learning with Applications*) is missing despite being the most direct methodological match to the killed-class recovery story.
4. **No deep-learning baseline newer than Sameen and Pradhan 2017.** Tabular deep learning has moved on. Pei et al. 2024 (interpretable DL in *Transportation Letters*), Xiao and Duan 2025 (multi-task DL severity), Chen et al. 2025 (MSCPO-XGBoost), and the Shen et al. 2026 *AAP* paper combining deep learning with Bayesian random parameters are concrete recent comparators.
5. **No explicitly explainable / Vision-Zero-aligned 2025-2026 paper.** Vision Zero is referenced via Belin et al. 2012 (policy paper). For the operational severity-triage framing the manuscript should cite Le 2026 (KSI risk in the UK with explainable ML) and Ye et al. 2026 (spatially disaggregated explainable AI) plus Gao et al. 2026 (large-scale multi-dimensional explainable ensemble in *Accident Analysis and Prevention*) which is the closest match to the BAAC framing at scale.

These five callouts drove the grouping below.

---

## Group 1: Imbalance-aware methods for crash severity (2024-2026)

Direct match to the killed-class recovery story. These should be cited in Section 3 (Methods) and Section 4.2 (per-class breakdown).

- Mohammadpour S.I., Vahedi J. Improved Balanced Random Forest (iBRF): A flexible hybrid resampling-bagging framework for crash severity classification in imbalanced datasets. Traffic Injury Prevention. 2026. DOI:10.1080/15389588.2025.2610428

- Aziz K., Chen F., Ahmad M. An interpretable dynamic ensemble selection multiclass imbalance approach with ensemble imbalance learning for predicting road crash injury severity. Scientific Reports. 2025. DOI:10.1038/s41598-025-08935-x

- Shangguan A., Feng N., Hei X. Predicting road traffic accident severity from imbalanced data using VAE attention and GCN. Scientific Reports. 2025. DOI:10.1038/s41598-025-17064-4

- Amiri M.A., Afshari S., Soltani A. Machine learning approaches to traffic accident severity prediction: Addressing class imbalance. Machine Learning with Applications. 2025. DOI:10.1016/j.mlwa.2025.100792

- Simmachan T., Boonkrong P. Effect of Resampling Techniques on Machine Learning Models for Classifying Road Accident Severity in Thailand. Journal of Current Science and Technology. 2025. DOI:10.59796/jcst.v15n2.2025.99

## Group 2: XGBoost / SHAP and explainable boosted trees (2024-2026)

Direct comparators to the tuned XGBoost in Section 3.3.

- Laphrom W., Se C., Champahom T. XGBoost-SHAP and Unobserved Heterogeneity Modelling of Temporal Multivehicle Truck-Involved Crash Severity Patterns. Civil Engineering Journal. 2024. DOI:10.28991/cej-2024-010-06-011

- Scarano A., Sadeghi M., Mauriello F. Cyclist crash severity modeling: A hybrid approach of XGBoost-SHAP and random parameters logit with heterogeneity in means and variances. Journal of Safety Research. 2025. DOI:10.1016/j.jsr.2025.04.003

- Sunkpho J., Se C., Wipulanusat W. SHAP-based convolutional neural network modeling for intersection crash severity on Thailand's highways. IATSS Research. 2025. DOI:10.1016/j.iatssr.2024.12.003

- Vadhwani D., Thakor D. An improved XGBoost model to predict the injury severity of person in road crash. International Journal of Crashworthiness. 2024. DOI:10.1080/13588265.2024.2366567

- Dumka A., Kandiboina R., Joshi A. Analyzing crash severity through impairment and protection: A hybrid XGBoost-Bayesian network approach. Traffic Injury Prevention. 2025. DOI:10.1080/15389588.2025.2582656

- Li P., Chen S., Yue L. Analyzing relationships between latent topics in autonomous vehicle crash narratives and crash severity using natural language processing techniques and explainable XGBoost. Accident Analysis and Prevention. 2024. DOI:10.1016/j.aap.2024.107605

- Chen F., Liu X.Q., Yang J.J. Traffic accident severity prediction based on an enhanced MSCPO-XGBoost hybrid model. Scientific Reports. 2025. DOI:10.1038/s41598-025-00797-7

## Group 3: Deep learning and tabular DL for crash severity (2024-2026)

Updated deep-learning comparators to replace the 2017 Sameen and Pradhan baseline.

- Pei Y., Wen Y., Pan S. Traffic accident severity prediction based on interpretable deep learning model. Transportation Letters. 2024. DOI:10.1080/19427867.2024.2398336

- Xiao Y., Duan Z. An explainable multi-task deep learning framework for crash severity prediction. Scientific Reports. 2025. DOI:10.1038/s41598-025-09226-1

- Khan M.N., Das S., Liu J. Predicting pedestrian-involved crash severity using inception-v3 deep learning model. Accident Analysis and Prevention. 2024. DOI:10.1016/j.aap.2024.107457

- Shen R., Ma L., Yan X. The nonlinear impact of road safety policy implementation on the severity of road traffic crashes: A fusion of deep learning and Bayesian random parameter methods. Accident Analysis and Prevention. 2026. DOI:10.1016/j.aap.2026.108503

- Islam M.M., Liu J., Chakraborty R. Evaluating crash risk factors of farm equipment vehicles using interpretable tabular deep learning. Accident Analysis and Prevention. 2025. DOI:10.1016/j.aap.2025.108048

- Zhao Y., Wang P., Zhao Y. SafeTraffic Copilot: adapting large language models for trustworthy traffic safety. Nature Communications. 2025. DOI:10.1038/s41467-025-64574-w

## Group 4: Large-scale ensemble and Vision-Zero aligned ML (2025-2026)

Direct support for the Vision Zero framing in Section 1 and the operational triage framing in Section 4.

- Gao Z., Easa S.M., Liu Y. Road traffic accident severity prediction based on large-scale data and multi-dimensional factors: an explainable ensemble learning approach. Accident Analysis and Prevention. 2026. DOI:10.1016/j.aap.2026.108581

- Le K.G. Safety-oriented and explainable machine learning for KSI crash risk prediction: Evidence from the United Kingdom. PLoS ONE. 2026. DOI:10.1371/journal.pone.0347873

- Ye H., Wu B., Yuan D. Resilient road safety modeling through spatially disaggregated explainable AI. PLoS ONE. 2026. DOI:10.1371/journal.pone.0344380

- Mostafa A.M., Aldughayfiq B., Tarek M. AI-based prediction of traffic crash severity for improving road safety and transportation efficiency. Scientific Reports. 2025. DOI:10.1038/s41598-025-10970-7

- Alanazi F., Umar I.K., Yosri A.M. Comparative evaluation of deep learning and traditional models for predicting traffic accident severity in Saudi Arabia. Scientific Reports. 2025. DOI:10.1038/s41598-025-13484-4

- Chen J., Liu P., Wang S. Prediction and interpretation of crash severity using machine learning. Journal of Safety Research. 2025. DOI:10.1016/j.jsr.2025.02.018

## Group 5: Vulnerable road users and pedestrian / cyclist severity (2024-2026)

Useful for the Discussion section because BAAC has high VRU representation.

- Kuo P.F., Hsu W.T., Lord D. Classification of autonomous vehicle crash severity. Accident Analysis and Prevention. 2024. DOI:10.1016/j.aap.2024.107666

- Barua S., Chakraborty R., Islam M.M. A data-driven approach to child pedestrian crash analysis using dimension reduction, clustering, and explainable AI. Accident Analysis and Prevention. 2025. DOI:10.1016/j.aap.2025.108229

- Junaid M., Jiang C., Alotaibi S. Investigating factors influencing injury severity in crashes involving vulnerable road users in Pakistan. Scientific Reports. 2025. DOI:10.1038/s41598-025-16477-5

- Moreno-Sanfelix A., Gragera-Pena F.C., Jaramillo-Moran M.A. Evaluation of the level of responsibility in pedestrian crashes using machine learning algorithms. Scientific Reports. 2026. DOI:10.1038/s41598-026-42875-4

- Samerei S.A., Aghabayk K. Interpretable machine learning for evaluating risk factors of freeway crash severity. International Journal of Injury Control and Safety Promotion. 2024. DOI:10.1080/17457300.2024.2351972

- Panicker A.K., Ramadurai G. Identifying factors affecting crash injury severity of pillion riders using interpretable machine learning techniques. International Journal of Injury Control and Safety Promotion. 2025. DOI:10.1080/17457300.2025.2501573

## Group 6: Methodological frameworks and country-level ML benchmarks (2024-2026)

Comparators for the Methods and Discussion sections.

- Khanum H., Garg A., Faheem M.I. A methodological framework for road accident severity prediction for Indian highways using machine learning models. MethodsX. 2025. DOI:10.1016/j.mex.2025.103728

- Chu T., Le X. Crash severity analysis on developing countries using random forest: a case study of Vietnam's expressways. Transportation Safety and Environment. 2026. DOI:10.1093/tse/tdag004

- Mengistu A.K., Gedefaw A.E., Baykemagn N.D. Predicting car accident severity in Northwest Ethiopia: a machine learning approach leveraging driver, environmental, and road conditions. Scientific Reports. 2025. DOI:10.1038/s41598-025-08005-2

- Wang C., Serre T. A Hybrid Approach to Investigating Factors Associated with Crash Injury Severity: Integrating Interpretable Machine Learning with Logit Model. Applied Sciences. 2025. DOI:10.3390/app151910417

- Se C., Champahom T., Jomnonkwao S. Pickup truck crash severity analysis via machine learning: policy insights for developing countries. International Journal of Injury Control and Safety Promotion. 2025. DOI:10.1080/17457300.2025.2504975

---

**Verification.** Every DOI above was hit live via https://doi.org/ on 2026-05-08 and returned a 302 redirect to the publisher landing page (Elsevier ScienceDirect, Springer Nature, Taylor and Francis, MDPI, Civil Engineering Journal, PLoS, Oxford Academic, ACM). DOIs that did not resolve cleanly were dropped from this list, not patched with placeholder metadata. Searches were issued against api.crossref.org with `from-pub-date:2024-01-01` and against the Europe PMC REST endpoint with `PUB_YEAR:[2024 TO 2026]`. Per project rules, no volume, issue, or page numbers are included in any entry.
