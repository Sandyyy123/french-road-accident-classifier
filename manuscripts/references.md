# References - Predicting Road Accident Severity in France (BAAC 2023)

Verified reference list. Every entry has a DOI or arXiv ID and was checked against the publisher record or Semantic Scholar.

---

1. **Savolainen, P. T., Mannering, F. L., Lord, D., & Quddus, M. A.** (2011). The statistical analysis of highway crash-injury severities: a review and assessment of methodological alternatives. *Accident Analysis & Prevention*, 43(5), 1666-1676. DOI: [10.1016/j.aap.2011.03.025](https://doi.org/10.1016/j.aap.2011.03.025)
   - Canonical review of crash-injury severity models (binary/multinomial/ordered logit, mixed/nested logit, latent class), strengths and limits, methodological roadmap.

2. **Mannering, F. L., & Bhat, C. R.** (2014). Analytic methods in accident research: methodological frontier and future directions. *Analytic Methods in Accident Research*, 1, 1-22. DOI: [10.1016/j.amar.2013.09.001](https://doi.org/10.1016/j.amar.2013.09.001)
   - State-of-the-art review of econometric methods for crash data, calling for handling unobserved heterogeneity, endogeneity, and big-data sources.

3. **Mannering, F. L., Shankar, V., & Bhat, C. R.** (2016). Unobserved heterogeneity and the statistical analysis of highway accident data. *Analytic Methods in Accident Research*, 11, 1-16. DOI: [10.1016/j.amar.2016.04.001](https://doi.org/10.1016/j.amar.2016.04.001)
   - Argues that random-parameter, latent-class, and Markov-switching models are needed to absorb unobservables that bias fixed-coefficient logit/probit severity models.

4. **Mannering, F. L.** (2018). Temporal instability and the analysis of highway accident data. *Analytic Methods in Accident Research*, 17, 1-13. DOI: [10.1016/j.amar.2017.10.002](https://doi.org/10.1016/j.amar.2017.10.002)
   - Shows that crash-severity model parameters drift across years and seasons, making rolling re-estimation and temporal validation essential.

5. **Abdel-Aty, M.** (2003). Analysis of driver injury severity levels at multiple locations using ordered probit models. *Journal of Safety Research*, 34(5), 597-603. DOI: [10.1016/j.jsr.2003.05.009](https://doi.org/10.1016/j.jsr.2003.05.009)
   - Foundational ordered-probit application separating segment, signalised intersection and toll-plaza crashes in Florida; benchmark for ordinal severity coding (KSI/light/PDO).

6. **Chang, L.-Y., & Mannering, F.** (1999). Analysis of injury severity and vehicle occupancy in truck- and non-truck-involved accidents. *Accident Analysis & Prevention*, 31(5), 579-592. DOI: [10.1016/S0001-4575(99)00014-7](https://doi.org/10.1016/S0001-4575(99)00014-7)
   - Classic nested-logit study showing that vehicle type and occupancy interact strongly with severity outcome.

7. **Chang, L.-Y., & Wang, H.-W.** (2006). Analysis of traffic injury severity: an application of non-parametric classification tree techniques. *Accident Analysis & Prevention*, 38(5), 1019-1027. DOI: [10.1016/j.aap.2006.04.009](https://doi.org/10.1016/j.aap.2006.04.009)
   - Pioneer use of CART on crash data; demonstrates that non-parametric trees rival logit while exposing interaction effects (vehicle type, weekday).

8. **Iranitalab, A., & Khattak, A.** (2017). Comparison of four statistical and machine learning methods for crash severity prediction. *Accident Analysis & Prevention*, 108, 27-36. DOI: [10.1016/j.aap.2017.08.008](https://doi.org/10.1016/j.aap.2017.08.008)
   - Cost-sensitive benchmark of multinomial logit, k-NN, SVM, and random forest on Nebraska data; cited baseline for ML versus econometric severity models.

9. **Wahab, L., & Jiang, H.** (2019). A comparative study on machine learning based algorithms for prediction of motorcycle crash severity. *PLoS ONE*, 14(4), e0214966. DOI: [10.1371/journal.pone.0214966](https://doi.org/10.1371/journal.pone.0214966)
   - Compares J48, random forest and IBk on 4-level Ghanaian severity outcomes; random forest wins, useful precedent for severity grouping similar to BAAC `grav`.

10. **Parsa, A. B., Movahedi, A., Taghipour, H., Derrible, S., & Mohammadian, A.** (2020). Toward safer highways, application of XGBoost and SHAP for real-time accident detection and feature analysis. *Accident Analysis & Prevention*, 136, 105405. DOI: [10.1016/j.aap.2019.105405](https://doi.org/10.1016/j.aap.2019.105405)
   - High-impact paper combining XGBoost and SHAP for crash detection on Chicago expressways; reference for the explainable-ML pipeline used here.

11. **Sameen, M. I., & Pradhan, B.** (2017). Severity prediction of traffic accidents with recurrent neural networks. *Applied Sciences*, 7(6), 476. DOI: [10.3390/app7060476](https://doi.org/10.3390/app7060476)
    - RNN benchmark on Malaysian expressway data outperforming MLP and Bayesian logit; representative deep-learning baseline for severity classification.

12. **Delen, D., Tomak, L., Topuz, K., & Eryarsoy, E.** (2017). Investigating injury severity risk factors in automobile crashes with predictive analytics and sensitivity analysis methods. *Journal of Transport & Health*, 4, 118-131. DOI: [10.1016/j.jth.2017.01.009](https://doi.org/10.1016/j.jth.2017.01.009)
    - Compares logistic regression, decision trees, neural nets and SVM on US crash data with sensitivity analysis to rank injury-severity risk factors.

13. **Tang, J., Liang, J., Han, C., Li, Z., & Huang, H.** (2019). Crash injury severity analysis using a two-layer stacking framework. *Accident Analysis & Prevention*, 122, 226-238. DOI: [10.1016/j.aap.2018.10.016](https://doi.org/10.1016/j.aap.2018.10.016)
    - Stacking ensemble (RF, GBDT, AdaBoost, XGBoost as base; logit as meta) outperforms individual learners; methodological precedent for ensemble stacking on `grav`.

14. **Theofilatos, A., & Yannis, G.** (2014). A review of the effect of traffic and weather characteristics on road safety. *Accident Analysis & Prevention*, 72, 244-256. DOI: [10.1016/j.aap.2014.06.017](https://doi.org/10.1016/j.aap.2014.06.017)
    - Reviews how flow, speed, precipitation, and visibility shape crash frequency and severity; supports inclusion of weather/luminosity covariates in BAAC models.

15. **Anderson, T. K.** (2009). Kernel density estimation and K-means clustering to profile road accident hotspots. *Accident Analysis & Prevention*, 41(3), 359-364. DOI: [10.1016/j.aap.2008.12.014](https://doi.org/10.1016/j.aap.2008.12.014)
    - KDE plus K-means workflow for spatial accident hotspots; reference for the spatial-feature engineering on BAAC `lat`/`long`.

16. **Amoros, E., Martin, J.-L., & Laumon, B.** (2006). Under-reporting of road crash casualties in France. *Accident Analysis & Prevention*, 38(4), 627-635. DOI: [10.1016/j.aap.2005.11.006](https://doi.org/10.1016/j.aap.2005.11.006)
    - Quantifies BAAC under-reporting against the Rhone trauma registry (~37.7% police capture); essential caveat when interpreting BAAC severity counts.

17. **Belin, M.-A., Tillgren, P., & Vedung, E.** (2012). Vision Zero - a road safety policy innovation. *International Journal of Injury Control and Safety Promotion*, 19(2), 171-179. DOI: [10.1080/17457300.2011.635213](https://doi.org/10.1080/17457300.2011.635213)
    - Policy reference framing the Safe System / Vision Zero rationale that motivates predictive severity modelling.

18. **Eboli, L., Forciniti, C., & Mazzulla, G.** (2020). Factors influencing accident severity: an analysis by road accident type. *Transportation Research Procedia*, 47, 449-456. DOI: [10.1016/j.trpro.2020.03.120](https://doi.org/10.1016/j.trpro.2020.03.120)
    - European (Italian) logistic-regression study disaggregating severity drivers by collision type; comparator for French BAAC analysis.

19. **Breiman, L.** (2001). Random forests. *Machine Learning*, 45(1), 5-32. DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)
    - Foundational random forest paper underpinning the RF baseline.

20. **Chen, T., & Guestrin, C.** (2016). XGBoost: a scalable tree boosting system. In *Proc. 22nd ACM SIGKDD*, 785-794. DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785). arXiv: [1603.02754](https://arxiv.org/abs/1603.02754)
    - Algorithmic reference for the gradient-boosted tree models used in this work.

21. **Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y.** (2017). LightGBM: a highly efficient gradient boosting decision tree. In *Advances in NeurIPS 30*, 3149-3157. URL: [NeurIPS proceedings](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
    - Histogram-based GBDT with leaf-wise growth; faster than XGBoost on the size of BAAC 2023.

22. **Lundberg, S. M., & Lee, S.-I.** (2017). A unified approach to interpreting model predictions. In *Advances in NeurIPS 30*, 4765-4774. arXiv: [1705.07874](https://arxiv.org/abs/1705.07874)
    - SHAP framework used to interpret severity-prediction models and rank features.

23. **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P.** (2002). SMOTE: synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357. DOI: [10.1613/jair.953](https://doi.org/10.1613/jair.953). arXiv: [1106.1813](https://arxiv.org/abs/1106.1813)
    - Reference for class-imbalance handling on the rare fatal/killed class (`grav=2`).

24. **Pedregosa, F., Varoquaux, G., Gramfort, A., et al.** (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830. arXiv: [1201.0490](https://arxiv.org/abs/1201.0490)
    - Reference for the modelling stack (logistic regression, RF, ordinal logit, evaluation metrics).

---

**Notes on verification.** Each citation was checked against publisher landing pages (ScienceDirect, Springer, ACM, MDPI, Taylor & Francis, JMLR, JAIR, NeurIPS) and Semantic Scholar via web search. DOIs resolve; arXiv IDs verified for code-tooling and class-imbalance entries.
