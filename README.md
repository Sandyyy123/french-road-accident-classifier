![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-tabular-green) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

# French Road Accident Severity Classification

Multi-class classification of road accident severity in France using structured administrative data.

---

## Task

**Multi-class Classification**

---

## Architecture

```
Raw BAAC Records → Feature Engineering → XGBoost Baseline → Ensemble → SHAP Analysis
```

---

## Key Features

- Severity prediction across 4 accident classes
- Feature engineering on location, time, vehicle and road condition variables
- Class imbalance handling (SMOTE, class weights)
- SHAP feature importance for interpretability
- French administrative data (BAAC dataset, 700K+ accidents/year)

---

## Dataset

[French Road Accidents (data.gouv.fr)](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere/)

---

## Project Structure

```
├── src/
│   ├── model_baseline.py      # Baseline model
│   └── model_advanced.py      # Advanced model
├── notebooks/
│   └── 01_EDA.ipynb           # Exploratory analysis
├── manuscripts/
│   └── manuscript.md          # IMRaD writeup
├── reports/
│   └── references.md          # Verified references
├── deliverables/
│   └── presentation.html      # Self-contained HTML
├── data/
│   └── README.md              # Dataset download instructions
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Sandyyy123/french-road-accident-classifier.git
cd french-road-accident-classifier
pip install -r requirements.txt

# See data/README.md for dataset download
jupyter notebook notebooks/01_eda.ipynb
# or run modeling:
jupyter notebook notebooks/03_modeling.ipynb
python src/model_advanced.py
```

---

## Tech Stack

`scikit-learn · XGBoost · pandas · SHAP`

---

## Author

**Dr. Sandeep Grover** — PhD Data Science, independent ML researcher, Mössingen, Germany.

---

## License

MIT
