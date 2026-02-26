# Customer Churn Prediction & Analysis

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?style=flat-square)
![EDA](https://img.shields.io/badge/📊-Exploratory%20Data%20Analysis-important?style=flat-square)
![Imbalanced Learning](https://img.shields.io/badge/🔄-Imbalanced%20Learning-ff69b4?style=flat-square)

End-to-end machine learning project to **understand, quantify, and predict customer churn** in telecom and other subscription-based businesses such as banking and SaaS.

This project focuses on predictive performance, imbalance handling, business cost optimization, and decision threshold calibration for real retention campaigns.

---

## 1. Business Context

Customer churn is one of the most expensive revenue leaks in subscription businesses. Acquiring a new customer often costs significantly more than retaining an existing one.

**Executive Problem**

> “We are losing customers every month. We do not clearly know who will leave next, nor which segments are most vulnerable.”

### Objective

Develop a predictive system that:

- Identifies customers likely to churn in the next 30 days  
- Optimizes recall while controlling campaign cost  
- Enables targeted retention campaigns  
- Produces interpretable churn drivers  

The goal is business impact, not just model accuracy.

---

## 2. Dataset

Primary benchmark dataset: **Telco Customer Churn**

| Dataset                          | Rows  | Churn Rate | Use Case             | Source |
|----------------------------------|-------|------------|----------------------|--------|
| WA_Fn-UseC_-Telco-Customer-Churn | 7043  | ~26.5%     | Telecom benchmark    | Kaggle |
| Bank Customer Churn             | ~10k  | ~20%       | Banking churn risk   | Kaggle |
| E-commerce Churn                | 5–20k | 15–30%     | Retail churn         | Kaggle |

**Repository file:** `data/Telco-Customer-Churn.csv`

### Feature Categories

**Demographics**
- Gender  
- SeniorCitizen  
- Partner  
- Dependents  

**Service-related**
- InternetService  
- PhoneService  
- StreamingTV  
- OnlineSecurity  
- TechSupport  

**Account-related**
- Tenure  
- Contract  
- MonthlyCharges  
- TotalCharges  

**Target**
- Churn (Yes / No)

---

## 3. Exploratory Data Analysis

Key observations:

- Month-to-month contracts churn 3–4× more than long-term contracts  
- High monthly charges + low tenure represent highest-risk segment  
- Customers without OnlineSecurity or TechSupport show elevated churn  
- Electronic check payment correlates with churn  
- Senior citizens have slightly higher churn probability  

Dataset imbalance: ~27% churn.

Implications:

- Stratified cross-validation required  
- PR-AUC prioritized over ROC-AUC  
- Recall-based threshold optimization  

---

## 4. Feature Engineering

Enhancements applied:

- Tenure buckets (0–6, 6–12, 12–24, 24+ months)  
- Revenue-to-tenure ratio  
- Service count aggregation  
- Binary encoding for contract type  
- Interaction term: tenure × monthly charges  
- Log transform of skewed monetary variables  

Final feature space: 18–26 engineered features.

---

## 5. Modeling Strategy

### Validation Design

- 5-fold stratified cross-validation  
- 20% hold-out test set  
- Random seed 2025  
- Full pipeline to prevent data leakage  
- Hyperparameter tuning via Optuna (60–120 trials per model)

### Evaluation Metric

Primary metric:

**Recall at ~30% precision**

Interpretation:

Contact approximately 30% of customers and capture 75–80% of churners.

PR-AUC is prioritized due to class imbalance.

---

## 6. Model Performance Comparison

(5-fold CV + hold-out validation)

| Rank | Model                | ROC-AUC | PR-AUC | Recall @ ~30% Precision | F1 (Churn) |
|------|----------------------|---------|--------|--------------------------|------------|
| 1    | CatBoost             | 0.90    | 0.69   | 0.79                     | 0.60       |
| 2    | LightGBM             | 0.895   | 0.685  | 0.78                     | 0.595      |
| 3    | XGBoost              | 0.892   | 0.678  | 0.77                     | 0.59       |
| 4    | HistGradientBoosting | 0.885   | 0.66   | 0.75                     | 0.57       |
| 5    | Random Forest        | 0.87    | 0.63   | 0.72                     | 0.54       |
| 6    | Logistic Regression  | 0.85    | 0.59   | 0.68                     | 0.51       |

### Why CatBoost Performs Best

- Native categorical handling  
- Minimal preprocessing  
- Stable under tuning  
- Strong bias-variance balance  

---

## 7. Business Impact Simulation

Assumptions:

- 7,000 customers  
- ~1,500 churn per cycle  
- Model recall = 79%  
- Customers contacted ≈ 2,200  
- Retention offer cost = $20  
- Campaign success rate = 15%  
- ARPU = $60  

Estimated retained customers:
~330

Potential saved monthly revenue:
~$19,800

Campaign cost:
~$44,000

Conclusion:
Threshold selection and lifetime value modeling are critical for profitability.

---

## 8. Model Explainability

SHAP applied to CatBoost predictions.

Top churn drivers:

- Short tenure  
- Month-to-month contract  
- High monthly charges  
- Lack of online security  
- Electronic check payment  

This supports actionable interventions such as:

- Early onboarding offers  
- Contract migration incentives  
- Bundled service promotions  
- Security feature upselling  

---

## 9. Production Architecture

Typical deployment flow:

1. Nightly batch scoring  
2. Store churn probability in CRM  
3. Risk segmentation  
4. Automated campaign triggering  
5. Quarterly retraining and drift monitoring  

Optional enhancements:

- FastAPI model serving  
- MLflow model registry  
- Docker containerization  
- Airflow scheduled retraining  
- Drift monitoring with Evidently  

---

## 10. Project Structure (Detailed)

```
Customer-Churn-Analysis/
│
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv
│   ├── processed/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── feature_matrix.parquet
│   └── external/
│
├── notebooks/
│   ├── 01-eda.ipynb
│   ├── 02-feature-engineering.ipynb
│   ├── 03-modeling-experiments.ipynb
│   └── 04-threshold-optimization.ipynb
│
├── src/
│   ├── config/
│   │   └── config.yaml
│   │
│   ├── data/
│   │   ├── load_data.py
│   │   ├── validate_data.py
│   │   └── split_data.py
│   │
│   ├── features/
│   │   ├── build_features.py
│   │   ├── feature_selection.py
│   │   └── encoding.py
│   │
│   ├── models/
│   │   ├── train_model.py
│   │   ├── tune_model.py
│   │   ├── evaluate.py
│   │   ├── threshold.py
│   │   └── explain.py
│   │
│   ├── pipeline/
│   │   └── churn_pipeline.py
│   │
│   ├── monitoring/
│   │   ├── drift.py
│   │   └── performance_tracking.py
│   │
│   └── utils/
│       ├── logger.py
│       ├── metrics.py
│       └── helpers.py
│
├── models/
│   ├── catboost_model.pkl
│   ├── feature_importance.csv
│   └── shap_values.npy
│
├── reports/
│   ├── eda_report.html
│   ├── model_comparison.csv
│   └── business_impact_summary.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

### Folder Responsibilities

### `data/`

Structured separation prevents contamination.

- `raw/`  
  Immutable source data. Never modified.

- `processed/`  
  Cleaned, transformed, split datasets. Reproducible via pipeline.

- `external/`  
  Optional enrichment datasets such as macro indicators or customer lifetime value tables.

---

### `notebooks/`

Used only for experimentation and exploration.

Each notebook has a clear purpose:
- `01-eda.ipynb` → Distribution analysis, imbalance review, churn drivers  
- `02-feature-engineering.ipynb` → Feature construction experiments  
- `03-modeling-experiments.ipynb` → Model comparison and hyperparameter trials  
- `04-threshold-optimization.ipynb` → Precision-recall tradeoff and business cost tuning  

Production logic is not embedded here.

---

### `src/`

Core production-grade code. All notebooks should eventually call functions from here.

#### `config/`
Central configuration file for:
- Random seeds  
- Feature lists  
- Model hyperparameters  
- Threshold selection  

Prevents hardcoding.

---

#### `data/`
Responsible for:
- Loading datasets  
- Data validation checks  
- Stratified splitting  

Ensures reproducibility.

---

#### `features/`
Contains transformation logic:
- Encoding categorical variables  
- Creating interaction features  
- Feature selection  
- Aggregations  

All transformations wrapped inside sklearn-compatible pipelines.

---

#### `models/`
Handles full ML lifecycle.

- `train_model.py` → Train chosen model  
- `tune_model.py` → Optuna-based hyperparameter tuning  
- `evaluate.py` → ROC-AUC, PR-AUC, confusion matrix  
- `threshold.py` → Precision-recall threshold selection  
- `explain.py` → SHAP interpretation  

This separation improves testability.

---

#### `pipeline/`
End-to-end orchestration script.

`churn_pipeline.py` performs:
1. Data loading  
2. Feature engineering  
3. Model training  
4. Evaluation  
5. Model saving  

Used for production batch runs.

---

#### `monitoring/`
Prepares system for real deployment.

- `drift.py` → Detect feature distribution shift  
- `performance_tracking.py` → Monitor recall, precision over time  

Prevents silent model degradation.

---

#### `utils/`
Shared utilities:
- Logging configuration  
- Custom metric functions  
- Helper functions  

Prevents duplication.

---

### `models/`

Stores serialized artifacts:

- Trained model file  
- Feature importance export  
- SHAP outputs  

This folder is usually excluded from version control in real production.

---

### `reports/`

Business-facing deliverables:

- EDA visual summary  
- Model comparison table  
- Business impact documentation  

Separates technical output from business communication.

---

## Why This Structure Matters

This is no longer a notebook project.

It demonstrates:

- Separation of concerns  
- Reproducibility  
- Production thinking  
- Monitoring readiness  
- Business alignment  

Most churn projects online are just notebooks. This structure signals engineering maturity.

---

## 11. Quick Start

```bash
git clone https://github.com/stageor/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

jupyter lab notebooks/01-eda.ipynb

python src/models/train_model.py --model catboost --save
```

---

## 12. Future Improvements

- Survival analysis for time-to-churn modeling  
- Cost-sensitive learning  
- Uplift modeling  
- Real-time scoring pipeline  
- Customer lifetime value integration  
- Continuous drift monitoring  
