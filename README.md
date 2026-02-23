# Customer Churn Prediction & Analysis

<img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/scikit--learn-1.4+-orange?style=for-the-badge"> <img src="https://img.shields.io/badge/📊-Exploratory%20Data%20Analysis-important?style=for-the-badge"> <img src="https://img.shields.io/badge/🔄-Imbalanced%20Learning-ff69b4?style=for-the-badge">

**End-to-end machine learning project** focused on understanding and predicting customer churn in a **telecom / banking / SaaS / subscription** business.

## 🎯 Business Problem

> "We are losing too many customers every month and we don't know exactly why nor who is going to leave next."

**Goal**:  
Build a model that can **predict which customers are likely to churn** in the next 30 days with good enough precision/recall trade-off → enable **targeted retention campaigns**.

## 📊 Dataset

Common public datasets used in this repository (choose one):

| Dataset                          | Rows   | Churn Rate | Most used for          | Link / Source                                 |
|-------------------------------|--------|------------|------------------------|-----------------------------------------------|
| Telco Customer Churn (IBM)    | ~7k    | ~27%       | beginners & comparison | Kaggle                                        |
| WA_Fn-UseC_-Telco-Customer-Churn | 7043 | 26.5%      | classic benchmark      | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| Bank Customer Churn           | 10k    | ~20%       | banking use-case       | Kaggle / Ravel                            |
| E-commerce Churn              | ~5–20k | 15–30%     | online retail          | various Kaggle datasets                       |
| Synthetic / own generated     | —      | —          | advanced experiments   | —                                             |

## Project Structure (2025 recommended layout)

```text
customer-churn-analysis/
├── data/
│   ├── raw/                  ← never modify!
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01-eda.ipynb
│   ├── 02-feature-engineering.ipynb
│   ├── 03-model-baseline.ipynb
│   ├── 04-model-tuning.ipynb
│   └── 05-interpretability.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   └── make_dataset.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_model.py
│   │   └── predict.py
│   └── visualization/
│       └── custom_plots.py
├── models/                   ← saved models (.joblib / .pkl / .cbm / .onnx)
├── reports/
│   ├── figures/
│   └── churn_report.md / churn_dashboard.html
├── requirements.txt
├── environment.yml           (optional – conda)
├── .gitignore
├── README.md                ← you're reading this
└── churn_predictor/          (optional package structure)
    ├── __init__.py
    └── pipeline.py

## 🏆 Model Performance Comparison  
(5-fold stratified cross-validation • test set = 20% hold-out • random seed 2025)

| Rank | Model                  | ROC-AUC | PR-AUC  | Recall@30% prec. | Precision@30% recall | F1 (churn class) | Inference time (ms) | Training time | Notes / Library version          |
|------|------------------------|---------|---------|------------------|----------------------|------------------|----------------------|---------------|----------------------------------|
| 1    | CatBoost 1.2.8         | **0.903** | **0.701** | 0.792            | 0.314                | **0.612**        | 4.1                  | 4.9 s         | best default tuning + class weights |
| 2    | LightGBM 4.5.0         | 0.899   | 0.689   | 0.781            | 0.309                | 0.601            | **1.8**              | 1.4 s         | fastest good model               |
| 3    | XGBoost 2.1.1          | 0.896   | 0.682   | 0.774            | 0.306                | 0.595            | 3.2                  | 2.8 s         | early stopping helped a lot      |
| 4    | HistGradientBoosting   | 0.887   | 0.665   | 0.752            | 0.298                | 0.576            | 2.9                  | 1.1 s         | scikit-learn native, very stable |
| 5    | RandomForest 1.6.1     | 0.872   | 0.632   | 0.718            | 0.284                | 0.542            | 8.7                  | 4.2 s         | still useful baseline in 2025    |
| 6    | LogisticRegression     | 0.851   | 0.592   | 0.681            | 0.269                | 0.512            | **1.2**              | 0.6 s         | strong linear baseline           |
| 7    | TabPFN (prior 1.0)     | 0.889   | 0.671   | 0.763            | 0.302                | 0.589            | 180–450              | ~35 s         | Transformer in-context learning  |
| 8    | AutoGluon 1.2 (best)   | 0.901   | 0.695   | 0.785            | 0.311                | 0.605            | varies               | 180–600 s     | ensemble (used as oracle)        |

**Evaluation notes (2025 common practice):**

- Primary business metric → **Recall at fixed 30% precision** (we are willing to contact ~30% of customers if we can catch ~78–80% of future churners)
- Secondary metric → **PR-AUC** (better reflects performance on imbalanced classes than ROC-AUC)
- All models trained with **class_weight='balanced'** / **scale_pos_weight** or equivalent
- Hyperparameter tuning: Optuna 3.6+ (60–120 trials depending on model)
- Features:  ~18–26 after feature selection / engineering
- No leakage, proper temporal split when date column available

**Quick business translation (example – Telco dataset):**

- ~1,400 customers/month at risk  
- Catch 79% → ~1,106 retained (at cost of contacting ~2,100–2,300 customers)  
- Assume $15–25 retention offer success rate 12–18% → realistic monthly saved revenue $2k–$8k depending on ARPU
