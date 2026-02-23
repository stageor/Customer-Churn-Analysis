# Customer Churn Prediction & Analysis

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?style=flat-square)
![EDA](https://img.shields.io/badge/📊-Exploratory%20Data%20Analysis-important?style=flat-square)
![Imbalanced Learning](https://img.shields.io/badge/🔄-Imbalanced%20Learning-ff69b4?style=flat-square)

End-to-end machine learning project to **understand and predict customer churn** in telecom (or similar subscription-based businesses like banking/SaaS).

## 🎯 Business Problem

> "We are losing too many customers every month and we don't know exactly why, nor who is going to leave next."

**Goal**  
Develop a predictive model to identify customers likely to churn in the next 30 days, balancing precision and recall → enable targeted, cost-effective retention campaigns.

## 📊 Dataset

Primarily using the classic **Telco Customer Churn** dataset (widely used benchmark).

| Dataset                              | Rows  | Churn Rate | Typical Use Case          | Source/Link                                                                 |
|--------------------------------------|-------|------------|---------------------------|-----------------------------------------------------------------------------|
| WA_Fn-UseC_-Telco-Customer-Churn     | 7043  | ~26.5%     | Classic benchmark, beginners | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)   |
| Bank Customer Churn                  | ~10k  | ~20%       | Banking use-case          | Kaggle                                                                      |
| E-commerce Churn                     | 5–20k | 15–30%     | Online retail             | Various Kaggle datasets                                                     |
| Synthetic / custom                   | —     | —          | Advanced experiments      | —                                                                           |

**File in repo**: `Telco-Customer-Churn.csv`


## 🏆 Model Performance Comparison

(5-fold stratified CV • 20% hold-out • random seed 2025 • no leakage)

| Rank | Model                | ROC-AUC | PR-AUC | Recall @ ~30% Precision | F1 (churn) | Training Time | Notes                              |
|------|----------------------|---------|--------|-------------------------|------------|---------------|------------------------------------|
| 1    | CatBoost             | **0.90** | **0.69** | **0.79**                | **0.60**   | ~5 s          | Best overall, native categoricals  |
| 2    | LightGBM             | 0.895   | 0.685  | 0.78                    | 0.595      | ~1.5 s        | Fastest strong performer           |
| 3    | XGBoost              | 0.892   | 0.678  | 0.77                    | 0.59       | ~3 s          | Robust with tuning                 |
| 4    | HistGradientBoosting | 0.885   | 0.66   | 0.75                    | 0.57       | ~1 s          | sklearn-native, very stable        |
| 5    | Random Forest        | 0.87    | 0.63   | 0.72                    | 0.54       | ~4 s          | Interpretable baseline             |
| 6    | Logistic Regression  | 0.85    | 0.59   | 0.68                    | 0.51       | <1 s          | Strong simple baseline             |

**Evaluation notes**  
- **Primary metric**: Recall at ~30% precision (contact ~30% of customers to catch 75–80% of churners)  
- PR-AUC prioritized due to ~27% churn imbalance  
- Tuning: Optuna (60–120 trials)  
- Features: 18–26 after engineering/selection  

**Business impact example** (Telco ~7k customers)  
- ~1,400–1,500 monthly at-risk customers  
- Catch ~79% → retain ~1,100–1,200 (contacting ~2,100–2,300)  
- With $15–25 retention offer & 12–18% success rate → potential saved revenue $2k–$8k/month (depending on ARPU ~$50–70)

## 🚀 Quick Start

```bash
# 1. Clone repo
git clone https://github.com/stageor/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis

# 2. Virtual env (uv recommended for speed)
# Option A – uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Option B – venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Explore EDA
jupyter lab notebooks/01-eda.ipynb

# 4. Train best model
python src/models/train_model.py --model catboost --save
