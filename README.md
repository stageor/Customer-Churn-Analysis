# Customer Churn Prediction and Revenue Impact Modeling

Developed a machine learning model to predict customer churn for telecom, SaaS, and subscription-based businesses.

The project focused on delivering measurable business impact by optimizing constrained recall, improving campaign ROI, and building deployable, production-ready machine learning systems.

---

## 1. Business Context

Churn is usually recognized only after revenue erosion becomes measurable. At that point, recovery becomes significantly more expensive.

Retention is structurally cheaper than acquisition. However, many organizations still rely on broad, untargeted retention campaigns because predictive targeting capabilities are weak.

**Executive Question**

Customers are most likely to churn in next 30 days, & within fixed campaign budget, many of them can realistically be retained?

### Project Objective

Build a churn prediction framework that:

- Identifies high-risk customers  
- Maximizes recall under outreach capacity constraints  
- Quantifies expected revenue preservation  
- Provides interpretable drivers for action  

The primary goal is financial impact, not accuracy optimization.

---

## 2. Dataset Overview

**Primary Dataset:** Telco Customer Churn  

- Total records: 7,043  
- Churn rate: ~26.5%  

**File Path**
```
data/Telco-Customer-Churn.csv
```

### Feature Categories

**Demographics**
- Gender  
- SeniorCitizen  
- Partner  
- Dependents  

**Service Usage**
- InternetService  
- StreamingTV  
- OnlineSecurity  
- TechSupport  

**Account Attributes**
- Tenure  
- Contract  
- MonthlyCharges  
- TotalCharges  

**Target Variable**
- Churn (Yes / No)

The class distribution is moderately imbalanced, with churn accounting for roughly 27% of the observations.

---

## 3. Exploratory Data Analysis Findings

Key patterns observed:

- Month-to-month contracts churn at 3 to 4 times the rate of long-term contracts  
- Low tenure combined with high monthly charges defines the highest-risk cohort  
- Absence of OnlineSecurity and TechSupport increases churn probability  
- Electronic check payment method correlates with churn  
- Senior citizens exhibit slightly higher churn rates  

### Modeling Implications

- Stratified cross-validation is mandatory  
- Precision-Recall curves are more informative than ROC curves  
- Threshold tuning is not optional  

A default probability threshold of 0.5 ignores campaign economics and is therefore unrealistic.

---

## 4. Feature Engineering

Enhancements introduced:

- Tenure banding  
- Revenue-to-tenure ratio  
- Aggregated service count  
- Binary encoding of contract types  
- Interaction term between tenure and monthly charges  
- Log transformation for skewed monetary variables  

Final engineered feature set contains approximately 20 variables.

All preprocessing steps are encapsulated within **scikit-learn pipelines** to prevent data leakage.


---

## 5. Modeling Framework

### Validation Design

- 5-fold stratified cross-validation  
- 20% hold-out test split  
- Fixed random seed set to 2025  
- Optuna-based hyperparameter optimization with 60 to 120 trials per model  
- End-to-end pipeline validation  

### Primary Business Metric

Recall at approximately 30% precision.

Interpretation:

If outreach capacity allows contacting 30% of customers, how many churners are captured?

PR-AUC is prioritized due to class imbalance.

---

## 6. Model Benchmarking

| Model                | ROC-AUC | PR-AUC | Recall @ ~30% Precision | F1 (Churn) |
|----------------------|---------|--------|--------------------------|------------|
| CatBoost             | 0.90    | 0.69   | 0.79                     | 0.60       |
| LightGBM             | 0.895   | 0.685  | 0.78                     | 0.595      |
| XGBoost              | 0.892   | 0.678  | 0.77                     | 0.59       |
| HistGradientBoosting | 0.885   | 0.66   | 0.75                     | 0.57       |
| Random Forest        | 0.87    | 0.63   | 0.72                     | 0.54       |
| Logistic Regression  | 0.85    | 0.59   | 0.68                     | 0.51       |

### Interpretation

A small difference in ROC performance is rarely actionable.

However, a 10% improvement in recall under a fixed campaign capacity can directly increase retained revenue.

For 1,500 churners per cycle:

- CatBoost identifies approximately 1,185  
- Logistic Regression identifies approximately 1,020  

That is 165 additional high-risk customers identified per cycle. Model choice affects revenue outcomes.

---

## 7. Final Model Selection

Selected model: CatBoost

Rationale:

- Highest PR-AUC  
- Strong recall in the high-recall operating region  
- Stable cross-validation performance  
- Native handling of categorical variables  
- Minimal preprocessing overhead  

Churn behavior is nonlinear. Gradient boosting methods consistently outperform linear baselines in this context.

---

## 8. Business Impact Simulation

Assumptions:

- 7,000 customers  
- 1,500 churn events per cycle  
- Recall of 79%  
- 2,200 customers contacted  
- Retention cost per contact: $20  
- Offer conversion rate: 15%  
- Average revenue per user: $60  

Estimated retained customers: approximately 330  
Monthly revenue preserved: approximately $19,800  
Campaign cost: approximately $44,000  

Conclusion:

Recall without cost modeling is insufficient. Profitability depends on threshold optimization and lifetime value integration.

---

## 9. Explainability Layer

SHAP analysis applied to CatBoost.

Top drivers of churn:

- Short tenure  
- Month-to-month contracts  
- High monthly charges  
- Absence of online security  
- Electronic check payments  

These drivers support targeted, actionable retention strategies rather than generic discounting.

---

## 10. Deployment Architecture

Production-oriented flow:

1. Nightly batch scoring  
2. Store churn probability in CRM  
3. Risk-based segmentation  
4. Automated campaign triggering  
5. Scheduled retraining each quarter  
6. Data and prediction drift monitoring  

Optional tooling stack:

- FastAPI  
- MLflow  
- Docker  
- Airflow  
- Evidently  

The architecture is built for operational deployment rather than notebook-level experimentation.

---

## 11. Repository Structure

```
Customer-Churn-Analysis/
│
├── data/
├── notebooks/
├── src/
├── models/
├── reports/
├── requirements.txt
└── README.md
```


All production logic resides in `src/`.  
Notebooks are used strictly for exploration.

---

## 12. Setup Instructions

```bash
git clone https://github.com/stageor/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python src/models/train_model.py --model catboost --save

---

## 13. Future Extensions

- Survival analysis for time-to-churn  
- Cost-sensitive learning  
- Uplift modeling  
- Real-time scoring API  
- Customer lifetime value integration  
- Continuous performance monitoring  

---
