# Credit-Risk-Scoring-Dashboard
# Credit Risk Analysis & Default Prediction

Analyzing loan default patterns and predicting default risk using Python and Power BI.

## Overview
This project explores which borrower and loan characteristics are associated
with loan default, and compares three classification models for predicting it.
Findings are presented in three Power BI dashboards.

## Dataset
- Source: Kaggle "Credit Risk Dataset" ([https://www.kaggle.com/datasets/laotse/credit-risk-dataset](url))
- 32,581 loan records after cleaning, 12 original columns
- Target: `loan_status` (1 = default, 0 = repaid)
- Default rate: 21.8% of loans

## Workflow
1. **Cleaning:** filtered unrealistic values (age, employment length),
   imputed missing `person_emp_length` and `loan_int_rate` with the median
2. **EDA:** default rate by home ownership, loan purpose, loan grade,
   and credit history; comparison of average financial metrics by outcome
3. **Preprocessing:** one-hot encoding of categorical variables
4. **Modeling:** stratified 80/20 train/test split; Logistic Regression,
   Decision Tree, Random Forest
5. **Evaluation:** Accuracy, ROC-AUC, precision/recall/F1, confusion matrix,
   feature importance
6. **Visualization:** 3 Power BI dashboards

## Results
| Model               | Accuracy | ROC-AUC | F1 (default class) |
|---------------------|----------|---------|--------------------|
| Logistic Regression | 85.01%   | 85.59%  |        57%         |
| Decision Tree       | 89.17%   | 84.92%  |        76%         |
| **Random Forest**   | 93.47%   | 93.7%   |        83%         |

**Random Forest, default class:** precision 0.96, recall 0.73.
The model is very reliable when it flags a default, but it misses about 27%
of actual defaults. In credit risk, missed defaults are usually costlier than
false alarms, so recall is the main metric to improve.

### Top drivers (Random Forest feature importance)
1. `loan_percent_income` (0.215)
2. `person_income` (0.151)
3. `loan_int_rate` (0.122)

Renters (`person_home_ownership_RENT`) and grade D loans also rank high.
Feature importance shows contribution to prediction, not causation.

## Dashboards
[Overview Khách hàng]
[Overview rủi ro]
[Mô hình dự báo]

## Limitations & next steps
- Median imputation was computed before the train/test split
  (minor leakage); next step is to use a `Pipeline`
- `loan_grade` and `loan_int_rate` reflect the lender's own risk assessment,
  so the model partly relearns that assessment
- Default 0.5 threshold, default hyperparameters, no cross-validation
- Planned: class weighting, threshold tuning, 5-fold stratified CV

## How to run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
# download the dataset from Kaggle and place it next to the script
python credit_risk.py
```

## Tech stack
Python (pandas, scikit-learn, matplotlib, seaborn), Power BI
