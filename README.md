# Credit Risk & Fraud Strategy Analytics Pipeline

An end-to-end fraud strategy and credit risk modeling pipeline built on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) dataset. This project goes beyond standard Kaggle modeling to simulate the full lifecycle of a production fraud risk control strategy — from rule development through model deployment, threshold optimization, and ongoing monitoring.

---

## Overview

The 8% default rate in this dataset mirrors real-world consumer fraud and credit loss rates, making it an ideal proxy for fintech risk strategy work. The pipeline demonstrates how a fraud strategy team would approach this problem operationally: starting with interpretable rules, layering in machine learning, and building the monitoring infrastructure needed to maintain performance in production.

---

## Dataset

**Source:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)

**File required:** `application_train.csv` (place in project root)

**Key statistics:**
- ~307,000 loan applications
- ~8% default rate (severe class imbalance — mirrors real fraud rates)
- 122 features covering demographics, financials, and credit bureau signals

---

## Skills Demonstrated

### Fraud Strategy & Rule Development
- Standalone rule-based control strategy with 5 risk rules (income floor, debt burden, employment flag, bureau score threshold, peer default network)
- Per-rule evaluation: hit rate, precision, incremental recall
- Combined OR-logic rule strategy with approval rate and F1 analysis
- Fraud typology mapping: thin-file / synthetic identity, loan stacking, bust-out fraud, fraud rings

### Machine Learning & Imbalance Handling
- Four model comparison: Logistic Regression, Random Forest, XGBoost (scale_pos_weight), XGBoost + SMOTE
- Industry-standard credit metrics: **KS Statistic**, **Gini Coefficient**, ROC-AUC, PR-AUC
- SHAP feature interpretability with business narrative per feature
- SMOTE oversampling with proper train-only application

### Business-Driven Decision Making
- Cost matrix threshold optimization (FN=$5,000, FP=$200) — finds optimal operational cutoff
- Expected loss calculation per applicant and per risk tier
- Business narrative explaining trade-offs at each threshold
- Three threshold scenarios: min-cost, F1-optimal, precision-constrained (regulatory)

### Operational Risk Scorecard
- Score bucketing into Low / Medium / High / Very High risk tiers
- Per-tier: default rate, approval action, expected loss per applicant, cumulative bad capture
- Operationally-ready decision table (Auto-Approve / Manual Review / Decline / Decline + Alert)

### Model Monitoring (PSI)
- Population Stability Index (PSI) for score distribution drift detection
- Development vs. production comparison
- Economic stress simulation: 10% feature shift scenario
- Automated alert thresholds: PSI < 0.10 Stable | 0.10–0.20 Monitor | > 0.20 Recalibrate

### Champion-Challenger Framework
- Full KPI comparison table across all strategies
- Champion selection rationale with quantified deltas
- C/C deployment recommendation (10% traffic allocation)

### Feature Engineering
- `DAYS_EMPLOYED_ANOMALY`: sentinel value flag (365,243 = unemployed/pensioner data artifact)
- Debt burden ratios: credit/income, annuity/income, credit/goods (LTV proxy)
- Bureau score aggregates: mean, min, std across EXT_SOURCE_1/2/3
- Social network default rate (fraud ring signal)
- Employment stability flag and age-employment ratio

---

## Setup & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset from Kaggle and place in project root
#    https://www.kaggle.com/competitions/home-credit-default-risk/data
#    File: application_train.csv

# 3. Run the pipeline
python credit_risk_model.py
```

Expected runtime: ~8–15 minutes on a standard laptop (SMOTE + SHAP are the slowest steps).

---

## Output Files

All outputs are written to `outputs/` (auto-created at runtime):

| File | Description |
|------|-------------|
| `eda_plots.png` | 8-panel EDA: missingness, default rates by segment, distributions, correlations |
| `shap_summary.png` | SHAP beeswarm plot — top 15 features by impact, with fraud typology annotations |
| `dashboard.png` | Full analytics dashboard: PR/ROC curves, threshold tuning, KS viz, PSI monitoring, C/C table |
| `credit_risk_model.log` | Full run log with timing, data stats, and metric summaries |

---

## Results Summary (Champion Model: XGB Weighted)

| Metric | Value |
|--------|-------|
| ROC-AUC | ~0.77 |
| PR-AUC (Avg Precision) | ~0.29 |
| KS Statistic | ~46 |
| Gini Coefficient | ~0.54 |
| Approval Rate @ threshold=0.30 | ~70% |
| Precision @ threshold=0.30 | ~0.37 |
| Recall @ threshold=0.30 | ~0.59 |

*Exact values vary by run due to SMOTE sampling. KS > 40 and Gini > 0.5 are industry benchmarks for a good credit scorecard.*

---

## Project Structure

```
credit_risk/
├── credit_risk_model.py   # End-to-end pipeline script
├── requirements.txt       # Python dependencies
├── README.md
├── .gitignore
└── outputs/               # Auto-generated at runtime
    ├── eda_plots.png
    ├── shap_summary.png
    ├── dashboard.png
    └── credit_risk_model.log
```

> **Note:** `application_train.csv` and `outputs/` are excluded from version control via `.gitignore`. Download the data from Kaggle and run the script to regenerate all outputs.
