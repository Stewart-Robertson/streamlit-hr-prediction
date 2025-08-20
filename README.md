# Business Turnover & Savings Estimator

_Predict employee attrition risk, run what-if scenarios, and quantify net savings under real-world budget constraints._

This project showcases a practical, decision-support data product aimed at HR leadership and Finance—clear enough for non-technical users, with enough detail for data scientists and analysts to trust.

> **Built fast by partnering with AI.**
> I intentionally used AI as a collaborative teammate throughout this project—accelerating planning, coding, documentation, and iteration.
> This repo is a transparent example of how a data analyst can ship useful, end-to-end products quickly by pair-programming with AI while owning the **reasoning**, **design**, **governance**, and **validation**.

---

## Links

| Resource | URL |
|----------|-----|
| **Live App** | [Streamlit Cloud](https://hr-app-sr.streamlit.app/#business-turnover-savings-predictor) |
| **Documentation** | [Usage Guide](https://stewart-robertson.github.io/streamlit-hr-prediction/) |
| **LinkedIn** | [Connect with me](https://www.linkedin.com/in/stewart-robertson-data/) |

---

## What This App Does

- **Scores attrition risk** for each employee using a calibrated logistic regression model (time-aware split: train ≤ 2022, test 2023)

- **Two targeting strategies:**
  - **Threshold**: Treat everyone above a chosen risk level
  - **Top-K**: Treat the highest-risk K% with optional coverage

- **Savings Sandbox**: Adjust replacement cost multiple, intervention cost, effectiveness, planning horizon (months), and monthly vs total spend

- **Economically optimal threshold**: Compute the profit-maximizing cut-off from your assumptions

- **What-if scenarios**: Test impact of reducing workload and improving manager quality; the model includes an engineered manager × workload interaction

- **Interpretability for stakeholders:**
  - "Executive summary" with Net savings, Flagged, Precision, Recall
  - ROC curve and Lift by decile with plain-language explainers
  - Threshold-free (coverage-adjusted) savings for Top-K pilots

---

## Repository Structure

```
.
├── app
│   └── main.py
├── config
│   ├── high_vif_candidates.json
│   └── selected_features.json
├── data
│   └── processed
│       └── hr_attrition_clean.csv
├── docs
│   └── ... (documentation files)
├── models
│   └── attrition_lr_calibrated_train_to_2022_skl171.pkl
├── notebooks
│   └── ... (Jupyter notebooks)
├── src
│   ├── data_cleaning.py
│   └── train_calibrated_lr.py
├── .gitignore
├── mkdocs.yml
├── README.md
└── requirements.txt
```

> **Note**: Data is synthetic and used solely to demo model + app mechanics.

## Model & Methodology

### Algorithm
- **Logistic Regression** with one-hot encoding + scaling
- **Probability calibration** (sigmoid or isotonic) using next-year data

### Validation
- **Time-based split**: Train ≤ 2022, test 2023
- **Option for grouped CV** by team to guard against team-level leakage

### Feature Engineering
- Dropped leaky columns (e.g., exit interview, offboarding)
- Reduced multicollinearity (comp/benefits stack)
- Added `mgmt_workload_score` (worse manager × higher workload)

### Performance (Live in App)
- **AUC**: ≈ 0.65–0.68
- **Strong lift** in top deciles
- **Good calibration** (predicted ≈ observed)

---

## Savings Calculations (Stakeholder-Friendly)

### Net Savings (Scenario)
Threshold-based calculation:
```
Net = TP × effectiveness × replacement_cost − (TP+FP) × intervention_cost
```

### Threshold-Free Savings (Coverage-Adjusted)
Top-K or Threshold cohort with Coverage %; sums per-person expected value from risk reductions (even if not crossing a threshold).

### Planning Horizon
- **Costs scale** to your selected months (e.g., 3-month view)
- **Replacement cost**: `avg_salary × replacement_multiple × (horizon/12)`
- **Intervention cost**: Total program cost or monthly × horizon (toggle in UI)
- **Economically optimal threshold**: Computed from current assumptions; offered as reference

---

## How to Use (2-Minute Tour)

1. **Set assumptions** in the left sidebar: replacement multiple, intervention cost, effectiveness, horizon (months)

2. **Choose strategy**: Threshold or Top-K (with Coverage %)

3. **Use what-if levers**: Reduce workload, improve manager quality; watch risk & savings update

4. **Read executive summary**: Net savings, Flagged, Precision/Recall; then check Model performance (ROC, Lift)

5. **Make a plan**: Start with Top-K + Coverage for a targeted pilot; export Top-risk employees

📖 **Full guide**: [https://stewart-robertson.github.io/streamlit-hr-prediction/](https://stewart-robertson.github.io/streamlit-hr-prediction/)

--- 

## Ethics, Fairness & Scope

- This app is **decision support**—not an automated decision system
- Data here is synthetic; in production, ensure bias audits, legal review, and employee privacy protections
- Avoid using sensitive or prohibited attributes
- Provide appeal and human review for any flagged list

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python 3.11 |
| **UI Framework** | Streamlit |
| **Data Processing** | pandas, numpy |
| **Machine Learning** | scikit-learn |
| **Visualization** | plotly |
| **Documentation** | MkDocs Material |
| **Model Persistence** | joblib |

---

## Contact

- **Documentation**: [https://stewart-robertson.github.io/streamlit-hr-prediction/](https://stewart-robertson.github.io/streamlit-hr-prediction/)
- **LinkedIn**: [https://www.linkedin.com/in/stewartrobertson-data/](https://www.linkedin.com/in/stewartrobertson-data/)

If you try it, I'd love feedback—especially from HR and Finance leaders on what would make this even more useful in the real world.
