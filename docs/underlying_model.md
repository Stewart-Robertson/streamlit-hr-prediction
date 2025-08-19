
# The Prediction Model (Plain English + Under the Hood)

!!! note "TL;DR"  
    The app uses a calibrated logistic regression model to estimate each employee’s probability of leaving.
    It’s trained on historical snapshots up to 2022 and validated on 2023 to mimic how it would perform in the next year.
    Results are turned into actionable savings estimates using your assumptions (intervention cost, effectiveness, replacement cost).

## What the model does

Think of the model like a **weather forecast for attrition**:

- Each person gets a risk score (0–100%) — "how likely is rain (attrition) for this person?"
- We don't claim certainty for any individual. Instead, we use these probabilities to prioritise attention (like taking an umbrella if rain is likely)
- The app lets you test what-if levers (e.g., reduce workload) and instantly see how the forecast and savings might change

### Analogy

You could think of logistic regression as a **weighted checklist**. For example, heavier workload and long commute may push risk up; higher pay competitiveness may push risk down. The model learns how much each factor matters based on past outcomes.

---

## What the Model Does NOT Do

| Limitation | Description |
|------------|-------------|
| **Decision Making** | It doesn't decide who to keep or let go — it's decision support |
| **Future Prediction** | It doesn't know the future. It extrapolates from patterns in your historical snapshots |
| **Post-Decision Learning** | It doesn't see post-decision outcomes (e.g., exit interviews, tickets) that would leak future information into training |

---

## Key Design Choices

### Transparency by Design
Logistic regression is easy to inspect and explain, a better fit for HR decision-making than black-box models when accuracy is similar.

### Calibrated Probabilities
We calibrate the model so "30% risk" roughly means "30 out of 100 similar employees leave."

### Time-Aware Validation
Train on ≤2022, evaluate on 2023. This prevents "peeking" into the future and gives a realistic performance estimate.

### Scenario-Ready
We engineered a feature (`mgmt_workload_score`) so what-if changes to workload and manager quality flow through to risk.

---

## Model Inputs (Features)

From the cleaned dataset (`data/processed/hr_attrition_clean.csv`), the model uses:

### Job & Pay
- `base_salary` (pay level)
- _`salary_band`_ (dropped to avoid duplication)
- _`compa_ratio`_ (dropped in final model due to collinearity)

### Performance & Progression
- `performance_rating`
- _`avg_raise_3y`_ (dropped)
- _`internal_moves_last_2y`_ (dropped)
- `time_since_last_promo_yrs`

### Work Patterns & Wellbeing
- `workload_score`
- `overtime_hours_month`
- `sick_days`
- `pto_days_taken`

### Engagement
- `engagement_score`
- `manager_quality` (missingness is informative; also used in interaction below)

### Logistics
- `commute_km`
- `onsite/remote` (if present)
- `night_shift`

### Organizational Context
- `department`
- `role`
- `team_id` (as categorical signals)

### Engineered Features
- `mgmt_workload_score = (10 − manager_quality) × workload_score`<br>
  *Captures that high workload under poorer management is especially risky*

### Explicitly Excluded Features

| Category | Features | Reason |
|----------|----------|---------|
| **Identifiers** | `employee_id` | Not predictive |
| **Post-Decision Signals** | `exit_interview_scheduled`, `offboarding_ticket_created` | Data leakage |
| **Duplicates/Collinear** | `salary_band`, high-VIF pay proxies (`compa_ratio`, `avg_raise_3y`, `benefit_score`) | Redundancy/collinearity |
| **Split Key** | `snapshot_year` | Used to split, not to predict |

!!! note "Missing Values"
    Missing `manager_quality` can itself be a signal. The pipeline treats missing values via encoding/scaling, and the engineered interaction uses a conservative floor (e.g., treats missing as 1 for the "quality" term in the what-if mechanism).

---

## Technical Pipeline

### 1. Preprocessing
- **Numerical**: `StandardScaler(with_mean=False)`
- **Categorical**: `OneHotEncoder(handle_unknown="ignore", sparse=True)`
- **Combined**: `ColumnTransformer`

### 2. Estimator
- `LogisticRegression(max_iter=1000)` trained on data ≤2022

### 3. Calibration
- `CalibratedClassifierCV(cv="prefit")` with isotonic or sigmoid chosen by lower Brier score on the 2023 set

### 4. Persistence
- **Model**: `models/attrition_lr_calibrated_train_to_2022_skl171.pkl`
- **Metrics**: `models/attrition_lr_calibrated_metrics_train_to_2022_skl171.json`

!!! info "Why Calibration?"
    Uncalibrated models can be over- or under-confident. Calibration aligns predicted probabilities with observed rates — crucial when you convert probabilities into expected savings.

---

## Validation & Performance

### Data Split
- **Train**: Snapshots ≤2022
- **Calibrate/Evaluate**: 2023

### Key Metrics

| Metric | Purpose |
|--------|---------|
| **ROC AUC** | Overall ranking quality (closer to 1 is better) |
| **PR AUC** | Useful when attrition is rare |
| **Lift by Decile** | Business-friendly: top 10% risk vs average rate |
| **Calibration** | Through the chosen calibration method (implicit) |

!!! tip "Interpreting Lift"
    If Decile 1 shows 3× lift, your top-risk 10% is three times as likely to contain real leavers as a random 10%. That's strong targeting signal.

---

## From Probability to Business Value

Two complementary savings views power the app:

### 1. Threshold-Based Net Savings

Pick a risk threshold; treat everyone above it.

```
Saved cost = (True Positives × effectiveness × replacement_cost)
Spend = (Flagged × intervention_cost)
Net = Saved − Spend
```

**Good for**: Operational reporting, precision/recall trade-offs

### 2. Threshold-Free Expected Value

Sum expected value across the treated cohort (Top-K or Threshold with Coverage):

```
Per person EV = effectiveness × replacement_cost × predicted_risk − intervention_cost
```

**Good for**: Comparing levers & strategies without being sensitive to a single threshold

---

## Why Not a Complex Model?

We tried Random Forest and XGBoost variants; in this dataset, logistic regression performed comparably (AUC around ~0.65) but offered **much better interpretability and governance**. When accuracy differences are marginal, simpler + explainable is the right HR choice.

---

## Limits & Caveats

### Causality
The model captures associations, not guaranteed causes. Use it to prioritise conversations and support, not as an automated decision engine.

### Data Drift
If your org changes (hybrid policies, comp structure), refresh training and re-calibrate. The app's time-aware split is a safeguard, not a guarantee.

### Group Effects
`team_id` can encode manager/team culture. If governance requires, add GroupKFold validation by team to quantify sensitivity.

### Fairness
Always review risk & intervention rates by relevant groups (e.g., function, location). Add governance checks before production use.

---

## Model Interpretation

### Coefficients
In logistic regression, each feature has a weight:
- **Positive weight** → increases log-odds of attrition
- **Negative weight** → decreases log-odds of attrition

You can export coefficients and a per-feature report for HR/legal review.

### Partial Effects
Use decile tables or partial dependence for top drivers (e.g., workload ↑, manager quality ↓).

### Calibration Check
Compare predicted vs actual rate in bins (the app's lift + AUC and calibration choice are proxies; you can add a reliability curve if needed).

---

## Reproducibility & Versioning

| Component | Location |
|-----------|----------|
| **Training Code** | `src/train_calibrated_lr.py` |
| **Serving Code** | `app/main.py` |
| **Data** | `data/processed/hr_attrition_clean.csv` |
| **Environment** | `requirements.txt` and `runtime.txt` (Python 3.11) |
| **Artifacts** | Model `.pkl` and metrics `.json` versioned by train cutoff year and sklearn tag |

!!! tip "Enterprise Tip"
    For enterprise, log training runs (data hash, params, metrics) in MLflow or DVC, and schedule periodic re-calibration.

---

## Governance Checklist

Ready for productionization:

- Data lineage documented (source, refresh cadence, snapshot definition)
- Feature list reviewed (no prohibited attributes; leakage removed)
- Validation includes time-based and, if required, grouped CV
- Calibration validated on out-of-time data; reliability plot archived
- Fairness report (rates by segment, false positive/negative parity where applicable)
- Monitoring plan (drift, performance drop alerts)
- Human-in-the-loop SOP (how HR acts on risk, audit trail)

---

## Frequently Asked Questions

### Is 65% AUC "good enough"?
For HR attrition with limited features, ~0.6–0.7 is common. Value comes from targeting and what-if planning, not perfect prediction.

### Why does Δ Net Savings sometimes show 0 while threshold-free savings changes?
Your scenario may shift probabilities without moving many people across the chosen threshold. The threshold-free view captures that subtle improvement.

### Can we add more drivers?
Yes — especially manager signals, work pattern telemetry, career progression, and comp competitiveness vs market (with governance).