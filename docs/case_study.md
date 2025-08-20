# **Fictional Case Study**: Reducing Avoidable Turnover at XY Financial

## Summary

!!! success "Key Result"
    With **25,000 employees at 17% attrition**, XY turned a costly annual problem into a **data-backed, quarterly savings plan** worth $5–7M per quarter, using transparent risk scoring, horizon-scaled economics, and scenario testing.

## Company Profile

| Attribute | Details |
|-----------|---------|
| **Company** | XY Financial *(fictional, based on real project scale)* |
| **Industry** | Financial Services |
| **Employees** | 25,000 |
| **Baseline Attrition** | ~17% annually (≈4,250 exits/year) |
| **Goal** | Reduce costly turnover and evaluate ROI of proactive retention programs under financial constraints |

---

## 1. Business Problem

XY was losing **~4,250 employees per year**, straining recruitment and productivity. Replacing each leaver cost **well over a year's salary** when accounting for:

- Hiring costs
- Onboarding expenses  
- Lost productivity

HR wanted to pilot targeted interventions (e.g., manager coaching, workload adjustments, retention bonuses), but Finance insisted on a **rigorous, data-backed business case** before releasing budget.

---

## 2. Approach Using the Predictor App

We used the **Attrition Savings Predictor** to:

1. **Estimate risk** of departure for every employee (25k rows scored)
2. **Compare intervention strategies**: fixed risk thresholds vs Top-K targeting
3. **Apply horizon scaling** (quarterly vs annual) so CFOs saw costs and savings on a realistic budget lens
4. **Run what-if scenarios** (reducing workload, improving manager quality) to simulate program levers
5. **Locate the "economically optimal threshold"** — where expected savings outweigh costs the most

---

## 3. Assumptions & Parameters

### Financial Parameters
- **Replacement cost**: 1.2× annual salary (scaled to horizon)
- **Average salary**: $82,000
- **Effective replacement cost (annual view)**: ~$98,400
- **Intervention cost**: $750 per participant (lump sum)
- **Effectiveness**: 30–35% (probability that a likely leaver stays if treated)
- **Planning horizon**: tested annual vs quarterly

### Model Performance
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **AUC (ROC)** | ~0.68 | Good ranking ability |
| **Precision @ Top 10%** | ~0.41 | 4 out of 10 flagged will actually leave |
| **Lift (Top 10% vs baseline)** | ~2.4× | Top 10% risk group 2.4× more likely to leave |
| **Calibration** | ±3 points | Predicted risk matches observed outcomes |

---

## 4. Strategy Experiments

### A) Threshold-Based Plan

- **Threshold**: ~0.28
- **Employees flagged**: ~6,000
- **Expected true leavers**: ~1,400

#### Savings (Annual View)
| Component | Calculation | Amount |
|-----------|-------------|---------|
| **Gross prevented attrition** | 1,400 × 0.35 × $98,400 | **$48M** |
| **Intervention cost** | 6,000 × $750 | **$4.5M** |
| **Net savings** | Gross - Cost | **$43.5M** |

### B) Top-K Strategy (Top 15%, Coverage 60%)

- **Cohort**: 25,000 × 15% × 60% ≈ 2,250 treated employees
- **Characteristics**: Riskier group than threshold baseline; higher precision
- **Net savings (annual view)**: ~**$21M**
- **Quarterly horizon**: drops to ~**$5.2M**, but still positive ROI

### C) What-If Leverage

Applied improvements to treated cohort:
- **Workload reduction**: -1.0
- **Manager quality improvement**: +0.5

#### Results
- **Aggregate risk delta**: ~24 points reduction
- **Δ Net savings**: +$6M annually vs baseline plan
- **Key insight**: Gain achieved **without extra per-head program cost** — from shifting predictors the model recognized as causal

---

## 5. Outcome & Decision

### Executive Decision
- **Chose**: Top-K pilot (15% × 60% coverage) in two divisions
- **Horizon**: 3-month scaling for budget-relevant costs and benefits
- **Finance approval**: Based on app-demonstrated ROI

### Financial Justification
| Scenario | Quarterly Savings |
|----------|-------------------|
| **Baseline ROI** | ~$5.2M |
| **With What-If Applied** | ~$7M |

### Implementation
- HR built an **A/B test** into the pilot to validate the assumed 30–35% effectiveness

---

## 6. Key Learnings

### Scale Impact
**Scale is everything.** At 25k headcount, even modest effect sizes create **multi-million dollar swings**.

### Transparency Value
**Transparency matters.** CFOs trusted the **threshold breakeven formula** and auditors valued the calibrated logistic regression.

### Financial Alignment
**Horizon scaling bridged HR and Finance.** Annual savings looked huge, but quarterly scaling made it **realistic for cashflow planning**.

### Targeting Efficiency
**Targeted interventions pay.** Top-K + Coverage gave tighter ROI than a blunt threshold.  