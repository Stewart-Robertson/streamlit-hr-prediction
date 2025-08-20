# Savings explained

We show savings in two complementary ways.

## 1) Threshold-based net savings
**What it is:** We pick a risk **threshold** and treat everyone above it.

**Formula (conceptual):**  

Net = (True Positives × effectiveness × replacement_cost) - (Treatments × intervention_cost)

- **True Positives**: flagged people who would have left
- **Treatments**: everyone flagged (TP + FP)

**Use it for:** operational reporting (how many flagged, precision/recall).

---

## 2) Threshold-free savings (expected value)
**What it is:** We sum expected value per person over the **treated cohort** (Top-K or Threshold, with Coverage), without relying on a single cut-off.

**Per person Expected Value:**  
effectiveness × replacement_cost × predicted_risk − intervention_cost

**Use it for:** comparing scenarios and strategy settings; it reflects how risk actually shifts.

---

## Reading the numbers (example)

| Metric | Value | 
|--------|------|
| Net savings (base) | $60,217,234 |
| Net savings (scenario) | $60,217,234 | 
| Δ Net savings | $0 | 
| Threshold-free savings (coverage-adjusted) | −$3,421,600 |

**What this means:**

_As work conditions improve, people's attrition risk decreases, so at some mixture of settings these people will cross the threshold and impact the net savings._

- The **threshold-based** result: Net savings (scenario) vs Net Savings (base) didn’t change because your what-if didn’t move enough people across the chosen threshold (so flagged counts and TP/FP stayed the same). 
- The **threshold-free** result is **negative**, meaning the scenario made the treated cohort *less* profitable in expectation (e.g., you spent money treating lower-risk people or your effectiveness/cost assumptions make the lever uneconomical).

**What to try next:**
- Increase the lever strength (e.g., larger workload reduction)
- Target smarter (**Top-K** with moderate **Coverage**)
- Recheck assumptions: effectiveness ↑ or cost ↓ often flips EV positive