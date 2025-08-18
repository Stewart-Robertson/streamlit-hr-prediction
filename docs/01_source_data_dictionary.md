# Employee Attrition Dataset - Data Dictionary

A comprehensive data dictionary covering all columns in the employee attrition prediction dataset, including derived fields and embedded signals for modeling.

---

## 📋 Identifiers & Dates

| Column | Type | Values / Range | Description | Embedded Signal / Notes |
|--------|------|----------------|-------------|-------------------------|
| `employee_id` | Integer | Unique | Employee identifier | Do not use in modelling (identifier only). |
| `snapshot_date` | Date (YYYY-MM-DD) | 2020-01-01 → 2024-12-31 | Observation date for features | Use for time-aware splits (e.g., train ≤2023, test = 2024). |
| `hire_date` | Date (YYYY-MM-DD) | Earlier than snapshot_date | Original hire date | Used to derive tenure. |
| `tenure_years` | Float (2 d.p.) | ≥ 0 | Years at company at snapshot | U-shape risk: higher in first year, dip in years 1–3, slight rise after ~7y. |
| `months_since_hire` | Float (1 d.p.) | ≥ 0 | Months since hire at snapshot | Same signal as tenure; convenience for models. |

---

## 🏢 Organization & Role

| Column | Type | Values / Range | Description | Embedded Signal / Notes |
|--------|------|----------------|-------------|-------------------------|
| `region` | Categorical | AMER, EMEA, APAC | Work region | Mild mix effects only (no strong bias). |
| `department` | Categorical | Sales, Engineering, Customer Success, Marketing, Finance, HR, Operations, Product | Functional area | Small base-rate shifts by department (e.g., Sales slightly higher). |
| `role` | Categorical | Dept-specific titles (e.g., SWE II, PM, AE, CSM, etc.) | Job title | Captures labour-market pull and career path nuance. |
| `level` | Categorical | IC1, IC2, IC3, IC4, Manager, Director, VP | Seniority level | Drives market pay reference; leadership relates to lower attrition. |
| `is_manager` | Binary (0/1) | 0 or 1 | Manager indicator | Managers tend to attrit less, all else equal. |
| `team_id` | Integer | 1–400 | Team cluster ID | Adds random effects → correlated outcomes within teams. |

---

## 💰 Compensation & Progression

| Column | Type | Values / Range | Description | Embedded Signal / Notes |
|--------|------|----------------|-------------|-------------------------|
| `base_salary` | Float (0 d.p.) | ≈ 45k–∞ (currency-agnostic) | Annual base pay | Relative to market reference matters more than absolute. |
| `compa_ratio` | Float (2 d.p.) | ≈ 0.6–1.5 (typ.) | base_salary / market_ref(dept×level) | Below 1.0 increases risk; above 1.0 reduces risk. |
| `avg_raise_3y` | Float (3 d.p.) | ≥ 0 (typ. ~0–0.08) | Mean merit raise % over last 3 cycles | Higher raises → lower risk. |
| `time_since_last_promo_yrs` | Float (2 d.p.) | ≥ 0 | Years since last promotion | Risk rises after ~3 years without promotion. |
| `internal_moves_last_2y` | Integer | 0+ | Internal transfers in last 2 years | More internal mobility → lower risk. |
| `stock_grants` | Float (0 d.p.) | 0 or >0 for senior/IC4+ | Equity value (approx.) | Higher equity slightly reduces risk. |
| `salary_band` | Categorical | Q1…Q5 | Pay quantile across the snapshot | Convenience feature; derived from base_salary. |

---

## 📈 Performance, Engagement & Development

| Column | Type | Values / Range | Description | Embedded Signal / Notes |
|--------|------|----------------|-------------|-------------------------|
| `performance_rating` | Integer | 1–5 | Most recent performance score | Higher rating → lower risk (also interacts with progression). |
| `engagement_score` | Float (2 d.p.) | 1–10 or NaN | Survey engagement | Lower → higher risk. MNAR missingness (disengaged more likely missing). |
| `manager_quality` | Float (2 d.p.) | 1–10 or NaN | Perceived manager quality | Lower → higher risk. MNAR missingness (worse mgr → more missing). |
| `workload_score` | Float (2 d.p.) | 1–10 | Self-reported workload | Higher workload increases risk, especially if manager quality is low (interaction). |
| `learning_hours_last_yr` | Float (1 d.p.) | ≥ 0 | Formal learning hours last year | Slight retention effect (more learning → slightly lower risk). |
| `benefit_score` | Float (2 d.p.) | 1–10 | Perceived benefits quality | Higher benefits → lower risk. |

---

## 🏠 Work Pattern, Schedule & Commute

| Column | Type | Values / Range | Description | Embedded Signal / Notes |
|--------|------|----------------|-------------|-------------------------|
| `remote_status` | Categorical | Remote, Hybrid, Onsite | Work arrangement | Onsite/Hybrid carry small added risk vs Remote. |
| `commute_km` | Float (1 d.p.) | 0–high | One-way commute distance | Longer commute → higher risk, strongest for Onsite (interaction). |
| `overtime_hours_month` | Float (1 d.p.) | ≥ 0 | Avg monthly overtime | Heavier OT correlates with risk via workload sentiment. |
| `night_shift` | Binary (0/1) | 0 or 1 | Night-shift indicator | Increases risk. |
| `schedule_flex` | Binary (0/1) | 0 or 1 | Flexible schedule option | Reduces risk. |

---

## 🏖️ Time Off & Leaves

| Column | Type | Values / Range | Description | Embedded Signal / Notes |
|--------|------|----------------|-------------|-------------------------|
| `sick_days` | Float (1 d.p.) | ≥ 0 | Sick days taken in last year | Small non-linear signal (both unusually high/low carry info). |
| `pto_days_taken` | Float (1 d.p.) | 0–40 (typ.) | Paid time off taken in last year | Contextual; moderate PTO often neutral/healthy. |
| `leave_last_yr` | Binary (0/1) | 0 or 1 | Any extended leave last year | Slight positive risk bump. |

---

## 🎯 Target & Known Leakage Columns

| Column | Type | Values / Range | Description | Embedded Signal / Notes |
|--------|------|----------------|-------------|-------------------------|
| `attrited` | Binary (0/1) | 0 or 1 | **Target**: left in the next period | Drawn from structured probability combining all signals (incl. team & macro drift). |
| `exit_interview_scheduled` | Binary (0/1) | 0 or 1 | Process flag | ⚠️ **LEAKAGE**: strongly tied to outcome; exclude at train time. |
| `offboarding_ticket_created` | Binary (0/1) | 0 or 1 | IT/HR offboarding flag | ⚠️ **LEAKAGE**: strongly tied to outcome; exclude at train time. |

