# HR Attrition Dataset — Data Dictionary & Embedded Signal

This comprehensive data dictionary provides detailed information about each variable in the HR attrition dataset, including embedded signals that may indicate likelihood of employee turnover.

---

## 1. Employee Demographics

| Column | Type | Description | Example | Embedded Signal for Attrition |
|--------|------|-------------|---------|------------------------------|
| `employee_id` | Integer | Unique identifier for each employee | 10234 | None |
| `age` | Integer | Employee's age in years | 35 | Employees under 25 or over 55 have slightly higher attrition risk |
| `gender` | Categorical<br>(Male, Female, Other) | Gender of employee | Male | No direct signal; may interact with other variables |
| `education_level` | Categorical<br>(High School, Bachelors, Masters, PhD) | Highest education attained | Bachelors | Certain roles may have different turnover rates by education |

---

## 2. Job & Employment Details

| Column | Type | Description | Example | Embedded Signal for Attrition |
|--------|------|-------------|---------|------------------------------|
| `department` | Categorical<br>(Sales, Engineering, HR, Finance, Operations) | Department of employment | Engineering | Sales tends to have higher turnover |
| `job_role` | Categorical | Specific role within department | Software Engineer | Roles with high external demand have higher attrition |
| `job_level` | Integer (1–5) | Seniority level | 3 | Lower levels have higher attrition risk |
| `years_at_company` | Float | Years since joining | 2.5 | Risk spikes around 1–2 years and after 8+ years |
| `years_in_current_role` | Float | Years in current role | 1.0 | Short tenure in role with low satisfaction increases risk |
| `manager_id` | Integer | ID of current manager | 205 | Certain managers may have higher attrition in their teams |

---

## 3. Compensation & Benefits

| Column | Type | Description | Example | Embedded Signal for Attrition |
|--------|------|-------------|---------|------------------------------|
| `monthly_income` | Float | Monthly base pay | 5,200.00 | Below-median pay increases attrition likelihood |
| `bonus_percentage` | Float | Annual bonus as % of salary | 8.0 | Lower bonuses correlate with higher attrition |
| `stock_option_level` | Integer (0–3) | Level of stock options granted | 1 | Level 0 → higher attrition; Level 3 → lower attrition |
| `benefits_score` | Float (0–1) | Perceived benefits satisfaction | 0.65 | Lower score → higher attrition |

---

## 4. Performance & Engagement

| Column | Type | Description | Example | Embedded Signal for Attrition |
|--------|------|-------------|---------|------------------------------|
| `performance_rating` | Integer (1–5) | Annual performance score | 4 | Low performers (1–2) and high performers (5) more likely to leave (low for firing risk, high for external offers) |
| `job_satisfaction` | Integer (1–4) | Self-reported satisfaction | 2 | Low satisfaction strongly predicts attrition |
| `work_life_balance` | Integer (1–4) | Work-life balance rating | 3 | Poor balance (1–2) increases risk |
| `training_hours_last_year` | Integer | Hours of training received | 25 | Extremely low training may signal disengagement |
| `promotions_last_5_years` | Integer | Promotions received in last 5 years | 1 | No promotions in 5+ years → higher attrition |

---

## 5. Workplace Relationships

| Column | Type | Description | Example | Embedded Signal for Attrition |
|--------|------|-------------|---------|------------------------------|
| `distance_from_home` | Float | Distance to workplace (km) | 18.5 | >30 km has higher attrition risk |
| `num_projects` | Integer | Number of projects currently assigned | 4 | Very high workload can raise attrition risk |
| `team_size` | Integer | Team headcount | 7 | Extremely small or large teams may influence risk |
| `manager_relationship_score` | Float (0–1) | Perceived relationship quality with manager | 0.75 | Low score strongly predicts attrition |

---

## 6. Exit Label

| Column | Type | Description | Example | Embedded Signal for Attrition |
|--------|------|-------------|---------|------------------------------|
| `attrition` | Binary (Yes/No) | Whether employee left in the last year | Yes | **Target variable** |

---

### Key Insights

- **High-risk indicators**: Low job satisfaction, poor work-life balance, below-median compensation, and weak manager relationships are strong predictors of attrition
- **Counterintuitive patterns**: Both low and high performers may be at risk for different reasons (performance issues vs. external opportunities)
- **Tenure effects**: Critical periods around 1–2 years and 8+ years show elevated attrition risk
- **Role-specific factors**: Sales roles and positions with high external market demand typically show higher turnover rates