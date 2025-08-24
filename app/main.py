import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, confusion_matrix
)

# Configure Streamlit page
st.set_page_config(page_title="Employee Turnover & Savings Estimator", layout="wide")

# -----------------------
# Documentation
# ----------------------- 
DOCS_URL = "https://stewart-robertson.github.io/streamlit-hr-prediction/"  
LINKEDIN_URL = "https://www.linkedin.com/in/stewart-robertson-data/"  

st.title("Employee Turnover & Savings Estimator")
st.caption("Predict attrition risk • Test what-ifs • Quantify savings")
col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1, 1, 1, 1, 1, 1, 1, 1])
with col1:
    st.link_button("How to use", DOCS_URL, help="Open the user guide", type="primary")
with col2:
    st.link_button("LinkedIn", LINKEDIN_URL, help="Open my LinkedIn profile")

# Resolve repo root regardless of where Streamlit is launched
REPO_ROOT = Path(__file__).resolve().parents[1]

# -----------------------
# Paths (anchored to repo root)
# -----------------------
DATA_PATH = REPO_ROOT / "data/processed/hr_attrition_clean.csv"
MODEL_PATH = REPO_ROOT / "models/attrition_lr_calibrated_train_to_2022_skl171.pkl"
METRICS_PATH = REPO_ROOT / "models/attrition_lr_calibrated_metrics_train_to_2022_skl171.json"

# -----------------------
# Helpers (formatting, policy levers)
# -----------------------
def fmt_money(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)

def apply_policy_levers(df: pd.DataFrame, wl_delta: float, mq_delta: float, coverage_pct: float, topk_mask_series: pd.Series | None) -> tuple[pd.DataFrame, pd.Series]:
    """Return a copy of df with workload/manager adjustments applied to a covered cohort (all or Top‑K). Returns (modified_df, covered_mask)."""
    df2 = df.copy()
    # Determine who is covered by the intervention
    if topk_mask_series is not None:
        covered = topk_mask_series.copy()
        if 0 < coverage_pct < 100:
            idx = covered[covered].index
            k = max(1, int(len(idx) * (coverage_pct / 100.0)))
            keep = df2.loc[idx].sort_values("attrition_risk", ascending=False).head(k).index
            covered.loc[:] = False
            covered.loc[keep] = True
    else:
        covered = pd.Series(True, index=df2.index)

    # Apply deltas
    if "workload_score" in df2.columns and wl_delta:
        df2.loc[covered, "workload_score"] = np.clip(df2.loc[covered, "workload_score"] - wl_delta, 0, None)
    if "manager_quality" in df2.columns and mq_delta:
        df2.loc[covered, "manager_quality"] = np.clip(df2.loc[covered, "manager_quality"].fillna(1) + mq_delta, 1, 10)
    return df2, covered

# --- Helpers: engineered features and probability-delta savings ---
def recompute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute engineered interaction features so scenarios affect the model as intended."""
    df2 = df.copy()
    if "manager_quality" in df2.columns and "workload_score" in df2.columns:
        mq = df2["manager_quality"].fillna(1)
        wl = df2["workload_score"]
        # Binary flag (for insights only)
        df2["mgmt_workload_risk"] = ((df2["manager_quality"].isna()) | ((mq < 5) & (wl > 7))).astype(int)
        # Continuous interaction used by the model
        df2["mgmt_workload_score"] = (10 - mq) * wl
    return df2

def prob_delta_savings(base_probs: pd.Series,
                       scen_probs: pd.Series,
                       covered_mask: pd.Series,
                       replacement_cost: float,
                       intervention_cost: float,
                       effectiveness: float) -> float:
    """
    Savings based on reduction in predicted attrition probability for the covered cohort.
    Prevented cost ≈ effectiveness × sum(max(base_p - scen_p, 0)) × replacement_cost
    Spend = intervention_cost × (# covered)
    """
    if covered_mask is None:
        covered_mask = pd.Series(True, index=base_probs.index)
    delta = (base_probs - scen_probs).clip(lower=0.0)
    prevented = effectiveness * delta[covered_mask].sum() * replacement_cost
    spend = intervention_cost * covered_mask.sum()
    return prevented - spend

# Columns the model was trained to expect (drop list handled inside the pipeline)
TARGET_COL = "attrited"
YEAR_COL = "snapshot_year"     # created by training script
TEAM_COL = "team_id"           # optional, for grouping
DEPT_COL = "department"        # optional, for grouping

# -----------------------
# Utilities
# -----------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Data file not found at: **{DATA_PATH}**. "
                 "Make sure you've run the cleaning/training step or the path is correct.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: **{MODEL_PATH}**. "
                 "Run the trainer (`python -m src.train_calibrated_lr`) to create it.")
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner=False)
def score_df(df: pd.DataFrame) -> pd.DataFrame:
    pipe = load_model()
    df_in = recompute_engineered_features(df)
    drop_cols = [c for c in [TARGET_COL, "attrition_risk"] if c in df_in.columns]
    X = df_in.drop(columns=drop_cols, errors="ignore")
    probs = pipe.predict_proba(X)[:, 1]
    out = df_in.copy()
    out["attrition_risk"] = probs
    return out

def topk_mask(scores: pd.Series, k_pct: float) -> pd.Series:
    k = max(1, int(len(scores) * k_pct / 100.0))
    cutoff = np.sort(scores.values)[-k]
    return scores >= cutoff

def metrics_at_threshold(y_true: pd.Series, y_prob: pd.Series, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    fpr  = fp / (fp + tn) if (fp + tn) else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=prec, recall=rec, fpr=fpr)


def econ_optimal_threshold(y_true: pd.Series,
                           y_prob: pd.Series,
                           avg_replacement_cost: float,
                           intervention_cost: float,
                           effectiveness: float,
                           grid: np.ndarray | None = None):
    """
    Find the threshold that maximizes expected net savings given costs/effectiveness.
    Returns (best_thr, best_ev).
    """
    if grid is None:
        # Dense grid to avoid overfitting to specific probability values
        grid = np.linspace(0.0, 1.0, 501)
    best_thr, best_ev = 0.5, -np.inf
    # Precompute sorted indices for speed (optional)
    for thr in grid:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        ev = expected_value(tp, fp, avg_replacement_cost, intervention_cost, effectiveness)
        if ev > best_ev:
            best_ev = ev
            best_thr = float(thr)
    return best_thr, best_ev

def expected_value(tp, fp, avg_replacement_cost, intervention_cost, effectiveness):
    # effectiveness = probability the intervention prevents a true leaver
    saved = tp * (effectiveness * avg_replacement_cost)
    spent = (tp + fp) * intervention_cost
    return saved - spent

def lift_table(y_true: pd.Series, y_prob: pd.Series, bins=10) -> pd.DataFrame:
    """
    Build a decile (or bins) table even when predicted probabilities have many ties.
    We use a deterministic rank to break ties, then qcut on the rank so we always
    get the requested number of bins.
    """
    df = pd.DataFrame({"y": y_true.values, "p": y_prob.values})
    # Deterministic tie-break so we don't lose bins due to duplicate edges
    df["rank"] = df["p"].rank(method="first", ascending=True)
    df["decile"] = pd.qcut(df["rank"], q=bins, labels=False)

    agg = df.groupby("decile").agg(
        n=("y", "size"),
        positives=("y", "sum"),
        avg_p=("p", "mean")
    ).reset_index()
    agg["rate"] = agg["positives"] / agg["n"]
    # Show highest-risk decile first
    agg = agg.sort_values("decile", ascending=False).reset_index(drop=True)
    agg["cum_positives"] = agg["positives"].cumsum()
    agg["cum_n"] = agg["n"].cumsum()
    overall_rate = df["y"].mean()
    agg["lift"] = agg["rate"] / overall_rate if overall_rate > 0 else 0.0
    return agg, overall_rate

# -----------------------
# App
# -----------------------

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    df_raw = load_data()
    scored = score_df(df_raw)

    # Precompute risk scores for cohort logic
    y_prob_sidebar = scored["attrition_risk"]
    flagged_mask_sidebar = y_prob_sidebar >= 0  # placeholder, real threshold applied below

    # Threshold & Top-K
    default_thr = float(np.quantile(scored["attrition_risk"], 0.80))
    thr = st.slider("Risk threshold (flag employees above this risk)", 0.0, 1.0, value=round(default_thr, 2), step=0.01, key="thr", help="Anyone with predicted risk ≥ threshold is flagged for intervention.")
    topk = st.slider("Focus Top‑K (%) for cohort-based actions", 1, 50, value=15, step=1, help="Define the Top‑K cohort for targeted interventions and scenario coverage.")

    # Build masks for cohort selection
    flagged_mask_sidebar = y_prob_sidebar >= thr
    topk_mask_sidebar = topk_mask(y_prob_sidebar, topk)

    st.markdown("**Intervention strategy**")
    strategy = st.radio(
        label="Choose who you will intervene with",
        options=["Threshold (flagged)", "Top‑K"],
        index=1,
        help="Define the cohort that receives the intervention. Threshold = anyone above the risk threshold. Top‑K = highest‑risk K% of all employees."
    )

    if strategy == "Threshold (flagged)":
        cohort_mask_selected = flagged_mask_sidebar
    else:  # Top‑K
        cohort_mask_selected = topk_mask_sidebar

    st.markdown("---")
    st.subheader("Savings Sandbox")
    avg_replacement_cost = st.number_input("Replacement cost multiplier (× annual salary)", min_value=0.1, max_value=2.0, value=1.0, step=0.1, help="Typical fully-loaded cost to replace a leaver is ~0.8–1.5× annual salary.")
    intervention_cost = st.number_input("Intervention cost per flagged employee ($)", min_value=0.0, value=800.0, step=50.0, help="e.g., coaching, retention bonus, workload program cost.")
    effectiveness = st.slider("Intervention effectiveness (%)", 0, 100, value=30, step=5, help="Probability that the intervention prevents a true leaver.") / 100.0
    coverage_pct = st.slider("Coverage within chosen cohort (%)", 10, 100, value=100, step=5, help="Apply interventions to this % of the targeted cohort (e.g., Top‑K).")

    st.markdown("**Time horizon & cash flow**")
    horizon_months = st.slider("Planning horizon (months)", 1, 12, value=3, step=1, help="Scale replacement cost and intervention spend to this horizon.")
    cost_is_monthly = st.checkbox("My intervention cost input is per month", value=False, help="If checked, total intervention spend = monthly cost × horizon; otherwise your cost is total program cost per person over the horizon.")

    # Derive replacement & intervention costs on the same horizon basis
    if "base_salary" in scored.columns:
        _avg_salary = scored["base_salary"].mean()
    else:
        _avg_salary = 50000.0
    _repl_cost_eff = _avg_salary * avg_replacement_cost * (horizon_months / 12.0)
    _intervention_cost_eff = intervention_cost * (horizon_months if cost_is_monthly else 1.0)

    # --- Economically optimal threshold (reference) ---
    y_true_sidebar = scored[TARGET_COL] if TARGET_COL in scored.columns else None
    if y_true_sidebar is not None:
        opt_thr, opt_ev = econ_optimal_threshold(
            y_true=y_true_sidebar,
            y_prob=y_prob_sidebar,
            avg_replacement_cost=_repl_cost_eff,
            intervention_cost=_intervention_cost_eff,
            effectiveness=effectiveness,
        )
        st.info(f"**Economically optimal threshold:** {opt_thr:.2f} — maximizes expected net savings at your current assumptions.")
        if st.button("Use optimal threshold", help="Set the slider to the profit‑maximizing cut‑off based on your costs and effectiveness."):
            st.session_state['thr'] = round(opt_thr, 2)

    st.markdown("---")
    st.subheader("What‑if levers")
    wl_delta = st.slider("Reduce workload score by", 0.0, 5.0, value=0.0, step=0.5, help="Simulate lowering workload via hiring/backfill/rebalancing.")
    mq_delta = st.slider("Improve manager quality by", 0.0, 5.0, value=0.0, step=0.5, help="Simulate impact of coaching/training/manager changes.")

# Compute base metrics
y_true = scored[TARGET_COL] if TARGET_COL in scored.columns else None
y_prob = scored["attrition_risk"]

# Executive summary — fixed panel
st.markdown("### Executive summary")
with st.container():
    st.markdown('<div class="exec-summary">', unsafe_allow_html=True)

    # Common quantities
    n_emps = len(scored)
    mean_risk = y_prob.mean()

    # Threshold-based metrics & savings
    if y_true is not None:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = (tp / (tp + fp)) if (tp + fp) else 0.0
        recall    = (tp / (tp + fn)) if (tp + fn) else 0.0
        flagged_total = int((y_pred == 1).sum())

        # Replacement & intervention costs on the selected horizon
        repl_cost = _repl_cost_eff
        ev_now = expected_value(tp, fp, repl_cost, _intervention_cost_eff, effectiveness)

        # Row 1
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1: st.metric("Net savings (@threshold)", fmt_money(ev_now))
        with r1c2: st.metric("Flagged employees", f"{flagged_total:,}")
        with r1c3: st.metric("Precision", f"{precision:.2f}")

        # Row 2
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1: st.metric("Recall", f"{recall:.2f}")
        with r2c2: st.metric("Employees", f"{n_emps:,}")
        with r2c3: st.metric("Mean risk", f"{mean_risk:.2f}")

        # Small performance caption
        auc = roc_auc_score(y_true, y_prob)
        ap  = average_precision_score(y_true, y_prob)
        st.caption(f"Model AUC / PR AUC: **{auc:.2f} / {ap:.2f}**  •  Threshold = {thr:.2f}  •  Net = prevented leaver cost − intervention spend (given your sandbox assumptions).")
    else:
        # When ground-truth not present, show stable org-level stats
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1: st.metric("Employees", f"{n_emps:,}")
        with r1c2: st.metric("Mean risk", f"{mean_risk:.2f}")
        with r1c3: st.metric("Threshold", f"{thr:.2f}")
        st.caption("Provide ground‑truth labels to view precision/recall, AUC and threshold‑based savings.")

    st.markdown("</div>", unsafe_allow_html=True)

# Tabs
tab_overview, tab_individuals, tab_segments, tab_whatif, tab_method = st.tabs(
    ["Overview", "Individuals", "Segments", "What-if", "Methodology"]
)

# ========== Overview ==========
with tab_overview:
    st.subheader("Risk distribution")
    fig_hist = px.histogram(scored, x="attrition_risk", nbins=40, opacity=0.9)
    st.plotly_chart(fig_hist, use_container_width=True)

    if y_true is not None:
        # --- Model performance section heading ---
        st.markdown("### Model performance")
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
        fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig_roc, use_container_width=True)
        st.caption("**ROC curve:** shows how well the model balances catching real leavers vs false positives. A curve closer to the top‑left is better. **AUC** (Area Under the Curve) is the overall score (1.0 is perfect, 0.5 is chance).")

        # PR AUC
        

        # Lift
        lift_df, overall_rate = lift_table(y_true, y_prob, bins=10)
        fig_lift = px.bar(lift_df, x=lift_df.index, y="lift", labels={"x":"Decile (High→Low risk)"})
        fig_lift.update_layout(title=f"Lift by Decile (overall attrition rate = {overall_rate:.2%})")
        st.plotly_chart(fig_lift, use_container_width=True)
        st.caption("**Lift by decile:** ranks employees by predicted risk, splits into ten equally‑sized buckets (deciles), and shows how much higher the observed attrition rate is in each decile compared to the company average. E.g., in the top 10% highest risk employees (first bar), we're 3x more likely to find actual leavers than if we picked randomly.")

    # EV at chosen threshold
    if y_true is not None:
        m = metrics_at_threshold(y_true, y_prob, thr)
        repl_cost = _repl_cost_eff
        ev = expected_value(m["tp"], m["fp"], repl_cost, _intervention_cost_eff, effectiveness)
        st.info(f"**Expected Value at threshold {thr:.2f}: ${ev:,.0f}** "
                f"(TP={m['tp']}, FP={m['fp']}, precision={m['precision']:.2f}, recall={m['recall']:.2f})")
    st.caption("Use the sidebar **Savings Sandbox** to tune assumptions (replacement cost, intervention cost, effectiveness) and the **Risk threshold** to see how savings and precision/recall trade off.")

# ========== Individuals ==========
with tab_individuals:
    st.subheader("Top-risk employees")
    mask_topk = topk_mask(y_prob, topk)
    cols_show = [c for c in ["employee_id", "department", "role", "base_salary", "manager_quality", "workload_score", "mgmt_workload_score"] if c in scored.columns]
    top_tbl = scored.loc[mask_topk, cols_show + ["attrition_risk"]].sort_values("attrition_risk", ascending=False)
    st.dataframe(top_tbl, use_container_width=True)
    csv_top = top_tbl.to_csv(index=False).encode("utf-8")
    st.download_button("Download Top‑risk CSV", data=csv_top, file_name="top_risk_employees.csv", mime="text/csv")

# ========== Segments ==========
with tab_segments:
    st.subheader("Segment view")
    seg_cols = []
    if DEPT_COL in scored.columns: seg_cols.append(DEPT_COL)
    if TEAM_COL in scored.columns: seg_cols.append(TEAM_COL)
    if not seg_cols:
        st.write("No segment columns found (e.g., department/team).")
    else:
        seg_choice = st.selectbox("Group by", seg_cols)
        seg = scored.groupby(seg_choice).agg(
            employees=("attrited","size") if TARGET_COL in scored.columns else ("attrition_risk","size"),
            mean_risk=("attrition_risk","mean"),
            positives=("attrited","sum") if TARGET_COL in scored.columns else ("attrition_risk","size")
        ).reset_index()
        fig_seg = px.bar(seg.sort_values("mean_risk", ascending=False).head(20),
                         x=seg_choice, y="mean_risk", title="Average predicted risk by segment")
        st.plotly_chart(fig_seg, use_container_width=True)
        st.dataframe(seg.sort_values("mean_risk", ascending=False), use_container_width=True)

# ========== What-if ==========
with tab_whatif:
    st.subheader("Scenario builder — simulate policy levers on risk and savings")
    # Determine cohort for application (selected strategy)
    cohort_mask = cohort_mask_selected  # determined by chosen intervention strategy
    df_scn, covered_mask = apply_policy_levers(scored, wl_delta=wl_delta, mq_delta=mq_delta, coverage_pct=coverage_pct, topk_mask_series=cohort_mask)
    # Recompute engineered features so changes to workload/manager feed the model interaction
    df_scn = recompute_engineered_features(df_scn)

    # Re-score scenario
    pipe = load_model()
    y_prob_scn = pipe.predict_proba(df_scn.drop(columns=[c for c in [TARGET_COL, "attrition_risk"] if c in df_scn.columns], errors="ignore"))[:, 1]

    # Metrics and savings deltas at the same threshold
    if y_true is not None:
        y_pred_base = (y_prob >= thr).astype(int)
        y_pred_scn  = (y_prob_scn >= thr).astype(int)
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true, y_pred_base).ravel()
        tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_true, y_pred_scn).ravel()

        # Replacement & intervention costs on the selected horizon
        repl_cost = _repl_cost_eff

        ev_base = expected_value(tp_b, fp_b, repl_cost, _intervention_cost_eff, effectiveness)
        ev_scn  = expected_value(tp_s, fp_s, repl_cost, _intervention_cost_eff, effectiveness)
        delta_ev = ev_scn - ev_base

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Net savings (base)", fmt_money(ev_base))
        with c2: st.metric("Net savings (scenario)  [what is this?](%s/savings/#1-threshold-based-net-savings)" % DOCS_URL, fmt_money(ev_scn))
        with c3: st.metric("Δ Net savings", fmt_money(delta_ev))

        # Probability-delta based savings (threshold-free), using covered cohort only
        ev_prob = prob_delta_savings(
            base_probs=y_prob,
            scen_probs=y_prob_scn,
            covered_mask=covered_mask,
            replacement_cost=repl_cost,
            intervention_cost=_intervention_cost_eff,
            effectiveness=effectiveness,
        )
        st.success(
            f"Threshold‑free savings (for {horizon_months}‑mo horizon, coverage‑adjusted): "
            f"**{fmt_money(ev_prob)}** "
            f"[(what is this?)]({DOCS_URL}/savings/#2-threshold-free-savings-expected-value)"
        )
        st.caption("Changes with horizon only when what‑ifs reduce risk and/or when intervention cost is monthly.")

    # Show distribution shift
    df_compare = pd.DataFrame({"base": y_prob, "scenario": y_prob_scn})
    df_compare["delta"] = df_compare["scenario"] - df_compare["base"]
    fig_delta = px.histogram(df_compare, x="delta", nbins=40, title="Risk change (scenario vs base): negative = improved")
    st.plotly_chart(fig_delta, use_container_width=True)

# ========== Methodology ==========
with tab_method:
    st.subheader("Methodology")
    st.markdown("""
    - Model: Logistic Regression with one-hot & scaling, **calibrated** (sigmoid or isotonic) on the next year.
    - Validation: Train ≤ **2022**, evaluate on **2023** (time-aware split).
    - Features: Removed leakage (offboarding/exit interview), deduped collinear pay metrics, added an interaction
      **(worse manager × higher workload)**.
    - This app is **decision support**: it prioritises who to check-in with; it doesn't make decisions by itself.
    """)

    # Show training calibration metrics if present
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        st.json(metrics)
    else:
        st.caption("Calibration metrics file not found.")

st.caption(
    "© Stewart Robertson — Demo portfolio app. "
    "•  📘 [How to use](%s)  •  🔗 [LinkedIn](%s)"
    % (DOCS_URL, LINKEDIN_URL)
)

st.caption("Disclaimer: This app is a proof of concept and should not be used for decision-making at your organisation.")
