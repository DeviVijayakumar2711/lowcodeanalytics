import streamlit as st
import pandas as pd
import numpy as np
import requests
import smtplib
from email.mime.text import MIMEText

# Time series
from prophet import Prophet
from sklearn.linear_model import LinearRegression

# Regression & classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

# Clustering / PCA / neighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Anomaly detection
from pyod.models.iforest import IForest
from sklearn.preprocessing import StandardScaler

# Advanced anomaly pipeline
import ruptures as rpt
import seaborn as sns
import matplotlib.pyplot as plt

# Optional causal inference
try:
    from dowhy import CausalModel
    DOWHY = True
except ImportError:
    DOWHY = False

st.set_page_config(page_title="Low-Code Analytics Platform", layout="wide")


# ─── 1) Landing / Hero Section ────────────────────────────────────────────────

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"/>

<style>
  .hero { background:#fff; padding:2.5rem 2rem; border:1px solid #e5e7eb;
           border-radius:1rem; text-align:center; box-shadow:0 4px 12px rgba(0,0,0,0.05);
           margin-bottom:2rem; }
  .hero h1 { font-size:2.25rem; margin-bottom:0.5rem; color:#111827; }
  .hero p  { font-size:1rem; color:#4B5563; margin-bottom:1rem; }
  .hero .btn { padding:0.75rem 1.5rem; background:#3b82f6; color:#fff;
               font-weight:600; border-radius:0.5rem; text-decoration:none; }
  .features { display:flex; justify-content:space-around; flex-wrap:wrap;
               gap:1rem; margin-bottom:2rem; }
  .card { flex:1 1 22%; background:#fff; padding:1.5rem; border:1px solid #e5e7eb;
           border-radius:0.75rem; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.03);}
  .card i { font-size:2.25rem; margin-bottom:0.75rem; color:#3b82f6; }
  .card h3 { margin:0.5rem 0; font-size:1.125rem; color:#111827; }
  .card p { font-size:0.95rem; color:#6B7280; line-height:1.4; }
  @media (max-width:800px) { .features{flex-direction:column;} .card{flex:auto;} }
</style>

<div class="hero">
  <h1><i class="fas fa-brain"></i> Low-Code Analytics Platform</h1>
  <p>
    Upload your data, choose forecasting, anomaly-detection, regression,
    clustering, causal & temporal insights—and get interactive charts,
    tables, alerts & downloads without writing a single line of code.
  </p>
  <a href="#upload" class="btn"><i class="fas fa-rocket"></i> Get Started</a>
</div>

<div class="features">
  <div class="card">
    <i class="fas fa-chart-line"></i><h3>Forecasting</h3>
    <p>Prophet + linear fallback; trend & seasonality charts.</p>
  </div>
  <div class="card">
    <i class="fas fa-exclamation-triangle"></i><h3>Anomaly Detection</h3>
    <p>IQR, dynamic bounds, IForest, drift & multivariate outliers.</p>
  </div>
  <div class="card">
    <i class="fas fa-sliders-h"></i><h3>Regression</h3>
    <p>Linear & logistic models with MSE, accuracy & confusion matrices.</p>
  </div>
  <div class="card">
    <i class="fas fa-project-diagram"></i><h3>Clustering & PCA</h3>
    <p>KMeans clusters, PCA reduction & nearest-neighbor recommendations.</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── 2) Helpers ────────────────────────────────────────────────────────────────

def download_df(df, filename):
    csv = df.to_csv(index=False)
    st.download_button("📥 Download Results", csv, file_name=filename, mime="text/csv")

def notify_email(subject, message, to):
    host, port = "smtp.office365.com", 587
    frm = st.secrets.get("EMAIL","")
    pwd = st.secrets.get("EMAIL_PASSWORD","")
    msg = MIMEText(message)
    msg["Subject"], msg["From"], msg["To"] = subject, frm, to
    try:
        s = smtplib.SMTP(host, port)
        s.starttls()
        s.login(frm, pwd)
        s.sendmail(frm, [to], msg.as_string())
        s.quit()
        st.success("✅ Email sent")
    except Exception as e:
        st.error(f"Email failed: {e}")

def notify_teams(url, text):
    try:
        r = requests.post(url, json={"text": text})
        if r.ok:    st.success("✅ Teams sent")
        else:       st.error("Teams alert failed")
    except Exception as e:
        st.error(f"Teams error: {e}")


# ─── 3) Upload & Task Picker ───────────────────────────────────────────────────

st.write("## Step 1: Upload & pick your analysis")
uploaded = st.file_uploader("📂 Upload CSV", type="csv", key="upload")
if not uploaded:
    st.stop()
df = pd.read_csv(uploaded)
df.columns = df.columns.str.strip()
st.write("### Data Preview", df.head())

task = st.selectbox("🛠️ Analysis Task", [
    "Time Series Forecasting",
    "Anomaly Detection (Advanced)",
    "Linear Regression",
    "Logistic Regression",
    "Correlation Analysis",
    "Clustering (KMeans)",
    "PCA (Dimensionality Reduction)",
    "Recommendation Engine"
], key="task")


# ─── 4) Sidebar Guide ─────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📘 How to use")
    if task=="Time Series Forecasting":
        st.markdown("""
        1. Upload CSV with one **date** column + one **numeric** column  
        2. Select **Date** & **Value** columns  
        3. Prophet may occasionally fail; falls back to linear trend  
        4. View chart, table & download forecast  
        """)
    elif task=="Anomaly Detection (Advanced)":
        st.markdown("""
        1. Pick **time** & ≥1 **numeric** metric  
        2. Adjust rolling window (for dynamic & drift)  
        3. Inspect per-metric IQR, dynamic, IForest & drift flags  
        4. Combined overlay & **multivariate** outliers  
        5. **Causal**, **Temporal clustering**, **Correlation**  
        6. Download & alert via Email/Teams  
        """)
    elif task=="Linear Regression":
        st.markdown("""
        1. Choose one **numeric** target & features  
        2. View **MSE** & prediction table  
        3. Download results  
        """)
    elif task=="Logistic Regression":
        st.markdown("""
        1. Choose one **binary** target & numeric features  
        2. View **accuracy**, **confusion matrix**  
        3. Download results  
        """)
    elif task=="Correlation Analysis":
        st.markdown("""
        1. Select ≥2 numeric columns  
        2. View & download **correlation matrix**  
        """)
    elif task=="Clustering (KMeans)":
        st.markdown("""
        1. Select numeric features & cluster count  
        2. View assignments & download  
        """)
    elif task=="PCA (Dimensionality Reduction)":
        st.markdown("""
        1. Select ≥2 numeric columns  
        2. Choose # principal components  
        3. View PC scores & download  
        """)
    else:
        st.markdown("""
        1. Upload a user-item matrix CSV  
        2. Select item columns for NearestNeighbors  
        3. View & download recommendations  
        """)


# ─── 5) Flexible Multi-Column Filter ────────────────────────────────────────────

st.markdown("### 🔎 Filters (optional)")
cols_to_filter = st.multiselect("Pick columns to filter", df.columns.tolist(), key="filter_cols")
filtered_df = df.copy()
for col in cols_to_filter:
    vals = st.multiselect(f"Values for '{col}'", df[col].dropna().unique(), key=col)
    if vals:
        filtered_df = filtered_df[filtered_df[col].isin(vals)]
if cols_to_filter:
    st.info(f"Filtered on: { {c: filtered_df[c].unique().tolist() for c in cols_to_filter} }")
    st.write("Filtered data", filtered_df)
df = filtered_df


# ─── 6) Task Branches ───────────────────────────────────────────────────────────

# 1) Time Series Forecasting
if task=="Time Series Forecasting":
    st.info("Forecasting with Prophet (or linear fallback)")
    date_col = st.selectbox("Date column", df.columns, key="ts_date")
    nums = df.select_dtypes(include=np.number).columns.tolist()
    values = [c for c in nums if c!=date_col]
    if not values:
        st.error("No numeric columns apart from date.")
        st.stop()
    value_col = st.selectbox("Value column", values, key="ts_val")

    ts = df[[date_col,value_col]].rename(columns={date_col:"ds",value_col:"y"})
    ts["ds"] = pd.to_datetime(ts["ds"], errors="coerce")
    ts = ts.dropna(subset=["ds","y"]).sort_values("ds")
    if len(ts)<2:
        st.error("Need ≥2 valid rows.")
    else:
        periods = st.slider("Days to forecast",7,90,30,key="fc_periods")
        try:
            m = Prophet().fit(ts)
            future = m.make_future_dataframe(periods=periods)
            fc = m.predict(future)
            st.line_chart(fc.set_index("ds")["yhat"])
            st.write("### Forecast Table", fc[["ds","yhat","yhat_lower","yhat_upper"]].tail(10))
            download_df(fc[["ds","yhat","yhat_lower","yhat_upper"]],"forecast.csv")
        except:
            st.warning("Prophet failed → linear fallback.")
            lr_df = ts.copy()
            lr_df["t"] = (lr_df["ds"]-lr_df["ds"].min()).dt.days
            lr = LinearRegression().fit(lr_df[["t"]], lr_df["y"])
            lr_df["trend"] = lr.predict(lr_df[["t"]])
            fig,ax = plt.subplots(figsize=(6,3))
            ax.plot(lr_df["ds"],lr_df["y"],label="Actual")
            ax.plot(lr_df["ds"],lr_df["trend"],"--",label="Trend")
            ax.legend(); ax.grid(ls="--",alpha=0.3); ax.tick_params(rotation=45)
            st.pyplot(fig)
            st.write("### Trend Table", lr_df[["ds","trend"]].tail(10))
            download_df(lr_df[["ds","trend"]],"trend.csv")


# 2) Anomaly Detection (Advanced)
elif task == "Anomaly Detection (Advanced)":
    import ruptures as rpt
    from ruptures.exceptions import BadSegmentationParameters
    from pyod.models.iforest import IForest
    import matplotlib.pyplot as plt

    st.info("⚠️ Advanced anomaly detection: static IQR, dynamic 5–95%, Isolation Forest & PELT drift")

    # pick up both object & category columns, filter out ultra-high-cardinality
        # ─── 1) Optional group-by ────────────────────────────────────────────────────
    # include any column with ≤50 unique values
    groupable = [c for c in df.columns if df[c].nunique() <= 50]
    group_col = st.selectbox("Group by (optional)", ["None"] + groupable)

    if group_col != "None":
        chosen = st.multiselect(
            f"Select {group_col}(s)",
            df[group_col].dropna().astype(str).unique().tolist()
        )
        if not chosen:
            st.warning("Please pick at least one group or choose 'None'")
            st.stop()

        # **filter** and then **cast that column to string** in the filtered df
        df_ad = df[df[group_col].astype(str).isin(chosen)].copy()
        df_ad[group_col] = df_ad[group_col].astype(str)

        groups = chosen
    else:
        df_ad, group_col, groups = df.copy(), "ALL", ["ALL"]
        df_ad["ALL"] = "ALL"

    # ─── 2) Time & metric selection ───────────────────────────────────────────────
    time_col = st.selectbox("⏰ Time column", df_ad.columns, key="ad_time")
    metrics  = st.multiselect("📊 Numeric metric(s)", 
                              df_ad.select_dtypes(include="number").columns.tolist(),
                              key="ad_metrics")
    if not time_col or not metrics:
        st.warning("Pick a time column and at least one metric")
        st.stop()

    window = st.slider("🔄 Rolling window for dynamic/drift (days)", 10, 180, 60, key="ad_win")

    all_frames = []
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # ─── 3) Per‐metric × Per‐group plotting + narrative ─────────────────────────
    for m in metrics:
        st.markdown(f"### 🔍 Metric: **{m}**")
        fig, ax = plt.subplots(figsize=(6,2.2))
        narr = []

        last_lo = last_hi = last_trend = last_ds = None

        for i, grp in enumerate(groups):
            sub = df_ad if grp == "ALL" else df_ad[df_ad[group_col] == grp]
            ts = (
                sub[[time_col, m]]
                .dropna(subset=[time_col, m])
                .assign(ds=lambda d: pd.to_datetime(d[time_col], errors="coerce"))
                .dropna(subset=["ds"])
                .rename(columns={m: "y"})
                .sort_values("ds").reset_index(drop=True)
            )
            if ts.empty:
                continue

            # static trend & IQR
            ts["trend"] = ts["y"].rolling(7, min_periods=1).mean()
            q1, q3 = ts["y"].quantile([.25, .75])
            lo, hi = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
            ts["iqr_out"] = ts["y"].lt(lo) | ts["y"].gt(hi)

            # dynamic band
            ts["dyn_lo"]  = ts["y"].rolling(window, min_periods=1).quantile(0.05)
            ts["dyn_hi"]  = ts["y"].rolling(window, min_periods=1).quantile(0.95)
            ts["dyn_out"] = ts["y"].lt(ts["dyn_lo"]) | ts["y"].gt(ts["dyn_hi"])

            # Isolation Forest
            iso = IForest(contamination=0.025)
            ts["if_out"] = iso.fit_predict(ts[["y"]]) == 1

            # PELT drift
            try:
                algo = rpt.Pelt(model="rbf").fit(ts["y"].values)
                cps = algo.predict(pen=3)
            except BadSegmentationParameters:
                cps = []
            ts["drift"] = False
            for cp in cps[:-1]:
                if 0 <= cp < len(ts):
                    ts.at[cp, "drift"] = True

            # stash for combined
            ts["metric"], ts["group"] = m, grp
            all_frames.append(ts)

            # plot Actual
            col = color_cycle[i % len(color_cycle)]
            ax.plot(ts["ds"], ts["y"], color=col, lw=1.3,
                    label=f"{grp}", zorder=3)

            # low-opacity flags
            ax.scatter(ts.loc[ts["dyn_out"], "ds"], ts.loc[ts["dyn_out"], "y"],
                       marker="s", color="#8FBC8F", alpha=0.5, s=30, zorder=4)
            ax.scatter(ts.loc[ts["if_out"],  "ds"], ts.loc[ts["if_out"],  "y"],
                       marker="^", color="#F4A460", alpha=0.5, s=30, zorder=4)
            ax.scatter(ts.loc[ts["drift"],   "ds"], ts.loc[ts["drift"],   "y"],
                       marker="x", color="#DDA0DD", alpha=0.5, s=35, zorder=4)

            # bright IQR outliers
            iqr_idx = ts.loc[ts["iqr_out"], ["ds","y"]]
            ax.scatter(iqr_idx["ds"], iqr_idx["y"],
                       marker="o", color="#D62728", s=45,
                       label="IQR outlier", zorder=5)

            # narrative bullets
            if not iqr_idx.empty:
                d0 = iqr_idx["ds"].dt.strftime("%Y-%m-%d").iat[0]
                v0 = iqr_idx["y"].iat[0]
                narr.append(f"• [{grp}] {m}={v0:.1f} on {d0}, beyond IQR.")
            dn = ts.loc[ts["dyn_out"], "ds"]
            if len(dn):
                narr.append(f"• [{grp}] dynamic band breached on {dn.dt.strftime('%Y-%m-%d').iat[0]}.")

            last_lo, last_hi = lo, hi
            last_trend, last_ds = ts["trend"], ts["ds"]

        # static IQR fill + trend
        if last_ds is not None:
            ax.fill_between(last_ds, last_lo, last_hi,
                            color="#E3F2FD", label="Normal IQR range",
                            zorder=0)
            ax.plot(last_ds, last_trend,
                    "--", color="#888888", lw=1, label="Trend", zorder=1)

        ax.set_ylabel(m, fontsize=9)
        ax.grid(ls="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=45, labelsize=8)

        # legend *outside*
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels,
                  bbox_to_anchor=(1.02, 1), loc='upper left',
                  borderaxespad=0, fontsize="x-small", ncol=1)

        plt.tight_layout(pad=2)
        st.pyplot(fig)

        # 3-lines of narrative
        if narr:
            st.markdown("\n".join(narr[:3]))
        else:
            st.markdown("• No anomalies detected for any group.")

    # ─── 4) Combined overlay with dual encoding ─────────────────────────────────
    all_df = pd.concat(all_frames, ignore_index=True)

        # color per metric
        # === 4) Combined Overlay with IQR outlier highlighting & summary ===
    # (Call this after you've built `all_df` = pd.concat(all_frames, ignore_index=True))

    # 1) Map each metric → color
    unique_metrics  = all_df["metric"].unique()
    palette         = plt.get_cmap("tab10")
    metric_colors   = {m: palette(i) for i, m in enumerate(unique_metrics)}

    # 2) Map each group → linestyle
    unique_groups   = all_df["group"].unique()
    line_styles     = ["-", "--", "-.", ":"]
    group_styles    = {g: line_styles[i % len(line_styles)] for i, g in enumerate(unique_groups)}

    # 3) Draw the plot
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for (metric, group), seg in all_df.groupby(["metric", "group"]):
        ax.plot(
            seg["ds"], seg["y"],
            color      = metric_colors[metric],
            linestyle  = group_styles[group],
            linewidth  = 1.5,
            label      = f"{metric} — {group}",
            zorder     = 2
        )

    # 4) Overlay IQR outliers only
    out = all_df[all_df["iqr_out"]]
    ax.scatter(
        out["ds"], out["y"],
        s        = 60,
        color    = "#D62728",
        edgecolor= "white",
        linewidth = 0.8,
        label     = "IQR outlier",
        zorder    = 5
    )

    # 5) Format axes, title, grid
    ax.set_title("Combined Overlay", fontsize=14, pad=10)
    ax.set_ylabel("Value", fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")

    # 6) Tidy up x‐labels & legend
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        loc            = "upper left",
        bbox_to_anchor = (1.02, 1),
        frameon        = False,
        fontsize       = "small"
    )

    plt.tight_layout(pad=2)
    st.pyplot(fig)

    # 7) Three‐line narrative summary
    total_out = len(out)
    if total_out:
        first_pt = out.sort_values("ds").iloc[0]
        last_dt  = out["ds"].max().strftime("%Y-%m-%d")
        st.markdown(
            f"• **{total_out}** IQR outliers detected  \n"
            f"• First outlier on **{first_pt['ds'].strftime('%Y-%m-%d')}** (value={first_pt['y']:.1f})  \n"
            f"• Most recent outlier on **{last_dt}**"
        )
    else:
        st.markdown("• No IQR outliers detected across all metric/groups.")

    
    # ─── Multivariate, Causal, Temporal, Correlation, Alerts ──────────────────

    # 4) Multivariate
    with st.expander("🎯 Multivariate outliers (Mahalanobis)", expanded=False):
        pivot = (
            all_df
              .pivot_table(index="ds", columns="metric", values="y", aggfunc="mean")
              .dropna()
        )
        if pivot.shape[1] > 1:
            mu, cov = pivot.mean().values, np.cov(pivot.values.T)
            inv     = np.linalg.pinv(cov)
            md      = pivot.apply(lambda r: np.sqrt((r-mu)@inv@(r-mu)), axis=1)
            thr     = np.percentile(md, 97.5)
            df_mv   = (
                pd.DataFrame({"ds": md.index, "mahal": md.values})
                  .query("mahal > @thr")
                  .nlargest(5, "mahal")
                  .reset_index(drop=True)
            )
            st.table(df_mv.assign(mahal=lambda d: d.mahal.round(2)))
            st.caption(f"Total joint anomalies: {(md>thr).sum()}")
        else:
            st.info("Need ≥2 metrics for multivariate analysis.")
        st.markdown("Top Mahalanobis distances highlight joint metric anomalies.")

    # 5) Causal
    with st.expander("🧠 Causal Inference", expanded=False):
        if not DOWHY:
            st.info("Install `dowhy` to enable causal analysis.")
        elif len(metrics) < 2:
            st.info("Select at least two metrics.")
        else:
            cause  = st.selectbox("Cause",   metrics, key="ci_c")
            effect = st.selectbox("Effect",  [m for m in metrics if m!=cause], key="ci_e")
            confs  = [m for m in metrics if m not in (cause,effect)]
            df_ci  = pivot[[cause,effect]+confs].dropna().reset_index(drop=True)
            if df_ci.shape[0] < 5:
                st.warning("Need ≥5 rows after dropna.")
            else:
                try:
                    if not confs:
                        lr  = LinearRegression().fit(df_ci[[cause]], df_ci[effect])
                        st.success(f"OLS slope ({cause}→{effect}): {lr.coef_[0]:.3f}")
                    else:
                        med = df_ci[cause].median()
                        df_ci["_t"] = (df_ci[cause]>=med).astype(int)
                        model = CausalModel(
                            data=df_ci, treatment="_t", outcome=effect,
                            common_causes=confs
                        )
                        iden = model.identify_effect(proceed_when_unidentifiable=True)
                        est  = model.estimate_effect(
                            iden, method_name="backdoor.propensity_score_matching"
                        )
                        st.success(f"PSM estimate: {est.value:.3f}")
                except Exception as e:
                    st.error(f"Causal error: {e}")
        st.markdown("Estimates how shifting the cause metric affects the effect metric.")

    # 6) Temporal clustering
    with st.expander("⏱️ Temporal Clustering", expanded=False):
        times = all_df.loc[all_df["iqr_out"], "ds"]
        if times.empty:
            st.info("No IQR anomalies to cluster.")
        else:
            df_tc = pd.DataFrame({"hour": times.dt.hour, "weekday": times.dt.day_name()})
            tbl   = df_tc.groupby(["hour","weekday"]).size().unstack(fill_value=0)
            st.table(tbl.style.background_gradient("OrRd", axis=None))
            st.markdown(
                "Rows = hour, Cols = weekday; counts of IQR-flagged anomalies.\n"
                "Use this to tune your scheduling of alerts."
            )

    # 7) Correlation
    with st.expander("📈 Correlation Matrix", expanded=False):
        corr = pivot.corr()
        fig, ax = plt.subplots(figsize=(3.5,3.5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    square=True, cbar_kws={"shrink":.7}, ax=ax)
        st.pyplot(fig)
        st.markdown("Pairwise correlations—see which metrics co-move.")

    # 8) Final anomaly table + download + alerts
    st.markdown("---\n### 🧾 All IQR-Flagged Anomaly Rows")
    df_tmp = df_ad.copy()
    df_tmp["ds"] = pd.to_datetime(df_tmp[time_col], errors="coerce")

    df_out = df_tmp.merge(
        all_df.loc[all_df["iqr_out"], ["ds","metric"]],
        on="ds", how="inner"
    ).reset_index(drop=True)

    st.dataframe(df_out)
    download_df(df_out, "all_anomaly_rows.csv")

    if st.checkbox("📧 Send Outlook Alert"):
        rec = st.text_input("Recipient email")
        if st.button("Send Email"):
            notify_email("Anomalies detected", "Please review the dashboard.", rec)
    if st.checkbox("💬 Send Teams Alert"):
        hook = st.text_input("Teams webhook URL")
        if st.button("Send Teams message"):
            notify_teams(hook, "🚨 Anomalies detected!")





# 3) Linear Regression
elif task=="Linear Regression":
    st.info("Linear regression")
    target   = st.selectbox("Target",df.select_dtypes(include=np.number).columns)
    features = st.multiselect("Features",df.select_dtypes(include=np.number).columns.drop(target))
    if features:
        X,y = df[features], df[target]
        m   = LinearRegression().fit(X,y)
        df["pred"] = m.predict(X)
        mse = mean_squared_error(y,df["pred"])
        st.write(f"**MSE:** {mse:.2f}")
        st.write(df.head())
        download_df(df[["pred"]+features+[target]],"linreg.csv")


# 4) Logistic Regression
elif task=="Logistic Regression":
    st.info("Logistic regression")
    target   = st.selectbox("Binary target",df.select_dtypes(include=np.number).columns)
    features = st.multiselect("Features",df.select_dtypes(include=np.number).columns.drop(target))
    if features:
        X,y   = df[features], df[target]
        m     = LogisticRegression().fit(X,y)
        df["pred"] = m.predict(X)
        acc  = accuracy_score(y,df["pred"])
        cm   = confusion_matrix(y,df["pred"])
        st.write(f"**Accuracy:** {acc:.2f}")
        st.write("Confusion matrix",cm)
        download_df(df[["pred"]+features+[target]],"logreg.csv")


# 5) Correlation Analysis
elif task=="Correlation Analysis":
    st.info("Correlation matrix")
    corr = df.select_dtypes(include=np.number).corr()
    fig,ax = plt.subplots()
    sns.heatmap(corr,annot=True,cmap="coolwarm",ax=ax)
    st.pyplot(fig)
    download_df(corr.reset_index(),"correlation.csv")


# 6) KMeans Clustering
elif task=="Clustering (KMeans)":
    st.info("KMeans clustering")
    features = st.multiselect("Features",df.select_dtypes(include=np.number).columns)
    k        = st.slider("Clusters",2,10,3)
    if features:
        km=KMeans(n_clusters=k).fit(df[features].dropna())
        df["cluster"]=km.labels_
        st.write(df.head())
        download_df(df[["cluster"]+features],"clusters.csv")


# 7) PCA
elif task=="PCA (Dimensionality Reduction)":
    st.info("PCA")
    features = st.multiselect("Columns",df.select_dtypes(include=np.number).columns)
    if len(features)>=2:
        n  = st.slider("Components",1,len(features),min(2,len(features)))
        pc = PCA(n_components=n).fit_transform(df[features].dropna())
        pcf=pd.DataFrame(pc,columns=[f"PC{i+1}" for i in range(n)])
        st.write(pcf.head())
        download_df(pcf,"pca.csv")


# 8) Recommendation Engine
elif task=="Recommendation Engine":
    st.info("Nearest-neighbor recs")
    items = st.multiselect("Item cols",df.select_dtypes(include=np.number).columns)
    if len(items)>=2:
        nn = NearestNeighbors(n_neighbors=3).fit(df[items].dropna())
        dist,idx = nn.kneighbors(df[items].dropna())
        df["recs"]=[list(df.iloc[i][items].index) for i in idx]
        st.write(df[["recs"]].head())
        download_df(df[["recs"]],"recs.csv")
