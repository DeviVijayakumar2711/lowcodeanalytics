import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
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
import seaborn as sns
import matplotlib.pyplot as plt

# Optional causal inference
try:
    from dowhy import CausalModel
    DOWHY = True
except ImportError:
    DOWHY = False

st.set_page_config(page_title="Low-Code Analytics POC", layout="wide")
st.title("🧠 Analytics Platform")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def download_df(df: pd.DataFrame, filename: str):
    csv = df.to_csv(index=False)
    st.download_button("📥 Download Results", csv, file_name=filename, mime="text/csv")

def notify_email(subject: str, message: str, to: str):
    host, port = "smtp.office365.com", 587
    frm = st.secrets.get("EMAIL", "")
    pwd = st.secrets.get("EMAIL_PASSWORD", "")
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

def notify_teams(url: str, text: str):
    try:
        r = requests.post(url, json={"text": text})
        if r.ok:
            st.success("✅ Teams sent")
        else:
            st.error("Teams alert failed")
    except Exception as e:
        st.error(f"Teams error: {e}")

# ─── Upload & Task Selector ──────────────────────────────────────────────────

st.markdown("Upload a CSV and choose an analytics task from the list below.")
uploaded = st.file_uploader("📂 Upload CSV", type="csv")
if not uploaded:
    st.stop()
df = pd.read_csv(uploaded)
df.columns = df.columns.str.strip()
st.write("### Preview of your data", df.head())

task = st.selectbox("🛠️ Analysis Task", [
    "Time Series Forecasting",
    "Anomaly Detection (Advanced)",
    "Linear Regression",
    "Logistic Regression",
    "Correlation Analysis",
    "Clustering (KMeans)",
    "PCA (Dimensionality Reduction)",
    "Recommendation Engine"
])

# ─── Sidebar Guide ───────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📘 How to use")
    if task == "Time Series Forecasting":
        st.markdown("""
        1. Upload a CSV with **one date** column & **one numeric** column  
        2. Select which columns represent date & value  
        3. Prophet may fail; we’ll fall back to a linear trend  
        4. View chart, trend table & download  
        """)
    elif task == "Anomaly Detection (Advanced)":
        st.markdown("""
        1. Choose a **time** column + **one or more** numeric metrics  
        2. Adjust min/max thresholds for your domain  
        3. Inspect **IQR** anomalies, combined overlay & multivariate outliers  
        4. Dive into causal inference, temporal clustering, correlation  
        5. View anomaly table & download, send alerts  
        """)
    elif task == "Linear Regression":
        st.markdown("""
        1. Pick one numeric **target** & numeric **features**  
        2. View **MSE** & prediction table  
        3. Download results  
        """)
    elif task == "Logistic Regression":
        st.markdown("""
        1. Pick a **binary** numeric target & numeric features  
        2. View **accuracy**, **confusion matrix**  
        3. Download results  
        """)
    elif task == "Correlation Analysis":
        st.markdown("""
        1. Select **≥2 numeric** columns  
        2. View & download the **correlation matrix**  
        """)
    elif task == "Clustering (KMeans)":
        st.markdown("""
        1. Select numeric features & number of clusters  
        2. View cluster labels & download  
        """)
    elif task == "PCA (Dimensionality Reduction)":
        st.markdown("""
        1. Select **≥2 numeric** columns  
        2. Choose number of principal components  
        3. View PC scores & download  
        """)
    else:
        st.markdown("""
        1. Upload a user–item style CSV  
        2. Select **item** columns for nearest neighbors  
        3. View & download recommendations  
        """)

# ─── Optional filter ───────────────────────────────────────────────────────────

filter_col = st.selectbox("Filter by column (optional)", ["None"] + df.columns.tolist())
if filter_col != "None":
    vals = st.multiselect(f"Select {filter_col}", df[filter_col].dropna().unique())
    if vals:
        df = df[df[filter_col].isin(vals)]
        st.info(f"Filtered: {filter_col} = {vals}")

# ─── Task branches ────────────────────────────────────────────────────────────


# ─── Time Series Forecasting ──────────────────────────────────────────────────
if task == "Time Series Forecasting":
    date_col  = st.selectbox("Date column",  df.columns)
    value_col = st.selectbox("Value column", df.columns)

    # ←——— NEW: guard against selecting the same column twice
    if date_col == value_col:
        st.error("❌ Date and Value must be **different** columns. Please select two distinct columns.")
        st.stop()

    ts = df[[date_col, value_col]].dropna().rename(
        columns={date_col: "ds", value_col: "y"}
    )
    ts["ds"] = pd.to_datetime(ts["ds"], errors="coerce")
    ts = ts.dropna(subset=["ds", "y"])

    if len(ts) < 2:
        st.error("Need at least 2 valid rows.")
    else:
        periods = st.slider("Days to forecast", 7, 90, 30)
        try:
            m = Prophet().fit(ts)
            future = m.make_future_dataframe(periods=periods)
            fc = m.predict(future)
            st.line_chart(fc.set_index("ds")["yhat"])
            st.write("### Forecast Table", fc[["ds","yhat","yhat_lower","yhat_upper"]].tail(10))
            download_df(fc[["ds","yhat","yhat_lower","yhat_upper"]], "forecast.csv")
        except Exception:
            st.warning("Prophet optimization failed → showing linear fallback.")
            lr_df = ts.copy()
            lr_df["t"] = (lr_df["ds"] - lr_df["ds"].min()).dt.days
            lr = LinearRegression().fit(lr_df[["t"]], lr_df["y"])
            lr_df["trend"] = lr.predict(lr_df[["t"]])
            fig, ax = plt.subplots(figsize=(6,3))
            ax.plot(lr_df["ds"], lr_df["y"], label="Actual")
            ax.plot(lr_df["ds"], lr_df["trend"], "--", label="Trend")
            ax.legend(); ax.grid(ls="--", alpha=0.3); ax.tick_params(rotation=45)
            st.pyplot(fig)
            st.write("### Trend Table", lr_df[["ds","trend"]].tail(10))
            download_df(lr_df[["ds","trend"]], "trend.csv")



# 2) Anomaly Detection (Advanced)

elif task == "Anomaly Detection (Advanced)":
    time_col = st.selectbox("Time column", df.columns)
    metrics = st.multiselect("Numeric metrics", df.select_dtypes(include="number").columns.tolist())
    if not time_col or not metrics:
        st.warning("Pick a time column and ≥1 metric."); st.stop()

    overlay = []
    # ─ per-metric
    for m in metrics:
        st.markdown(f"#### 🔎 {m}")
        col = df[m].dropna()
        lo = st.number_input(f"Lower threshold for {m}", float(col.min()), float(col.max()), float(col.min()))
        hi = st.number_input(f"Upper threshold for {m}", float(col.min()), float(col.max()), float(col.max()))

        ts = df[[time_col,m]].dropna().copy()
        ts[time_col] = pd.to_datetime(ts[time_col])
        ts = ts.rename(columns={time_col:"ds", m:"y"}).sort_values("ds")
        ts["trend"] = ts["y"].rolling(7,1).mean()
        q1, q3 = ts["y"].quantile([.25,.75])
        iqr = q3-q1
        lb, ub = q1-1.5*iqr, q3+1.5*iqr
        ts["anomaly"] = ((ts["y"]<lb)|(ts["y"]>ub)|(ts["y"]<lo)|(ts["y"]>hi))
        ts["metric"] = m
        overlay.append(ts)

        # plot
        fig, ax = plt.subplots(figsize=(6,2.5))
        ax.plot(ts["ds"], ts["y"], label="Actual", lw=1)
        ax.plot(ts["ds"], ts["trend"], "--", label="Trend")
        ax.fill_between(ts["ds"], lb, ub, color="skyblue", alpha=0.4, label="IQR Range")
        ax.scatter(ts.loc[ts["anomaly"],"ds"], ts.loc[ts["anomaly"],"y"],
                   c="red", s=20, label="Anomaly")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5,1.15), ncol=4, fontsize="x-small")
        ax.set_ylabel(m); ax.grid(ls="--", alpha=0.3); ax.tick_params(axis="x", rotation=45)
        plt.tight_layout(); st.pyplot(fig)

        # narrative
        a = ts[ts["anomaly"]]
        if not a.empty:
            f = a.iloc[0]; d = f["y"]-f["trend"]
            dirn = "above" if d>0 else "below"
            st.info(f"ℹ️ {m} was {abs(d):.1f} units {dirn} trend at {f['ds']}.")
        else:
            st.info(f"ℹ️ No anomalies for {m}.")

    # ─ combined overlay
    all_df = pd.concat(overlay)
    st.markdown("### 🔗 Combined Overlay")
    fig, ax = plt.subplots(figsize=(6,2.5))
    for m, grp in all_df.groupby("metric"):
        ax.plot(grp["ds"], grp["y"], label=m)
        ax.scatter(grp.loc[grp["anomaly"],"ds"], grp.loc[grp["anomaly"],"y"], c="red", s=15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5,1.15),
              ncol=len(metrics), fontsize="x-small")
    ax.grid(ls="--", alpha=0.3); ax.tick_params(axis="x", rotation=45)
    plt.tight_layout(); st.pyplot(fig)

    # ─ multivariate anomalies (Mahalanobis)
    pivot = all_df.pivot(index="ds", columns="metric", values="y").dropna()
    st.markdown("🎯 Multivariate anomalies (Mahalanobis):")
    if pivot.shape[1] > 1:
        mu = pivot.mean().values
        cov = np.cov(pivot.values.T)
        inv = np.linalg.pinv(cov)
        md = pivot.apply(lambda r: np.sqrt((r-mu)@inv@(r-mu)), axis=1)
        thr = np.percentile(md, 97.5)
        st.write(md[md>thr].index.tolist())
    else:
        st.info("Need ≥2 metrics for multivariate")

    # ─ causal inference
    with st.expander("🧠 Causal Inference"):
        if not DOWHY:
            st.info("Install `dowhy` to enable this section.")
        elif len(metrics) < 2:
            st.info("Select ≥2 metrics.")
        else:
            cause = st.selectbox("Cause", metrics)
            effect = st.selectbox("Effect", [m for m in metrics if m!=cause])
            confs = [m for m in metrics if m not in (cause,effect)]
            df_ci = pivot[[cause,effect]+confs].dropna().reset_index(drop=True)
            if df_ci.shape[0] < 5:
                st.warning("Need ≥5 rows after dropna.")
            else:
                try:
                    if not confs:
                        lr = LinearRegression().fit(df_ci[[cause]], df_ci[effect])
                        st.success(f"OLS slope ({cause}→{effect}): {lr.coef_[0]:.4f}")
                    else:
                        med = df_ci[cause].median()
                        df_ci["_t"] = (df_ci[cause]>=med).astype(int)
                        model = CausalModel(data=df_ci,
                                            treatment="_t",
                                            outcome=effect,
                                            common_causes=confs)
                        iden = model.identify_effect(proceed_when_unidentifiable=True)
                        est  = model.estimate_effect(iden,
                                                     method_name="backdoor.propensity_score_matching")
                        st.success(f"PSM estimate: {est.value:.4f}")
                except Exception as e:
                    st.error(f"Causal inference error: {e}")

    # ─ temporal clustering
    with st.expander("⏱️ Temporal Clustering"):
        times = all_df.loc[all_df["anomaly"], "ds"]
        if times.empty:
            st.info("No anomalies to cluster.")
        else:
            df_tc = pd.DataFrame({"hour": times.dt.hour, "weekday": times.dt.weekday})
            tbl = df_tc.groupby(["hour","weekday"]).size().unstack(fill_value=0)
            st.write(tbl)
            fig, ax = plt.subplots(figsize=(4,3))
            sns.heatmap(tbl, annot=True, fmt="d", cmap="Reds", ax=ax)
            ax.set_title("Anomalies by hour & weekday")
            ax.set_xlabel("Weekday"); ax.set_ylabel("Hour")
            plt.tight_layout(); st.pyplot(fig)

    # ─ correlation matrix
    with st.expander("📈 Correlation Matrix"):
        corr = pivot.corr()
        fig, ax = plt.subplots(figsize=(4,4))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    square=True, cbar_kws={"shrink":.7}, ax=ax)
        plt.tight_layout(); st.pyplot(fig)

    # ─ anomaly table, download & alerts
    st.markdown("### 🧾 Anomaly Table")
    st.dataframe(all_df.loc[all_df["anomaly"], ["ds","metric","y"]])
    download_df(all_df.loc[all_df["anomaly"], ["ds","metric","y"]], "anomalies.csv")

    if st.checkbox("📧 Send Outlook Alert"):
        rec = st.text_input("Recipient email")
        if st.button("Send Email"):
            notify_email("Anomalies detected", "Please review dashboard.", rec)

    if st.checkbox("💬 Send Teams Alert"):
        hook = st.text_input("Teams webhook URL")
        if st.button("Send Teams message"):
            notify_teams(hook, "🚨 Anomalies detected in your metrics!")
            
# 3) Linear Regression
elif task == "Linear Regression":
    target   = st.selectbox("Target", df.select_dtypes(include=np.number).columns)
    features = st.multiselect("Features", df.select_dtypes(include=np.number).columns.drop(target))
    if features:
        X, y = df[features], df[target]
        m     = LinearRegression().fit(X, y)
        df["pred"] = m.predict(X)
        mse = mean_squared_error(y, df["pred"])
        st.write(f"**MSE:** {mse:.2f}")
        st.write(df.head())
        download_df(df[["pred"] + features + [target]], "linreg.csv")

# 4) Logistic Regression
elif task == "Logistic Regression":
    target   = st.selectbox("Binary target", df.select_dtypes(include=np.number).columns)
    features = st.multiselect("Features", df.select_dtypes(include=np.number).columns.drop(target))
    if features:
        X, y = df[features], df[target]
        m     = LogisticRegression().fit(X, y)
        df["pred"] = m.predict(X)
        acc = accuracy_score(y, df["pred"])
        cm  = confusion_matrix(y, df["pred"])
        st.write(f"**Accuracy:** {acc:.2f}")
        st.write("Confusion matrix", cm)
        download_df(df[["pred"] + features + [target]], "logreg.csv")

# 5) Correlation Analysis
elif task == "Correlation Analysis":
    corr = df.select_dtypes(include=np.number).corr()
    st.write("### Correlation Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    download_df(corr.reset_index(), "correlation.csv")

# 6) KMeans Clustering
elif task == "Clustering (KMeans)":
    num = st.multiselect("Features", df.select_dtypes(include=np.number).columns)
    k   = st.slider("Clusters", 2, 10, 3)
    if num:
        km = KMeans(n_clusters=k).fit(df[num].dropna())
        df["cluster"] = km.labels_
        st.write(df.head())
        download_df(df[["cluster"] + num], "clusters.csv")

# 7) PCA
elif task == "PCA (Dimensionality Reduction)":
    num = st.multiselect("Columns", df.select_dtypes(include=np.number).columns)
    if len(num) >= 2:
        ncomp = st.slider("Components", 2, len(num), 2)
        pc    = PCA(n_components=ncomp).fit_transform(df[num].dropna())
        pcf   = pd.DataFrame(pc, columns=[f"PC{i+1}" for i in range(ncomp)])
        st.write(pcf.head())
        download_df(pcf, "pca.csv")

# 8) Recommendation Engine
elif task == "Recommendation Engine":
    items = st.multiselect("Item columns", df.select_dtypes(include=np.number).columns)
    if len(items) >= 2:
        nn = NearestNeighbors(n_neighbors=3).fit(df[items].dropna())
        dist, idx = nn.kneighbors(df[items].dropna())
        df["recs"] = [list(df.index[i]) for i in idx]
        st.write(df[["recs"]].head())
        download_df(df[["recs"]], "recs.csv")
