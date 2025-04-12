import streamlit as st
import pandas as pd
import numpy as np
import requests
from prophet import Prophet
from pyod.models.iforest import IForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt
import io
import pyodbc

st.set_page_config(page_title="Low-Code Analytics POC", layout="wide")

st.title("🧠 Low-Code Analytics Platform")
st.markdown("""
Upload your dataset and explore powerful analytics tasks:
- ⏱️ Time Series Forecasting (Prophet)
- ⚠️ Anomaly Detection (PyOD)
- 📈 Linear & Logistic Regression
- 🔍 Correlation Analysis
- 🔢 Clustering, PCA
- 💡 Recommendation Engine (based on similarity)
- 🗄️ Azure SQL Integration
- 💬 Chat Assistant (LLM-enabled with fallback)
""")

with st.expander("📘 What kind of input files should I upload?"):
    st.markdown("""
    This app expects CSV files with **column headers**. Depending on the analysis task, here’s what you need:

    - **Time Series Forecasting**: One column with date (e.g., `Date`, `Operation Date`) and one column with values (e.g., `Sales`, `Odometer`, etc.)
    - **Regression (Linear/Logistic)**: One target column (numeric or binary) and multiple feature columns (numeric)
    - **PCA / Clustering / Anomaly Detection**: At least **2 numeric columns**
    - **Recommendation Engine**: Matrix-style data with user/item-like columns having numeric values
    - **Correlation Analysis**: Multiple numeric columns for comparison

    ✅ Tip: Upload a clean CSV with no empty header rows or merged columns.
    """)

uploaded_file = st.file_uploader("📂 Upload your CSV file", type="csv")

def download_button(df, filename):
    csv = df.to_csv(index=False)
    st.download_button("📥 Download Results", csv, file_name=filename, mime="text/csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Data Preview", df.head())

    task = st.selectbox("🛠️ Select Analysis Task", [
        "Time Series Forecasting", 
        "Anomaly Detection", 
        "Linear Regression", 
        "Logistic Regression",
        "Correlation Analysis",
        "Clustering (KMeans)",
        "PCA (Dimensionality Reduction)",
        "Recommendation Engine"
    ])

    if task == "Time Series Forecasting":
        st.info("Input required: Date column + Value column")
        date_col = st.selectbox("📅 Select Date Column", df.columns)
        value_col = st.selectbox("📈 Select Value Column", df.columns)
        ts_df = df[[date_col, value_col]].copy()
        ts_df = ts_df.rename(columns={date_col: 'ds', value_col: 'y'})
        ts_df['ds'] = pd.to_datetime(ts_df['ds'], errors='coerce')
        ts_df = ts_df.dropna(subset=['ds', 'y'])
        if len(ts_df) < 2:
            st.error("🚫 The selected columns have less than 2 valid data rows. Please choose a date column and a numeric column with valid data.")
        else:
            model = Prophet()
            model.fit(ts_df)
            periods = st.slider("🔮 Forecast Days", 7, 90, 30)
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
            st.line_chart(forecast.set_index('ds')['yhat'])
            download_button(forecast[['ds', 'yhat']], "forecast.csv")

    elif task == "Anomaly Detection":
        st.info("Input required: Numeric columns")
        num_cols = st.multiselect("📌 Select Numeric Columns", df.select_dtypes(include=np.number).columns)
        if num_cols and len(df[num_cols].dropna()) > 1:
            X = df[num_cols]
            X_scaled = StandardScaler().fit_transform(X)
            clf = IForest()
            clf.fit(X_scaled)
            df['anomaly'] = clf.predict(X_scaled)
            st.write("### 🚨 Anomaly Detection Result", df.head())
            download_button(df, "anomaly_results.csv")

    elif task == "Linear Regression":
        st.info("Input required: Numeric target column + feature columns")
        target = st.selectbox("🎯 Select Target Column", df.select_dtypes(include=np.number).columns)
        features = st.multiselect("📌 Select Feature Columns", df.select_dtypes(include=np.number).columns.drop(target))
        if features and len(df[features].dropna()) > 1 and df[target].notna().sum() > 1:
            X = df[features]
            y = df[target]
            model = LinearRegression()
            model.fit(X, y)
            df['prediction'] = model.predict(X)
            mse = mean_squared_error(y, df['prediction'])
            st.write(f"### 📉 Mean Squared Error: {mse:.2f}")
            st.write(df.head())
            download_button(df, "linear_regression.csv")

    elif task == "Logistic Regression":
        st.info("Input required: Binary target column + numeric feature columns")
        target = st.selectbox("🎯 Select Binary Target Column", df.select_dtypes(include=np.number).columns)
        features = st.multiselect("📌 Select Feature Columns", df.select_dtypes(include=np.number).columns.drop(target))
        if features and len(df[features].dropna()) > 1 and df[target].notna().sum() > 1:
            X = df[features]
            y = df[target]
            model = LogisticRegression()
            model.fit(X, y)
            preds = model.predict(X)
            df['prediction'] = preds
            acc = accuracy_score(y, preds)
            st.write(f"### ✅ Accuracy: {acc:.2f}")
            st.write("### Confusion Matrix")
            st.write(confusion_matrix(y, preds))
            download_button(df, "logistic_regression.csv")

    elif task == "Correlation Analysis":
        st.info("Input required: At least 2 numeric columns")
        corr = df.corr()
        st.write("### 🔗 Correlation Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    elif task == "Clustering (KMeans)":
        st.info("Input required: Numeric features to cluster")
        num_cols = st.multiselect("📌 Select Features for Clustering", df.select_dtypes(include=np.number).columns)
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        if num_cols and len(df[num_cols].dropna()) > 1:
            X = df[num_cols]
            model = KMeans(n_clusters=n_clusters)
            df['cluster'] = model.fit_predict(X)
            st.write(df.head())
            download_button(df, "clustering_results.csv")

    elif task == "PCA (Dimensionality Reduction)":
        st.info("Input required: At least 2 numeric columns")
        num_cols = st.multiselect("📌 Select Columns for PCA", df.select_dtypes(include=np.number).columns)

        if len(num_cols) < 2:
            st.warning("Please select at least two numeric columns for PCA.")
        else:
            max_components = min(len(df), len(num_cols))
            if max_components < 2:
                st.warning("Not enough data points or columns to apply PCA.")
            else:
                n_components = st.slider("Number of Components", 1, max_components, min(2, max_components))
                X = StandardScaler().fit_transform(df[num_cols])
                pca = PCA(n_components=n_components)
                principal_components = pca.fit_transform(X)
                pca_df = pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(n_components)])
                st.write(pca_df.head())
                download_button(pca_df, "pca_results.csv")

    elif task == "Recommendation Engine":
        st.info("Input required: User-Item matrix (e.g., users as rows, items as columns, values as ratings)")
        item_cols = st.multiselect("📌 Select Item Features", df.select_dtypes(include=np.number).columns)
        if len(item_cols) >= 2 and df[item_cols].dropna().shape[0] > 1:
            X = df[item_cols]
            nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(X)
            distances, indices = nbrs.kneighbors(X)
            df['recommendations'] = [list(df.iloc[indices[i]][item_cols].index) for i in range(len(df))]
            st.write(df[[*item_cols, 'recommendations']].head())
            download_button(df, "recommendation_results.csv")

    st.markdown("---")

# Azure SQL Integration
with st.expander("🗄️ Connect to Azure SQL (Optional)"):
    sql_enabled = st.checkbox("Enable Azure SQL connection")
    if sql_enabled:
        server = st.text_input("Server Name", placeholder="yourserver.database.windows.net")
        database = st.text_input("Database Name")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Connect and Load Data"):
            try:
                conn = pyodbc.connect(
                    f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
                )
                query = st.text_area("SQL Query", "SELECT TOP 10 * FROM your_table")
                df_sql = pd.read_sql(query, conn)
                st.write("### 🔄 Data from Azure SQL", df_sql)
            except Exception as e:
                st.error(f"Connection failed: {e}")

# Chatbot using Hugging Face API with fallback

def ask_llm_with_fallback(question):
    try:
        api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        headers = {"Authorization": "Bearer YOUR_HF_API_KEY"}  # Replace with your HF token
        payload = {"inputs": question}
        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 200 and response.json():
            return response.json()[0].get('generated_text', 'No response generated.')
        else:
            return fallback_chatbot(question)
    except:
        return fallback_chatbot(question)

def fallback_chatbot(q):
    q = q.lower()
    if any(word in q for word in ["regression", "linear", "logistic"]):
        return "📘 Tip: Use linear regression for continuous outcomes, logistic regression for binary classification tasks."
    elif any(word in q for word in ["anomaly", "outlier", "detect"]):
        return "📘 Tip: Try anomaly detection with Isolation Forest to find unusual patterns in your numeric features."
    elif any(word in q for word in ["forecast", "predict future", "time series"]):
        return "📘 Tip: Use Prophet time series forecasting when you have date and value columns."
    elif any(word in q for word in ["correlation", "relationship"]):
        return "📘 Tip: Use the correlation matrix to understand how features are related."
    else:
        return "🤖 Sorry, I couldn't find a match. Try asking about regression, anomaly detection, forecasting, or correlation."

with st.expander("💬 Ask Chatbot (Mistral 7B with fallback)"):
    user_question = st.text_input("Ask me anything (data, analytics, general)")
    if user_question:
        with st.spinner("Thinking..."):
            answer = ask_llm_with_fallback(user_question)
            st.markdown(f"**🤖 Answer:** {answer}")
