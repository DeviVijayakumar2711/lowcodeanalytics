import streamlit as st
import pandas as pd
import numpy as np
import requests
import smtplib
from email.mime.text import MIMEText
import plotly.graph_objects as go
import plotly.express as px
from pyod.models.iforest import IForest
import ruptures as rpt
import stumpy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import os
from pathlib import Path
import re
import datetime

# LangChain & OpenAI imports
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from langchain.globals import set_verbose, get_verbose

# Enable verbose logging if needed
set_verbose(True)

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Analytics Platform", layout="wide")

# ─── 0) LLM & Agent Setup ─────────────────────────────────────────────────────
def get_secret_debug(key, default=None):
    return os.getenv(key, default)

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
AZURE_ENDPOINT    = os.getenv("AZURE_ENDPOINT")
AZURE_DEPLOYMENT  = os.getenv("AZURE_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

if not OPENAI_API_KEY:
    st.sidebar.error("OPENAI_API_KEY is missing. Please configure it in App settings.")
    st.stop()

llm = None
if AZURE_ENDPOINT and AZURE_DEPLOYMENT and OPENAI_API_KEY:
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=OPENAI_API_KEY,
            azure_deployment=AZURE_DEPLOYMENT,
            api_version=AZURE_API_VERSION,
            temperature=0.1,
        )
    except Exception as e:
        st.warning(f"AzureChatOpenAI init failed: {e}. Falling back to ChatOpenAI.")
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.1)
else:
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.1)

if llm is None:
    st.error("LLM initialization failed.")
    st.stop()

# ─── CACHED HELPERS ───────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data(show_spinner=False)
def convert_datetime(df: pd.DataFrame, col: str, fmt: str) -> pd.DataFrame:
    out = df.copy()
    if fmt == "Auto-Detect":
        out[col] = pd.to_datetime(out[col], errors="coerce")
    else:
        out[col] = pd.to_datetime(out[col], format=fmt, errors="coerce")
    return out

@st.cache_data(show_spinner=False)
def filter_date(df: pd.DataFrame, col: str, start, end) -> pd.DataFrame:
    mask = (df[col] >= pd.to_datetime(start)) & (df[col] <= pd.to_datetime(end))
    return df.loc[mask].reset_index(drop=True)

@st.cache_data(ttl=3600)
def daily_aggregate(df: pd.DataFrame, time_col: str, group_col: str, metrics: list[str]) -> pd.DataFrame:
    df2 = df.set_index(time_col)
    keys = [pd.Grouper(freq="D")]
    if group_col != 'ALL':
        keys.append(group_col)
    return df2.groupby(keys)[metrics].mean().reset_index()

# ─── DOWNLOAD & ALERT HELPERS ─────────────────────────────────────────────────
def download_df(df: pd.DataFrame, filename: str):
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, file_name=filename)

def notify_email(subject: str, message: str, to: str):
    host, port = "smtp.office365.com", 587
    frm = os.getenv("EMAIL")
    pwd = os.getenv("EMAIL_PASSWORD")
    msg = MIMEText(message)
    msg["Subject"], msg["From"], msg["To"] = subject, frm, to
    with smtplib.SMTP(host, port) as s:
        s.starttls()
        s.login(frm, pwd)
        s.sendmail(frm, [to], msg.as_string())
    st.success("Email sent ✅")

def notify_teams(url: str, text: str):
    r = requests.post(url, json={"text": text})
    if r.ok:
        st.success("Teams message sent ✅")
    else:
        st.error("Teams message failed")

# ─── CSV AGENT ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_csv_agent(path: str, llm_instance):
    return create_csv_agent(
        llm_instance, path,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )

# ─── 1) DATA SOURCE & PREVIEW ─────────────────────────────────────────────────
st.title("Low-Code Analytics Platform")
BASE_DIR = Path(__file__).parent
DEFAULT_CSV = BASE_DIR / "telematics_trip_new2.csv"

if DEFAULT_CSV.exists():
    df_orig = load_csv(str(DEFAULT_CSV))
    st.success(f"✅ Loaded default dataset ({df_orig.shape[0]} rows)")
else:
    df_orig = pd.DataFrame()

uploaded = st.file_uploader("📂 Or upload CSV", type="csv")
if uploaded:
    df_orig = load_csv(uploaded)
    st.success(f"✅ Loaded uploaded dataset ({df_orig.shape[0]} rows)")

if df_orig.empty:
    st.error("No dataset available.")
    st.stop()

df = df_orig.copy()
st.dataframe(df.head())

tmp_csv = BASE_DIR / "__tmp_telematics.csv"
df.to_csv(tmp_csv, index=False)
csv_agent = get_csv_agent(str(tmp_csv), llm)

# ─── Sidebar: AI Chatbot ───────────────────────────────────────────────────────
st.sidebar.header("🤖 Ask the AI Bot")
if "history" not in st.session_state:
    st.session_state.history = []

user_q = st.sidebar.chat_input("Your question…")
if user_q:
    st.session_state.history.append({"role": "user", "content": user_q})
    with st.spinner("Thinking…"):
        ans = csv_agent.invoke({"input": user_q})
        out = ans.get("output", "")
        if isinstance(out, AIMessage):
            out = out.content
        st.session_state.history.append({"role": "assistant", "content": out})

for msg in st.session_state.history:
    st.sidebar.chat_message(msg["role"]).write(msg["content"])

# ─── 2) TIME & DATE FILTER ───────────────────────────────────────────────────
st.markdown("---")
st.header("1. Global Data Filters")

# Time column selection
time_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
time_col = st.selectbox("Select Time column", time_cols)

# Date format
formats = {
    "Auto-Detect": "Auto-Detect",
    "DD/MM/YYYY": "%d/%m/%Y",
    "MM/DD/YYYY": "%m/%d/%Y",
    "YYYY-MM-DD": "%Y-%m-%d",
    "YYYY-MM-DD HH:MM:SS": "%Y-%m-%d %H:%M:%S",
}
fmt = st.selectbox("Date Format", list(formats.keys()), index=0)
df = convert_datetime(df, time_col, formats[fmt])
df.dropna(subset=[time_col], inplace=True)

# Date range slider
min_d = df[time_col].min().date()
max_d = df[time_col].max().date()
default_start = max_d - pd.DateOffset(months=3)
start, end = st.slider("Date range", min_value=min_d, max_value=max_d,
                       value=(default_start.date(), max_d), format="YYYY-MM-DD")
df = filter_date(df, time_col, start, end)
st.write(f"Displaying {len(df)} records from {start} to {end}")

# ... continue with the rest of the analysis sections exactly as in the file ...
