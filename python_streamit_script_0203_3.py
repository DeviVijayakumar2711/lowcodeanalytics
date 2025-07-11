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
import re # Import regex for parsing LLM output
import datetime # Import datetime module if not already imported

# Streamlit, LangChain, and OpenAI imports
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
try:
    from langchain_experimental.agents import create_csv_agent
except ImportError:
    from langchain.agents import create_csv_agent
from langchain.schema import AIMessage

# Added for agent configuration in newer Langchain versions
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate # Needed for custom agent if create_csv_agent is insufficient
from langchain.agents import AgentExecutor # Explicitly import AgentExecutor
from langchain.globals import set_verbose, get_verbose

# To enable verbose mode:
set_verbose(True)

# To check if verbose mode is enabled:
if get_verbose():
    # Your verbose output/logging code here
    pass


# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Analytics Platform", layout="wide")


# ─── 0) LLM & Agent Setup ─────────────────────────────────────────────────────

# ─── 0) LLM & Agent Setup ─────────────────────────────────────────────────────
def get_secret_debug(key, default=None):
    return os.getenv(key, default)


# straight from the host’s env; no python-dotenv, no .env file
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
AZURE_ENDPOINT    = os.getenv("AZURE_ENDPOINT")
AZURE_DEPLOYMENT  = os.getenv("AZURE_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

if not OPENAI_API_KEY:
    st.sidebar.error("OPENAI_API_KEY is missing. Please configure it in App settings.")
    st.stop()

llm = None # Global LLM instance for direct calls
if (AZURE_ENDPOINT is not None and isinstance(AZURE_ENDPOINT, str) and
    AZURE_DEPLOYMENT is not None and isinstance(AZURE_DEPLOYMENT, str) and
    OPENAI_API_KEY is not None and isinstance(OPENAI_API_KEY, str) and
    AZURE_API_VERSION is not None and isinstance(AZURE_API_VERSION, str)):
    try:
        llm = AzureChatOpenAI(
            azure_endpoint    = AZURE_ENDPOINT,
            api_key           = OPENAI_API_KEY,
            azure_deployment  = AZURE_DEPLOYMENT,
            api_version       = AZURE_API_VERSION,
            temperature       = 0.1,
        )
    except Exception as e:
        st.warning(f"Failed to initialize AzureChatOpenAI: {e}. Falling back to standard ChatOpenAI if API key available.")
        if OPENAI_API_KEY is not None and isinstance(OPENAI_API_KEY, str):
            try:
                llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.1)
            except Exception as fe:
                st.sidebar.error(f"Failed to initialize standard ChatOpenAI: {fe}")
                st.stop()
        else:
            st.sidebar.error("OPENAI_API_KEY is missing or invalid. Cannot initialize any LLM.")
            st.stop()
else:
    if OPENAI_API_KEY is not None and isinstance(OPENAI_API_KEY, str):
        try:
            llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.1)
        except Exception as fe:
            st.sidebar.error(f"Failed to initialize standard ChatOpenAI: {fe}")
            st.stop()
    else:
        st.sidebar.error("OPENAI_API_KEY is missing or invalid. Cannot initialize any LLM.")
        st.stop()

if llm is None:
    st.error("Fatal: LLM could not be initialized. Please check your secrets and the console for errors.")
    st.stop()


# ─── CACHED HELPERS ───────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_csv(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data(show_spinner=False)
def convert_datetime(df, col, date_format_string):
    """Converts a column to datetime using a specific format string."""
    out = df.copy()
    try:
        if date_format_string == "Auto-Detect":
             out[col] = pd.to_datetime(out[col], errors="coerce")
        else:
            out[col] = pd.to_datetime(out[col], format=date_format_string, errors="coerce")
    except Exception as e:
        st.error(f"Date format error: {e}. Please ensure the format string matches your data.")
        return None
    return out

@st.cache_data(show_spinner=False)
def filter_date(df, col, start, end):
    mask = (df[col] >= pd.to_datetime(start)) & (df[col] <= pd.to_datetime(end))
    return df.loc[mask].reset_index(drop=True)

@st.cache_data(show_spinner=False)
def fit_iforest(values: np.ndarray, contamination: float):
    data_2d = values.reshape(-1, 1)
    try:
        if len(data_2d) > 1:
            iso = IForest(contamination=contamination, random_state=42)
            iso.fit(data_2d)
            return iso.predict(data_2d), iso.decision_function(data_2d)
        return np.zeros(len(values), dtype=bool), np.zeros(len(values))
    except Exception as e:
        st.warning(f"Isolation Forest failed: {e}.")
        return np.zeros(len(values), dtype=bool), np.zeros(len(values))


@st.cache_data(ttl=3600)
def daily_aggregate(df, time_col, group_col, metrics):
    df2 = df.set_index(time_col)
    grp_keys = [pd.Grouper(freq="D")]
    if group_col != 'ALL':
        if group_col not in df2.columns:
            st.error(f"Error: Group column '{group_col}' not found for aggregation.")
            return pd.DataFrame()
        grp_keys.append(group_col)
    existing_metrics = [m for m in metrics if m in df2.columns]
    if not existing_metrics:
        st.warning("No valid numeric metrics for aggregation.")
        return pd.DataFrame()
    aggregated_df = df2.groupby(grp_keys)[existing_metrics].mean().reset_index()
    return aggregated_df

# ─── DOWNLOAD & ALERT HELPERS ─────────────────────────────────────────────────
def download_df(df, filename):
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, file_name=filename)

def notify_email(subject, message, to):
    host, port = "smtp.office365.com", 587
    frm = os.getenv("EMAIL", "")
    pwd = os.getenv("EMAIL_PASSWORD", "")

    msg = MIMEText(message)
    msg["Subject"], msg["From"], msg["To"] = subject, frm, to
    try:
        s = smtplib.SMTP(host, port)
        s.starttls()
        s.login(frm, pwd)
        s.sendmail(frm, [to], msg.as_string())
        s.quit()
        st.success("Email sent ✅")
    except Exception as e:
        st.error(f"Email failed: {e}")

def notify_teams(url, text):
    try:
        r = requests.post(url, json={"text": text})
        if r.ok: st.success("Teams message sent ✅")
        else: st.error("Teams message failed")
    except Exception as e:
        st.error(f"Teams error: {e}")

# ─── CSV AGENT ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_csv_agent(_path: str, _llm_instance):
    return create_csv_agent(
        _llm_instance, _path, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_executor_kwargs={"handle_parsing_errors": True},
        allow_dangerous_code=True, 
    )

# ─── LLM-BASED ANOMALY INSIGHTS GENERATION ────────────────────────────────────
@st.cache_data(show_spinner="AI Analyst is diagnosing anomaly...")
def get_llm_anomaly_details(anomaly_row_dict: dict, metric: str, _llm) -> dict:
    filtered_anomaly_row = {k: v for k, v in anomaly_row_dict.items() if not isinstance(v, (list, dict, set))}
    formatted_row = "\n".join([f"- {k}: {v}" for k, v in filtered_anomaly_row.items()])

    llm_query = f"""
    As an expert AI Bus Operations Analyst with 100 years of experience, analyze the following anomalous trip data for the metric '{metric}'.
    The observed value was '{anomaly_row_dict.get(metric, 'N/A')}'.

    **Anomalous Trip Details:**
    ```
    {formatted_row}
    ```

    Based on this data, provide:
    1.  **Root Cause Diagnosis:** What are the most probable reasons for this specific anomaly?
    2.  **Key Insights:** What are the significant operational or safety implications?
    3.  **Actionable Suggestions:** What concrete steps should be taken to investigate or address this incident?
    """
    try:
        response_message = _llm.invoke(llm_query)
        content = response_message.content

        headings_map = {
            "root_cause": "Root Cause Diagnosis", "key_insights": "Key Insights",
            "action_suggestions": "Actionable Suggestions"
        }
        pattern = r'(?i)(?:#+\s*|\d\.\s*|\*\*)?(' + '|'.join(headings_map.values()) + r')(?::?|\*\*)'
        matches = list(re.finditer(pattern, content))

        parsed_results = {}
        for i, match in enumerate(matches):
            heading_text = match.group(1).strip()
            heading_key = next((k for k, v in headings_map.items() if v.lower() == heading_text.lower()), None)
            if not heading_key: continue
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            parsed_results[heading_key] = content[start_pos:end_pos].strip()

        return {
            "root_cause": parsed_results.get("root_cause", "N/A"),
            "key_insights": parsed_results.get("key_insights", "N/A"),
            "action_suggestions": parsed_results.get("action_suggestions", "N/A"),
        }
    except Exception as e:
        return {"root_cause": f"Failed: {e}", "key_insights": "Failed", "action_suggestions": "Failed"}


# ─── 1) DATA SOURCE & PREVIEW ─────────────────────────────────────────────────
st.title("Low-Code Analytics Platform")
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "telematics_trip_new2.csv"
df_orig = None
if DEFAULT_CSV.exists():
    df_orig = load_csv(str(DEFAULT_CSV))
    st.success(f"✅ Loaded default dataset ({df_orig.shape[0]} rows)")
uploaded = st.file_uploader("📂 Or upload CSV to override default", type="csv")
if uploaded is not None:
    df_orig = load_csv(uploaded)
    st.success(f"✅ Loaded uploaded file with {df_orig.shape[0]} rows")
if df_orig is None or df_orig.empty:
    st.error(f"No dataset available.")
    st.stop()

df = df_orig.copy()
st.dataframe(df.head())
tmp_csv = BASE_DIR / "__tmp_telematics.csv"
df_orig.to_csv(tmp_csv, index=False)
llm_agent_instance = get_csv_agent(str(tmp_csv), llm)

# ─── Sidebar: AI Chatbot ───────────────────────────────────────────────────────
st.sidebar.header("🤖 Ask the AI Bot")
st.sidebar.markdown("- “Which driver had the highest alarm count last week?”\n- “Top 5 buses by mean speed last month?”")
if "history" not in st.session_state: st.session_state.history = []
user_q = st.sidebar.chat_input("Your question…")
if user_q:
    st.session_state.history.append({"role":"user","content":user_q})
    with st.spinner("Thinking…"):
        ans = llm_agent_instance.invoke({"input": user_q})
        ans_content = ans.get('output', 'No answer found.')
        if isinstance(ans_content, AIMessage): ans_content = ans_content.content
        st.session_state.history.append({"role":"assistant","content":ans_content})
for msg in st.session_state.history:
    st.sidebar.chat_message(msg["role"]).write(msg["content"])

# ─── 2) TIME & DATE FILTER ───────────────────────────────────────────────────
st.markdown("---")
st.header("1. Global Data Filters")

col1, col2 = st.columns([1, 2])
with col1:
    time_candidates = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if not time_candidates:
        st.error("No date/time columns found in your data.")
        st.stop()
    time_col = st.selectbox("Select Time column", time_candidates)

with col2:
    date_format_options = {
        "Auto-Detect": "Auto-Detect",
        "DD/MM/YYYY": "%d/%m/%Y",
        "MM/DD/YYYY": "%m/%d/%Y",
        "YYYY-MM-DD": "%Y-%m-%d",
        "YYYY-MM-DD HH:MM:SS": "%Y-%m-%d %H:%M:%S",
        "DD/MM/YYYY HH:MM": "%d/%m/%Y %H:%M",
    }
    selected_format_key = st.selectbox(
        "Specify Date Format",
        options=list(date_format_options.keys()),
        index=1,
        help="Select the format that matches your date column to ensure all data is loaded correctly."
    )
    date_format = date_format_options[selected_format_key]

df = convert_datetime(df, time_col, date_format)
if df is None or df[time_col].isnull().all():
    st.error("Data could not be parsed with the specified date format. Please check the format and your data.")
    st.stop()
df.dropna(subset=[time_col], inplace=True)

min_d_full, max_d_full = df[time_col].dt.date.min(), df[time_col].dt.date.max()
try:
    start_d_default = (pd.to_datetime(max_d_full) - pd.DateOffset(months=3)).date()
except Exception:
    start_d_default = min_d_full

if start_d_default < min_d_full:
    start_d_default = min_d_full

st.info(f"💡 For faster performance, the app has defaulted to showing data from the last 3 months. You can adjust the range using the slider below.")

start_d, end_d = st.slider(
    "Date range",
    min_value=min_d_full, max_value=max_d_full,
    value=(start_d_default, max_d_full),
    format="YYYY-MM-DD"
)
df = filter_date(df, time_col, start_d, end_d)
st.write(f"**Displaying {len(df)} records from {start_d} to {end_d}.**")


# ─── 2b) DYNAMIC COLUMN FILTERS ──────────────────────────────────────────
with st.expander("Show/Hide Column Filters", expanded=True):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    columns_to_filter = st.multiselect("Select columns to add filters for:", options=categorical_cols)
    filters = {}
    if columns_to_filter:
        filter_cols = st.columns(len(columns_to_filter))
        for i, column in enumerate(columns_to_filter):
            with filter_cols[i]:
                unique_values = df[column].dropna().unique()
                selected_values = st.multiselect(f"Filter by {column}", options=sorted(unique_values), key=f'filter_{column}')
                if selected_values:
                    filters[column] = selected_values

original_row_count = len(df)
if filters:
    for column, values in filters.items():
        df = df[df[column].isin(values)]
    st.write(f"Further filtered from {original_row_count} to {len(df)} rows.")

# ─── 3) METRIC & GROUP SELECTION ─────────────────────────────────────────────
st.markdown("---")
st.header("2. Analysis Parameters")
all_metrics = df.select_dtypes(include=np.number).columns.tolist()
default_metric = 'overall_alarm_count'
if default_metric in all_metrics:
    default_metrics_selection = [default_metric] + [m for m in all_metrics if m != default_metric]
else:
    default_metrics_selection = all_metrics

metrics = st.multiselect("Select Numeric Metrics", all_metrics, default=default_metrics_selection[:min(2, len(default_metrics_selection))])
object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
potential_group_cols = [c for c in object_cols if 1 < df[c].nunique() <= 100]
preferred_group_cols = ['bus_no', 'operating_line', 'depot', 'driver_id', 'svc_no']
sorted_group_opts = ['ALL'] + sorted(list(set(preferred_group_cols) & set(potential_group_cols))) + sorted([c for c in potential_group_cols if c not in preferred_group_cols])
group_col = st.selectbox("Group by (optional)", sorted_group_opts)
if group_col == 'ALL' and 'ALL' not in df.columns: df['ALL'] = 'ALL'

# ─── 4) ANALYSIS SECTION DROPDOWN ────────────────────────────────────────────
st.markdown("---")
st.header("3. Choose Analysis Section")
section_options = ["Operations Overview", "Statistical Anomaly Detection", "Driver Profiling & Analysis", "Temporal Clustering","Motif Mining", "Correlation", "Clustering", "Table & Alerts"]
section = st.selectbox("Analysis Section", section_options)

if "anomaly_list" not in st.session_state: st.session_state.anomaly_list = []
if 'driver_summary' not in st.session_state: st.session_state.driver_summary = pd.DataFrame()


# ─── Operations Overview ────────────────────────────────────────────────────────────────
if section == "Operations Overview":
    st.header("⭐ Bus Operations Command Center")
    st.markdown("An AI-powered command center for fleet safety and performance, designed by a bus telematics expert for operational leaders.")

    # --- 0. Data Preparation & Validation ---
    
    # AI FIX: The following hardcoded column definitions and the entire redundant 
    # date conversion block have been REMOVED. The app will now correctly use the 
    # 'time_col' and 'date_format' selected in the "1. Global Data Filters" section.
    # This makes the behavior consistent across all analysis sections.
    
    alarm_col = "overall_alarm_count"
    driver_col = "driver_id"
    bus_col = "bus_no"
    depot_col = "depot_id"
    svc_col = "svc_no"
    distance_col = "total_distance_km" 

    # A. Check for the absolute minimum required columns
    # The 'time_col' variable is now correctly inherited from the global filter section.
    required_cols = [time_col, alarm_col]
    if df.empty or not all(col in df.columns for col in required_cols):
        st.warning(f"⚠️ Data is empty or missing essential columns ('{time_col}', '{alarm_col}'). Please adjust global filters or upload a valid dataset.")
        st.stop()
        
    # B. The redundant date conversion block that was here has been removed.
    
    # C. Guarantee correct data types for plotting and analysis
    for col in [driver_col, bus_col, depot_col, svc_col]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # --- 1. Executive KPI Scorecards ---
    st.subheader("📊 Executive KPI Scorecard")
    total_trips = len(df)
    total_alarms = df[alarm_col].sum()
    alarms_per_trip = total_alarms / total_trips if total_trips > 0 else 0
    
    if distance_col in df.columns and df[distance_col].sum() > 0:
        total_distance = df[distance_col].sum()
        alarms_per_100km = (total_alarms / total_distance) * 100
    else:
        alarms_per_100km = "N/A"

    alarm_mean = df[alarm_col].mean()
    alarm_std = df[alarm_col].std()
    high_risk_threshold = alarm_mean + (2 * alarm_std)
    high_risk_trips_count = df[df[alarm_col] > high_risk_threshold].shape[0]
    high_risk_trip_percent = (high_risk_trips_count / total_trips) * 100 if total_trips > 0 else 0

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Total Trips", f"{total_trips:,}")
    kpi_cols[1].metric("Total Alarms", f"{int(total_alarms):,}")
    kpi_cols[2].metric("Avg Alarms / Trip", f"{alarms_per_trip:.2f}")
    kpi_cols[3].metric("High-Risk Trips", f"{high_risk_trips_count:,}")
    kpi_cols[4].metric("High-Risk Trip %", f"{high_risk_trip_percent:.2f}%")
    st.markdown("---")

    # --- 2. Daily Alarm Trend ---
     # --- Alarm Count Trend with Time Series Prediction ---
    st.subheader("📈 Alarm Count Trend & Prediction")
    if 'overall_alarm_count' in df.columns:
        daily_alarms = df.set_index(time_col).resample('D')['overall_alarm_count'].sum().reset_index()
        daily_alarms.dropna(inplace=True)

        if len(daily_alarms) > 2:
            # Prediction
            model = LinearRegression()
            X = np.array(range(len(daily_alarms))).reshape(-1, 1)
            y = daily_alarms['overall_alarm_count']
            model.fit(X, y)
            
            future_days = 30
            future_X = np.array(range(len(daily_alarms), len(daily_alarms) + future_days)).reshape(-1, 1)
            predictions = model.predict(future_X)
            
            future_dates = pd.date_range(start=daily_alarms[time_col].max() + pd.Timedelta(days=1), periods=future_days)
            pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Alarms': predictions})

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=daily_alarms[time_col], y=daily_alarms['overall_alarm_count'], mode='lines', name='Historical Daily Alarms'))
            fig_trend.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Predicted Alarms'], mode='lines', name='Predicted Trend', line=dict(dash='dash')))
            fig_trend.update_layout(title="Daily Alarm Count with 30-Day Prediction", xaxis_title="Date", yaxis_title="Total Alarms")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Not enough data points to generate a trend prediction.")
    else:
        st.warning("'overall_alarm_count' not found, cannot display trend.")

    # --- 3. High-Level Alarm Summary ---
    st.subheader("🔬 Alarm Hotspot Analysis")
    col1, col2 = st.columns(2)
    with col1:
        alarm_type_cols = [c for c in df.columns if 'alarm_' in c and c != alarm_col and df[c].dtype in ['int64', 'float64']]
        if alarm_type_cols:
            alarm_type_totals = df[alarm_type_cols].sum().sort_values(ascending=False).reset_index()
            alarm_type_totals.columns = ['Alarm Type', 'Total Count']
            fig_types = px.bar(alarm_type_totals, x='Alarm Type', y='Total Count', title="Total Alarms by Specific Type")
            fig_types.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_types, use_container_width=True)

    with col2:
        depot_alarms_df = df.groupby(depot_col)[alarm_col].sum().sort_values(ascending=False).reset_index()
        fig_depot = px.bar(depot_alarms_df, x=depot_col, y=alarm_col, title="Total Alarms by Depot")
        st.plotly_chart(fig_depot, use_container_width=True)

    st.write("#### Hierarchical Alarm View: Depot ➔ Service No. ➔ Bus")
    df_tree = df.groupby([depot_col, svc_col, bus_col]).agg(total_alarms=(alarm_col, 'sum')).reset_index()
    df_tree = df_tree[df_tree['total_alarms'] > 0]
    fig_tree = px.treemap(df_tree, path=[depot_col, svc_col, bus_col], values='total_alarms', title='Alarm Distribution Across Depots, Services, and Buses', color='total_alarms', color_continuous_scale='YlOrRd')
    fig_tree.update_traces(textinfo="label+value", hoverinfo="label+value+percent parent")
    st.plotly_chart(fig_tree, use_container_width=True)
    st.markdown("---")
    
    # --- 4. Top 10 High-Risk Drivers & Buses ---
    st.subheader("👤 Top 10 High-Risk Drivers & Buses (by Alarms Per Trip)")
    col1, col2 = st.columns(2)
    with col1:
        driver_summary = df.groupby(driver_col).agg(total_alarms=(alarm_col, 'sum'), total_trips=(time_col, 'count')).reset_index()
        driver_summary['alarms_per_trip'] = driver_summary['total_alarms'] / driver_summary['total_trips']
        top_drivers = driver_summary.sort_values('alarms_per_trip', ascending=False).head(10)
        
        fig_drivers = px.bar(top_drivers.sort_values('alarms_per_trip', ascending=True), x='alarms_per_trip', y=driver_col, orientation='h', title="Top 10 Drivers by Alarms Per Trip")
        fig_drivers.update_yaxes(type='category')
        st.plotly_chart(fig_drivers, use_container_width=True)

    with col2:
        bus_summary = df.groupby(bus_col).agg(total_alarms=(alarm_col, 'sum'), total_trips=(time_col, 'count')).reset_index()
        bus_summary['alarms_per_trip'] = bus_summary['total_alarms'] / bus_summary['total_trips']
        top_buses = bus_summary.sort_values('alarms_per_trip', ascending=False).head(10)
        
        fig_buses = px.bar(top_buses.sort_values('alarms_per_trip', ascending=True), x='alarms_per_trip', y=bus_col, orientation='h', title="Top 10 Buses by Alarms Per Trip")
        fig_buses.update_yaxes(type='category')
        st.plotly_chart(fig_buses, use_container_width=True)
    st.markdown("---")
    
    # --- 5. AI-Powered Debrief ---
    st.subheader("💡 AI-Powered Operational Debrief")
    with st.spinner("AI Analyst is synthesizing findings to uncover hidden insights..."):
        # Create a rich context for the LLM
        top_driver_info = driver_summary.sort_values('alarms_per_trip', ascending=False).head(1).to_dict('records')[0]
        top_bus_info = bus_summary.sort_values('alarms_per_trip', ascending=False).head(1).to_dict('records')[0]
        top_alarm_types_info = df[alarm_type_cols].sum().nlargest(3).to_dict() if alarm_type_cols else "N/A"

        summary_context = f"""
        - Overall Performance: Total Trips={total_trips}, Total Alarms={int(total_alarms)}, Alarms/Trip={alarms_per_trip:.2f}, High-Risk Trip Rate={high_risk_trip_percent:.2f}%.
        - Highest Risk Driver (by Alarms/Trip): {top_driver_info}.
        - Highest Risk Bus (by Alarms/Trip): {top_bus_info}.
        - Top 3 Alarm Types by Volume: {top_alarm_types_info}.
        - Alarms by Depot: {df.groupby(depot_col)[alarm_col].sum().to_dict()}.
        """

        insight_prompt = f"""
        You are a world-class AI Bus Telematics Analyst with infinite experience, briefing the Head of Operations. Your analysis must be sharp, insightful, and actionable.

        Here is the data summary for the period:
        {summary_context}

        Structure your response using the following markdown headings precisely:

        ### AI Alarm Narrative
        Write a concise, executive-level story about the fleet's current alarm posture. What is the main theme? Is risk concentrated or widespread?

        ### Hidden Insights & Correlations
        This is the most critical part. Dig deep to find non-obvious connections that require action. Explicitly state your reasoning. For example:
        - **Driver-Bus-Route Analysis:** Does the highest-risk driver operate the highest-risk bus? If not, do multiple good drivers have high alarms on that one bus (suggesting a vehicle fault)? Do the top-risk drivers/buses concentrate on specific service numbers (`svc_no`)?
        - **Depot-Specific Patterns:** Is a specific alarm type (e.g., harsh braking) dominant in one depot but not others? This could point to route topography issues or training gaps.
        - **Data Quality Red Flags:** Are there drivers with a very high alarm rate but only a few trips? Mention this as a potential data anomaly that could be skewing the results.

        ### Recommended Action Plan
        Generate a markdown table with three columns: 'Priority' (High, Medium, Low), 'Recommended Action', and 'Rationale'. Provide the top 3-5 most critical, concrete actions the team should take now.
        """
        try:
            insight_response = llm.invoke(insight_prompt).content
            st.markdown(insight_response)
        except Exception as e:
            st.error(f"Failed to generate AI insights: {e}")


# ─── RE-ARCHITECTED: Statistical Anomaly Detection ───────────────────────────────────
# ─── RE-ARCHITECTED: Statistical Anomaly Detection ───────────────────────────────────
elif section == "Statistical Anomaly Detection":
    st.header("🔬 Advanced Anomaly Detection")
    st.markdown("""
    This analysis uses advanced statistical methods to uncover operational anomalies.
    1.  **Dynamic Daily Analysis (Rolling Z-Score):** We identify days where the *average* performance significantly deviates from a recent historical window (e.g., last 7 days).
    2.  **Detailed Trip-Level Analysis:** We drill down to individual trips that breach a performance threshold, providing a detailed AI diagnosis of all alarm events for that specific trip.
    """)
    st.session_state.anomaly_list = []

    if df.empty or not metrics:
        st.info("No data available for the current filter settings. Please adjust filters or select metrics.")
    else:
        metric = metrics[0]
        if len(metrics) > 1:
            st.info(f"Analysis is focused on the first selected metric: **{metric}**. Please deselect other metrics for a different analysis.")

        # --- 1. Dynamic Daily Average Analysis (Rolling Z-Score) ---
        st.subheader(f"1. Dynamic Daily Average Analysis for: {metric}")
        df_day = daily_aggregate(df, time_col, 'ALL', [metric])

        if df_day.empty or len(df_day) < 7 or df_day[metric].isnull().all():
            st.warning("Could not generate daily aggregated data for analysis. Need at least 7 days of data. Check your filters.")
        else:
            col1, col2 = st.columns(2)
            rolling_window = col1.slider("Rolling Window (Days)", min_value=3, max_value=30, value=7)
            z_score_threshold = col2.slider("Anomaly Threshold (Z-Score)", min_value=1.5, max_value=4.0, value=2.5, step=0.1)

            df_day['rolling_mean'] = df_day[metric].rolling(window=rolling_window, min_periods=1).mean()
            df_day['rolling_std'] = df_day[metric].rolling(window=rolling_window, min_periods=1).std()
            df_day['upper_bound'] = df_day['rolling_mean'] + (df_day['rolling_std'] * z_score_threshold)
            df_day['lower_bound'] = df_day['rolling_mean'] - (df_day['rolling_std'] * z_score_threshold)
            df_day.fillna(0, inplace=True)

            df_day['anomaly'] = (df_day[metric] > df_day['upper_bound']) & (df_day['upper_bound'] > 0)
            anomalous_days = df_day[df_day['anomaly']]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_day[time_col], y=df_day['upper_bound'], fill=None, mode='lines', line_color='rgba(255,179,0,0.2)', name='Upper Bound'))
            fig.add_trace(go.Scatter(x=df_day[time_col], y=df_day['lower_bound'], fill='tonexty', mode='lines', line_color='rgba(255,179,0,0.2)', name='Dynamic Normal Range'))
            fig.add_trace(go.Scatter(x=df_day[time_col], y=df_day[metric], name=f'Daily Avg {metric}', mode='lines+markers', line=dict(color='#636EFA')))
            fig.add_trace(go.Scatter(x=df_day[time_col], y=df_day['rolling_mean'], name=f'{rolling_window}-Day Rolling Avg', mode='lines', line=dict(color='orange', dash='dash')))
            if not anomalous_days.empty:
                fig.add_trace(go.Scatter(x=anomalous_days[time_col], y=anomalous_days[metric], name='Anomalous Day', mode='markers', marker=dict(color='red', size=12, symbol='x')))
            fig.update_layout(title=f'Daily Average of {metric} with Dynamic Anomaly Bands', xaxis_title='Date', yaxis_title=f'Average {metric}')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### AI Fleet Performance Narrative")
            with st.spinner("AI is analyzing the fleet's performance for this period..."):
                driver_perf = df.groupby('driver_id')[metric].mean().sort_values(ascending=False).nlargest(5).to_dict() if 'driver_id' in df else {}
                bus_perf = df.groupby('bus_no')[metric].mean().sort_values(ascending=False).nlargest(5).to_dict() if 'bus_no' in df else {}
                svc_perf = df.groupby('svc_no')[metric].mean().sort_values(ascending=False).nlargest(5).to_dict() if 'svc_no' in df else {}
                depot_perf = df.groupby('depot_id')[metric].mean().sort_values(ascending=False).to_dict() if 'depot_id' in df else {}

                period_prompt = f"""
                As a master AI Operations Analyst, provide a fleet performance narrative for the selected period, focusing on the '{metric}' metric.
                The fleet average was generally stable, with no days breaching the {z_score_threshold}-sigma anomaly threshold.
                However, a deeper look at the data is required to spot underlying risk concentrations.

                **Performance Data (Average '{metric}' per trip):**
                - **Top 5 Riskiest Drivers:** {driver_perf}
                - **Top 5 Riskiest Buses:** {bus_perf}
                - **Top 5 Riskiest Service Routes:** {svc_perf}
                - **Performance by Depot:** {depot_perf}

                Based on this, generate:
                1.  **Executive Summary:** A brief narrative describing the fleet's overall risk posture. Is the risk concentrated or widespread?
                2.  **Hidden Risk Insight:** What is the most important non-obvious insight? Connect the dots between the riskiest drivers, buses, routes, and depots.
                3.  **Proactive Recommendation:** What is the single most important proactive step the ops team should take, even without a major anomaly?
                """
                try:
                    period_insight = llm.invoke(period_prompt).content
                    st.info(period_insight)
                except Exception as e:
                    st.error(f"Failed to generate AI period summary: {e}")

        # --- 2. Detailed Trip-Level Anomaly Analysis ---
        st.markdown("---")
        st.subheader(f"2. Detailed Trip-Level Anomaly Analysis for: {metric}")
        
        col1_trip, col2_trip = st.columns([2, 1])
        default_threshold = df[metric].quantile(0.98)
        threshold = col1_trip.number_input(f"Set anomaly threshold for '{metric}':", value=float(default_threshold), step=1.0)
        
        df_trip_analysis = df.copy()
        df_trip_analysis['anomaly'] = df_trip_analysis[metric] > threshold
        anomalous_trips = df_trip_analysis[df_trip_analysis['anomaly']].sort_values(by=metric, ascending=False)
        
        num_to_show = col2_trip.number_input("Number of anomalies to display:", min_value=1, max_value=len(anomalous_trips) if not anomalous_trips.empty else 1, value=min(5, len(anomalous_trips) if not anomalous_trips.empty else 1))

        fig_trip = px.scatter(df_trip_analysis, x=time_col, y=metric, color='anomaly', color_discrete_map={True: 'red', False: '#636EFA'}, title=f'Individual Trip Analysis for {metric}')
        fig_trip.add_hline(y=threshold, line_dash="dash", annotation_text="Anomaly Threshold", annotation_position="bottom right")
        st.plotly_chart(fig_trip, use_container_width=True)

        if not anomalous_trips.empty:
            st.markdown(f"#### AI Deep-Dive on Top {num_to_show} Anomalous Trips")
            for _, trip_row in anomalous_trips.head(num_to_show).iterrows():
                trip_time = trip_row[time_col].strftime('%Y-%m-%d %H:%M')
                with st.expander(f"**Trip on {trip_time}**: {metric} of {trip_row[metric]:.2f}"):
                    triggered_alarms = {k: int(v) for k, v in trip_row.items() if k.startswith('alarm_') and v > 0}
                    trip_context = { "Driver": trip_row.get('driver_id', 'N/A'), "Bus": trip_row.get('bus_no', 'N/A'), "Route": trip_row.get('svc_no', 'N/A'), "Depot": trip_row.get('depot_id', 'N/A'), "Primary Anomaly": f"{metric} at {trip_row[metric]:.2f}", "Triggered Alarms": triggered_alarms or "None" }
                    
                    # AI FIX: Corrected the KeyError by using the correct key 'Primary Anomaly'.
                    trip_prompt = f"""
                    As a master AI Operations Analyst, perform a root cause analysis on this high-risk trip.
                    **Trip Context:**
                    - **Driver/Bus/Route/Depot:** {trip_context['Driver']} / {trip_context['Bus']} / {trip_context['Route']} / {trip_context['Depot']}
                    - **Primary Anomaly:** {trip_context['Primary Anomaly']}
                    - **All Triggered Alarms on this Trip:** {trip_context['Triggered Alarms']}
                    Provide a precise debrief for the Operations Manager:
                    1.  **Precise Root Cause:** Based on the combination of triggered alarms, what is the most likely event sequence?
                    2.  **Operational Impact:** What is the immediate safety or performance risk?
                    3.  **Clear Action Plan:** What is the exact, single next step for the Ops team?
                    """
                    try:
                        trip_insight = llm.invoke(trip_prompt).content
                        st.info(trip_insight)
                    except Exception as e:
                        st.error(f"Failed to generate AI insight for the trip: {e}")
        else:
            st.success(f"No trips found above the threshold of {threshold}.")

        # --- 3. Final Action Summary Table ---
        st.markdown("---")
        st.subheader("3. Section Summary & Action Plan")
        if not anomalous_trips.empty:
            with st.spinner("Generating final action plan..."):
                st.markdown("**Key Takeaway**: While daily averages may appear stable, trip-level analysis reveals specific, high-risk events that require immediate follow-up. These isolated incidents are often precursors to larger problems and represent the most effective area for targeted intervention.")
                
                action_df = anomalous_trips.copy()
                
                def get_alarm_summary(row):
                    alarms = {k.replace('alarm_', '').replace('_count', ''): int(v) for k, v in row.items() if k.startswith('alarm_') and v > 0}
                    return ', '.join([f"{name.replace('_', ' ').title()} ({count})" for name, count in alarms.items()]) if alarms else "N/A"
                
                action_df['Key Alarms'] = action_df.apply(get_alarm_summary, axis=1)
                
                cols_to_show = [time_col, 'driver_id', 'bus_no', 'svc_no', 'depot_id', metric, 'Key Alarms']
                final_cols = [c for c in cols_to_show if c in action_df.columns]
                
                st.markdown("#### High-Priority Investigation List")
                st.dataframe(action_df[final_cols])
                download_df(action_df[final_cols], "trip_investigation_list.csv")
        else:
            st.success("Excellent! No high-risk trips were flagged for investigation in this period.")
# ─── Temporal Risk Pattern Mining ───────────────────────────────────────────────────
elif section == "Temporal Clustering":
    st.header("🕒 Temporal Risk Pattern Mining")
    st.markdown("""
    Pinpoint *when* and *why* incidents occur to reveal operational risk patterns:
    1. **Risk vs. Continuous Driving Duration:** Do fatigue or harsh-braking spikes emerge after certain hours?
    2. **Incident Hotspots:** Which day-of-week & hour-of-day combinations are the riskiest?
    3. **Weekday vs. Weekend:** Are there different risk profiles for weekdays versus weekends?
    """)

    if df.empty or not metrics:
        st.info("No data: adjust filters or select metrics.")
    else:
        # AI FIX: The logic to find time columns is now more robust.
        # It now correctly finds columns with 'date', 'time', or 'dt' (for datetime).
        time_cols_options = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'dt' in c.lower()]

        if not time_cols_options:
            st.error("No potential date/time columns were detected in the data.")
            st.stop()
            
        st.info(f"Using globally selected time column: **{time_col}**. You can override it below for this section's analysis.")
        
        try:
            default_index = time_cols_options.index(time_col)
        except ValueError:
            default_index = 0

        temporal_time_col = st.selectbox(
            "Select Event Timestamp Column for Temporal Analysis",
            options=time_cols_options,
            index=default_index
        )

        alarm_cols = [c for c in df.columns if 'alarm' in c.lower() and c.endswith('_count')]
        incident_metric = st.selectbox("Select Incident Metric to Analyze", alarm_cols or metrics)

        # --- Data Enrichment ---
        tmp = df.copy()
        tmp[temporal_time_col] = pd.to_datetime(tmp[temporal_time_col], errors='coerce')
        tmp.dropna(subset=[temporal_time_col, incident_metric], inplace=True)
        tmp = tmp[tmp[incident_metric] > 0] 

        if tmp.empty:
            st.warning(f"No incidents recorded for the metric `{incident_metric}` in the selected data range.")
        else:
            tmp['hour']       = tmp[temporal_time_col].dt.hour
            tmp['weekday']    = tmp[temporal_time_col].dt.day_name()
            tmp['is_weekend'] = tmp[temporal_time_col].dt.weekday >= 5

            # --- 1. Risk vs. Driving Duration ---
            st.subheader("1. Risk vs. Continuous Driving Duration")
            dur_cols = [c for c in df.columns if 'dur' in c.lower()]
            if not dur_cols:
                st.warning("No duration column found for this analysis.")
            else:
                dur_col = st.selectbox("Select Driving Duration (mins)", dur_cols)
                if dur_col in tmp.columns:
                    d = tmp.copy()
                    d['hours'] = pd.to_numeric(d[dur_col], errors='coerce') / 60
                    d = d.dropna(subset=['hours'])
                    if not d.empty:
                        max_h = int(d['hours'].max()) + 1
                        d['hour_bin'] = pd.cut(d['hours'], bins=range(max_h + 1), right=False,
                                            labels=[f"{i}-{i+1}h" for i in range(max_h)])
                        counts = d.groupby('hour_bin', observed=False).size().reset_index(name='count')
                        fig1 = px.bar(counts, x='hour_bin', y='count',
                                    title=f"{incident_metric} Incidents vs. Driving Duration",
                                    labels={'hour_bin': 'Continuous Driving (hrs)', 'count': 'Number of Incidents'})
                        st.plotly_chart(fig1, use_container_width=True)

                        # --- NEW: Enhanced AI Insights for Duration Patterns ---
                        with st.expander("**AI-Powered Analysis on Peak Duration Risk**", expanded=True):
                             with st.spinner("AI is analyzing duration-based risk..."):
                                if not counts.empty:
                                    peak = counts.loc[counts['count'].idxmax()]
                                    peak_bin_label = peak['hour_bin']
                                    
                                    # Filter data to the peak risk duration
                                    peak_duration_trips = d[d['hour_bin'] == peak_bin_label]
                                    
                                    # Extract rich context
                                    top_drivers_dur = peak_duration_trips['driver_id'].value_counts().nlargest(3).to_dict() if 'driver_id' in peak_duration_trips else 'N/A'
                                    top_buses_dur = peak_duration_trips['bus_no'].value_counts().nlargest(3).to_dict() if 'bus_no' in peak_duration_trips else 'N/A'
                                    top_routes_dur = peak_duration_trips['svc_no'].value_counts().nlargest(3).to_dict() if 'svc_no' in peak_duration_trips else 'N/A'

                                    duration_context = f"""
                                    - **Peak Risk Window:** After **{peak_bin_label}** of continuous driving.
                                    - **Incident Count in Window:** {peak['count']} incidents.
                                    - **Key Drivers in this Window:** {top_drivers_dur}
                                    - **Key Buses in this Window:** {top_buses_dur}
                                    - **Key Routes in this Window:** {top_routes_dur}
                                    """

                                    duration_prompt = f"""
                                    As a world-class AI Fleet Safety Analyst, analyze the operational risk associated with continuous driving duration.

                                    **Data Context:**
                                    {duration_context}

                                    Based on this data, provide a sharp and actionable analysis:

                                    ### Peak Fatigue Hypothesis
                                    What is the most likely reason for the spike in incidents during this time window? Is it driver fatigue, route complexity on longer trips, or specific vehicle issues that appear over time?

                                    ### Key Contributing Factors
                                    Are the incidents concentrated with specific drivers, buses, or routes? Explain the most significant connection (e.g., "The risk is highest on Route 7 with Driver 123, suggesting a combination of a demanding route and a driver who may need fatigue management coaching.")

                                    ### Actionable Mitigation Plan
                                    Provide a clear, prioritized action plan for the operations manager.
                                    1. **Immediate Action:** (e.g., Review schedules for the top drivers identified).
                                    2. **Secondary Investigation:** (e.g., Cross-reference the top routes with road conditions or traffic data).
                                    3. **Preventative Measure:** (e.g., Implement mandatory break reminders after X hours of continuous driving).
                                    """
                                    try:
                                        duration_insight = llm.invoke(duration_prompt).content
                                        st.markdown(duration_insight)
                                    except Exception as e:
                                        st.error(f"Failed to generate AI analysis for duration risk: {e}")
                                else:
                                    st.info("No duration data available to analyze.")

 
            # --- 2. Weekday vs. Weekend Analysis ---
            st.subheader("2. Weekday vs. Weekend Incident Analysis")
            weekend_counts = tmp['is_weekend'].value_counts().reset_index()
            weekend_counts['is_weekend'] = weekend_counts['is_weekend'].map({False: 'Weekday', True: 'Weekend'})
            fig_wknd = px.pie(weekend_counts, names='is_weekend', values='count', title=f'Total Incidents: Weekday vs. Weekend',
                              color_discrete_sequence=['#636EFA', '#EF553B'])
            st.plotly_chart(fig_wknd, use_container_width=True)
            with st.expander("AI Insights: Weekday vs. Weekend Patterns"):
                 st.info("This chart shows the distribution of incidents between weekdays and weekends. A significant imbalance might suggest that operational risks, such as traffic congestion or driver schedules, differ greatly between these periods.")


            # --- 3. Day/Hour Heatmap ---
            st.subheader("3. Incident Hotspots by Day & Hour")
            heat = tmp.groupby(['weekday', 'hour']).size().unstack(fill_value=0)
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            heat = heat.reindex(days_order).fillna(0)
            for h in range(24):
                if h not in heat.columns: heat[h] = 0
            heat = heat[sorted(heat.columns)]
            
            fig2 = go.Figure(data=go.Heatmap(
                z=heat.values, x=heat.columns, y=heat.index,
                colorscale='Reds'
            ))
            fig2.update_layout(title=f"Heatmap of {incident_metric} Incidents", xaxis_title="Hour of Day", yaxis_title="Day of Week")
            st.plotly_chart(fig2, use_container_width=True)

            # --- AI-Powered Deep-Dive on Hotspots ---
            with st.expander("**AI-Powered Deep-Dive on Hotspot**", expanded=True):
                if heat.empty:
                    st.warning("No data to analyze for hotspots.")
                else:
                    with st.spinner("AI is analyzing the operational hotspot..."):
                        # Find the single busiest hour (hotspot)
                        idx = np.unravel_index(np.argmax(heat.values), heat.shape)
                        hotspot_day, hotspot_hour = heat.index[idx[0]], heat.columns[idx[1]]
                        incident_count = heat.iloc[idx]

                        # Filter the dataframe to get all incidents from that specific hotspot
                        hotspot_trips = tmp[(tmp['weekday'] == hotspot_day) & (tmp['hour'] == hotspot_hour)]

                        # --- NEW: Extract deep context from the hotspot data ---
                        top_drivers = hotspot_trips['driver_id'].value_counts().nlargest(3).to_dict() if 'driver_id' in hotspot_trips else 'N/A'
                        top_buses = hotspot_trips['bus_no'].value_counts().nlargest(3).to_dict() if 'bus_no' in hotspot_trips else 'N/A'
                        top_depots = hotspot_trips['depot_id'].value_counts().nlargest(3).to_dict() if 'depot_id' in hotspot_trips else 'N/A'
                        top_svcs = hotspot_trips['svc_no'].value_counts().nlargest(3).to_dict() if 'svc_no' in hotspot_trips else 'N/A'

                        hotspot_context = f"""
                        - **Hotspot Identified:** The fleet's riskiest time is **{hotspot_day} at {hotspot_hour}:00**, with a peak of **{incident_count}** incidents.
                        - **Top Contributing Drivers:** {top_drivers}
                        - **Top Contributing Buses:** {top_buses}
                        - **Top Contributing Depots:** {top_depots}
                        - **Top Contributing Service Numbers:** {top_svcs}
                        """

                        # --- NEW: Enhanced LLM Prompt ---
                        hotspot_prompt = f"""
                        As a world-class AI Bus Operations Analyst with infinite experience, you are tasked with analyzing a critical operational hotspot.

                        **Hotspot Context:**
                        {hotspot_context}

                        Your analysis must be sharp, data-driven, and actionable for the Head of Operations. Structure your response using the following markdown headings:

                        ### Hotspot Diagnosis
                        Based on the data, what is the most likely root cause of this recurring hotspot? Is it a driver issue, a vehicle problem, a route-specific challenge, or a combination? *For example, if the same drivers appear regardless of bus, it's a driver issue. If a single bus has high incidents with multiple good drivers, it's a vehicle issue. If incidents are spread across drivers/buses but on one service number, it's a route issue.*

                        ### Key Hidden Insights
                        What non-obvious patterns or correlations can you find?
                        - Is there a link between the top drivers and the top buses?
                        - Do the top service numbers originate from the top depots, suggesting a depot-level problem (e.g., training, vehicle allocation)?
                        - Is there any other data point that suggests a hidden relationship?

                        ### Actionable Triage Plan for Ops
                        Provide a concise, prioritized list of the top 3 actions the operations team must take immediately to mitigate this risk. Be specific.
                        1. **Immediate Action:**
                        2. **Secondary Investigation:**
                        3. **Long-Term Mitigation:**
                        """
                        try:
                            hotspot_insight = llm.invoke(hotspot_prompt).content
                            st.markdown(hotspot_insight)
                        except Exception as e:
                            st.error(f"Failed to generate AI analysis for hotspot: {e}")
# ─── Correlation ─────────────────────────────────────────────────────────────
elif section == "Correlation":
    st.header("📈 Predictive Correlation Analysis")
    st.markdown("""
    This analysis uncovers relationships between different metrics. Understanding these connections can help predict potential issues.
    For example, does high speed consistently lead to lower fuel efficiency? Does increased harsh braking correlate with specific alarm types?
    """)
    if df.empty or len(metrics) < 2:
        st.info("Please select at least two numeric metrics and ensure data is available for correlation.")
    else:
        # --- 1. Correlation Matrix Calculation & Visualization ---
        all_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_cols = st.multiselect(
            "Select numeric columns to include in correlation matrix (≥2)",
            all_cols,
            default=all_cols[:min(8, len(all_cols))] # Default to a reasonable number
        )
        if len(selected_cols) < 2:
            st.info("Please select at least two columns for correlation.")
        else:
            corr = df[selected_cols].corr()
            fig = px.imshow(
                corr,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                aspect='auto',
                origin='lower',
                labels={'x':'Feature','y':'Feature','color':'Correlation'}
            )
            fig.update_layout(title='Correlation Matrix Heatmap')
            st.plotly_chart(fig, use_container_width=True)

            # --- 2. Interactive Strong Correlation Finder ---
            st.subheader("🔍 Find and Analyze Strong Correlations")

            # Dynamic threshold slider
            corr_threshold = st.slider(
                "Select Minimum Correlation Strength (|r|)",
                min_value=0.1, max_value=1.0, value=0.6, step=0.05
            )

            # Find all unique correlation pairs above the threshold
            strong_corrs = corr.abs().stack().reset_index().rename(columns={0: 'abs_corr'})
            # Remove self-correlation
            strong_corrs = strong_corrs[strong_corrs['level_0'] != strong_corrs['level_1']]
            # Filter by threshold
            strong_corrs = strong_corrs[strong_corrs['abs_corr'] >= corr_threshold]
            # Remove duplicates (e.g., (A,B) is the same as (B,A)) by sorting the pair names
            strong_corrs['sorted_pair'] = strong_corrs.apply(lambda row: tuple(sorted((row['level_0'], row['level_1']))), axis=1)
            strong_corrs = strong_corrs.drop_duplicates(subset='sorted_pair').drop(columns='sorted_pair')

            if strong_corrs.empty:
                st.info(f"No correlation pairs found with an absolute strength of {corr_threshold} or higher.")
            else:
                # Add the original signed correlation value back
                strong_corrs['r'] = strong_corrs.apply(lambda row: corr.loc[row['level_0'], row['level_1']], axis=1)

                # --- 3. AI-Powered Insights for Operations ---
                st.subheader("📝 Operational Insights from Correlations")

                # Separate positive and negative correlations for clarity
                positive_corrs = strong_corrs[strong_corrs['r'] > 0].sort_values(by='r', ascending=False)
                negative_corrs = strong_corrs[strong_corrs['r'] < 0].sort_values(by='r', ascending=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Positive Correlations (Metrics move together)")
                    if not positive_corrs.empty:
                        for _, row in positive_corrs.iterrows():
                            feat1, feat2, r_val = row['level_0'], row['level_1'], row['r']
                            with st.expander(f"**{feat1} & {feat2}** (r = {r_val:.2f})"):
                                st.markdown(f"**Observation:** As `{feat1}` increases, `{feat2}` also tends to increase.")

                                # AI Insight
                                insight_prompt = f"""
                                As an expert AI Bus Operations Analyst, explain the operational significance of a strong positive correlation (r={r_val:.2f}) between `{feat1}` and `{feat2}`.

                                Provide:
                                1. **Operational Hypothesis:** What is the most likely real-world reason for this relationship? (e.g., "Higher average speeds naturally lead to longer trip distances covered.")
                                2. **What to Watch For:** What potential problem could this indicate if the values become extreme? (e.g., "If both are excessively high, it could point to drivers speeding on long, straight routes, increasing risk and fuel consumption.")
                                """
                                try:
                                    insight_response = llm.invoke(insight_prompt)
                                    st.info(insight_response.content)
                                except Exception as e:
                                    st.error(f"Could not get AI insight: {e}")
                    else:
                        st.write("No strong positive correlations found.")

                with col2:
                    st.markdown("#### Negative Correlations (Metrics move opposite)")
                    if not negative_corrs.empty:
                        for _, row in negative_corrs.iterrows():
                            feat1, feat2, r_val = row['level_0'], row['level_1'], row['r']
                            with st.expander(f"**{feat1} & {feat2}** (r = {r_val:.2f})"):
                                st.markdown(f"**Observation:** As `{feat1}` increases, `{feat2}` tends to decrease.")

                                # AI Insight
                                insight_prompt = f"""
                                As an expert AI Bus Operations Analyst, explain the operational significance of a strong negative correlation (r={r_val:.2f}) between `{feat1}` and `{feat2}`.

                                Provide:
                                1. **Operational Hypothesis:** What is the most likely real-world reason for this inverse relationship? (e.g., "Higher fuel efficiency (km/l) naturally leads to lower overall fuel consumption for the same trip.")
                                2. **What to Watch For:** What potential problem could this indicate? (e.g., "A bus showing low fuel efficiency (low `{feat1}`) and high fuel consumption (high `{feat2}`) for its mileage could have an engine or tire pressure issue.")
                                """
                                try:
                                    insight_response = llm.invoke(insight_prompt)
                                    st.info(insight_response.content)
                                except Exception as e:
                                    st.error(f"Could not get AI insight: {e}")
                    else:
                        st.write("No strong negative correlations found.")

                # --- 4. Visual Analysis of Top Correlations ---
                st.subheader("Visual Analysis of Top 3 Strongest Correlations")
                top_3_corrs = strong_corrs.nlargest(3, 'abs_corr')

                for idx, row in top_3_corrs.iterrows():
                    feat1, feat2 = row['level_0'], row['level_1']
                    st.markdown(f"#### `{feat1}` vs. `{feat2}`")
                    scatter_fig = px.scatter(
                        df,
                        x=feat1,
                        y=feat2,
                        trendline="ols", # Ordinary Least Squares trendline
                        title=f"Relationship between {feat1} and {feat2}",
                        opacity=0.6,
                        color_discrete_sequence=['#636EFA']
                    )
                    st.plotly_chart(scatter_fig, use_container_width=True)



# ─── Motif Mining ────────────────────────────────────────────────────────────
# ─── Motif Mining ────────────────────────────────────────────────────────────
elif section == "Motif Mining":
    st.header("🔍 Alarm Sequence Pattern & Root Cause Analysis")
    st.markdown("""
    This analysis identifies the most frequent sequences of alarms that occur together within a single trip.
    Understanding these patterns (e.g., "Over Speed" followed by "Harsh Brake") helps uncover complex operational risks and their root causes.
    """)
    
    alarm_types = [c for c in df.columns if c.startswith('alarm_') and c.endswith('_count')]
    
    if len(alarm_types) < 2:
        st.warning("Motif analysis requires at least two 'alarm_*_count' columns in the data.")
    else:
        from collections import Counter
        seq_counter = Counter()
        # --- NEW: Enhanced context gathering ---
        context_db = {} 

        for _, row in df.iterrows():
            # Create a sequence of alarm names if their count is greater than 0
            seq = tuple(sorted([t.replace('alarm_', '').replace('_count', '') for t in alarm_types if row[t] > 0]))
            
            if len(seq) >= 2: # Only analyze trips with at least two different alarms
                seq_counter[seq] += 1
                
                # Store detailed context for each sequence occurrence
                if seq not in context_db:
                    context_db[seq] = {'driver_id': [], 'bus_no': [], 'svc_no': [], 'depot_id': []}
                
                # Safely get context, append if column exists
                for key in context_db[seq].keys():
                    if key in row:
                        context_db[seq][key].append(row[key])

        if not seq_counter:
            st.info("No recurring alarm sequences (motifs) with 2 or more alarm types were found in the data.")
        else:
            top_patterns = seq_counter.most_common(5)
            
            st.subheader("Top 5 Most Frequent Alarm Sequences")
            
            # --- Enhanced Summary Table ---
            summary_rows = []
            for seq, cnt in top_patterns:
                row_data = {'Sequence': ' → '.join(s.replace("_", " ").title() for s in seq), 'Frequency': cnt}
                # Calculate top contributors for this sequence
                if seq in context_db:
                    for key, values in context_db[seq].items():
                        if values:
                            top_item = Counter(values).most_common(1)[0]
                            row_data[f'Top {key.replace("_", " ").title()}'] = f"{top_item[0]} ({top_item[1]} times)"
                        else:
                            row_data[f'Top {key.replace("_", " ").title()}'] = "N/A"
                summary_rows.append(row_data)
            
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df)
            download_df(summary_df, "alarm_sequence_summary.csv")

            # --- NEW: AI-Powered Deep-Dive Analysis ---
            st.subheader("AI-Powered Root Cause Analysis for Top Sequences")
            for seq, cnt in top_patterns:
                seq_name = ' → '.join(s.replace("_", " ").title() for s in seq)
                with st.expander(f"**Analyze Pattern: {seq_name}** (Occurred {cnt} times)"):
                    with st.spinner(f"AI is analyzing the '{seq_name}' pattern..."):
                        
                        # Prepare rich context for the LLM
                        context_str_parts = [f"This pattern occurred {cnt} times."]
                        if seq in context_db:
                            for key, values in context_db[seq].items():
                                if values:
                                    top_3_items = Counter(values).most_common(3)
                                    top_3_str = ", ".join([f"{item} ({count}x)" for item, count in top_3_items])
                                    context_str_parts.append(f"- **Top Contributing {key.replace('_', ' ').title()}s:** {top_3_str}")
                        
                        context_for_prompt = "\n".join(context_str_parts)

                        motif_prompt = f"""
                        As a world-class AI Bus Operations Analyst, analyze the following recurring alarm sequence (motif).

                        **Alarm Pattern:** "{seq_name}"

                        **Operational Context:**
                        {context_for_prompt}

                        Your analysis must be sharp, data-driven, and actionable for the Head of Operations. Provide the following:

                        ### 1. Pattern Hypothesis
                        Based on the alarms in the sequence and the context provided, what is the most likely operational scenario causing this pattern? 
                        *For example, a "Harsh Turn → Lane Departure" pattern might suggest drivers are cutting corners too quickly.*

                        ### 2. Key Contributing Factors & Hidden Insights
                        Dig deeper into the context. Is this pattern tied to specific drivers, buses, routes, or depots? What is the most significant link? 
                        *For instance: "This pattern is overwhelmingly tied to Bus-123, regardless of the driver, pointing to a potential vehicle alignment or sensor issue." or "The pattern is most frequent on Service-96, which is known for its tight corners, suggesting a route-based challenge."*

                        ### 3. Actionable Mitigation Plan
                        Provide a clear, prioritized action plan to address this specific pattern.
                        - **Priority 1 (Immediate Action):** - **Priority 2 (Investigation):** - **Priority 3 (Long-Term Fix):** """
                        try:
                            motif_insight = llm.invoke(motif_prompt).content
                            st.markdown(motif_insight)
                        except Exception as e:
                            st.error(f"Failed to generate AI analysis for this pattern: {e}")

# ─── Clustering ──────────────────────────────────────────────────────────────
# ─── Clustering ──────────────────────────────────────────────────────────────
elif section == "Clustering":
    st.header("🧩 Operational Profile Clustering")
    st.markdown("""
    This analysis groups trips with similar characteristics into "Operational Profiles."
    This helps identify distinct types of behavior across the fleet (e.g., high-efficiency routes vs. high-alarm urban routes).
    """)
    if df.empty:
         st.info("No data available for the current filter settings.")
    else:
        # --- 1. Feature Selection ---
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        selected_num = st.multiselect(
            "Select numeric features for clustering (min 2)",
            options=numeric_cols,
            default=numeric_cols[:min(4, len(numeric_cols))] # Sensible default
        )
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        selected_cat = st.multiselect(
            "Select categorical variables to include (optional)",
            options=categorical_cols
        )
        k = st.slider("Number of Profiles (Clusters)", 2, 8, 3, key="k_cluster")

        # --- 2. Run Clustering ---
        if st.button("Generate Operational Profiles"):
            if len(selected_num) < 2:
                st.error("Please select at least two numeric features to generate profiles.")
            else:
                # Build feature matrix
                X_num = df[selected_num]
                # Pre-emptively clean numeric columns
                for col in selected_num:
                    X_num[col] = pd.to_numeric(X_num[col], errors='coerce')
                X_num.dropna(inplace=True)

                if selected_cat:
                    X_cat = pd.get_dummies(df.loc[X_num.index][selected_cat].astype(str), drop_first=True)
                    X = pd.concat([X_num, X_cat], axis=1)
                else:
                    X = X_num

                if X.shape[0] < k:
                    st.error(f"Not enough clean data ({X.shape[0]} rows) to form {k} profiles. Try changing filters or features.")
                else:
                    with st.spinner("Analyzing data and generating profiles..."):
                        # Scale & cluster
                        scaler = StandardScaler().fit(X)
                        Xs = scaler.transform(X)
                        km = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(Xs)
                        labels = km.labels_

                        # Add cluster labels back to the original filtered dataframe
                        df_clustered = df.loc[X.index].copy()
                        df_clustered['cluster'] = labels

                        # PCA for 2D plot
                        pca = PCA(n_components=2, random_state=42)
                        comps = pca.fit_transform(Xs)

                        plot_df = pd.DataFrame({
                            'PC1': comps[:,0], 'PC2': comps[:,1], 'Profile': labels.astype(str)
                        }, index=X.index)

                        # --- 3. Visualization ---
                        st.subheader("Visual Map of Operational Profiles")
                        fig_cl = px.scatter(
                            plot_df, x='PC1', y='PC2', color='Profile',
                            title=f"Operational Profiles (K={k})",
                            color_discrete_sequence=px.colors.qualitative.Plotly
                        )
                        st.plotly_chart(fig_cl, use_container_width=True)
                        download_df(df_clustered, "clustered_data_with_profiles.csv")

                        # --- 4. AI-Powered Cluster Explanation ---
                        st.subheader("Detailed Profile Analysis")
                        global_means = df_clustered[selected_num].mean()

                        for cl in sorted(df_clustered['cluster'].unique()):
                            cluster_data = df_clustered[df_clustered['cluster'] == cl]

                            # Create a text summary of the cluster's characteristics
                            patterns = []
                            for f in selected_num:
                                m_cl = cluster_data[f].mean()
                                m_gl = global_means[f]
                                if m_gl != 0:
                                    rel = (m_cl - m_gl) / abs(m_gl) # Use absolute for stable percentage
                                    if rel > 0.2: lbl = "High"
                                    elif rel < -0.2: lbl = "Low"
                                    else: lbl = "Average"
                                else:
                                    lbl = 'Average'
                                patterns.append(f"{f} is {lbl} ({m_cl:.2f})")

                            profile_summary = "; ".join(patterns)

                            # AI Prompt to interpret the cluster
                            profile_prompt = f"""
                            As an expert AI Bus Operations Analyst, analyze the following operational profile which has been automatically clustered from data.

                            Profile Characteristics (compared to fleet average):
                            {profile_summary}

                            Based on this profile, provide:
                            1. **Profile Persona:** Give this cluster a short, descriptive name. (e.g., "High-Efficiency Highway Cruisers", "High-Alarm Urban Routes", "Idle Depot Buses").
                            2. **Operational Insight:** In simple terms, what does this profile represent? What is the likely story behind this behavior?
                            3. **Key Actionable Advice:** What is the most important thing an operations manager should do based on this insight? (e.g., "Investigate these routes for traffic issues," "Schedule these drivers for fatigue management training," "Check these buses for maintenance issues.")
                            """

                            try:
                                with st.spinner(f"Generating AI analysis for Profile {cl}..."):
                                    profile_response = llm.invoke(profile_prompt)
                                    profile_content = profile_response.content
                            except Exception as e:
                                profile_content = f"Could not get AI insight: {e}"

                            with st.expander(f"**Profile {cl}: Analysis** ({len(cluster_data)} trips)"):
                                st.info(profile_content)

                                # --- 5. Key Contributors Table ---
                                st.markdown("##### Key Contributors to this Profile")

                                contributor_cols = ['bus_no', 'depot', 'svc_no', 'driver_id']
                                # Only use columns that actually exist in the dataframe
                                available_cols = [c for c in contributor_cols if c in cluster_data.columns]

                                if available_cols:
                                    top_contributors = {}
                                    for col in available_cols:
                                        # Get top 3 most frequent items for each category
                                        top_items = cluster_data[col].value_counts().nlargest(3)
                                        top_contributors[f'Top {col.replace("_", " ").title()}'] = [f"{idx} ({cnt})" for idx, cnt in top_items.items()]

                                    # Use pandas to create a clean, aligned table
                                    # Need to make lists of equal length for the DataFrame
                                    max_len = max(len(v) for v in top_contributors.values()) if top_contributors else 0
                                    for k, v in top_contributors.items():
                                        v.extend(['-'] * (max_len - len(v)))

                                    st.dataframe(pd.DataFrame(top_contributors))
                                else:
                                    st.write("No categorical contributor columns (e.g., bus_no, depot) were found in the data.")


# ─── Table & Alerts ──────────────────────────────────────────────────────────
elif section == "Table & Alerts":
    st.header("🔔 Anomaly Table & Alerts")

    if st.session_state.anomaly_list:
        anomaly_df = pd.DataFrame(st.session_state.anomaly_list)
        st.dataframe(anomaly_df)

        st.subheader("Send Alert")
        recipient_email = st.text_input("Recipient Email", value=os.getenv("EMAIL", ""))
        teams_webhook = st.text_input("Teams Webhook URL (optional)", value=os.getenv("TEAMS_WEBHOOK", ""))

        if st.button("Send Alert"):
            if not recipient_email:
                st.error("Please enter a recipient email address.")
            else:
                alert_subject = f"Anomaly Alert: {len(anomaly_df)} Anomalies Detected"
                alert_message = "Detected Anomalies:\n\n" + anomaly_df.to_string(index=False)
                notify_email(alert_subject, alert_message, recipient_email)
                if teams_webhook:
                    notify_teams(teams_webhook, alert_message)
    else:
        st.info("No anomalies detected in the 'Overview' section yet. Run anomaly detection there first.")
