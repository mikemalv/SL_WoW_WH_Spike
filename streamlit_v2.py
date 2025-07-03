import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from snowflake.snowpark.context import get_active_session
import numpy as np

# Optional imports for enhanced features
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Get the current session
session = get_active_session()

# Page config
st.set_page_config(
    page_title="Snowflake Warehouse Monitor",
    page_icon="❄️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    .metric-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
        font-weight: 600;
        line-height: 1.2;
    }
    .metric-card .metric {
        font-size: 1.3rem;
        font-weight: bold;
        margin: 0;
        line-height: 1.2;
        word-wrap: break-word;
    }
    .metric-card .subtext {
        font-size: 0.9rem;
        margin-top: 0.3rem;
        line-height: 1.2;
    }
    
    .spike-card {
        background-color: #fdf2e9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #e67e22;
        margin: 0.5rem 0;
        word-wrap: break-word;
        overflow-wrap: break-word;
        border: 1px solid #f39c12;
    }
    .spike-card h5 {
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
        font-weight: 600;
        line-height: 1.2;
        color: #b8470f;
    }
    .spike-card .spike-metric {
        font-size: 1.4rem;
        font-weight: bold;
        color: #b8470f;
        line-height: 1.2;
    }
    .spike-card .spike-details {
        font-size: 0.9rem;
        margin-top: 0.5rem;
        color: #8b4513;
        line-height: 1.3;
    }
    
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 0.5rem 0;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    .success-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: #155724;
        line-height: 1.2;
    }
    .success-card .metric {
        font-size: 1.3rem;
        font-weight: bold;
        margin: 0;
        color: #155724;
        line-height: 1.2;
    }
    .success-card .subtext {
        font-size: 0.9rem;
        margin-top: 0.3rem;
        color: #155724;
        line-height: 1.2;
    }
    
    .alert-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 0.5rem 0;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    .alert-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: #721c24;
        line-height: 1.2;
    }
    .alert-card .metric {
        font-size: 1.3rem;
        font-weight: bold;
        margin: 0;
        color: #721c24;
        line-height: 1.2;
    }
    .alert-card .subtext {
        font-size: 0.9rem;
        margin-top: 0.3rem;
        color: #721c24;
        line-height: 1.2;
    }
    
    .efficiency-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 0.5rem 0;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    .efficiency-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: #0d47a1;
        line-height: 1.2;
    }
    .efficiency-card .metric {
        font-size: 1.3rem;
        font-weight: bold;
        margin: 0;
        color: #0d47a1;
        line-height: 1.2;
        word-wrap: break-word;
    }
    .efficiency-card .subtext {
        font-size: 0.9rem;
        margin-top: 0.3rem;
        color: #0d47a1;
        line-height: 1.2;
    }
    
    .stMetric > label {
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    .stMetric > div {
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    /* Target delta values specifically */
    .stMetric [data-testid="metric-container"] [data-testid="metric-delta"] {
        font-size: 14px !important;
        font-weight: 400 !important;
    }
    .stMetric small {
        font-size: 12px !important;
        font-weight: 400 !important;
    }
    .stMetric span[data-testid="stMetricDelta"] {
        font-size: 14px !important;
        font-weight: 400 !important;
    }
    .forecast-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("❄️ Snowflake Warehouse Usage Monitor")
st.markdown("Monitor week-over-week warehouse credit usage and identify consumption spikes")


# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Date range selector
    weeks_back = st.slider("Weeks of history", min_value=2, max_value=12, value=4)
    
    # Cloud services inclusion
    include_cloud_services = st.checkbox("☁️ Include Cloud Services", value=True, 
                                       help="Include CLOUD_SERVICES_ONLY warehouse data in all metrics and charts")
    
    # Forecasting options
    if PROPHET_AVAILABLE:
        st.subheader("🔮 Forecasting")
        enable_forecasting = st.checkbox("Enable Forecasting", value=True)
        forecast_weeks = st.slider("Forecast weeks ahead", min_value=1, max_value=8, value=4, disabled=not enable_forecasting)
    else:
        enable_forecasting = False
        forecast_weeks = 4
    
    # Budget tracking
    st.subheader("💰 Budget Settings")
    monthly_budget = st.number_input(
        "Monthly budget ($)",
        min_value=0.0,
        value=1000.0,
        step=100.0,
        help="Set your monthly Snowflake budget for alerts"
    )
    
    # Refresh button
    refresh_button = st.button("🔄 Refresh Data", type="primary")
    
    # Debug mode toggle
    debug_mode = st.checkbox("Debug Mode", value=False)
    st.session_state['debug_mode'] = debug_mode

# Initialize session state
if 'df_usage' not in st.session_state:
    st.session_state.df_usage = None
    st.session_state.df_current = None
    st.session_state.refresh_needed = True

# Always ensure df_queries is initialized (for backward compatibility)
if 'df_queries' not in st.session_state:
    st.session_state.df_queries = None

# Handle refresh logic
if refresh_button:
    st.session_state.refresh_needed = True
elif 'refresh_needed' not in st.session_state:
    st.session_state.refresh_needed = True
else:
    st.session_state.refresh_needed = False

# Query history settings
with st.sidebar:
    st.subheader("🔍 Query Analysis")
    query_days_back = st.slider("Query history (days)", min_value=1, max_value=14, value=7)

# Function to get query history data
@st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes (more recent data)
def get_query_history(days_back=7):
    try:
        query = f"""
        SELECT 
            DATE(start_time) as query_date,
            warehouse_name,
            warehouse_size,
            user_name,
            role_name,
            database_name,
            schema_name,
            query_type,
            execution_status,
            total_elapsed_time,
            execution_time,
            compilation_time,
            queued_provisioning_time,
            queued_overload_time,
            queued_repair_time,
            bytes_scanned,
            bytes_written,
            bytes_deleted,
            rows_produced,
            credits_used_cloud_services,
            partitions_scanned,
            partitions_total,
            bytes_spilled_to_local_storage,
            bytes_spilled_to_remote_storage,
            outbound_data_transfer_bytes,
            inbound_data_transfer_bytes,
            error_code,
            error_message,
            query_id,
            query_text,
            start_time,
            end_time
        FROM snowflake.account_usage.query_history
        WHERE start_time >= DATEADD('day', -{days_back}, CURRENT_DATE())
            AND start_time < CURRENT_DATE()
            AND warehouse_name NOT ILIKE 'COMPUTE_SERVICE%'
            AND execution_status IN ('SUCCESS', 'FAILED_WITH_ERROR', 'CANCELLED')
        ORDER BY start_time DESC;
        --LIMIT 10000;
        """
        
        result = session.sql(query)
        df = pd.DataFrame(result.collect())
        
        if df.empty:
            return df
            
        # Rename columns to lowercase
        df.columns = df.columns.str.lower()
        
        # Convert time columns to proper numeric types (milliseconds)
        time_columns = ['total_elapsed_time', 'execution_time', 'compilation_time', 
                       'queued_provisioning_time', 'queued_overload_time', 'queued_repair_time']
        for col in time_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Convert bytes columns
        bytes_columns = ['bytes_scanned', 'bytes_written', 'bytes_deleted', 
                        'bytes_spilled_to_local_storage', 'bytes_spilled_to_remote_storage',
                        'outbound_data_transfer_bytes', 'inbound_data_transfer_bytes']
        for col in bytes_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Convert other numeric columns
        numeric_columns = ['rows_produced', 'rows_examined', 'credits_used_cloud_services',
                          'partitions_scanned', 'partitions_total']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Convert timestamps
        for col in ['start_time', 'end_time']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Add calculated fields
        df['execution_time_seconds'] = df['execution_time'] / 1000
        df['total_elapsed_time_seconds'] = df['total_elapsed_time'] / 1000
        df['bytes_scanned_gb'] = df['bytes_scanned'] / (1024**3)
        df['bytes_written_gb'] = df['bytes_written'] / (1024**3)
        df['scan_efficiency'] = np.where(df['bytes_scanned'] > 0, 
                                       df['rows_produced'] / df['bytes_scanned'] * 1024**3, 0)
        
        return df
    except Exception as e:
        st.error(f"Error fetching query history data: {str(e)}")
        return pd.DataFrame()

# Enhanced function to get warehouse usage data
@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_warehouse_usage(weeks_back):
    try:
        query = f"""
        SELECT 
            DATE_TRUNC('week', start_time)::DATE as week_start_date,
            warehouse_name as wh_name,
            SUM(credits_used) as total_credits,
            SUM(credits_used_compute) as compute_credits,
            SUM(credits_used_cloud_services) as cloud_credits,
            COUNT(*) as usage_count,
            SUM(TIMESTAMPDIFF(second, start_time, end_time))/3600.0 as total_hours,
            AVG(credits_used) as avg_credits_per_period,
            STDDEV(credits_used) as stddev_credits
        FROM snowflake.account_usage.warehouse_metering_history
        WHERE start_time >= DATEADD('week', -{weeks_back}, CURRENT_DATE())
            AND start_time < CURRENT_DATE()
            AND warehouse_name IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 2, 1;
        """
        
        result = session.sql(query)
        df = pd.DataFrame(result.collect())
        
        if df.empty:
            return df
        
        # Rename columns
        df.columns = ['week_start', 'warehouse_name', 'total_credits', 'compute_credits', 
                     'cloud_credits', 'usage_count', 'total_hours', 'avg_credits_per_period', 'stddev_credits']
        
        # Ensure proper data types
        df['week_start'] = pd.to_datetime(df['week_start'])
        for col in ['total_credits', 'compute_credits', 'cloud_credits', 'usage_count', 'total_hours', 'avg_credits_per_period', 'stddev_credits']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Calculate week-over-week changes
        df = df.sort_values(['warehouse_name', 'week_start'])
        df['prev_week_credits'] = df.groupby('warehouse_name')['total_credits'].shift(1)
        df['prev_week_usage_count'] = df.groupby('warehouse_name')['usage_count'].shift(1)
        df['rolling_avg_4w'] = df.groupby('warehouse_name')['total_credits'].transform(lambda x: x.rolling(4, min_periods=1).mean())
        
        # Calculate percentage changes
        df['credit_change_pct'] = ((df['total_credits'] - df['prev_week_credits']) / df['prev_week_credits'] * 100).round(1)
        df['usage_change_pct'] = ((df['usage_count'] - df['prev_week_usage_count']) / df['prev_week_usage_count'] * 100).round(1)
        df['credit_change_pct'] = df['credit_change_pct'].fillna(0)
        df['usage_change_pct'] = df['usage_change_pct'].fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Error fetching warehouse usage data: {str(e)}")
        return pd.DataFrame()

# Function to get current week snapshot
@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_current_week_snapshot():
    try:
        query = """
        SELECT 
            warehouse_name,
            SUM(credits_used) as credits_this_week,
            COUNT(*) as usage_count_this_week,
            MAX(start_time) as last_used
        FROM snowflake.account_usage.warehouse_metering_history
        WHERE start_time >= DATE_TRUNC('week', CURRENT_DATE())
        GROUP BY 1
        ORDER BY 2 DESC;
        """
        df = session.sql(query).to_pandas()
        df.columns = df.columns.str.lower()
        return df
    except Exception as e:
        st.error(f"Error fetching current week snapshot: {str(e)}")
        return pd.DataFrame()

# Simple forecasting function
def create_simple_forecast(df, warehouse_name, periods=4):
    """Create simple forecast using Prophet"""
    if not PROPHET_AVAILABLE or df.empty:
        return None
    
    try:
        warehouse_data = df[df['warehouse_name'] == warehouse_name].copy()
        if len(warehouse_data) < 4:
            return None
        
        forecast_data = warehouse_data[['week_start', 'total_credits']].copy()
        forecast_data.columns = ['ds', 'y']
        forecast_data = forecast_data.sort_values('ds')
        
        model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        model.fit(forecast_data)
        
        future = model.make_future_dataframe(periods=periods, freq='W')
        forecast = model.predict(future)
        
        return forecast
    except Exception as e:
        if debug_mode:
            st.error(f"Error creating forecast: {str(e)}")
        return None

# Anomaly detection function
def detect_anomalies(df, sensitivity=0.1):
    """Simple anomaly detection"""
    if not SKLEARN_AVAILABLE or df.empty:
        return df
    
    try:
        features = ['total_credits', 'usage_count', 'total_hours']
        anomaly_data = df[features].fillna(0)
        
        if len(anomaly_data) < 5:
            df['anomaly'] = False
            return df
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(anomaly_data)
        
        iso_forest = IsolationForest(contamination=sensitivity, random_state=42)
        anomaly_labels = iso_forest.fit_predict(scaled_data)
        
        df['anomaly'] = anomaly_labels == -1
        return df
    except Exception as e:
        df['anomaly'] = False
        return df

# Fetch data
if st.session_state.refresh_needed or st.session_state.df_usage is None:
    with st.spinner("Fetching warehouse usage data..."):
        try:
            if refresh_button and hasattr(get_warehouse_usage, 'clear'):
                try:
                    get_warehouse_usage.clear()
                    get_current_week_snapshot.clear()
                except:
                    pass  # Ignore cache clear errors
            
            st.session_state.df_usage = get_warehouse_usage(weeks_back)
            st.session_state.df_current = get_current_week_snapshot()
            st.session_state.df_queries = get_query_history(query_days_back)
            
            # Add anomaly detection if available
            if SKLEARN_AVAILABLE and not st.session_state.df_usage.empty:
                st.session_state.df_usage = detect_anomalies(st.session_state.df_usage)
            
            st.session_state.refresh_needed = False
            
            if st.session_state.df_usage is not None and not st.session_state.df_usage.empty:
                st.success("✅ Data loaded successfully!")
            else:
                st.warning("⚠️ No warehouse usage data found. This could be due to:\n- No warehouse activity in the selected period\n- Data latency (up to 3 hours)\n- Permission issues")
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("Please check:\n- Your role has access to SNOWFLAKE.ACCOUNT_USAGE schema\n- The WAREHOUSE_METERING_HISTORY table exists and is accessible")
            st.session_state.df_usage = pd.DataFrame()
            st.session_state.df_current = pd.DataFrame()
            st.session_state.df_queries = pd.DataFrame()

# Check if data was loaded successfully
if st.session_state.df_usage is None or st.session_state.df_current is None:
    st.error("Failed to load data. Please check your permissions to access SNOWFLAKE.ACCOUNT_USAGE schema.")
    st.stop()

# Get data from session state
df = st.session_state.df_usage
df_current = st.session_state.df_current
df_queries = st.session_state.df_queries

# Ensure dataframes are not None and have proper structure
if df is None:
    df = pd.DataFrame()
else:
    df = df.loc[:, ~df.columns.duplicated()]
    
if df_current is None:
    df_current = pd.DataFrame()
else:
    df_current = df_current.loc[:, ~df_current.columns.duplicated()]

if df_queries is None:
    df_queries = pd.DataFrame()
else:
    df_queries = df_queries.loc[:, ~df_queries.columns.duplicated()]

# Apply cloud services filtering based on user preference
if not include_cloud_services:
    # Filter out CLOUD_SERVICES_ONLY from warehouse data
    if df is not None and not df.empty and 'warehouse_name' in df.columns:
        df = df[df['warehouse_name'] != 'CLOUD_SERVICES_ONLY'].copy()
    
    if df_current is not None and not df_current.empty and 'warehouse_name' in df_current.columns:
        df_current = df_current[df_current['warehouse_name'] != 'CLOUD_SERVICES_ONLY'].copy()
    
    # Filter out CLOUD_SERVICES_ONLY from query data
    if df_queries is not None and not df_queries.empty and 'warehouse_name' in df_queries.columns:
        df_queries = df_queries[df_queries['warehouse_name'] != 'CLOUD_SERVICES_ONLY'].copy()

# Additional check for empty dataframes
if df is None or df.empty or df_current is None:
    st.warning("No warehouse usage data found for the selected time period.")
    st.info("This could mean:\n- No warehouses have been used in the selected timeframe\n- You don't have access to SNOWFLAKE.ACCOUNT_USAGE schema\n- There's a delay in usage data (up to 3 hours)")
    # Create empty tabs to maintain UI structure
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Summary", "🚨 Spike Detection", "📈 Trends & Forecasting", "🔍 Deep Analytics", "🔍 Query Analysis", "📋 Raw Data"])
    for tab, message in zip([tab1, tab2, tab3, tab4, tab5, tab6], 
                           ["executive summary", "spike detection", "trends and forecasting", "deep analytics", "query analysis", "raw data"]):
        with tab:
            st.info(f"No data available for {message}.")
    st.stop()



# Enhanced tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Summary", "🚨 Spike Detection", "📈 Trends & Forecasting", "🔍 Deep Analytics", "🔍 Query Analysis", "💰 Expensive Queries", "📋 Raw Data"])

# TAB 1: EXECUTIVE SUMMARY
with tab1:
    st.header("📊 Summary")
    
    # Key metrics
    if df_current is not None and not df_current.empty:
        total_credits_this_week = df_current['credits_this_week'].sum()
        total_usage_count_this_week = df_current['usage_count_this_week'].sum()
        active_warehouses = len(df_current)
        top_warehouse = df_current.iloc[0]['warehouse_name']
        
        # Calculate week-over-week changes
        if not df.empty:
            current_week = df['week_start'].max()
            current_week_total = df[df['week_start'] == current_week]['total_credits'].sum()
            prev_week_total = df[df['week_start'] == (current_week - pd.Timedelta(weeks=1))]['total_credits'].sum()
            wow_change = ((current_week_total - prev_week_total) / prev_week_total * 100) if prev_week_total > 0 else 0
        else:
            wow_change = 0
            
        avg_credits_per_wh = total_credits_this_week / active_warehouses if active_warehouses > 0 else 0
    else:
        total_credits_this_week = 0
        total_usage_count_this_week = 0
        active_warehouses = 0
        top_warehouse = "N/A"
        wow_change = 0
        avg_credits_per_wh = 0
    
    # Main metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💳 Total Credits (This Week)", 
            f"{total_credits_this_week:,.2f}",
            delta=f"{wow_change:+.1f}%" if wow_change != 0 else None
        )
    
    with col2:
        st.metric("⚡ Usage Events", f"{total_usage_count_this_week:,}")
    
    with col3:
        st.metric("🏭 Active Warehouses", active_warehouses)
    
    with col4:
        st.metric("📊 Avg Credits/Warehouse", f"{avg_credits_per_wh:.2f}")
    
    st.markdown("---")
    
    # Enhanced Quick Insights with more metrics
    st.subheader("🔍 Quick Insights")
    
    if not df.empty:
        insights_col1, insights_col2, insights_col3, insights_col4 = st.columns(4)
        
        with insights_col1:
            # Most volatile warehouse
            if 'stddev_credits' in df.columns:
                most_volatile = df.loc[df['stddev_credits'].idxmax(), 'warehouse_name'] if df['stddev_credits'].max() > 0 else "N/A"
                volatility_score = df['stddev_credits'].max()
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🎢 Most Volatile</h4>
                    <div class="metric">{most_volatile}</div>
                    <div class="subtext">Std Dev: {volatility_score:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with insights_col2:
            # Fastest growing warehouse
            if 'credit_change_pct' in df.columns:
                latest_week = df['week_start'].max()
                latest_data = df[df['week_start'] == latest_week]
                if not latest_data.empty and latest_data['credit_change_pct'].max() > 0:
                    fastest_growing = latest_data.loc[latest_data['credit_change_pct'].idxmax(), 'warehouse_name']
                    growth_rate = latest_data['credit_change_pct'].max()
                else:
                    fastest_growing = "N/A"
                    growth_rate = 0
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🚀 Fastest Growing</h4>
                    <div class="metric">{fastest_growing}</div>
                    <div class="subtext">+{growth_rate:.1f}% this week</div>
                </div>
                """, unsafe_allow_html=True)
        
        with insights_col3:
            # Most efficient warehouse (lowest credits per hour)
            df_efficiency = df.copy()
            df_efficiency['credits_per_hour'] = df_efficiency['total_credits'] / df_efficiency['total_hours'].replace(0, np.nan)
            df_efficiency = df_efficiency.dropna(subset=['credits_per_hour'])
            if not df_efficiency.empty:
                most_efficient = df_efficiency.loc[df_efficiency['credits_per_hour'].idxmin(), 'warehouse_name']
                efficiency_score = df_efficiency['credits_per_hour'].min()
            else:
                most_efficient = "N/A"
                efficiency_score = 0
            st.markdown(f"""
            <div class="efficiency-card">
                <h4>⚡ Most Efficient</h4>
                <div class="metric">{most_efficient}</div>
                <div class="subtext">{efficiency_score:.2f} credits/hour</div>
            </div>
            """, unsafe_allow_html=True)
        
        with insights_col4:
            # Anomaly status
            anomaly_count = df['anomaly'].sum() if 'anomaly' in df.columns else 0
            total_records = len(df)
            anomaly_rate = (anomaly_count / total_records * 100) if total_records > 0 else 0
            
            if anomaly_count > 0:
                st.markdown(f"""
                <div class="alert-card">
                    <h4>🚨 Anomalies Detected</h4>
                    <div class="metric">{anomaly_count}</div>
                    <div class="subtext">{anomaly_rate:.1f}% of records</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-card">
                    <h4>✅ All Normal</h4>
                    <div class="metric">0</div>
                    <div class="subtext">No anomalies detected</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Budget tracking alert
    if monthly_budget > 0:
        # Estimate monthly usage based on current week
        days_in_month = 30
        days_elapsed = 7  # Assuming weekly tracking
        projected_monthly_credits = (total_credits_this_week / 7) * days_in_month
        projected_monthly_cost = projected_monthly_credits * 2.0  # Assuming $2 per credit
        
        budget_usage_pct = (projected_monthly_cost / monthly_budget * 100) if monthly_budget > 0 else 0
        
        if budget_usage_pct > 100:
            st.error(f"⚠️ **Budget Alert**: Projected monthly cost (${projected_monthly_cost:.2f}) exceeds budget by {budget_usage_pct - 100:.1f}%!")
        elif budget_usage_pct > 80:
            st.warning(f"⚡ **Budget Warning**: Currently at {budget_usage_pct:.1f}% of monthly budget")
        else:
            st.success(f"✅ **Budget Status**: Using {budget_usage_pct:.1f}% of monthly budget")
    
    # Top warehouses visualization
    st.subheader("🏆 Top Performing Warehouses")
    if df_current is not None and len(df_current) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_top = px.bar(
                df_current.head(5), 
                x='warehouse_name', 
                y='credits_this_week',
                title="Credit Usage by Top Warehouses",
                labels={'credits_this_week': 'Credits Used', 'warehouse_name': 'Warehouse'},
                color='credits_this_week',
                color_continuous_scale='Blues'
            )
            fig_top.update_layout(xaxis_tickangle=-45, showlegend=False, height=400)
            st.plotly_chart(fig_top, use_container_width=True)
        
        with col2:
            # Warehouse size recommendations
            st.markdown("### 📏 Sizing Recommendations")
            
            # Get warehouse sizes from query history if available
            if not df_queries.empty and 'warehouse_size' in df_queries.columns:
                warehouse_sizes = df_queries.groupby(['warehouse_name', 'warehouse_size']).size().reset_index(name='query_count')
                
                # Calculate recommendations based on usage patterns
                recommendations = []
                
                for wh in df_current.head(5)['warehouse_name']:
                    wh_queries = df_queries[df_queries['warehouse_name'] == wh]
                    if not wh_queries.empty:
                        avg_exec_time = wh_queries['execution_time_seconds'].mean()
                        spill_rate = ((wh_queries['bytes_spilled_to_local_storage'] > 0).sum() / len(wh_queries) * 100) if len(wh_queries) > 0 else 0
                        
                        if avg_exec_time > 300 or spill_rate > 20:
                            recommendations.append(f"**{wh}**: Consider ⬆️ upsizing")
                        elif avg_exec_time < 30 and spill_rate < 5:
                            recommendations.append(f"**{wh}**: Consider ⬇️ downsizing")
                        else:
                            recommendations.append(f"**{wh}**: ✅ Size appropriate")
                
                if recommendations:
                    for rec in recommendations[:3]:  # Show top 3
                        st.write(rec)
                else:
                    st.info("Analyzing usage patterns...")
            else:
                st.info("Need query history for sizing recommendations")

# TAB 2: SPIKE DETECTION with enhanced styling
with tab2:
    st.header("🚨 Spike Detection")
    
    if df is not None and not df.empty:
        # Spike threshold settings
        col1, col2 = st.columns(2)
        with col1:
            spike_threshold = st.slider(
                "Spike threshold (% increase)",
                min_value=10,
                max_value=200,
                value=50,
                step=10
            )
        with col2:
            min_credit_threshold = st.number_input(
                "Minimum credits for spike detection",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=5.0,
                help="Only flag spikes for warehouses using at least this many credits"
            )
        
        # Find spikes with minimum credit threshold
        spikes = df[
            (df['credit_change_pct'] > spike_threshold) & 
            (df['credit_change_pct'].notna()) &
            (df['credit_change_pct'] != float('inf')) &
            (df['total_credits'] >= min_credit_threshold)  # Only flag if above minimum threshold
        ].sort_values('credit_change_pct', ascending=False)
        
        if len(spikes) > 0:
            st.warning(f"⚠️ Found {len(spikes)} spikes above {spike_threshold}% increase")
            
            # Enhanced spike cards - similar to quick insights
            st.subheader("🎯 Top 6 Largest Spikes")
            spike_count = min(6, len(spikes))
            
            # Display spikes in a 3x2 grid
            for row in range(2):  # Two rows
                cols = st.columns(3)  # Three columns per row
                for col in range(3):
                    idx = row * 3 + col
                    if idx < spike_count:
                        spike_row = spikes.iloc[idx]
                        
                        with cols[col]:
                            # Safe datetime formatting
                            if pd.notnull(spike_row['week_start']):
                                try:
                                    week_str = spike_row['week_start'].strftime('%Y-%m-%d')
                                except:
                                    week_str = str(spike_row['week_start'])[:10]
                            else:
                                week_str = 'Unknown'
                            
                            prev_credits = spike_row['prev_week_credits'] if pd.notnull(spike_row['prev_week_credits']) else 0
                            credit_diff = spike_row['total_credits'] - prev_credits
                            
                            st.markdown(f"""
                            <div class="spike-card">
                                <h5>#{idx + 1} {spike_row['warehouse_name']}</h5>
                                <div class="spike-metric">+{spike_row['credit_change_pct']:.1f}%</div>
                                <div class="spike-details">
                                    Week: {week_str}<br>
                                    Credits: {spike_row['total_credits']:.2f} (+{credit_diff:.2f})<br>
                                    Hours: {spike_row['total_hours']:.1f}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.info(f"No spikes found above {spike_threshold}% threshold")
        
        # Heatmap section
        st.subheader("Week-over-Week Change Heatmap")
        
        # Heatmap options
        col1, col2 = st.columns(2)
        with col1:
            show_only_spikes = st.checkbox("Show only warehouses with spikes", value=False,
                                         help="Hide warehouses with 0% change across all weeks")
        with col2:
            spike_threshold_heatmap = st.number_input("Minimum spike % to show", 
                                                    value=10, min_value=0, max_value=100, step=5,
                                                    help="Only show warehouses with at least one spike above this threshold")
        
        try:
            # Create heatmap data
            heatmap_df = df[['warehouse_name', 'week_start', 'credit_change_pct']].copy()
            heatmap_df['week_start'] = heatmap_df['week_start'].astype(str).str[:10]
            heatmap_df = heatmap_df.groupby(['warehouse_name', 'week_start'])['credit_change_pct'].mean().reset_index()
            
            heatmap_data = heatmap_df.pivot(
                index='warehouse_name',
                columns='week_start',
                values='credit_change_pct'
            ).fillna(0)
            
            # Filter warehouses based on user selection
            if show_only_spikes:
                # Keep only warehouses that have at least one spike above threshold
                # Also check if warehouse ever exceeds minimum credit threshold
                warehouse_credits = df.groupby('warehouse_name')['total_credits'].max()
                warehouse_has_spike = (heatmap_data.abs() >= spike_threshold_heatmap).any(axis=1)
                warehouse_has_sufficient_credits = warehouse_credits >= min_credit_threshold
                # Only show warehouses that have spikes AND meet credit threshold
                heatmap_data = heatmap_data[warehouse_has_spike & warehouse_has_sufficient_credits]
            
            if len(heatmap_data) > 0 and len(heatmap_data.columns) > 0:
                # Create text annotations with hyphens for zero values
                text_values = heatmap_data.values.round(1)
                text_annotations = []
                for i in range(len(text_values)):
                    row_texts = []
                    for j in range(len(text_values[i])):
                        val = text_values[i][j]
                        if val == 0:
                            row_texts.append("-")
                        else:
                            row_texts.append(f"{val}%")
                    text_annotations.append(row_texts)
                
                # Create custom colorscale with white for zero values
                colorscale = [
                    [0.0, '#4575b4'],    # Blue for negative (decreases)
                    [0.48, '#ffffbf'],   # Light yellow
                    [0.495, '#ffffff'],  # White for zero
                    [0.505, '#ffffff'],  # White for zero
                    [0.52, '#ffffbf'],   # Light yellow
                    [1.0, '#d73027']     # Red for positive (spikes)
                ]
                
                # Determine range for color scale
                abs_max = max(abs(heatmap_data.values.min()), abs(heatmap_data.values.max()))
                if abs_max == 0:
                    abs_max = 1
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=heatmap_data.values,
                    x=heatmap_data.columns.tolist(),
                    y=heatmap_data.index.tolist(),
                    colorscale=colorscale,
                    zmid=0,
                    zmin=-abs_max,
                    zmax=abs_max,
                    text=text_annotations,
                    texttemplate='%{text}',
                    textfont={"size": 10, "color": "black"},
                    colorbar=dict(title="% Change", ticksuffix="%"),
                    hovertemplate='<b>%{y}</b><br>Week: %{x}<br>Change: %{z:.1f}%<extra></extra>'
                ))
                
                # Calculate height and make scrollable if needed
                height = max(400, len(heatmap_data.index) * 30)
                max_height = 800
                scrollable = height > max_height
                
                fig_heatmap.update_layout(
                    title="Week-over-Week Credit Usage Change (%)",
                    xaxis_title="Week Starting",
                    yaxis_title="Warehouse",
                    height=min(height, max_height),
                    xaxis=dict(tickangle=-45),
                    margin=dict(l=150)
                )
                
                if scrollable:
                    st.info(f"📊 Showing {len(heatmap_data)} warehouses (scrollable)")
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("Not enough data to generate heatmap. Need at least 2 weeks of data.")
        except Exception as e:
            st.info(f"Unable to generate heatmap: {str(e)}")
    else:
        st.info("No warehouse usage data available for spike detection.")

# TAB 3: SIMPLIFIED TRENDS & FORECASTING
with tab3:
    st.header("📈 Trends & Forecasting")
    
    # Warehouse selector
    if df is not None and not df.empty and 'warehouse_name' in df.columns:
        warehouses = sorted(df['warehouse_name'].unique())
        top_warehouses = df.groupby('warehouse_name')['total_credits'].sum().nlargest(5).index.tolist()
        
        selected_warehouses = st.multiselect(
            "Select warehouses to analyze",
            options=warehouses,
            default=top_warehouses[:3] if top_warehouses else []
        )
    else:
        st.warning("No warehouse data available for the selected time period")
        selected_warehouses = []
    
    if selected_warehouses and df is not None and not df.empty:
        # Filter data
        df_filtered = df[df['warehouse_name'].isin(selected_warehouses)].copy()
        
        # Trend chart
        st.subheader("📊 Credit Usage Trends")
        fig_trend = px.line(
            df_filtered,
            x='week_start',
            y='total_credits',
            color='warehouse_name',
            title="Weekly Credit Usage Trend",
            labels={'total_credits': 'Credits Used', 'week_start': 'Week Starting'},
            markers=True
        )
        fig_trend.update_layout(hovermode='x unified', height=500)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Simple Forecasting
        if enable_forecasting and PROPHET_AVAILABLE:
            st.subheader("🔮 Credit Usage Forecasting")
            
            forecast_warehouse = st.selectbox(
                "Select warehouse for forecasting",
                options=selected_warehouses
            )
            
            if forecast_warehouse:
                with st.spinner(f"Generating forecast for {forecast_warehouse}..."):
                    forecast_data = create_simple_forecast(df, forecast_warehouse, forecast_weeks)
                
                if forecast_data is not None:
                    # Create forecast visualization
                    fig_forecast = go.Figure()
                    
                    # Historical data
                    historical = df[df['warehouse_name'] == forecast_warehouse].sort_values('week_start')
                    fig_forecast.add_trace(go.Scatter(
                        x=historical['week_start'],
                        y=historical['total_credits'],
                        mode='lines+markers',
                        name='Historical',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Forecast data
                    future_dates = forecast_data['ds'].tail(forecast_weeks)
                    future_values = forecast_data['yhat'].tail(forecast_weeks)
                    
                    fig_forecast.add_trace(go.Scatter(
                        x=future_dates,
                        y=future_values,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='red', dash='dash', width=2)
                    ))
                    
                    fig_forecast.update_layout(
                        title=f"📊 Credit Usage Forecast - {forecast_warehouse}",
                        xaxis_title="Week",
                        yaxis_title="Credits Used",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Forecast summary
                    forecast_avg = future_values.mean()
                    historical_avg = historical['total_credits'].mean()
                    trend = "increasing" if forecast_avg > historical_avg else "decreasing"
                    change_pct = ((forecast_avg - historical_avg) / historical_avg * 100) if historical_avg > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Forecasted Average", f"{forecast_avg:.2f}", delta=f"{change_pct:+.1f}%")
                    with col2:
                        st.metric("Trend Direction", trend.title())
                    with col3:
                        next_week = future_values.iloc[0] if len(future_values) > 0 else 0
                        st.metric("Next Week Prediction", f"{next_week:.2f}")
                else:
                    st.info(f"Unable to generate forecast for {forecast_warehouse}. Need more historical data.")
        
        elif enable_forecasting and not PROPHET_AVAILABLE:
            st.info("📦 Forecasting requires the Prophet library. Install it to enable predictions.")
    
    else:
        st.info("Please select at least one warehouse to view trends and forecasts")

# TAB 4: DEEP ANALYTICS
with tab4:
    st.header("🔍 Deep Analytics")
    
    if not df.empty:
        # Statistical Overview
        st.subheader("📊 Statistical Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_records = len(df)
            unique_warehouses = df['warehouse_name'].nunique()
            st.metric("📋 Total Records", f"{total_records:,}")
            st.metric("🏭 Unique Warehouses", unique_warehouses)
        
        with col2:
            avg_credits = df['total_credits'].mean()
            std_credits = df['total_credits'].std()
            st.metric("💰 Avg Credits/Week", f"{avg_credits:.2f}")
            st.metric("📊 Std Deviation", f"{std_credits:.2f}")
        
        with col3:
            max_credits = df['total_credits'].max()
            min_credits = df['total_credits'].min()
            st.metric("📈 Max Weekly Credits", f"{max_credits:.2f}")
            st.metric("📉 Min Weekly Credits", f"{min_credits:.2f}")
        
        with col4:
            if 'anomaly' in df.columns:
                anomaly_count = df['anomaly'].sum()
                anomaly_rate = (anomaly_count / total_records * 100) if total_records > 0 else 0
                st.metric("🚨 Anomalies", anomaly_count)
                st.metric("⚠️ Anomaly Rate", f"{anomaly_rate:.1f}%")
            else:
                st.metric("🚨 Anomalies", "N/A")
                st.metric("⚠️ Anomaly Rate", "N/A")
        
        # Efficiency Analysis
        st.subheader("⚡ Efficiency Analysis")
        
        if 'total_hours' in df.columns:
            df_efficiency = df.copy()
            df_efficiency['credits_per_hour'] = df_efficiency['total_credits'] / df_efficiency['total_hours'].replace(0, np.nan)
            df_efficiency['hours_per_period'] = df_efficiency['total_hours'] / df_efficiency['usage_count'].replace(0, np.nan)
            
            efficiency_summary = df_efficiency.groupby('warehouse_name').agg({
                'credits_per_hour': ['mean', 'std'],
                'hours_per_period': 'mean',
                'total_credits': 'sum',
                'usage_count': 'sum'
            }).round(2)
            
            # Flatten column names
            efficiency_summary.columns = ['_'.join(col).strip() for col in efficiency_summary.columns.values]
            efficiency_summary = efficiency_summary.reset_index()
            
            # Rename columns for clarity
            column_mapping = {
                'credits_per_hour_mean': 'Avg Credits/Hour',
                'credits_per_hour_std': 'Credits/Hour Std Dev',
                'hours_per_period_mean': 'Avg Hours/Period',
                'total_credits_sum': 'Total Credits',
                'usage_count_sum': 'Total Usage Count'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in efficiency_summary.columns:
                    efficiency_summary.rename(columns={old_col: new_col}, inplace=True)
            
            st.dataframe(efficiency_summary, use_container_width=True)
            
            # Efficiency visualization
            if 'Avg Credits/Hour' in efficiency_summary.columns:
                fig_efficiency = px.bar(
                    efficiency_summary.head(10),
                    x='warehouse_name',
                    y='Avg Credits/Hour',
                    title="Average Credits per Hour by Warehouse",
                    labels={'warehouse_name': 'Warehouse', 'Avg Credits/Hour': 'Credits per Hour'},
                    color='Avg Credits/Hour',
                    color_continuous_scale='RdYlBu_r'
                )
                fig_efficiency.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_efficiency, use_container_width=True)
        
        # Usage Patterns
        st.subheader("📈 Usage Patterns")
        
        if 'week_start' in df.columns:
            # Weekly usage trends
            weekly_summary = df.groupby('week_start').agg({
                'total_credits': ['sum', 'count', 'mean'],
                'warehouse_name': 'nunique'
            }).round(2)
            
            weekly_summary.columns = ['Total Credits', 'Usage Events', 'Avg Credits/Event', 'Active Warehouses']
            weekly_summary = weekly_summary.reset_index()
            
            # Usage pattern visualization
            fig_pattern = go.Figure()
            
            fig_pattern.add_trace(go.Scatter(
                x=weekly_summary['week_start'],
                y=weekly_summary['Total Credits'],
                mode='lines+markers',
                name='Total Credits',
                yaxis='y1'
            ))
            
            fig_pattern.add_trace(go.Scatter(
                x=weekly_summary['week_start'],
                y=weekly_summary['Active Warehouses'],
                mode='lines+markers',
                name='Active Warehouses',
                yaxis='y2'
            ))
            
            fig_pattern.update_layout(
                title="Weekly Usage Patterns",
                xaxis_title="Week",
                yaxis=dict(title="Total Credits", side="left"),
                yaxis2=dict(title="Active Warehouses", side="right", overlaying="y"),
                height=400
            )
            
            st.plotly_chart(fig_pattern, use_container_width=True)
        
        # Anomaly Detection Results
        if 'anomaly' in df.columns and SKLEARN_AVAILABLE:
            st.subheader("🚨 Anomaly Detection Results")
            
            anomalies = df[df['anomaly'] == True]
            
            if not anomalies.empty:
                st.warning(f"Found {len(anomalies)} anomalies in the data")
                
                # Show anomalies
                anomaly_display = anomalies[['week_start', 'warehouse_name', 'total_credits', 'usage_count', 'total_hours']].copy()
                anomaly_display['week_start'] = anomaly_display['week_start'].dt.strftime('%Y-%m-%d')
                
                st.dataframe(
                    anomaly_display.sort_values('total_credits', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Anomaly visualization with size based on severity
                df_plot = df.copy()
                df_plot['anomaly_severity'] = np.where(df_plot['anomaly'], 
                                                     np.abs(df_plot['total_credits'] - df_plot['total_credits'].mean()) / df_plot['total_credits'].std(),
                                                     1)
                df_plot['point_size'] = np.where(df_plot['anomaly'], 
                                               np.clip(df_plot['anomaly_severity'] * 15 + 10, 10, 50),
                                               8)
                
                fig_anomaly = px.scatter(
                    df_plot,
                    x='total_credits',
                    y='usage_count',
                    color='anomaly',
                    size='point_size',
                    title="Credit Usage vs Usage Count (Anomalies Highlighted)",
                    labels={'total_credits': 'Total Credits', 'usage_count': 'Usage Count'},
                    color_discrete_map={True: 'red', False: 'blue'},
                    hover_data=['warehouse_name', 'week_start']
                )
                st.plotly_chart(fig_anomaly, use_container_width=True)
            else:
                st.success("No anomalies detected in the current dataset")
        
        # Warehouse Performance Ranking
        st.subheader("🏆 Warehouse Performance Ranking")
        
        ranking_metrics = df.groupby('warehouse_name').agg({
            'total_credits': ['sum', 'mean', 'std'],
            'usage_count': 'sum',
            'total_hours': 'sum'
        }).round(2)
        
        ranking_metrics.columns = ['Total Credits', 'Avg Credits', 'Credits Std Dev', 'Total Usage', 'Total Hours']
        ranking_metrics = ranking_metrics.reset_index()
        
        # Calculate performance score (lower is better for efficiency)
        if 'Total Hours' in ranking_metrics.columns and ranking_metrics['Total Hours'].sum() > 0:
            ranking_metrics['Efficiency Score'] = (ranking_metrics['Total Credits'] / ranking_metrics['Total Hours'].replace(0, np.nan)).round(2)
            
            # Performance metric cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                most_efficient_idx = ranking_metrics['Efficiency Score'].idxmin()
                most_efficient = ranking_metrics.loc[most_efficient_idx]
                st.markdown(f"""
                <div class="success-card">
                    <h4>⭐ Most Efficient</h4>
                    <div class="metric">{most_efficient['warehouse_name']}</div>
                    <div class="subtext">{most_efficient['Efficiency Score']:.2f} credits/hour</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                least_efficient_idx = ranking_metrics['Efficiency Score'].idxmax()
                least_efficient = ranking_metrics.loc[least_efficient_idx]
                st.markdown(f"""
                <div class="alert-card">
                    <h4>⚠️ Least Efficient</h4>
                    <div class="metric">{least_efficient['warehouse_name']}</div>
                    <div class="subtext">{least_efficient['Efficiency Score']:.2f} credits/hour</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                highest_usage_idx = ranking_metrics['Total Credits'].idxmax()
                highest_usage = ranking_metrics.loc[highest_usage_idx]
                st.markdown(f"""
                <div class="metric-card">
                    <h4>💰 Highest Usage</h4>
                    <div class="metric">{highest_usage['warehouse_name']}</div>
                    <div class="subtext">{highest_usage['Total Credits']:,.0f} total credits</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                most_active_idx = ranking_metrics['Total Hours'].idxmax()
                most_active = ranking_metrics.loc[most_active_idx]
                st.markdown(f"""
                <div class="efficiency-card">
                    <h4>🕒 Most Active</h4>
                    <div class="metric">{most_active['warehouse_name']}</div>
                    <div class="subtext">{most_active['Total Hours']:,.0f} total hours</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Visual representation - Efficiency vs Usage scatter plot
            st.subheader("📊 Efficiency vs Usage Analysis")
            
            fig_performance = px.scatter(
                ranking_metrics,
                x='Total Credits',
                y='Efficiency Score',
                size='Total Hours',
                color='Total Usage',
                hover_name='warehouse_name',
                title="Warehouse Performance: Efficiency vs Total Usage",
                labels={
                    'Total Credits': 'Total Credits Used',
                    'Efficiency Score': 'Credits per Hour (Lower = More Efficient)',
                    'Total Hours': 'Total Active Hours',
                    'Total Usage': 'Total Usage Events'
                },
                color_continuous_scale='Viridis'
            )
            
            fig_performance.update_layout(
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig_performance, use_container_width=True)
            
            # Performance ranking table
            st.subheader("📋 Detailed Performance Ranking")
        
        # Sort by total credits (descending)
        ranking_metrics = ranking_metrics.sort_values('Total Credits', ascending=False)
        
        st.dataframe(ranking_metrics, use_container_width=True, hide_index=True)
        
        # Warehouse Utilization Heatmap
        st.subheader("🔥 Warehouse Utilization Heatmap")
        
        if not df_queries.empty and 'start_time' in df_queries.columns:
            # Create hourly utilization data
            df_utilization = df_queries.copy()
            df_utilization['hour'] = df_utilization['start_time'].dt.hour
            df_utilization['day_of_week'] = df_utilization['start_time'].dt.day_name()
            
            # Create pivot table for heatmap
            utilization_pivot = df_utilization.groupby(['day_of_week', 'hour']).size().reset_index(name='query_count')
            
            # Ensure all days and hours are represented
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            hours = list(range(24))
            
            # Create complete grid
            complete_grid = pd.DataFrame([(d, h) for d in days_order for h in hours], 
                                       columns=['day_of_week', 'hour'])
            utilization_pivot = complete_grid.merge(utilization_pivot, on=['day_of_week', 'hour'], how='left')
            utilization_pivot['query_count'] = utilization_pivot['query_count'].fillna(0)
            
            # Pivot for heatmap
            heatmap_data = utilization_pivot.pivot(index='day_of_week', columns='hour', values='query_count')
            heatmap_data = heatmap_data.reindex(days_order)
            
            # Create heatmap
            fig_heatmap = px.imshow(
                heatmap_data,
                labels=dict(x="Hour of Day", y="Day of Week", color="Query Count"),
                x=[f"{h:02d}:00" for h in range(24)],
                y=days_order,
                color_continuous_scale="YlOrRd",
                title="Query Activity Heatmap by Day and Hour"
            )
            
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Peak usage insights
            peak_hour = utilization_pivot.loc[utilization_pivot['query_count'].idxmax()]
            st.info(f"📊 **Peak Usage**: {peak_hour['day_of_week']} at {peak_hour['hour']:02d}:00 with {int(peak_hour['query_count'])} queries")
        
        # Performance Score Card
        st.subheader("🎯 Warehouse Performance Score")
        
        if not df.empty:
            # Calculate performance scores for each warehouse
            performance_scores = []
            
            for warehouse in df['warehouse_name'].unique():
                wh_data = df[df['warehouse_name'] == warehouse]
                wh_queries = df_queries[df_queries['warehouse_name'] == warehouse] if not df_queries.empty else pd.DataFrame()
                
                # Calculate metrics
                total_credits = wh_data['total_credits'].sum()
                credit_stability = 100 - (wh_data['credit_change_pct'].std() if 'credit_change_pct' in wh_data.columns else 0)
                
                if not wh_queries.empty:
                    query_efficiency = 100 - min(100, wh_queries['execution_time_seconds'].mean() / 10)
                    spill_score = 100 - min(100, ((wh_queries['bytes_spilled_to_local_storage'] > 0).sum() / len(wh_queries) * 100))
                else:
                    query_efficiency = 50
                    spill_score = 50
                
                # Calculate overall score
                overall_score = (credit_stability * 0.3 + query_efficiency * 0.4 + spill_score * 0.3)
                
                performance_scores.append({
                    'Warehouse': warehouse,
                    'Overall Score': round(overall_score, 1),
                    'Stability': round(credit_stability, 1),
                    'Efficiency': round(query_efficiency, 1),
                    'Spill Score': round(spill_score, 1),
                    'Total Credits': round(total_credits, 2)
                })
            
            # Display top performers
            scores_df = pd.DataFrame(performance_scores).sort_values('Overall Score', ascending=False)
            
            col1, col2, col3 = st.columns(3)
            
            # Top 3 performers
            for idx, (col, rank) in enumerate(zip([col1, col2, col3], ['🥇 Gold', '🥈 Silver', '🥉 Bronze'])):
                if idx < len(scores_df):
                    with col:
                        wh = scores_df.iloc[idx]
                        score_color = 'success' if wh['Overall Score'] > 80 else 'alert' if wh['Overall Score'] < 60 else 'metric'
                        st.markdown(f"""
                        <div class="{score_color}-card">
                            <h4>{rank}</h4>
                            <div class="metric">{wh['Warehouse']}</div>
                            <div class="subtext">Score: {wh['Overall Score']}/100</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Full performance table
            st.markdown("#### 📊 Full Performance Scorecard")
            st.dataframe(scores_df.head(10), use_container_width=True, hide_index=True)
        
    else:
        st.info("No data available for deep analytics")

# TAB 5: QUERY ANALYSIS
with tab5:
    st.header("🔍 Query Analysis")
    
    if not df_queries.empty:
        # Query Analysis Overview
        st.subheader("📊 Query Performance Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_queries = len(df_queries)
            successful_queries = len(df_queries[df_queries['execution_status'] == 'SUCCESS'])
            success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
            st.metric("📈 Total Queries", f"{total_queries:,}")
            st.metric("✅ Success Rate", f"{success_rate:.1f}%")
        
        with col2:
            avg_execution_time = df_queries['execution_time_seconds'].mean()
            median_execution_time = df_queries['execution_time_seconds'].median()
            st.metric("⏱️ Avg Execution Time", f"{avg_execution_time:.2f}s")
            st.metric("⏹️ Median Execution Time", f"{median_execution_time:.2f}s")
        
        with col3:
            total_data_scanned = df_queries['bytes_scanned_gb'].sum()
            avg_data_scanned = df_queries['bytes_scanned_gb'].mean()
            st.metric("💽 Total Data Scanned", f"{total_data_scanned:.2f} GB")
            st.metric("📊 Avg Data/Query", f"{avg_data_scanned:.2f} GB")
        
        with col4:
            cloud_services_credits = df_queries['credits_used_cloud_services'].sum()
            unique_users = df_queries['user_name'].nunique()
            st.metric("☁️ Cloud Services Credits", f"{cloud_services_credits:.3f}")
            st.metric("👥 Unique Users", unique_users)
        
        st.markdown("---")
        
        # Query Performance by Warehouse
        st.subheader("🏭 Query Performance by Warehouse")
        
        if 'warehouse_name' in df_queries.columns:
            warehouse_metrics = df_queries.groupby('warehouse_name').agg({
                'execution_time_seconds': ['count', 'mean', 'median', 'max'],
                'bytes_scanned_gb': 'sum',
                'rows_produced': 'sum',
                'credits_used_cloud_services': 'sum'
            }).round(3)
            
            # Flatten column names
            warehouse_metrics.columns = ['Query Count', 'Avg Exec Time (s)', 'Median Exec Time (s)', 
                                       'Max Exec Time (s)', 'Total Data Scanned (GB)', 
                                       'Total Rows Produced', 'Cloud Services Credits']
            warehouse_metrics = warehouse_metrics.reset_index()
            
            # Add efficiency metrics
            warehouse_metrics['Queries per Hour'] = warehouse_metrics['Query Count'] / (query_days_back * 24)
            warehouse_metrics['GB per Query'] = warehouse_metrics['Total Data Scanned (GB)'] / warehouse_metrics['Query Count'].replace(0, np.nan)
            
            st.dataframe(warehouse_metrics, use_container_width=True, hide_index=True)
            
            # Warehouse performance visualization
            fig_wh_perf = px.scatter(
                warehouse_metrics,
                x='Query Count',
                y='Avg Exec Time (s)',
                size='Total Data Scanned (GB)',
                color='Cloud Services Credits',
                hover_name='warehouse_name',
                title="Warehouse Query Performance: Count vs Execution Time",
                labels={
                    'Query Count': 'Number of Queries',
                    'Avg Exec Time (s)': 'Average Execution Time (seconds)',
                    'Total Data Scanned (GB)': 'Data Scanned (GB)',
                    'Cloud Services Credits': 'Cloud Services Credits'
                }
            )
            fig_wh_perf.update_layout(height=500)
            st.plotly_chart(fig_wh_perf, use_container_width=True)
        
        # Query Type Analysis
        st.subheader("📋 Query Type Distribution")
        
        if 'query_type' in df_queries.columns:
            query_type_stats = df_queries.groupby('query_type').agg({
                'execution_time_seconds': ['count', 'mean'],
                'bytes_scanned_gb': 'mean'
            }).round(3)
            
            query_type_stats.columns = ['Count', 'Avg Execution Time (s)', 'Avg Data Scanned (GB)']
            query_type_stats = query_type_stats.reset_index()
            query_type_stats = query_type_stats.sort_values('Count', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Query type pie chart
                fig_pie = px.pie(
                    query_type_stats,
                    values='Count',
                    names='query_type',
                    title="Query Type Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Query type performance
                fig_bar = px.bar(
                    query_type_stats.head(10),
                    x='query_type',
                    y='Avg Execution Time (s)',
                    title="Average Execution Time by Query Type",
                    color='Avg Execution Time (s)',
                    color_continuous_scale='Reds'
                )
                fig_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # Top Expensive Queries
        st.subheader("💰 Most Expensive Queries")
        
        # Sort by execution time
        expensive_queries = df_queries.nlargest(10, 'execution_time_seconds')[
            ['query_id', 'warehouse_name', 'user_name', 'execution_time_seconds', 
             'bytes_scanned_gb', 'rows_produced', 'query_type']
        ].copy()
        
        if not expensive_queries.empty:
            expensive_queries['execution_time_seconds'] = expensive_queries['execution_time_seconds'].round(2)
            expensive_queries['bytes_scanned_gb'] = expensive_queries['bytes_scanned_gb'].round(3)
            
            st.dataframe(expensive_queries, use_container_width=True, hide_index=True)
        
        # Failed Queries Analysis
        failed_queries = df_queries[df_queries['execution_status'] != 'SUCCESS']
        if not failed_queries.empty:
            st.subheader("❌ Failed Queries Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                failure_reasons = failed_queries['error_code'].value_counts().head(10)
                if not failure_reasons.empty:
                    fig_failures = px.bar(
                        x=failure_reasons.values,
                        y=failure_reasons.index,
                        orientation='h',
                        title="Top Failure Reasons",
                        labels={'x': 'Count', 'y': 'Error Code'}
                    )
                    st.plotly_chart(fig_failures, use_container_width=True)
            
            with col2:
                failed_by_warehouse = failed_queries['warehouse_name'].value_counts().head(10)
                if not failed_by_warehouse.empty:
                    fig_wh_failures = px.bar(
                        x=failed_by_warehouse.index,
                        y=failed_by_warehouse.values,
                        title="Failed Queries by Warehouse",
                        labels={'x': 'Warehouse', 'y': 'Failed Query Count'}
                    )
                    fig_wh_failures.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_wh_failures, use_container_width=True)
        
        # Query Timeline
        st.subheader("📈 Query Activity Timeline")
        
        if 'start_time' in df_queries.columns:
            # Hourly query activity
            df_queries['hour'] = df_queries['start_time'].dt.hour
            hourly_activity = df_queries.groupby('hour').size().reset_index(name='query_count')
            
            fig_timeline = px.line(
                hourly_activity,
                x='hour',
                y='query_count',
                title="Query Activity by Hour of Day",
                labels={'hour': 'Hour of Day', 'query_count': 'Number of Queries'},
                markers=True
            )
            fig_timeline.update_layout(height=400)
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # User Activity Analysis
        st.subheader("👥 User Activity Analysis")
        
        user_stats = df_queries.groupby('user_name').agg({
            'execution_time_seconds': ['count', 'sum', 'mean'],
            'bytes_scanned_gb': 'sum',
            'credits_used_cloud_services': 'sum'
        }).round(3)
        
        user_stats.columns = ['Query Count', 'Total Exec Time (s)', 'Avg Exec Time (s)', 
                             'Total Data Scanned (GB)', 'Cloud Services Credits']
        user_stats = user_stats.reset_index()
        user_stats = user_stats.sort_values('Query Count', ascending=False).head(15)
        
        st.dataframe(user_stats, use_container_width=True, hide_index=True)
        
        # Efficiency Analysis
        st.subheader("⚡ Query Efficiency Analysis")
        
        if len(df_queries) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Spill analysis
                spill_queries = df_queries[
                    (df_queries['bytes_spilled_to_local_storage'] > 0) | 
                    (df_queries['bytes_spilled_to_remote_storage'] > 0)
                ]
                spill_rate = (len(spill_queries) / len(df_queries) * 100) if len(df_queries) > 0 else 0
                
                st.metric("🌊 Queries with Spillage", f"{len(spill_queries):,}")
                st.metric("📊 Spillage Rate", f"{spill_rate:.1f}%")
                
                if len(spill_queries) > 0:
                    avg_local_spill = spill_queries['bytes_spilled_to_local_storage'].mean() / (1024**3)
                    avg_remote_spill = spill_queries['bytes_spilled_to_remote_storage'].mean() / (1024**3)
                    st.metric("💾 Avg Local Spill", f"{avg_local_spill:.2f} GB")
                    st.metric("☁️ Avg Remote Spill", f"{avg_remote_spill:.2f} GB")
            
            with col2:
                # Partition pruning analysis
                pruned_queries = df_queries[df_queries['partitions_scanned'] < df_queries['partitions_total']]
                pruning_effectiveness = (len(pruned_queries) / len(df_queries) * 100) if len(df_queries) > 0 else 0
                
                st.metric("🎯 Queries with Pruning", f"{len(pruned_queries):,}")
                st.metric("📈 Pruning Effectiveness", f"{pruning_effectiveness:.1f}%")
                
                if len(df_queries) > 0:
                    avg_scan_ratio = (df_queries['partitions_scanned'] / df_queries['partitions_total'].replace(0, np.nan)).mean()
                    avg_scan_ratio = avg_scan_ratio if not pd.isna(avg_scan_ratio) else 0
                    st.metric("📊 Avg Scan Ratio", f"{avg_scan_ratio:.2%}")
    
    else:
        st.info("No query history data available. This could be due to:\n- No query activity in the selected period\n- Insufficient permissions to access QUERY_HISTORY table\n- Data latency (up to 45 minutes)")

# TAB 6: EXPENSIVE QUERIES
with tab6:
    st.header("💰 Expensive Queries Analysis")
    
    if not df_queries.empty:
        # Cost settings
        col1, col2, col3 = st.columns(3)
        with col1:
            credit_cost = st.number_input(
                "Cost per credit ($)", 
                min_value=0.0, 
                value=2.0, 
                step=0.1,
                help="Approximate cost per Snowflake credit in USD"
            )
        with col2:
            expense_threshold = st.slider(
                "Expense threshold (seconds)",
                min_value=10,
                max_value=300,
                value=60,
                step=10,
                help="Queries taking longer than this are considered expensive"
            )
        with col3:
            top_n = st.slider(
                "Show top N queries",
                min_value=10,
                max_value=100,
                value=25,
                step=5
            )
        
        st.markdown("---")
        
        # Calculate estimated costs
        df_queries['estimated_cost'] = df_queries['credits_used_cloud_services'] * credit_cost
        df_queries['is_expensive'] = df_queries['execution_time_seconds'] > expense_threshold
        
        # Expensive queries summary
        expensive_count = df_queries['is_expensive'].sum()
        expensive_cost = df_queries[df_queries['is_expensive']]['estimated_cost'].sum()
        total_cost = df_queries['estimated_cost'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🚨 Expensive Queries", f"{expensive_count:,}")
        with col2:
            st.metric("💸 Cost of Expensive Queries", f"${expensive_cost:.2f}")
        with col3:
            st.metric("💰 Total Query Cost", f"${total_cost:.2f}")
        with col4:
            cost_percentage = (expensive_cost / total_cost * 100) if total_cost > 0 else 0
            st.metric("📊 % of Total Cost", f"{cost_percentage:.1f}%")
        
        # Most expensive queries by different metrics
        st.subheader("🔝 Most Expensive Queries")
        
        metric_tab1, metric_tab2, metric_tab3, metric_tab4 = st.tabs([
            "⏱️ By Execution Time", 
            "💾 By Data Scanned", 
            "💰 By Estimated Cost",
            "🔄 By Resource Spillage"
        ])
        
        with metric_tab1:
            expensive_by_time = df_queries.nlargest(top_n, 'execution_time_seconds')[
                ['query_id', 'warehouse_name', 'user_name', 'execution_time_seconds', 
                 'bytes_scanned_gb', 'estimated_cost', 'query_type', 'start_time']
            ].copy()
            
            if not expensive_by_time.empty:
                expensive_by_time['execution_time_min'] = (expensive_by_time['execution_time_seconds'] / 60).round(2)
                expensive_by_time['estimated_cost'] = expensive_by_time['estimated_cost'].round(2)
                
                # Add severity coloring
                expensive_by_time['severity'] = pd.cut(
                    expensive_by_time['execution_time_seconds'],
                    bins=[0, 60, 300, 600, float('inf')],
                    labels=['⚪ Low', '🟡 Medium', '🟠 High', '🔴 Critical']
                )
                
                display_cols = ['severity', 'query_id', 'warehouse_name', 'user_name', 
                              'execution_time_min', 'bytes_scanned_gb', 'estimated_cost', 'query_type']
                
                st.dataframe(
                    expensive_by_time[display_cols],
                    use_container_width=True,
                    hide_index=True
                )
        
        with metric_tab2:
            expensive_by_data = df_queries.nlargest(top_n, 'bytes_scanned_gb')[
                ['query_id', 'warehouse_name', 'user_name', 'bytes_scanned_gb',
                 'execution_time_seconds', 'rows_produced', 'scan_efficiency', 'query_type']
            ].copy()
            
            if not expensive_by_data.empty:
                expensive_by_data['bytes_scanned_tb'] = (expensive_by_data['bytes_scanned_gb'] / 1024).round(3)
                expensive_by_data['execution_time_min'] = (expensive_by_data['execution_time_seconds'] / 60).round(2)
                
                st.dataframe(
                    expensive_by_data[['query_id', 'warehouse_name', 'user_name', 'bytes_scanned_tb',
                                     'execution_time_min', 'scan_efficiency', 'query_type']],
                    use_container_width=True,
                    hide_index=True
                )
        
        with metric_tab3:
            expensive_by_cost = df_queries.nlargest(top_n, 'estimated_cost')[
                ['query_id', 'warehouse_name', 'user_name', 'estimated_cost',
                 'credits_used_cloud_services', 'execution_time_seconds', 'bytes_scanned_gb']
            ].copy()
            
            if not expensive_by_cost.empty:
                expensive_by_cost['estimated_cost'] = expensive_by_cost['estimated_cost'].round(2)
                expensive_by_cost['execution_time_min'] = (expensive_by_cost['execution_time_seconds'] / 60).round(2)
                
                st.dataframe(expensive_by_cost, use_container_width=True, hide_index=True)
        
        with metric_tab4:
            # Queries with spillage
            spillage_queries = df_queries[
                (df_queries['bytes_spilled_to_local_storage'] > 0) | 
                (df_queries['bytes_spilled_to_remote_storage'] > 0)
            ].copy()
            
            if not spillage_queries.empty:
                spillage_queries['total_spill_gb'] = (
                    (spillage_queries['bytes_spilled_to_local_storage'] + 
                     spillage_queries['bytes_spilled_to_remote_storage']) / (1024**3)
                ).round(3)
                
                spillage_queries = spillage_queries.nlargest(top_n, 'total_spill_gb')[
                    ['query_id', 'warehouse_name', 'user_name', 'total_spill_gb',
                     'execution_time_seconds', 'bytes_scanned_gb', 'warehouse_size']
                ]
                
                spillage_queries['execution_time_min'] = (spillage_queries['execution_time_seconds'] / 60).round(2)
                
                st.dataframe(spillage_queries, use_container_width=True, hide_index=True)
            else:
                st.info("No queries with resource spillage found")
        
        # Query cost breakdown by warehouse
        st.subheader("💰 Cost Breakdown by Warehouse")
        
        warehouse_costs = df_queries.groupby('warehouse_name').agg({
            'estimated_cost': 'sum',
            'execution_time_seconds': 'sum',
            'query_id': 'count'
        }).round(2)
        
        warehouse_costs.columns = ['Total Cost ($)', 'Total Execution Time (s)', 'Query Count']
        warehouse_costs = warehouse_costs.sort_values('Total Cost ($)', ascending=False).head(10)
        warehouse_costs['Avg Cost per Query ($)'] = (
            warehouse_costs['Total Cost ($)'] / warehouse_costs['Query Count']
        ).round(2)
        
        fig_cost = px.bar(
            warehouse_costs.reset_index(),
            x='warehouse_name',
            y='Total Cost ($)',
            title="Estimated Query Costs by Warehouse",
            color='Avg Cost per Query ($)',
            color_continuous_scale='Reds'
        )
        fig_cost.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_cost, use_container_width=True)
        
        # Query patterns for expensive queries
        st.subheader("📊 Expensive Query Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Time distribution of expensive queries
            expensive_queries_df = df_queries[df_queries['is_expensive']].copy()
            if not expensive_queries_df.empty:
                expensive_queries_df['hour'] = expensive_queries_df['start_time'].dt.hour
                hourly_expensive = expensive_queries_df.groupby('hour').size().reset_index(name='count')
                
                fig_hourly = px.line(
                    hourly_expensive,
                    x='hour',
                    y='count',
                    title="Expensive Queries by Hour of Day",
                    markers=True
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # Query type distribution for expensive queries
            if not expensive_queries_df.empty and 'query_type' in expensive_queries_df.columns:
                type_dist = expensive_queries_df['query_type'].value_counts().head(10)
                
                fig_types = px.pie(
                    values=type_dist.values,
                    names=type_dist.index,
                    title="Expensive Query Types"
                )
                st.plotly_chart(fig_types, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Cost Optimization Recommendations")
        
        # Calculate recommendations
        recommendations = []
        
        # Check for queries with high spillage
        high_spillage = df_queries[
            ((df_queries['bytes_spilled_to_local_storage'] + 
              df_queries['bytes_spilled_to_remote_storage']) / (1024**3)) > 10
        ]
        if len(high_spillage) > 0:
            recommendations.append({
                "issue": "High Resource Spillage",
                "count": len(high_spillage),
                "recommendation": "Consider increasing warehouse size for these queries to reduce spillage",
                "potential_savings": f"${(len(high_spillage) * 0.5):.2f}"
            })
        
        # Check for inefficient scans
        inefficient_scans = df_queries[
            (df_queries['bytes_scanned_gb'] > 100) & 
            (df_queries['scan_efficiency'] < 0.01)
        ]
        if len(inefficient_scans) > 0:
            recommendations.append({
                "issue": "Inefficient Data Scans",
                "count": len(inefficient_scans),
                "recommendation": "Add clustering keys or partition pruning to reduce data scanned",
                "potential_savings": f"${(len(inefficient_scans) * 2):.2f}"
            })
        
        # Check for long-running queries
        long_running = df_queries[df_queries['execution_time_seconds'] > 600]
        if len(long_running) > 0:
            recommendations.append({
                "issue": "Long-Running Queries",
                "count": len(long_running),
                "recommendation": "Review query optimization, consider materialized views",
                "potential_savings": f"${(len(long_running) * 5):.2f}"
            })
        
        if recommendations:
            for rec in recommendations:
                with st.expander(f"🔧 {rec['issue']} ({rec['count']} queries)"):
                    st.write(f"**Recommendation:** {rec['recommendation']}")
                    st.write(f"**Potential Savings:** {rec['potential_savings']} per day")
        else:
            st.success("✅ No major cost optimization opportunities detected!")
    
    else:
        st.info("No query data available for expensive query analysis")

# TAB 7: RAW DATA
with tab7:
    st.header("📋 Raw Data")
    
    if not df.empty:
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            warehouse_filter = st.multiselect(
                "Filter by warehouse",
                df['warehouse_name'].unique(),
                default=[]
            )
        with col2:
            min_credits = st.number_input(
                "Minimum credits",
                min_value=0.0,
                value=0.0
            )
        
        # Apply filters
        df_display = df.copy(deep=True)
        if warehouse_filter:
            df_display = df_display[df_display['warehouse_name'].isin(warehouse_filter)]
        df_display = df_display[df_display['total_credits'] >= min_credits]
        
        # Format columns for display
        if 'week_start' in df_display.columns and not df_display.empty:
            df_display = df_display.copy()
            df_display['week_start'] = df_display['week_start'].astype(str).str[:10]
        
        # Round numeric columns
        numeric_columns = {
            'credit_change_pct': 1,
            'usage_change_pct': 1,
            'total_credits': 2,
            'compute_credits': 2,
            'cloud_credits': 2,
            'total_hours': 1,
            'avg_credits_per_period': 2,
            'stddev_credits': 2
        }
        
        for col, decimals in numeric_columns.items():
            if col in df_display.columns:
                df_display[col] = pd.to_numeric(df_display[col], errors='coerce').round(decimals)
        
        # Display statistics
        st.subheader("📊 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            unique_warehouses = len(df_display['warehouse_name'].unique()) if 'warehouse_name' in df_display.columns else 0
            st.metric("Total Warehouses", unique_warehouses)
        with col2:
            total_credits = df_display['total_credits'].sum() if 'total_credits' in df_display.columns else 0
            st.metric("Total Credits", f"{total_credits:,.2f}")
        with col3:
            total_hours = df_display['total_hours'].sum() if 'total_hours' in df_display.columns else 0
            st.metric("Total Hours", f"{total_hours:,.1f}")
        
        # Display data
        st.subheader("📋 Detailed Data")
        
        # Define columns to display
        display_columns = []
        if 'week_start' in df_display.columns:
            display_columns.append('week_start')
        if 'warehouse_name' in df_display.columns:
            display_columns.append('warehouse_name')
            
        # Add numeric columns in order
        for col in ['total_credits', 'compute_credits', 'cloud_credits', 'usage_count', 
                   'total_hours', 'credit_change_pct', 'usage_change_pct', 'avg_credits_per_period', 'stddev_credits']:
            if col in df_display.columns:
                display_columns.append(col)
        
        if display_columns:
            st.dataframe(
                df_display[display_columns],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No data columns available to display")
        
        # Download button
        if not df_display.empty:
            csv = df_display.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"warehouse_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No warehouse usage data available for the selected time period.")

# Footer with enhanced information in an expander
st.markdown("---")

with st.expander("💡 **Dashboard Information & Tips**", expanded=False):
    st.markdown("""
    ### 📊 **Data Sources & Refresh**
    - Data is cached for 10 minutes. Click 'Refresh Data' in the sidebar to get the latest information
    - Warehouse usage data in ACCOUNT_USAGE can have up to 3 hour latency
    
    ### 🔍 **Understanding the Metrics**
    - **Credits**: Snowflake compute credits consumed by warehouses
    - **Usage Count**: Number of metering periods (typically 1-second intervals when warehouse is active)
    - **Spikes**: Week-over-week percentage increases above your selected threshold
    - **Anomalies**: Unusual patterns detected using Isolation Forest algorithm (if scikit-learn is available)
    
    ### 🔐 **Permissions Required**
    - Your role must have access to `SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY` table
    - For query-level analysis, you need access to `SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY` table
    - Query history data has up to 45 minutes latency and covers the last 365 days
    
    ### 🎯 **Troubleshooting**
    - **No data**: Check warehouse activity, permissions, or data latency
    - **Forecasting not working**: Install Prophet library
    - **Anomaly detection not available**: Install scikit-learn library
    - **Slow performance**: Reduce the number of weeks in history or refresh data cache
    """)

# Feature availability status
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    status = "✅ Available" if PROPHET_AVAILABLE else "📦 Not Installed"
    st.markdown(f"**Forecasting**: {status}")

with col2:
    status = "✅ Available" if SKLEARN_AVAILABLE else "📦 Not Installed"
    st.markdown(f"**Anomaly Detection**: {status}")

with col3:
    try:
        import numpy
        numpy_status = "✅ Available"
    except ImportError:
        numpy_status = "📦 Not Installed"
    st.markdown(f"**Enhanced Analytics**: {numpy_status}")

with col4:
    st.markdown(f"**Debug Mode**: {'🔍 Enabled' if debug_mode else '🔒 Disabled'}")
