import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from snowflake.snowpark.context import get_active_session

# Get the current session
session = get_active_session()

# Page config
st.set_page_config(
    page_title="Snowflake Warehouse Monitor",
    page_icon="❄️",
    layout="wide"
)

# Title and description
st.title("❄️ Snowflake Warehouse Usage Monitor")
st.markdown("Monitor week-over-week warehouse credit usage and identify consumption spikes")
st.info("📊 This dashboard uses the WAREHOUSE_METERING_HISTORY table which tracks warehouse credit consumption in hourly intervals. For query-level details, a separate QUERY_HISTORY analysis would be needed.")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Date range selector
    weeks_back = st.slider("Weeks of history", min_value=2, max_value=12, value=4)
    
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
elif refresh_button:
    st.session_state.refresh_needed = True
else:
    st.session_state.refresh_needed = False

# Function to get warehouse usage data
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
            SUM(TIMESTAMPDIFF(second, start_time, end_time))/3600.0 as total_hours
        FROM snowflake.account_usage.warehouse_metering_history
        WHERE start_time >= DATEADD('week', -{weeks_back}, CURRENT_DATE())
            AND start_time < CURRENT_DATE()
            AND warehouse_name IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 2, 1;
        """
        
        # Execute query and get results
        result = session.sql(query)
        
        # Convert to pandas with explicit column names
        df = pd.DataFrame(result.collect())
        
        # Rename columns to ensure clarity
        df.columns = ['week_start', 'warehouse_name', 'total_credits', 'compute_credits', 
                     'cloud_credits', 'usage_count', 'total_hours']
        
        # Ensure proper data types
        df['week_start'] = pd.to_datetime(df['week_start'])
        df['total_credits'] = pd.to_numeric(df['total_credits'], errors='coerce')
        df['compute_credits'] = pd.to_numeric(df['compute_credits'], errors='coerce')
        df['cloud_credits'] = pd.to_numeric(df['cloud_credits'], errors='coerce')
        df['usage_count'] = pd.to_numeric(df['usage_count'], errors='coerce')
        df['total_hours'] = pd.to_numeric(df['total_hours'], errors='coerce')
        
        # Calculate week-over-week changes
        df = df.sort_values(['warehouse_name', 'week_start'])
        
        # Add previous week values
        df['prev_week_credits'] = df.groupby('warehouse_name')['total_credits'].shift(1)
        df['prev_week_usage_count'] = df.groupby('warehouse_name')['usage_count'].shift(1)
        
        # Calculate percentage changes
        df['credit_change_pct'] = ((df['total_credits'] - df['prev_week_credits']) / df['prev_week_credits'] * 100).round(1)
        df['usage_change_pct'] = ((df['usage_count'] - df['prev_week_usage_count']) / df['prev_week_usage_count'] * 100).round(1)
        
        # Fill NaN values with 0 for percentage changes
        df['credit_change_pct'] = df['credit_change_pct'].fillna(0)
        df['usage_change_pct'] = df['usage_change_pct'].fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Error fetching warehouse usage data: {str(e)}")
        return pd.DataFrame()  # Return empty dataframe on error

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
        return pd.DataFrame()  # Return empty dataframe on error

# Function to clean column names and ensure uniqueness
def clean_column_names(df):
    """Clean column names and ensure they are unique"""
    if df is None or df.empty:
        return df
    
    # Get column names
    cols = df.columns.tolist()
    
    # Clean names - remove spaces, convert to lowercase
    clean_cols = [col.strip().lower().replace(' ', '_') for col in cols]
    
    # Ensure uniqueness
    seen = {}
    unique_cols = []
    for col in clean_cols:
        if col in seen:
            seen[col] += 1
            unique_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            unique_cols.append(col)
    
    df.columns = unique_cols
    return df
@st.cache_data(ttl=600)
def get_query_stats(weeks_back):
    try:
        query = f"""
        SELECT 
            DATE_TRUNC('week', start_time) as week_start,
            warehouse_name,
            COUNT(DISTINCT query_id) as query_count,
            AVG(total_elapsed_time)/1000 as avg_query_seconds,
            MAX(total_elapsed_time)/1000 as max_query_seconds,
            MEDIAN(total_elapsed_time)/1000 as median_query_seconds
        FROM snowflake.account_usage.query_history
        WHERE start_time >= DATEADD('week', -{weeks_back}, CURRENT_DATE())
            AND start_time < CURRENT_DATE()
            AND warehouse_name IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 2;
        """
        
        df = session.sql(query).to_pandas()
        
        # Convert column names to lowercase for consistency
        df.columns = df.columns.str.lower()
        
        # Convert week_start to datetime
        df['week_start'] = pd.to_datetime(df['week_start'])
        
        return df
    except Exception as e:
        # Return None if query_history is not accessible
        return None

# Fetch data
if st.session_state.refresh_needed or st.session_state.df_usage is None:
    with st.spinner("Fetching warehouse usage data..."):
        try:
            # Clear cache if refresh is requested
            if refresh_button and hasattr(get_warehouse_usage, 'clear'):
                try:
                    get_warehouse_usage.clear()
                    get_current_week_snapshot.clear()
                except:
                    pass  # Ignore cache clear errors
            
            st.session_state.df_usage = get_warehouse_usage(weeks_back)
            st.session_state.df_current = get_current_week_snapshot()
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

# Check if data was loaded successfully
if st.session_state.df_usage is None or st.session_state.df_current is None:
    st.error("Failed to load data. Please check your permissions to access SNOWFLAKE.ACCOUNT_USAGE schema.")
    st.stop()

# Get data from session state
df = st.session_state.df_usage
df_current = st.session_state.df_current

# Ensure dataframes are not None and have proper structure
if df is None:
    df = pd.DataFrame()
else:
    # Clean the dataframe to ensure no duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
if df_current is None:
    df_current = pd.DataFrame()
else:
    # Clean the dataframe to ensure no duplicate columns
    df_current = df_current.loc[:, ~df_current.columns.duplicated()]

# Additional check for empty dataframes
if df is None or df.empty or df_current is None:
    st.warning("No warehouse usage data found for the selected time period.")
    st.info("This could mean:\n- No warehouses have been used in the selected timeframe\n- You don't have access to SNOWFLAKE.ACCOUNT_USAGE schema\n- There's a delay in usage data (up to 3 hours)")
    # Create empty tabs to maintain UI structure
    tab1, tab2, tab3, tab4 = st.tabs(["🚨 Spike Detection", "📈 Trends", "📊 Overview", "📋 Raw Data"])
    with tab1:
        st.info("No data available for spike detection.")
    with tab2:
        st.info("No data available for trends.")
    with tab3:
        st.info("No data available to display.")
    with tab4:
        st.info("No raw data available.")
    st.stop()

# Tabs for different views - REORDERED
tab1, tab2, tab3, tab4 = st.tabs(["🚨 Spike Detection", "📈 Trends", "📊 Overview", "📋 Raw Data"])

# TAB 1: SPIKE DETECTION (previously tab3)
with tab1:
    st.header("🚨 Spike Detection")
    
    if df is not None and not df.empty:
        # Spike threshold
        spike_threshold = st.slider(
            "Spike threshold (% increase)",
            min_value=10,
            max_value=200,
            value=50,
            step=10
        )
        
        # Find spikes
        spikes = df[
            (df['credit_change_pct'] > spike_threshold) & 
            (df['credit_change_pct'].notna()) &
            (df['credit_change_pct'] != float('inf'))
        ].sort_values('credit_change_pct', ascending=False)
        
        if len(spikes) > 0:
            st.warning(f"⚠️ Found {len(spikes)} spikes above {spike_threshold}% increase")
            
            # Spike details
            st.subheader("Top 5 Largest Spikes")
            spike_count = min(5, len(spikes))
            for idx in range(spike_count):
                spike_row = spikes.iloc[idx]
                
                # Safe datetime formatting
                if pd.notnull(spike_row['week_start']):
                    try:
                        week_str = spike_row['week_start'].strftime('%Y-%m-%d')
                    except:
                        week_str = str(spike_row['week_start'])[:10]
                else:
                    week_str = 'Unknown'
                
                with st.expander(
                    f"{idx + 1}. {spike_row['warehouse_name']} - Week of {week_str} "
                    f"(+{spike_row['credit_change_pct']:.1f}%)"
                ):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    prev_credits = spike_row['prev_week_credits'] if pd.notnull(spike_row['prev_week_credits']) else 0
                    credit_diff = spike_row['total_credits'] - prev_credits
                    
                    col1.metric(
                        "Current Week Credits",
                        f"{spike_row['total_credits']:.2f}",
                        f"+{credit_diff:.2f}"
                    )
                    
                    col2.metric(
                        "Previous Week Credits",
                        f"{prev_credits:.2f}"
                    )
                    
                    if pd.notnull(spike_row['prev_week_usage_count']):
                        usage_diff = int(spike_row['usage_count'] - spike_row['prev_week_usage_count'])
                        col3.metric(
                            "Usage Count",
                            f"{int(spike_row['usage_count']):,}",
                            f"+{usage_diff:,}"
                        )
                    else:
                        col3.metric(
                            "Usage Count",
                            f"{int(spike_row['usage_count']):,}"
                        )
                    
                    col4.metric(
                        "Change %",
                        f"{spike_row['credit_change_pct']:.1f}%"
                    )
                    
                    # Additional info in a second row
                    st.caption(f"Total Hours: {spike_row['total_hours']:.1f}")
        else:
            st.info(f"No spikes found above {spike_threshold}% threshold")
        
        # Heatmap section moved here
        st.subheader("Week-over-Week Change Heatmap")
        
        try:
            # Create a clean dataframe for the heatmap
            heatmap_df = df[['warehouse_name', 'week_start', 'credit_change_pct']].copy()
            
            # Convert week_start to string to avoid datetime issues
            heatmap_df['week_start'] = heatmap_df['week_start'].astype(str).str[:10]
            
            # Remove any duplicates by aggregating
            heatmap_df = heatmap_df.groupby(['warehouse_name', 'week_start'])['credit_change_pct'].mean().reset_index()
            
            # Prepare data for heatmap using pivot
            heatmap_data = heatmap_df.pivot(
                index='warehouse_name',
                columns='week_start',
                values='credit_change_pct'
            ).fillna(0)
            
            if len(heatmap_data) > 0 and len(heatmap_data.columns) > 0:
                # Create heatmap
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=heatmap_data.values,
                    x=heatmap_data.columns.tolist(),
                    y=heatmap_data.index.tolist(),
                    colorscale='RdYlBu_r',
                    zmid=0,
                    text=heatmap_data.values.round(1),
                    texttemplate='%{text}%',
                    textfont={"size": 10},
                    colorbar=dict(title="% Change")
                ))
                
                fig_heatmap.update_layout(
                    title="Week-over-Week Credit Usage Change (%)",
                    xaxis_title="Week Starting",
                    yaxis_title="Warehouse",
                    height=max(400, len(heatmap_data.index) * 25)
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("Not enough data to generate heatmap. Need at least 2 weeks of data.")
        except Exception as e:
            st.info(f"Unable to generate heatmap: {str(e)}")
    else:
        st.info("No warehouse usage data available for spike detection.")

# TAB 2: TRENDS (previously tab2)
with tab2:
    st.header("Week-over-Week Trends")
    
    # Warehouse selector
    if df is not None and not df.empty and 'warehouse_name' in df.columns:
        warehouses = sorted(df['warehouse_name'].unique())
        
        # Get top 5 warehouses by total credits
        top_warehouses = df.groupby('warehouse_name')['total_credits'].sum().nlargest(5).index.tolist()
        
        selected_warehouses = st.multiselect(
            "Select warehouses to compare",
            options=warehouses,
            default=top_warehouses[:5] if top_warehouses else []
        )
    else:
        st.warning("No warehouse data available for the selected time period")
        selected_warehouses = []
    
    if selected_warehouses and df is not None and not df.empty:
        # Filter data and create a clean copy
        df_filtered = df[df['warehouse_name'].isin(selected_warehouses)].copy()
        
        # Reset index to ensure no index issues
        df_filtered = df_filtered.reset_index(drop=True)
        
        # Debug information
        if st.session_state.get('debug_mode', False):
            with st.expander("Debug Information"):
                st.write("DataFrame Info:")
                st.write(f"Shape: {df_filtered.shape}")
                st.write(f"Columns: {df_filtered.columns.tolist()}")
                st.write(f"Column types: {df_filtered.dtypes.to_dict()}")
                st.write(f"Any duplicated columns: {df_filtered.columns.duplicated().any()}")
                if df_filtered.columns.duplicated().any():
                    st.write(f"Duplicated columns: {df_filtered.columns[df_filtered.columns.duplicated()].tolist()}")
                st.write("First few rows:")
                st.dataframe(df_filtered.head())
        
        # Create separate dataframes for each chart to avoid issues
        # Credit usage trend
        try:
            # Create a completely new dataframe for plotting
            plot_data = pd.DataFrame({
                'week_start': df_filtered['week_start'].values,
                'warehouse_name': df_filtered['warehouse_name'].values,
                'total_credits': df_filtered['total_credits'].values
            })
            
            fig_trend = px.line(
                plot_data,
                x='week_start',
                y='total_credits',
                color='warehouse_name',
                title="Weekly Credit Usage Trend",
                labels={'total_credits': 'Credits Used', 'week_start': 'Week Starting'},
                markers=True
            )
            fig_trend.update_layout(hovermode='x unified')
            st.plotly_chart(fig_trend, use_container_width=True)
        except Exception as e:
            # Fallback to plotly graph objects
            try:
                st.warning("Using alternative charting method")
                fig_trend = go.Figure()
                
                for warehouse in df_filtered['warehouse_name'].unique():
                    warehouse_data = df_filtered[df_filtered['warehouse_name'] == warehouse]
                    fig_trend.add_trace(go.Scatter(
                        x=warehouse_data['week_start'].tolist(),
                        y=warehouse_data['total_credits'].tolist(),
                        mode='lines+markers',
                        name=str(warehouse),
                        line=dict(width=2),
                        marker=dict(size=8)
                    ))
                
                fig_trend.update_layout(
                    title="Weekly Credit Usage Trend",
                    xaxis_title="Week Starting",
                    yaxis_title="Credits Used",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            except Exception as e2:
                st.error(f"Error creating chart: {str(e)}")
                if st.session_state.get('debug_mode', False):
                    st.write("Original error:", str(e))
                    st.write("Fallback error:", str(e2))
                    st.write("DataFrame columns:", df_filtered.columns.tolist())
                    st.write("DataFrame shape:", df_filtered.shape)
        
        # Usage count trend
        if 'usage_count' in df_filtered.columns:
            try:
                chart_data = df_filtered[['week_start', 'warehouse_name', 'usage_count']].copy()
                chart_data = chart_data.reset_index(drop=True)
                
                fig_usage = px.line(
                    chart_data,
                    x='week_start',
                    y='usage_count',
                    color='warehouse_name',
                    title="Weekly Usage Count Trend",
                    labels={'usage_count': 'Number of Usage Periods', 'week_start': 'Week Starting'},
                    markers=True
                )
                fig_usage.update_layout(hovermode='x unified')
                st.plotly_chart(fig_usage, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating usage trend chart: {str(e)}")
                if st.session_state.get('debug_mode', False):
                    st.exception(e)
        
        # Total hours trend
        if 'total_hours' in df_filtered.columns:
            try:
                chart_data = df_filtered[['week_start', 'warehouse_name', 'total_hours']].copy()
                chart_data = chart_data.reset_index(drop=True)
                
                fig_hours = px.line(
                    chart_data,
                    x='week_start',
                    y='total_hours',
                    color='warehouse_name',
                    title="Weekly Warehouse Hours Trend",
                    labels={'total_hours': 'Total Hours', 'week_start': 'Week Starting'},
                    markers=True
                )
                fig_hours.update_layout(hovermode='x unified')
                st.plotly_chart(fig_hours, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating hours trend chart: {str(e)}")
                if st.session_state.get('debug_mode', False):
                    st.exception(e)
    else:
        if not selected_warehouses:
            st.info("Please select at least one warehouse to view trends")
        else:
            st.info("No data available for selected warehouses")

# TAB 3: OVERVIEW (previously tab1)
with tab3:
    st.header("Current Week Overview")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    if df_current is not None and not df_current.empty:
        total_credits_this_week = df_current['credits_this_week'].sum() if 'credits_this_week' in df_current.columns else 0
        total_usage_count_this_week = df_current['usage_count_this_week'].sum() if 'usage_count_this_week' in df_current.columns else 0
        active_warehouses = len(df_current)
        top_warehouse = df_current.iloc[0]['warehouse_name'] if 'warehouse_name' in df_current.columns else "N/A"
    else:
        total_credits_this_week = 0
        total_usage_count_this_week = 0
        active_warehouses = 0
        top_warehouse = "N/A"
    
    col1.metric("Total Credits This Week", f"{total_credits_this_week:,.2f}")
    col2.metric("Usage Count", f"{total_usage_count_this_week:,}")
    col3.metric("Active Warehouses", active_warehouses)
    col4.metric("Top Consumer", top_warehouse)
    
    # Top warehouses bar chart
    st.subheader("Top 5 Warehouses by Credits (Current Week)")
    if df_current is not None and len(df_current) > 0:
        fig_top = px.bar(
            df_current.head(5), 
            x='warehouse_name', 
            y='credits_this_week',
            title="Credit Usage by Warehouse",
            labels={'credits_this_week': 'Credits Used', 'warehouse_name': 'Warehouse'},
            color='credits_this_week',
            color_continuous_scale='Blues'
        )
        fig_top.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.info("No warehouse usage data for the current week")

# TAB 4: RAW DATA (remains the same)
with tab4:
    st.header("Raw Data")
    
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
            # Convert week_start to string format safely
            df_display = df_display.copy()
            df_display['week_start'] = df_display['week_start'].astype(str).str[:10]
        
        # Round numeric columns
        numeric_columns = {
            'credit_change_pct': 1,
            'usage_change_pct': 1,
            'total_credits': 2,
            'compute_credits': 2,
            'cloud_credits': 2,
            'total_hours': 1
        }
        
        for col, decimals in numeric_columns.items():
            if col in df_display.columns:
                df_display[col] = pd.to_numeric(df_display[col], errors='coerce').round(decimals)
        
        # Display statistics
        st.subheader("Summary Statistics")
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
        st.subheader("Detailed Data")
        
        # Define columns to display
        display_columns = []
        if 'week_start' in df_display.columns:
            display_columns.append('week_start')
        if 'warehouse_name' in df_display.columns:
            display_columns.append('warehouse_name')
            
        # Add numeric columns in order
        for col in ['total_credits', 'compute_credits', 'cloud_credits', 'usage_count', 
                   'total_hours', 'credit_change_pct', 'usage_change_pct']:
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

# Footer
st.markdown("---")
st.markdown("💡 **Tips**:")
st.markdown("- Data is cached for 10 minutes. Click 'Refresh Data' in the sidebar to get the latest information.")
st.markdown("- Warehouse usage data in ACCOUNT_USAGE can have up to 3 hour latency.")
st.markdown("- This dashboard uses WAREHOUSE_METERING_HISTORY which tracks credit consumption periods, not individual queries.")
st.markdown("- For query-level details, you would need access to the QUERY_HISTORY table.")
st.markdown("- Ensure your role has access to SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY table.")
