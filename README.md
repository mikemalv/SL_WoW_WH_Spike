# SL_WoW_WH_Spike

# ❄️ Snowflake Warehouse Usage Monitor

A Streamlit-based dashboard for monitoring Snowflake warehouse credit consumption, identifying usage spikes, and analyzing week-over-week trends.

<img width="1415" alt="image" src="https://github.com/user-attachments/assets/cfad637d-a77c-4748-bd81-2042991cc279" />


## 🚀 Features

### 🚨 Spike Detection
- Automatically identifies warehouses with significant week-over-week credit usage increases
- Configurable spike threshold (10-200% increase)
- Shows top 5 largest spikes with detailed metrics
- Interactive heatmap visualization of week-over-week changes

### 📈 Trends Analysis
- Multi-warehouse comparison charts
- Weekly credit usage trends
- Usage count and total hours tracking
- Interactive line charts with unified hover mode

### 📊 Current Week Overview
- Real-time metrics for the current week
- Top 5 warehouses by credit consumption
- Active warehouse count
- Total usage statistics

### 📋 Raw Data Export
- Filterable data table
- Export to CSV functionality
- Summary statistics
- Warehouse and credit filtering options

## 📋 Prerequisites

- Python 3.10+
- Snowflake account with access to `SNOWFLAKE.ACCOUNT_USAGE` schema
- Snowflake role with permissions to query `WAREHOUSE_METERING_HISTORY` table
- Streamlit in Snowflake (SiS) environment

## 📦 Dependencies

```python
streamlit
pandas
plotly
snowflake-snowpark-python
```

## 🔧 Configuration

### Settings Panel
- **Weeks of history**: Adjustable from 2-12 weeks
- **Refresh Data**: Manual data refresh button
- **Debug Mode**: Toggle for additional diagnostic information

### Data Caching
- Warehouse usage data is cached for 10 minutes
- Manual refresh available via sidebar button

## 📊 Data Sources

The dashboard uses the following Snowflake system tables:
- `SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY` - Primary source for credit consumption data
- Data latency: Up to 3 hours

### Key Metrics Tracked
- **Total Credits**: Overall credit consumption
- **Compute Credits**: Credits used for compute operations
- **Cloud Services Credits**: Credits used for cloud services
- **Usage Count**: Number of warehouse usage periods
- **Total Hours**: Cumulative warehouse runtime

## 🎯 Use Cases

1. **Cost Optimization**
   - Identify warehouses with unexpected usage spikes
   - Monitor credit consumption trends
   - Plan capacity based on historical patterns

2. **Performance Monitoring**
   - Track warehouse utilization
   - Identify usage patterns
   - Detect anomalies in consumption

3. **Budget Planning**
   - Analyze week-over-week trends
   - Forecast future credit usage
   - Allocate resources efficiently

## 🚦 Usage Guide

1. **Launch the Dashboard**
   - Open the Streamlit app in Snowflake
   - Wait for initial data load

2. **Spike Detection (Default Tab)**
   - Adjust spike threshold slider
   - Review identified spikes
   - Analyze heatmap for patterns

3. **Trends Analysis**
   - Select warehouses to compare
   - Review credit, usage, and hours trends
   - Identify long-term patterns

4. **Current Week Overview**
   - Monitor real-time metrics
   - Identify top consumers
   - Track active warehouse count

5. **Export Data**
   - Apply filters as needed
   - Download CSV for offline analysis

## ⚠️ Important Notes

- **Data Latency**: ACCOUNT_USAGE views have up to 3-hour latency
- **Permissions**: Ensure your role has access to ACCOUNT_USAGE schema
- **Credit Tracking**: Dashboard tracks warehouse credits only, not overall account credits
- **Query-Level Details**: For query-specific analysis, additional access to QUERY_HISTORY table is required

## 🐛 Troubleshooting

### No Data Displayed
- Verify role permissions for ACCOUNT_USAGE schema
- Check if warehouses have been active in the selected timeframe
- Consider data latency (up to 3 hours)

### Permission Errors
```sql
-- Grant necessary permissions
GRANT SELECT ON SCHEMA SNOWFLAKE.ACCOUNT_USAGE TO ROLE your_role;
GRANT SELECT ON TABLE SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY TO ROLE your_role;
```

### Performance Issues
- Reduce the weeks of history
- Use warehouse filters to limit data
- Clear cache and refresh if needed

## 📈 Future Enhancements

- [ ] Email alerts for spike detection
- [ ] Integration with QUERY_HISTORY for detailed analysis
- [ ] Predictive analytics for credit forecasting
- [ ] Cost allocation by department/project
- [ ] Automated warehouse suspension recommendations
