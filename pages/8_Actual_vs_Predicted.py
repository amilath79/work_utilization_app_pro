import streamlit as st
import pandas as pd
from utils.sql_data_connector import load_utilization_vs_prediction
from utils.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from utils.display_utils import get_display_name 
from config import PUNCH_CODE_NAMES 
from utils.page_auth import check_live_ad_page_access   
st.set_page_config(page_title="Prediction vs Actuals", layout="wide")


check_live_ad_page_access()

st.title("🔍 Actual vs Predicted Performance")
st.markdown("Compare actual utilization data with model predictions to evaluate performance.")

# Sidebar filters
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2025-07-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-08-15"))

# Load data
df = load_utilization_vs_prediction(start_date, end_date)

if df.empty:
    st.warning("No data found for the selected date range.")
    st.stop()

# # Metrics
# st.subheader("📊 Model Performance Metrics")
# col1, col2, col3 = st.columns(3)

# with col1:
#     mae = mean_absolute_error(df["ActualHours"], df["PredictedHours"])
#     st.metric("MAE (Hours)", f"{mae:.2f}")

# with col2:
#     rmse = root_mean_squared_error(df["ActualHours"], df["PredictedHours"])
#     st.metric("RMSE (Hours)", f"{rmse:.2f}")

# with col3:
#     mape = mean_absolute_percentage_error(df["ActualHours"], df["PredictedHours"])
#     st.metric("MAPE (Hours)", f"{mape:.2f}%")

# Visual comparison
st.subheader("📈 Predicted vs Actual (Hours)")

unique_punchcodes = df["PunchCode"].unique()
punch_code_options = {}
for pc in unique_punchcodes:
    display_name = get_display_name(pc, use_table_format=True)
    punch_code_options[f"{pc}"] = pc

selected_punchcodes = st.multiselect(
    "Select Work Types to view",
    options=list(punch_code_options.keys()),
    default=list(punch_code_options.keys())[:10],  # Default to first 10
    format_func=lambda x: x  # Display the full formatted name
)

# Convert selected display names back to punch codes
selected_punch_codes = [punch_code_options[display_name] for display_name in selected_punchcodes]

filtered_df = df[df["PunchCode"].isin(selected_punch_codes)]

for punchcode in selected_punch_codes:
    display_name = get_display_name(punchcode, use_table_format=True)
    st.write(f"### {display_name} (Code: {punchcode})")
    pc_df = filtered_df[filtered_df["PunchCode"] == punchcode]
    chart_df = pc_df[["Date", "ActualHours", "PredictedHours"]].set_index("Date").sort_index()
    st.line_chart(chart_df)

with st.expander("🔍 View Raw Data"):
    # Create display dataframe with enhanced punch code information
    display_df = df.copy()
    display_df['Work Type'] = display_df['PunchCode'].apply(lambda x: get_display_name(x, use_table_format=True))
    
    # Reorder columns to show Work Type prominently
    cols = ['Date', 'Work Type', 'PunchCode'] + [col for col in display_df.columns if col not in ['Date', 'Work Type', 'PunchCode']]
    display_df = display_df[cols]
    
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "Work Type": st.column_config.TextColumn("Work Type", width="medium"),
            "PunchCode": st.column_config.TextColumn("Code", width="small"),
            "ActualHours": st.column_config.NumberColumn("Actual Hours", format="%.1f"),
            "PredictedHours": st.column_config.NumberColumn("Predicted Hours", format="%.1f")
        }
    )
