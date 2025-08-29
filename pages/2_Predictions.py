"""
Predictions page for the Work Utilization Prediction app.
Enhanced models for punch codes 206 & 213 using pipeline architecture.
Only column names updated for better display.
"""
import os
os.environ["STREAMLIT_SERVER_WATCH_CHANGES"] = "false"
import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import sys
import traceback
import plotly.graph_objects as go

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_engineering import EnhancedFeatureTransformer
from utils.prediction import predict_multiple_days
from utils.data_loader import load_enhanced_models
from utils.sql_data_connector import save_predictions_to_db
from utils.holiday_utils import is_non_working_day
from config import MODELS_DIR, ENHANCED_WORK_TYPES, PUNCH_CODE_NAMES
from utils.display_utils import get_display_name
from utils.page_auth import check_live_ad_page_access

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Workforce Predictions",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

check_live_ad_page_access()
# Initialize all session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'ts_data' not in st.session_state:
    st.session_state.ts_data = None
if 'enhanced_df' not in st.session_state:
    st.session_state.enhanced_df = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'enhanced_models' not in st.session_state:
    st.session_state.enhanced_models = None
if 'enhanced_metadata' not in st.session_state:
    st.session_state.enhanced_metadata = None
if 'enhanced_features' not in st.session_state:
    st.session_state.enhanced_features = None
if 'current_predictions' not in st.session_state:
    st.session_state.current_predictions = None
if 'current_hours_predictions' not in st.session_state:
    st.session_state.current_hours_predictions = None
if 'save_success_message' not in st.session_state:
    st.session_state.save_success_message = None

def ensure_data_and_models():
    """Ensure data and models are loaded"""
    try:
        # Load enhanced models if not loaded
        if st.session_state.enhanced_models is None:
            with st.spinner("Loading enhanced models..."):
                # Fix: load_enhanced_models() returns only 3 values, not 4
                enhanced_models, enhanced_metadata, enhanced_features = load_enhanced_models()
                
                if enhanced_models:
                    st.session_state.enhanced_models = enhanced_models
                    st.session_state.enhanced_metadata = enhanced_metadata
                    st.session_state.enhanced_features = enhanced_features
                    
                    # Load enhanced_df separately if needed
                    if st.session_state.enhanced_df is None:
                        st.session_state.enhanced_df = load_enhanced_data()
                    
                    logger.info(f"Loaded {len(enhanced_models)} enhanced models")
                else:
                    st.error("❌ Failed to load enhanced models")
                    return False
        
        return True
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        logger.error(f"Error loading models: {str(e)}")
        return False

def load_enhanced_data():
    """Load enhanced data separately"""
    try:
        training_data_path = os.path.join(MODELS_DIR, 'enhanced_training_data.pkl')
        if os.path.exists(training_data_path):
            df = pd.read_pickle(training_data_path)
            logger.info(f"✅ Loaded enhanced training data: {df.shape}")
            return df
        else:
            # Fallback to regular data loading
            from utils.data_loader import load_data
            df = load_data()
            logger.info("✅ Loaded fallback training data")
            return df
    except Exception as e:
        logger.warning(f"Could not load enhanced data: {str(e)}")
        return pd.DataFrame()  # Empty dataframe as fallback

def create_resource_plan_table(predictions, hours_predictions, selected_work_types, date_range):
    """Create a structured resource planning table"""
    resource_data = []
    
    for date in date_range:
        for work_type in selected_work_types:
            # Check if this is a working day for this punch code
            is_non_working, reason = is_non_working_day(date)
            
            if not is_non_working:
                # Working day - use actual predictions
                predicted_hours = predictions.get(date, {}).get(work_type, 0)
                hours_value = hours_predictions.get(date, {}).get(work_type, predicted_hours)
                workers_value = max(1, round(hours_value / 8.0)) if hours_value > 0 else 0
            else:
                # Non-working day - force to 0 (OVERRIDE any predictions)
                hours_value = 0
                workers_value = 0
            
            # Add rows for different metrics
            resource_data.extend([
                {
                    'Date': date,
                    'Day': date.strftime('%A'),
                    'PunchCode': work_type,
                    'Metric': 'Hours',
                    'Value': round(hours_value, 1)
                },
                {
                    'Date': date,
                    'Day': date.strftime('%A'),
                    'PunchCode': work_type,
                    'Metric': 'Workers',
                    'Value': workers_value
                }
            ])
    
    return pd.DataFrame(resource_data)

def main():
    st.header("Workforce Predictions")
    
    # Display save success message if it exists
    if st.session_state.save_success_message:
        st.success(st.session_state.save_success_message)
        # Clear the message after displaying it
        st.session_state.save_success_message = None
    
    # Check data and models
    if not ensure_data_and_models():
        return
    
    # Get available work types from enhanced models
    if st.session_state.enhanced_models:
        available_work_types = list(st.session_state.enhanced_models.keys())
    else:
        available_work_types = []
        st.warning("⚠️ No models loaded.")
        return

    # Date range selector
    col1, col2 = st.columns(2)
    
    with col1:
        # Use enhanced_df for date calculation
        if st.session_state.enhanced_df is not None:
            # Export for debugging (optional - can be removed)
            try:
                st.session_state.enhanced_df.to_excel('enhanced_df.xlsx')
            except Exception as e:
                logger.warning(f"Could not export enhanced_df: {str(e)}")
            
            latest_date = st.session_state.enhanced_df['Date'].max().date()
        else:
            # Fallback if enhanced_df not available
            latest_date = datetime.now().date()
        
        next_date = latest_date + timedelta(days=1)
        
        pred_start_date = st.date_input(
            "Start Date",
            value=next_date,
            min_value=latest_date,
            disabled=True,
            help="Select the start date for the prediction period"
        )
    
    with col2:
        # Number of days to predict
        num_days = st.slider(
            "Number of Days",
            min_value=1,
            max_value=365,
            value=7,
            help="Select the number of days to predict"
        )
        
        pred_end_date = pred_start_date + timedelta(days=num_days-1)
        st.write(f"End Date: {pred_end_date.strftime('%Y-%m-%d')}")
    
    # Work type selector (ORIGINAL - no display names)
    selected_work_types = st.multiselect(
        "Select Punch Codes",
        options=available_work_types,
        default=available_work_types,
        help="Select the work types for which you want to make predictions"
    )

    # Generate Predictions Button
    if st.button("🎯 Generate Predictions", type="primary"):
        if not selected_work_types:
            st.warning("Please select at least one work type for prediction")
        else:
            with st.spinner(f"Generating predictions for {num_days} days..."):
                try:
                    # Run predictions using enhanced models
                    # Filter models to only include selected work types
                    filtered_models = {wt: st.session_state.enhanced_models[wt] 
                                     for wt in selected_work_types 
                                     if wt in st.session_state.enhanced_models}
                    
                    # Call predict_multiple_days with correct parameters (4 parameters)
                    predictions, hours_predictions, metadata = predict_multiple_days(
                        st.session_state.enhanced_df,
                        filtered_models,
                        pred_start_date,
                        num_days
                    )
                    
                    if predictions:
                        st.session_state.current_predictions = predictions
                        st.session_state.current_hours_predictions = hours_predictions
                        st.success(f"✅ Predictions generated for {len(predictions)} days and {len(selected_work_types)} work types!")
                    else:
                        st.error("❌ Failed to generate predictions")
                        
                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
                    logger.error(f"Error generating predictions: {str(e)}")
                    logger.error(traceback.format_exc())

    # Only show predictions and save options if predictions exist
    if st.session_state.current_predictions:
        predictions = st.session_state.current_predictions
        hours_predictions = st.session_state.current_hours_predictions
        
        # Create a dataframe for display
        results_records = []
        
        for date, pred_dict in predictions.items():
            # Check if day is non-working
            is_non_working, reason = is_non_working_day(date)
            
            for work_type, value in pred_dict.items():
                # Only set to 0 if it's a non-working day
                display_value = 0 if is_non_working else value
                
                results_records.append({
                    'Date': date,
                    'Work Type': work_type,
                    'Predicted Hours': round(display_value, 1),
                    'Raw Prediction': round(value, 1),
                    'Day of Week': date.strftime('%A'),
                    'Is Non-Working Day': "Yes" if is_non_working else "No",
                    'Reason': reason if is_non_working else ""
                })
        
        results_df = pd.DataFrame(results_records)
        
        # Reconstruct date range for display
        first_date = min(predictions.keys())
        last_date = max(predictions.keys())

        # Add pivot table for resource planning
        st.subheader("Resource Planning View")

        # Generate the date range
        date_range = list(predictions.keys())
        selected_work_types_from_predictions = list(set(wt for pred_dict in predictions.values() for wt in pred_dict.keys()))
        
        # Create structured data with all selected work types
        resource_data = create_resource_plan_table(
            predictions, 
            hours_predictions, 
            selected_work_types_from_predictions,
            date_range
        )
        
        # ONLY CHANGE: Update column names in pivot tables using display names
        def get_column_display_name(punch_code):
            """Get display name for punch code in pivot table columns"""
            return get_display_name(punch_code, use_table_format=True)
        
        # Create daily pivot table with punch codes as columns and metrics as sub-columns
        daily_pivot = pd.pivot_table(
            resource_data,
            values='Value',
            index=['Date', 'Day'],
            columns=['PunchCode', 'Metric'],
            fill_value=0
        )
        
        # Rename columns to use display names
        new_columns = []
        for col in daily_pivot.columns:
            if isinstance(col, tuple) and len(col) == 2:
                punch_code, metric = col
                display_name = get_column_display_name(punch_code)
                new_columns.append((display_name, metric))
            else:
                new_columns.append(col)
        daily_pivot.columns = pd.MultiIndex.from_tuples(new_columns)
        
        # Create monthly pivot table
        resource_data['Month'] = resource_data['Date'].dt.strftime('%Y-%m')
        monthly_pivot = pd.pivot_table(
            resource_data,
            values='Value',
            index='Month',
            columns=['PunchCode', 'Metric'],
            aggfunc='sum',
            fill_value=0
        )
        
        # Rename columns to use display names for monthly pivot
        new_monthly_columns = []
        for col in monthly_pivot.columns:
            if isinstance(col, tuple) and len(col) == 2:
                punch_code, metric = col
                display_name = get_column_display_name(punch_code)
                new_monthly_columns.append((display_name, metric))
            else:
                new_monthly_columns.append(col)
        monthly_pivot.columns = pd.MultiIndex.from_tuples(new_monthly_columns)
        
        # Display options
        view_option = st.radio(
            "View Format:",
            ["Daily View", "Monthly Summary", "Raw Data"],
            horizontal=True
        )
        
        if view_option == "Daily View":
            st.write("### Daily Resource Planning")
            st.dataframe(daily_pivot, use_container_width=True)
            
        elif view_option == "Monthly Summary":
            st.write("### Monthly Resource Summary")
            st.dataframe(monthly_pivot, use_container_width=True)
            
        else:  # Raw Data
            st.write("### Raw Prediction Data")
            st.dataframe(results_df, use_container_width=True)
        
        # Save predictions section
        st.subheader("💾 Save Predictions")
        
        # save_description = st.text_input(
        #     "Description (Optional)",
        #     value=f"Prediction for {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}",
        #     help="Enter a description for this prediction batch"
        # )
        
        if st.button("💾 Save to Database", type="secondary"):
            with st.spinner("Saving predictions to database..."):
                try:
                    # Get current username
                    username = "streamlit_user"  # You might want to get this from session or config
                    
                    # Convert predictions to the correct format expected by save_predictions_to_db
                    # The function expects: (predictions_dict, hours_dict, username)
                    
                    # predictions should be: {date: {work_type: predicted_workers}}
                    # hours_predictions should be: {date: {work_type: predicted_hours}}
                    
                    # Save to database using the correct function signature
                    success = save_predictions_to_db(predictions, hours_predictions, username)
                    
                    if success:
                        st.session_state.save_success_message = f"✅ Successfully saved predictions to database!"
                        st.rerun()
                    else:
                        st.error("❌ Failed to save predictions to database")
                        
                except Exception as e:
                    st.error(f"Error saving predictions: {str(e)}")
                    logger.error(f"Error saving predictions: {str(e)}")

if __name__ == "__main__":
    main()