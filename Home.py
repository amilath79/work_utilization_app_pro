"""
Home page for the Work Utilization Prediction app.
Lightweight landing page with status checks only.
"""
import streamlit as st
import pandas as pd
import logging
import os
import sys
from datetime import datetime
from config import ENTERPRISE_CONFIG, enterprise_logger


# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import load_data, load_combined_models
from utils.sql_data_connector import extract_sql_data
from utils.state_manager import StateManager
from config import DATA_DIR, MODELS_DIR, APP_TITLE, SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION, APP_ICON
from utils.page_auth import check_live_ad_page_access
from utils.brand_styling import load_brand_css

# Configure page - SINGLE CONFIG ONLY
if ENTERPRISE_CONFIG.enterprise_mode:
    st.set_page_config(
        page_title=f"Enterprise - {APP_TITLE}",
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
else:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )


check_live_ad_page_access('Home.py')
# Configure logger
logger = logging.getLogger(__name__)

def check_database_connection():
    """Quick check if database is accessible"""
    try:
        test_query = "SELECT COUNT(*) as count FROM WorkUtilizationData WHERE Date >= '2025-01-01'"
        result = extract_sql_data(
            server=SQL_SERVER,
            database=SQL_DATABASE,
            query=test_query,
            trusted_connection=SQL_TRUSTED_CONNECTION
        )
        return result is not None and not result.empty
    except:
        return False


def check_data_availability():
    """Quick check of data availability without loading"""
    try:
        # Check database first
        if check_database_connection():
            return "Database", "✅ Connected"
        
        # Check for sample data
        sample_path = os.path.join(DATA_DIR, "sample_work_utilization.xlsx")
        if os.path.exists(sample_path):
            return "File", "📁 Sample data available"
        
        return "None", "❌ No data source found"
    except Exception as e:
        return "Error", f"❌ Error: {str(e)}"

def check_models_status():
    """Quick check if models are trained"""
    try:
        if os.path.exists(MODELS_DIR):
            # Check for enhanced models
            enhanced_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('enhanced_model_') and f.endswith('.pkl')]
            if enhanced_files:
                return "Enhanced", f"✅ {len(enhanced_files)} enhanced models ready"
            
            # Check for standard models
            model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('_model.pkl')]
            if model_files:
                return "Standard", f"✅ {len(model_files)} standard models ready"
        
        return "None", "❌ No trained models found"
    except Exception as e:
        return "Error", f"❌ Error checking models: {str(e)}"

def main():
    # Simple enterprise page configuration
    if ENTERPRISE_CONFIG.enterprise_mode:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🏢 Enterprise Mode")
        st.sidebar.markdown(f"🔒 Environment: {ENTERPRISE_CONFIG.environment.value}")

    
    st.title("Work Utilization Prediction")
    
    st.write("""
    Welcome to the Work Utilization Prediction application! This tool helps you predict
    workforce requirements based on historical data using machine learning.
    
    Use the sidebar to navigate between different pages.
    """)
    
    # Quick Status Dashboard
    st.header(" System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💾 Data Sources")
        data_type, data_status = check_data_availability()
        st.write(data_status)
        
        if data_type == "Database":
            st.info("💡 Data will be loaded automatically when you visit Data Overview page")
        elif data_type == "File":
            st.info("💡 Upload your data file in the Data Overview page")
        else:
            st.warning("💡 Please upload data or check database connection in Data Overview page")
    
    with col2:
        st.subheader("🤖 ML Models")
        model_type, model_status = check_models_status()
        st.write(model_status)
        
        if model_type == "None":
            st.warning("💡 Run `python train_models2.py` to train models")
        else:
            st.info("💡 Models ready for predictions")
    
    
    # Application Info
    with st.expander("ℹ️ Application Information"):
        st.write("**Supported Work Types (Punch Codes):** 202, 203, 206, 209, 210, 211, 213, 214, 215, 217")
        st.write("**Machine Learning Models:** Light GBM with Enhanced Feature Engineering")
        st.write("**Prediction Horizon:** 1-30 days ahead")
        
        if st.checkbox("Show technical details"):
            st.write("**Features:** Lag features, rolling windows, cyclical encoding, holiday detection")
            st.write("**Model Architecture:** Pipeline with preprocessing and RandomForest")
            st.write("**Performance Metrics:** MAE, RMSE, R², MAPE")

if __name__ == "__main__":
    main()
    StateManager.initialize()
    