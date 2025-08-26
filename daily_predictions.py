#!/usr/bin/env python3
"""
Daily Next Day Prediction Script
Generates predictions for the next day and saves them to the database
Uses existing functions: predict_next_day() and save_predictions_to_db()
Can be run for specific dates using command line arguments
"""

import os
import sys
import logging
import traceback
import argparse
from datetime import datetime, timedelta
import pandas as pd
import pickle
import json

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.prediction import predict_next_day
from utils.data_loader import load_enhanced_models
from utils.sql_data_connector import extract_sql_data, save_predictions_to_db
from config import (
    MODELS_DIR, ENHANCED_WORK_TYPES, SQL_SERVER, SQL_DATABASE, 
    SQL_TRUSTED_CONNECTION
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "daily_predictions.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("daily_predictions")

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

def load_training_data():
    """Load training data for predictions"""
    try:
        logger.info("Loading training data for next day prediction")
        query = """
        SELECT Date, PunchCode as WorkType, Hours, SystemHours, 
        CASE WHEN PunchCode IN (206, 213) THEN NoRows
        ELSE Quantity END as Quantity
        FROM WorkUtilizationData 
        WHERE PunchCode IN ('202', '203', '206', '209', '210', '211', '213', '214', '215', '217') 
        AND Hours > 0 
        AND SystemHours > 0 
        AND Date < CAST(GETDATE() AS DATE)
        ORDER BY Date
        """
        
        df = extract_sql_data(
            server=SQL_SERVER,
            database=SQL_DATABASE,
            query=query,
            trusted_connection=SQL_TRUSTED_CONNECTION
        )
        
        if df is None or df.empty:
            logger.error("No training data loaded")
            return None
            
        df['Date'] = pd.to_datetime(df['Date'])
        df['WorkType'] = df['WorkType'].astype(str)
        
        logger.info(f"Loaded {len(df)} records for prediction")
        logger.info(f"Data date range: {df['Date'].min()} to {df['Date'].max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def generate_predictions_for_date(target_date=None):
    """Generate predictions for a specific date"""
    try:
        if target_date:
            logger.info("=" * 60)
            logger.info(f"🚀 STARTING PREDICTION FOR SPECIFIC DATE: {target_date}")
            logger.info("=" * 60)
        else:
            logger.info("=" * 60)
            logger.info("🚀 STARTING DAILY NEXT DAY PREDICTION")
            logger.info("=" * 60)
        
        # Load training data
        df = load_training_data()
        if df is None:
            logger.error("❌ Failed to load training data")
            return False
            
        # Load enhanced models
        logger.info("Loading enhanced models...")
        models, metadata, features = load_enhanced_models()
        
        if not models:
            logger.error("❌ No enhanced models found")
            return False
            
        logger.info(f"✅ Loaded models for work types: {list(models.keys())}")
        
        # Generate predictions for specified date or next day
        logger.info("Generating predictions...")
        if target_date:
            next_date, predictions, hours_predictions = predict_next_day(df, models, date=target_date)
        else:
            next_date, predictions, hours_predictions = predict_next_day(df, models)
        
        if next_date is None:
            logger.error("❌ Failed to generate predictions")
            return False
            
        logger.info(f"📅 Generated predictions for: {next_date.strftime('%Y-%m-%d')}")
        
        # Log prediction summary
        logger.info("\n📊 PREDICTION SUMMARY:")
        logger.info("-" * 40)
        total_hours = 0
        total_workers = 0
        
        for work_type in sorted(predictions.keys()):
            workers = predictions[work_type]
            hours = hours_predictions[work_type]
            total_hours += hours
            total_workers += workers
            
            logger.info(f"Work Type {work_type}: {workers} workers, {hours:.1f} hours")
        
        logger.info("-" * 40)
        logger.info(f"TOTAL: {total_workers} workers, {total_hours:.1f} hours")
        
        # Prepare data for database saving
        predictions_dict = {next_date: predictions}
        hours_dict = {next_date: hours_predictions}
        
        # Get username for saving
        username = get_current_user()
        
        # Save predictions to database
        logger.info(f"💾 Saving predictions to database for user: {username}")
        success = save_predictions_to_db(predictions_dict, hours_dict, username)
        
        if success:
            logger.info("✅ Predictions successfully saved to database")
            
            # Log save summary
            logger.info(f"\n💾 SAVE SUMMARY:")
            logger.info(f"   Date: {next_date.strftime('%Y-%m-%d')}")
            logger.info(f"   Work Types: {len(predictions)}")
            logger.info(f"   Total Predictions: {len(predictions)}")
            logger.info(f"   User: {username}")
            logger.info(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return True
        else:
            logger.error("❌ Failed to save predictions to database")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in generate_predictions: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def generate_next_day_predictions():
    """Generate predictions for the next working day (backward compatibility)"""
    return generate_predictions_for_date()

def get_current_user():
    """Get current user for saving predictions"""
    try:
        import getpass
        return getpass.getuser()
    except:
        return "system_auto"

def verify_database_connection():
    """Verify database connectivity before running predictions"""
    try:
        logger.info("🔍 Verifying database connection...")
        
        test_query = "SELECT COUNT(*) as record_count FROM WorkUtilizationData"
        df = extract_sql_data(
            server=SQL_SERVER,
            database=SQL_DATABASE,
            query=test_query,
            trusted_connection=SQL_TRUSTED_CONNECTION
        )
        
        if df is not None and not df.empty:
            record_count = df.iloc[0]['record_count']
            logger.info(f"✅ Database connection verified. Records available: {record_count}")
            return True
        else:
            logger.error("❌ Database connection test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Database connection error: {str(e)}")
        return False

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Generate workforce predictions for next day or specific date')
    parser.add_argument('--date', '-d', type=str, 
                       help='Generate predictions for specific date in YYYY-MM-DD format')
    parser.add_argument('--tomorrow', '-t', action='store_true',
                       help='Generate predictions for tomorrow (default behavior)')
    
    args = parser.parse_args()
    
    try:
        start_time = datetime.now()
        
        # Determine target date
        if args.date:
            try:
                target_date = datetime.strptime(args.date, '%Y-%m-%d')
                logger.info(f"🎯 Target date specified: {target_date.strftime('%Y-%m-%d')} ({target_date.strftime('%A')})")
            except ValueError:
                logger.error("❌ Invalid date format. Use YYYY-MM-DD")
                return False
        else:
            target_date = None
            logger.info("🕒 Using default behavior: next working day")
        
        logger.info(f"🕒 Prediction process started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Verify database connection
        if not verify_database_connection():
            logger.error("❌ Database connection verification failed. Exiting.")
            return False
        
        # Check if target date is a working day
        if target_date:
            from utils.holiday_utils import is_non_working_day
            is_non_working, reason = is_non_working_day(target_date.date())
            
            if is_non_working:
                logger.warning(f"⚠️  {target_date.date()} is a non-working day: {reason}")
                logger.info("Predictions will be generated but may be zero for non-working punch codes")
            
        # Generate and save predictions
        success = generate_predictions_for_date(target_date)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        if success:
            logger.info("=" * 60)
            logger.info("🎉 PREDICTION PROCESS COMPLETED SUCCESSFULLY")
            logger.info(f"⏱️  Duration: {duration.total_seconds():.1f} seconds")
            logger.info("=" * 60)
        else:
            logger.error("=" * 60)
            logger.error("❌ PREDICTION PROCESS FAILED")
            logger.error(f"⏱️  Duration: {duration.total_seconds():.1f} seconds")
            logger.error("=" * 60)
            
        return success
        
    except Exception as e:
        logger.error(f"❌ Critical error in main: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    sys.exit(exit_code)