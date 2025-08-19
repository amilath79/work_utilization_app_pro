#!/usr/bin/env python3
"""
Send Missed Email Script
Manually send prediction email for a specific date (like yesterday)
"""

import os
import sys
import logging
import argparse
import pandas as pd
import traceback
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.sql_data_connector import extract_sql_data, load_demand_with_kpi_data
from utils.data_loader import load_enhanced_models
from utils.prediction import predict_next_day
from utils.demand_scheduler import get_next_working_day
from config import SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("send_missed_email")

def load_training_data():
    """Load training data for predictions"""
    try:
        logger.info("Loading training data for email predictions")
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
        
        logger.info(f"Loaded {len(df)} records for email prediction")
        return df
        
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        return None

def load_prediction_data(date_value):
    """Load existing prediction data from database"""
    try:
        sql_query = f"""
        SELECT ID, Date, PunchCode, NoOfMan, Hours, PredictionType, Username, 
               CreatedDate, LastModifiedDate
        FROM PredictionData WHERE PunchCode in (209,211, 213, 214, 215, 202, 203, 206, 210, 217)
        AND Date = '{date_value}'
        ORDER BY PunchCode
        """
        
        df = extract_sql_data(
            server=SQL_SERVER,
            database=SQL_DATABASE,
            query=sql_query,
            trusted_connection=SQL_TRUSTED_CONNECTION
        )
        
        if df is not None and not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            df['PunchCode'] = df['PunchCode'].astype(str)
            return df
        else:
            logger.warning("No prediction data found in database")
            return None
            
    except Exception as e:
        logger.error(f"Error loading prediction data: {str(e)}")
        return None

def create_comparison_data(original_predictions, improved_predictions_workers, improved_predictions_hours):
    """Create comparison data for email"""
    comparison_data = []
    
    # Process each punch code in original predictions
    for _, row in original_predictions.iterrows():
        punch_code = str(row['PunchCode'])
        original_workers = row['NoOfMan']
        original_hours = row['Hours']
        
        # Get improved predictions for this punch code
        improved_workers = improved_predictions_workers.get(punch_code, 0)
        improved_hours = improved_predictions_hours.get(punch_code, 0)
        
        # Calculate differences
        workers_diff = improved_workers - original_workers
        hours_diff = improved_hours - original_hours
        
        comparison_data.append({
            'PunchCode': punch_code,
            'Original Workers': original_workers,
            'Improved Workers': improved_workers,
            'Original Hours': original_hours,
            'Improved Hours': improved_hours,
            'Workers Difference': workers_diff,
            'Hours Difference': hours_diff
        })
    
    # Add entries for punch codes that are only in improved predictions
    for punch_code in improved_predictions_workers.keys():
        if punch_code not in original_predictions['PunchCode'].astype(str).values:
            improved_workers = improved_predictions_workers[punch_code]
            improved_hours = improved_predictions_hours[punch_code]
            
            comparison_data.append({
                'PunchCode': punch_code,
                'Original Workers': 0,
                'Improved Workers': improved_workers,
                'Original Hours': 0,
                'Improved Hours': improved_hours,
                'Workers Difference': improved_workers,
                'Hours Difference': improved_hours
            })
    
    return comparison_data

def send_email_for_date(target_date):
    """Send prediction email for a specific date"""
    try:
        logger.info(f"📧 Generating email for date: {target_date}")
        
        # Load training data and models
        df = load_training_data()
        if df is None:
            logger.error("Failed to load training data")
            return False
            
        models, metadata, features = load_enhanced_models()
        if not models:
            logger.error("Failed to load enhanced models")
            return False
            
        # Load existing predictions from database for this date
        original_predictions = load_prediction_data(target_date.strftime('%Y-%m-%d'))
        if original_predictions is None:
            logger.warning(f"No existing predictions found in database for {target_date}")
            # Generate new predictions for comparison
            next_pred_date, new_predictions, new_hours = predict_next_day(df, models, target_date)
            if next_pred_date is None:
                logger.error("Failed to generate new predictions")
                return False
            
            # Create dummy original predictions with zeros
            original_predictions = pd.DataFrame([
                {'PunchCode': wt, 'NoOfMan': 0, 'Hours': 0}
                for wt in new_predictions.keys()
            ])
        else:
            # Generate improved predictions for this specific date
            next_pred_date, new_predictions, new_hours = predict_next_day(df, models, target_date)
            if next_pred_date is None:
                logger.error("Failed to generate improved predictions")
                return False
        
        # Create comparison data
        comparison_data = create_comparison_data(original_predictions, new_predictions, new_hours)
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate totals
        workers_total_original = comparison_df['Original Workers'].sum()
        workers_total_improved = comparison_df['Improved Workers'].sum()
        hours_total_original = comparison_df['Original Hours'].sum()
        hours_total_improved = comparison_df['Improved Hours'].sum()
        
        # Add total row
        total_row = {
            'PunchCode': 'TOTAL',
            'Original Workers': workers_total_original,
            'Improved Workers': workers_total_improved,
            'Original Hours': hours_total_original,
            'Improved Hours': hours_total_improved,
            'Workers Difference': workers_total_improved - workers_total_original,
            'Hours Difference': hours_total_improved - hours_total_original
        }
        comparison_df = pd.concat([comparison_df, pd.DataFrame([total_row])], ignore_index=True)
        
        # Send email for this specific date
        success = send_prediction_email(
            comparison_df,
            datetime.now().date(),  # Current date as generation date
            target_date,            # Target date for predictions
            workers_total_original,
            workers_total_improved,
            hours_total_original,
            hours_total_improved
        )
        
        if success:
            logger.info(f"✅ Email sent successfully for {target_date}")
            return True
        else:
            logger.error(f"❌ Failed to send email for {target_date}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error sending email for {target_date}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Send missed prediction email for specific date')
    parser.add_argument('--date', '-d', type=str, 
                       help='Date in YYYY-MM-DD format (default: yesterday)')
    parser.add_argument('--yesterday', '-y', action='store_true',
                       help='Send email for yesterday')
    
    args = parser.parse_args()
    
    # Determine target date
    if args.yesterday or args.date is None:
        # Default to yesterday
        target_date = datetime.now().date() - timedelta(days=1)
        logger.info("Using yesterday as target date")
    else:
        # Parse provided date
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            logger.error("Invalid date format. Use YYYY-MM-DD")
            return False
    
    logger.info(f"🎯 Target date: {target_date} ({target_date.strftime('%A')})")
    
    # Check if it was a working day
    from utils.holiday_utils import is_non_working_day
    is_non_working, reason = is_non_working_day(target_date)
    
    if is_non_working:
        logger.warning(f"⚠️  {target_date} was a non-working day: {reason}")
        response = input("Do you still want to send the email? (y/n): ")
        if response.lower() != 'y':
            logger.info("Email cancelled by user")
            return False
    
    # Send the email
    success = send_email_for_date(target_date)
    
    if success:
        print(f"✅ Email sent successfully for {target_date}!")
        print("Check your email inbox for the report.")
    else:
        print(f"❌ Failed to send email for {target_date}")
        print("Check logs for details.")
    
    return success

if __name__ == "__main__":
    main()