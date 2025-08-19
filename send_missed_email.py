#!/usr/bin/env python3
"""
Send Missed Email Script
Manually send prediction email for a specific date (like yesterday)
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from email_scheduler import (
    load_training_data, load_prediction_data, create_comparison_data, 
    send_email, load_enhanced_models
)
from utils.prediction import predict_next_day
from utils.demand_scheduler import get_next_working_day

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("send_missed_email")

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
        success = send_email(
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
    import pandas as pd  # Add this import
    main()