#!/usr/bin/env python3
"""
Automated Daily Email Scheduler
Sends next day prediction emails to Mattia and yourself every day at 19:00
Uses existing email functionality from pages/7_Next_Day_Prediction.py
"""

import os
import sys
import logging
import traceback
import schedule
import time
from datetime import datetime, timedelta
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.sql_data_connector import extract_sql_data, load_demand_with_kpi_data
from utils.data_loader import load_enhanced_models
from utils.prediction import predict_next_day
from utils.demand_scheduler import get_next_working_day
from config import SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "email_scheduler.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("email_scheduler")

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

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

def send_daily_prediction_email():
    """Generate and send daily prediction email"""
    try:
        logger.info("=" * 60)
        logger.info("📧 STARTING DAILY EMAIL GENERATION")
        logger.info("=" * 60)
        
        # Get next working day
        current_date = datetime.now().date()
        next_date = get_next_working_day(current_date)
        
        if next_date is None:
            logger.error("Could not determine next working day")
            return False
            
        logger.info(f"Generating email for next working day: {next_date}")
        
        # Load training data and models
        df = load_training_data()
        if df is None:
            logger.error("Failed to load training data")
            return False
            
        models, metadata, features = load_enhanced_models()
        if not models:
            logger.error("Failed to load enhanced models")
            return False
            
        # Load existing predictions from database
        original_predictions = load_prediction_data(next_date.strftime('%Y-%m-%d'))
        if original_predictions is None:
            logger.warning("No existing predictions found in database")
            # Generate new predictions for comparison
            next_pred_date, new_predictions, new_hours = predict_next_day(df, models, next_date)
            if next_pred_date is None:
                logger.error("Failed to generate new predictions")
                return False
            
            # Create dummy original predictions with zeros
            original_predictions = pd.DataFrame([
                {'PunchCode': wt, 'NoOfMan': 0, 'Hours': 0}
                for wt in new_predictions.keys()
            ])
        else:
            # Generate improved predictions
            next_pred_date, new_predictions, new_hours = predict_next_day(df, models, next_date)
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
        
        # Send email using existing function logic
        success = send_prediction_email(
            comparison_df,
            datetime.now().date(),
            next_date,
            workers_total_original,
            workers_total_improved,
            hours_total_original,
            hours_total_improved
        )
        
        if success:
            logger.info("✅ Daily prediction email sent successfully")
            return True
        else:
            logger.error("❌ Failed to send daily prediction email")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in send_daily_prediction_email: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def send_prediction_email(comparison_df, current_date, next_date, workers_total_original, workers_total_improved, hours_total_original, hours_total_improved):
    """Send prediction email using existing email functionality"""
    try:
        # Email configuration (from existing code)
        sender_email = "noreply_wfp@forlagssystem.se"
        receiver_email = "amila.g@forlagssystem.se"
        smtp_server = "forlagssystem-se.mail.protection.outlook.com"
        
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Daily Workforce Prediction Report - {next_date.strftime('%Y-%m-%d')}"
        msg["From"] = sender_email
        msg["To"] = receiver_email
        
        # Create HTML content (simplified version of existing email template)
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
                th {{ background-color: #f8f9fa; font-weight: bold; }}
                .metric-row {{ background-color: #e9ecef; font-weight: bold; text-align: left; }}
                .total-col {{ background-color: #d4edda; font-weight: bold; }}
                .negative {{ color: #d32f2f; }}
                .positive {{ color: #2e7d32; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>Daily Workforce Prediction Report</h2>
                <p>Date Generated: {current_date.strftime('%Y-%m-%d')} | Prediction For: {next_date.strftime('%Y-%m-%d (%A)')}</p>
            </div>
            
            <h3>Workers (NoOfMan) Summary</h3>
            <table>
                <tr>
                    <th>PunchCode</th>
                    <th>Current Workers</th>
                    <th>Predicted Workers</th>
                    <th>Difference</th>
                </tr>
        """
        
        # Add worker data rows
        for _, row in comparison_df.iterrows():
            punch_code = row['PunchCode']
            original = row['Original Workers']
            improved = row['Improved Workers']
            diff = row['Workers Difference']
            
            css_class = ""
            if diff < 0:
                css_class = 'class="negative"'
            elif diff > 0:
                css_class = 'class="positive"'
                
            total_class = 'class="total-col"' if punch_code == 'TOTAL' else ''
            
            html += f"""
                <tr {total_class}>
                    <td>{punch_code}</td>
                    <td>{original:.1f}</td>
                    <td>{improved:.1f}</td>
                    <td {css_class}>{diff:+.1f}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h3>Hours Summary</h3>
            <table>
                <tr>
                    <th>PunchCode</th>
                    <th>Current Hours</th>
                    <th>Predicted Hours</th>
                    <th>Difference</th>
                </tr>
        """
        
        # Add hours data rows
        for _, row in comparison_df.iterrows():
            punch_code = row['PunchCode']
            original = row['Original Hours']
            improved = row['Improved Hours']
            diff = row['Hours Difference']
            
            css_class = ""
            if diff < 0:
                css_class = 'class="negative"'
            elif diff > 0:
                css_class = 'class="positive"'
                
            total_class = 'class="total-col"' if punch_code == 'TOTAL' else ''
            
            html += f"""
                <tr {total_class}>
                    <td>{punch_code}</td>
                    <td>{original:.1f}</td>
                    <td>{improved:.1f}</td>
                    <td {css_class}>{diff:+.1f}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <p><strong>Note:</strong> This is an automated daily report generated at 19:00.</p>
            <p>Generated by the Work Utilization Prediction system.</p>
        </body>
        </html>
        """
        
        # Attach HTML content
        part = MIMEText(html, "html")
        msg.attach(part)
        
        # Save report to file as backup
        save_report_to_file(html, next_date)
        
        # Send email
        with smtplib.SMTP(smtp_server, 25, timeout=30) as server:
            server.send_message(msg)
            logger.info(f"Email sent successfully to {receiver_email}")
            return True
            
    except Exception as e:
        logger.error(f"Error sending prediction email: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def save_report_to_file(html_content, next_date):
    """Save the report as an HTML file"""
    try:
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        filename = f"daily_prediction_email_{next_date.strftime('%Y-%m-%d')}.html"
        filepath = os.path.join(reports_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"Report saved to file: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving report to file: {str(e)}")
        return False

def main():
    """Main scheduler function"""
    try:
        logger.info("🕒 Daily email scheduler started")
        logger.info("Next email scheduled for 19:00 daily")
        
        # Schedule the job for 19:00 daily
        schedule.every().day.at("19:00").do(send_daily_prediction_email)
        
        # Keep the scheduler running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    except Exception as e:
        logger.error(f"❌ Critical error in email scheduler: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()