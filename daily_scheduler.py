#!/usr/bin/env python3
"""
Daily Model Training Scheduler
Runs train_models2.py every day at 2:00 AM
"""

import schedule
import time
import subprocess
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("scheduler")

def run_training():
    """Execute the model training script"""
    try:
        logger.info("Starting daily model training...")
        
        # Run train_models2.py
        result = subprocess.run(
            ['python', 'train_models2.py'],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            logger.info("✅ Model training completed successfully")
            logger.info(f"Output: {result.stdout}")
        else:
            logger.error("❌ Model training failed")
            logger.error(f"Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Model training timed out after 1 hour")
    except Exception as e:
        logger.error(f"❌ Error running training: {str(e)}")

def main():
    """Main scheduler function"""
    logger.info("Daily model training scheduler started")
    logger.info("Next training scheduled for 2:00 AM daily")
    
    # Schedule the job for 2:00 AM daily
    schedule.every().day.at("02:00").do(run_training)
    
    # Keep the scheduler running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()