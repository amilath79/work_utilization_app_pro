#!/usr/bin/env python3
"""
Test Email Script
Manually trigger the daily prediction email for testing
"""

import os
import sys
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the email function
from email_scheduler import send_daily_prediction_email

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Test the email functionality"""
    print("🧪 Testing daily prediction email...")
    
    success = send_daily_prediction_email()
    
    if success:
        print("✅ Test email sent successfully!")
        print("Check your email inbox for the report.")
    else:
        print("❌ Test email failed!")
        print("Check logs/email_scheduler.log for details.")

if __name__ == "__main__":
    main()