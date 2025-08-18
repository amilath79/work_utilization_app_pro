@echo off
echo ========================================
echo Starting email scheduler at %date% %time%
echo ========================================

cd /d "C:\forlogssystems\app"

echo Starting daily email scheduler...
"C:\Python313\python.exe" email_scheduler.py >> logs\email_scheduler.log 2>&1

echo Email scheduler started at %date% %time% >> logs\email_operations.log