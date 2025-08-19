@echo off
echo ========================================
echo Sending missed email for yesterday
echo ========================================

cd /d "C:\forlogssystems\app"

echo Generating and sending yesterday's prediction email...
"C:\Python313\python.exe" send_missed_email.py --yesterday

echo.
echo Email process completed at %date% %time%
pause