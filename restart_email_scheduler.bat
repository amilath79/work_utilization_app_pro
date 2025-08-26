@echo off
echo ========================================
echo Restarting Email Scheduler
echo ========================================

cd /d "C:\forlogssystems\app"

echo Step 1: Stopping any existing Python processes...
taskkill /f /im python.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✓ Stopped existing Python processes
) else (
    echo ✓ No Python processes were running
)

echo.
echo Step 2: Waiting 3 seconds for cleanup...
timeout /t 3 /nobreak >nul

echo.
echo Step 3: Starting fresh email scheduler...
start "Email Scheduler" "C:\Python313\python.exe" email_scheduler.py

echo.
echo Step 4: Checking if scheduler started...
timeout /t 2 /nobreak >nul
tasklist | findstr python.exe >nul
if %ERRORLEVEL% EQU 0 (
    echo ✓ Email scheduler is now running
    echo ✓ Emails will be sent daily at 19:00
) else (
    echo ✗ Email scheduler failed to start
    echo Check for errors manually
)

echo.
echo ========================================
echo Restart completed at %date% %time%
echo ========================================
pause