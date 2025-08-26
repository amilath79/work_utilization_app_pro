@echo off
echo ========================================
echo Email Scheduler Status Check
echo ========================================

cd /d "C:\forlogssystems\app"

echo Checking if email scheduler is running...
tasklist | findstr python.exe >nul
if %ERRORLEVEL% EQU 0 (
    echo ✓ Python process is running
    echo.
    echo Running Python processes:
    wmic process where "name='python.exe'" get ProcessId,CommandLine 2>nul
) else (
    echo ✗ No Python processes found
    echo Email scheduler is NOT running
)

echo.
echo ========================================
echo Log Files Status:
echo ========================================

if exist logs\email_scheduler*.log (
    echo ✓ Email scheduler log files found:
    dir logs\email_scheduler*.log /o-d
    echo.
    echo Latest log entries:
    powershell "Get-Content logs\email_scheduler*.log | Select-Object -Last 10"
) else (
    echo ✗ No email scheduler log files found
)

echo.
echo ========================================
echo Next Steps:
echo ========================================
echo - If scheduler is NOT running: Run restart_email_scheduler.bat
echo - If scheduler is running: It will send emails daily at 19:00
echo - To send missed email: Run send_missed_email.py --yesterday
echo ========================================
pause