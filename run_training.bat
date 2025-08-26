@echo off
echo ========================================
echo Starting daily ML operations at %date% %time%
echo ========================================

cd /d "C:\forlagssystem\app"

echo.
echo [1/2] Training models...
"python" train_models2.py >> logs\training.log 2>&1

if %ERRORLEVEL% EQU 0 (
    echo ✓ Model training completed successfully
    
    echo.
    echo [2/2] Generating next day predictions...
    "python" daily_predictions.py >> logs\predictions.log 2>&1
    
    if %ERRORLEVEL% EQU 0 (
        echo ✓ Next day predictions completed successfully
        echo.
        echo ========================================
        echo All operations completed at %date% %time%
        echo ========================================
    ) else (
        echo ✗ Next day predictions failed
        echo Check logs\predictions.log for details
    )
) else (
    echo ✗ Model training failed
    echo Check logs\training.log for details
    echo Skipping predictions due to training failure
)

echo Operation completed at %date% %time% >> logs\daily_operations.log