@echo off
echo ============================================================
echo TRADECHAT WEEKLY HEAVY TRAINING - DEEP OPTIMIZATION
echo ============================================================
echo.
echo WARNING: This process is resource-intensive and may take hours.
echo It will retrain ALL models with full Autotune.
echo.
set /p confirm="Do you want to proceed? (y/n): "
if /i "%confirm%" neq "y" exit /b

python scripts\diagnose_assets.py --assets ALL --train --autotune
echo.
echo ============================================================
echo HEAVY TRAINING COMPLETE. AI models are now fully optimized.
echo ============================================================
pause
