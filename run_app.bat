@echo off
title Multi-Indicator Ensemble Stock Screener
cd /d "%~dp0"

echo ========================================================
echo   Technical Analysis Multi-Indicator Screener
echo ========================================================
echo.
echo Starting Streamlit server...
echo The dashboard will automatically open in your web browser.
echo Press Ctrl + C to stop the server.
echo.

python -m streamlit run app.py

pause
