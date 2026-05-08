@echo off
echo ========================================
echo ProtAI - Starting Backend Server
echo ========================================
echo.
echo Backend will start on http://localhost:5000
echo Frontend will be accessible at http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

cd /d "%~dp0backend"
py -3.11 app.py