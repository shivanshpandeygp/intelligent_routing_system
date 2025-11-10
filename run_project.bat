@echo off
title Intelligent RL-Based Routing System
color 0A

echo ============================================================
echo    INTELLIGENT RL-BASED ROUTING SYSTEM
echo    M.Tech Project - MMMUT Gorakhpur
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.8+ from https://www.python.org
    pause
    exit /b 1
)

echo [1/4] Checking Python installation...
python --version
echo.

REM Check if virtual environment exists
if not exist ".venv\" (
    echo [2/4] Virtual environment not found. Creating...
    python -m venv .venv
    echo Virtual environment created successfully!
) else (
    echo [2/4] Virtual environment found.
)
echo.

REM Activate virtual environment
echo [3/4] Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install/update dependencies
echo [4/4] Checking dependencies...
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo Dependencies installed!
echo.

echo ============================================================
echo    STARTING APPLICATION...
echo ============================================================
echo.
echo Opening in browser: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run Streamlit app
streamlit run frontend\app.py

pause
