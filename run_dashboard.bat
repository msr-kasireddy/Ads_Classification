@echo off
REM ==========================================================================
REM  DOUBLE-CLICK THIS FILE on Windows to open the Ads Detection dashboard.
REM  (First run sets everything up; later runs just open the dashboard.)
REM ==========================================================================
cd /d "%~dp0"
echo Newspaper Ads Detection - starting up...
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo Python is not installed. Get it from https://www.python.org/downloads/
  echo During install, tick "Add Python to PATH". Then double-click this file again.
  pause
  exit /b 1
)

if not exist ".venv" (
  echo First-time setup: creating a private environment...
  python -m venv .venv
)
call .venv\Scripts\activate.bat

python -c "import streamlit, streamlit_drawable_canvas, cv2" >nul 2>nul
if errorlevel 1 (
  echo Installing required libraries (one-time, a few minutes)...
  python -m pip install --upgrade pip >nul
  python -m pip install -r requirements.txt
)

echo.
echo Opening the dashboard in your browser...
echo (Leave this window open while you work. Close it to stop.)
echo.
python -m streamlit run app/dashboard.py
pause
