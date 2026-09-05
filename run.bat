@echo off
echo ==============================================
echo   Senior AI Code Companion - Telegram Bot
echo ==============================================

if not exist venv (
    echo [1/3] Creating virtual environment...
    python -m venv venv
)

echo [2/3] Activating venv and checking dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo [3/3] Starting bot...
python bot.py
pause
