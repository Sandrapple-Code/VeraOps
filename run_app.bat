@echo off
echo Starting VeraOps using virtual environment...
if not exist .venv (
    echo Error: .venv virtual environment not found!
    echo Please run: python -m venv .venv
    echo And then: .venv\Scripts\pip.exe install -r requirements.txt
    pause
    exit /b
)
.venv\Scripts\python.exe -m streamlit run app.py
pause
