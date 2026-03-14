@echo off
echo ========================================
echo   Fine-Tuning vs RAG: Live Demo
echo ========================================
echo.

:: Check for virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install core dependencies if needed
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install streamlit plotly pandas python-dotenv openai
    echo.
    echo For FinBERT live inference, also run:
    echo   pip install torch transformers
    echo.
)

:: Check for .env
if not exist ".env" (
    echo No .env file found. Demo will use simulated mode.
    echo Copy .env.example to .env and add API keys for live mode.
    echo.
)

:: Launch
echo Starting Streamlit app...
echo Open http://localhost:8501 in your browser
echo.
streamlit run app/app.py --server.headless true

pause
