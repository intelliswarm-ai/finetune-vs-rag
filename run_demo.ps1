# Fine-Tuning vs RAG Demo - PowerShell Launcher

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Fine-Tuning vs RAG: Live Demo" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check for virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Install core dependencies if needed
$streamlitInstalled = pip show streamlit 2>$null
if (-not $streamlitInstalled) {
    Write-Host "Installing dependencies (first run)..." -ForegroundColor Yellow
    pip install streamlit plotly pandas python-dotenv openai
    Write-Host ""
    Write-Host "For FinBERT live inference, also run:" -ForegroundColor Yellow
    Write-Host "  pip install torch transformers" -ForegroundColor White
    Write-Host ""
}

# Check for .env file
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Write-Host "No .env file found. Copy .env.example to .env and add your API keys." -ForegroundColor Yellow
        Write-Host "The demo works without API keys (simulated mode)." -ForegroundColor Yellow
        Write-Host ""
    }
}

# Launch
Write-Host "Starting Streamlit app..." -ForegroundColor Green
Write-Host "Open http://localhost:8501 in your browser" -ForegroundColor Green
Write-Host ""
streamlit run app/app.py --server.headless true
