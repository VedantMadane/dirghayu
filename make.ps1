# Dirghayu Development Helper Script (Windows PowerShell)
# Quick commands for common development tasks

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

function Show-Help {
    Write-Host "Dirghayu Development Commands:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  install          " -NoNewline -ForegroundColor Green
    Write-Host "Install dependencies with uv (fast!)"
    Write-Host "  install-pip      " -NoNewline -ForegroundColor Green
    Write-Host "Install dependencies with pip (slower)"
    Write-Host "  install-dev      " -NoNewline -ForegroundColor Green
    Write-Host "Install with dev dependencies"
    Write-Host "  data             " -NoNewline -ForegroundColor Green
    Write-Host "Download/create sample data"
    Write-Host "  demo             " -NoNewline -ForegroundColor Green
    Write-Host "Run end-to-end demo"
    Write-Host "  api              " -NoNewline -ForegroundColor Green
    Write-Host "Start API server"
    Write-Host "  test             " -NoNewline -ForegroundColor Green
    Write-Host "Run tests"
    Write-Host "  lint             " -NoNewline -ForegroundColor Green
    Write-Host "Check code quality"
    Write-Host "  format           " -NoNewline -ForegroundColor Green
    Write-Host "Format code"
    Write-Host "  clean            " -NoNewline -ForegroundColor Green
    Write-Host "Clean temporary files"
    Write-Host "  quick-start      " -NoNewline -ForegroundColor Green
    Write-Host "Install, download data, and run demo"
    Write-Host "  dev-setup        " -NoNewline -ForegroundColor Green
    Write-Host "Setup development environment"
    Write-Host ""
    Write-Host "Usage: .\make.ps1 <command>" -ForegroundColor Yellow
}

function Install-Dependencies {
    Write-Host "[*] Installing dependencies with uv..." -ForegroundColor Cyan
    uv pip install -r requirements.txt
}

function Install-Dependencies-Pip {
    Write-Host "[*] Installing dependencies with pip..." -ForegroundColor Cyan
    pip install -r requirements.txt
}

function Install-Dev {
    Write-Host "[*] Installing with dev dependencies..." -ForegroundColor Cyan
    uv pip install -e ".[dev]"
}

function Download-Data {
    Write-Host "[*] Downloading/creating sample data..." -ForegroundColor Cyan
    python scripts/download_data.py
}

function Run-Demo {
    Write-Host "[*] Running end-to-end demo..." -ForegroundColor Cyan
    python demo.py data/sample.vcf
}

function Start-API {
    Write-Host "[*] Starting API server..." -ForegroundColor Cyan
    python src/api/server.py
}

function Run-Tests {
    Write-Host "[*] Running tests..." -ForegroundColor Cyan
    pytest tests/ -v
}

function Check-Lint {
    Write-Host "[*] Checking code quality..." -ForegroundColor Cyan
    ruff check src/ scripts/ demo.py
}

function Format-Code {
    Write-Host "[*] Formatting code..." -ForegroundColor Cyan
    ruff format src/ scripts/ demo.py
}

function Clean-Files {
    Write-Host "[*] Cleaning temporary files..." -ForegroundColor Cyan
    Get-ChildItem -Path . -Include __pycache__,*.pyc,.pytest_cache,.ruff_cache,.mypy_cache,*.egg-info -Recurse -Force | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue
    Write-Host "[OK] Clean complete!" -ForegroundColor Green
}

function Quick-Start {
    Install-Dependencies
    Download-Data
    Run-Demo
}

function Dev-Setup {
    Install-Dev
    Download-Data
}

# Execute command
switch ($Command.ToLower()) {
    "help" { Show-Help }
    "install" { Install-Dependencies }
    "install-pip" { Install-Dependencies-Pip }
    "install-dev" { Install-Dev }
    "data" { Download-Data }
    "demo" { Run-Demo }
    "api" { Start-API }
    "test" { Run-Tests }
    "lint" { Check-Lint }
    "format" { Format-Code }
    "clean" { Clean-Files }
    "quick-start" { Quick-Start }
    "dev-setup" { Dev-Setup }
    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host ""
        Show-Help
    }
}
