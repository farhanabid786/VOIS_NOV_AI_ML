$ErrorActionPreference = "Stop"

Write-Host "Checking Python 3.10..."
py -3.10 --version
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.10 was not found. Install Python 3.10 x64 and verify with: py -0p"
}

$backend = Join-Path $PSScriptRoot "backend"
$venvPython = Join-Path $backend ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating backend virtual environment..."
    py -3.10 -m venv (Join-Path $backend ".venv")
}

Write-Host "Installing pinned backend dependencies..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $backend "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Backend dependency installation failed." }

Write-Host "Training Random Forest from the supplied CSV..."
Push-Location $backend
try {
    & $venvPython train_model.py
    if ($LASTEXITCODE -ne 0) { throw "Model training failed." }

    Write-Host "Installing test dependencies..."
    & $venvPython -m pip install -r requirements-dev.txt
    if ($LASTEXITCODE -ne 0) { throw "Test dependency installation failed." }

    Write-Host "Running API tests..."
    & $venvPython -m pytest tests -q
    if ($LASTEXITCODE -ne 0) { throw "API tests failed." }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Backend setup completed. Start the API with:"
Write-Host "cd backend"
Write-Host ".\.venv\Scripts\Activate.ps1"
Write-Host "python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000"
