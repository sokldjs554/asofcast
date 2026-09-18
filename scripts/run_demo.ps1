param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    throw 'Create .venv and install the project using the README commands first.'
}
Push-Location $Root
try {
    & $Python -m asofcast verify --artifacts artifacts/demo
    if ($LASTEXITCODE -ne 0) { throw 'Bundle verification failed. Server was not started.' }
    Write-Host "Open http://127.0.0.1:$Port . Press Ctrl+C to stop."
    & $Python -m asofcast serve --artifacts artifacts/demo --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'Server exited with an error.' }
} finally {
    Pop-Location
}
