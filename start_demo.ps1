param(
    [int]$Rows = 6000,
    [double]$FraudRate = 0.05,
    [int]$Epochs = 12
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Create the project virtual environment and install requirements.txt first.'
}

$runId = Get-Date -Format 'yyyyMMdd-HHmmss'
$dataDir = Join-Path $projectRoot "data\demo-runs\$runId"
$modelDir = Join-Path $projectRoot "artifacts\demo-runs\$runId"
New-Item -ItemType Directory -Path $dataDir, $modelDir -Force | Out-Null
$v1Csv = Join-Path $dataDir 'finguard-v1-synthetic.csv'
$v2Csv = Join-Path $dataDir 'creditcard-synthetic.csv'
$frontendCsv = Join-Path $dataDir 'frontend-scan-synthetic.csv'
$v1Model = Join-Path $modelDir 'finguard-v1'
$v2Model = Join-Path $modelDir 'creditcard-v2'
$frontendModel = Join-Path $modelDir 'frontend-scan'

& $python -m engine.simulate_finguard_data --output $v1Csv --rows $Rows --fraud-rate $FraudRate
if ($LASTEXITCODE -ne 0) { throw 'Failed to generate the v1 synthetic dataset.' }
& $python -m engine.train_torch --data-file $v1Csv --preset finguard-v1 `
    --label-column is_fraud --data-classification synthetic --output-dir $v1Model --epochs $Epochs
if ($LASTEXITCODE -ne 0) { throw 'Failed to train the v1 PyTorch bundle.' }

& $python -m engine.simulate_creditcard_data --output $v2Csv --rows $Rows --fraud-rate $FraudRate
if ($LASTEXITCODE -ne 0) { throw 'Failed to generate the v2 synthetic dataset.' }
& $python -m engine.train_torch --data-file $v2Csv --preset creditcard-kaggle `
    --label-column Class --data-classification synthetic --output-dir $v2Model --epochs $Epochs
if ($LASTEXITCODE -ne 0) { throw 'Failed to train the v2 PyTorch bundle.' }

& $python -m engine.simulate_frontend_data --output $frontendCsv --rows $Rows --fraud-rate $FraudRate
if ($LASTEXITCODE -ne 0) { throw 'Failed to generate the frontend-scan synthetic dataset.' }
& $python -m engine.train_torch --data-file $frontendCsv --preset frontend-demo `
    --label-column is_fraud --data-classification synthetic --output-dir $frontendModel --epochs $Epochs
if ($LASTEXITCODE -ne 0) { throw 'Failed to train the frontend-scan PyTorch bundle.' }

$env:FINGUARD_ENV = 'development'
$env:FINGUARD_MODEL_DIR = $v1Model
$env:FINGUARD_CREDITCARD_MODEL_DIR = $v2Model
$env:FINGUARD_FRONTEND_MODEL_DIR = $frontendModel
$env:FINGUARD_DB_PATH = Join-Path $dataDir 'audit.db'
Write-Host "FinGuard synthetic demo is starting. Data and models are in $dataDir and $modelDir"
Write-Host 'Open http://127.0.0.1:8000/docs; use Ctrl+C to stop the server.'
& $python -m uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1
exit $LASTEXITCODE
