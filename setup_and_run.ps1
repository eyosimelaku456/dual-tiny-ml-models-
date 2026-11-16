# ============================================================
# setup_and_run.ps1
# PowerShell version - run if the .bat file doesn't work
# Right-click → "Run with PowerShell"
# OR from PowerShell terminal: .\setup_and_run.ps1
# ============================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  TinyML Battery SOH - Setup and Run" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $ProjectRoot

# ── Step 0: Find Python ──────────────────────────────────────
Write-Host "[0] Locating Python3..." -ForegroundColor Yellow
$PYTHON = $null
foreach ($cmd in @("python3", "python")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python 3") {
            $PYTHON = $cmd
            Write-Host "    Found: $ver" -ForegroundColor Green
            break
        }
    } catch {}
}
if (-not $PYTHON) {
    Write-Host "[ERROR] Python 3 not found! Install from https://python.org" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# ── Step 1: Create venv ──────────────────────────────────────
Write-Host ""
Write-Host "[1] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "    venv already exists, skipping." -ForegroundColor Gray
} else {
    & $PYTHON -m venv venv
    Write-Host "    venv created!" -ForegroundColor Green
}

# ── Step 2: Activate venv ────────────────────────────────────
Write-Host ""
Write-Host "[2] Activating venv..." -ForegroundColor Yellow
$activate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    Write-Host "[ERROR] Activation script not found: $activate" -ForegroundColor Red
    exit 1
}
& $activate
Write-Host "    venv activated!" -ForegroundColor Green

# ── Step 3: Upgrade pip ──────────────────────────────────────
Write-Host ""
Write-Host "[3] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# ── Step 4: Install dependencies ─────────────────────────────
Write-Host ""
Write-Host "[4] Installing dependencies..." -ForegroundColor Yellow

$packages = @(
    @("Core scientific", "numpy pandas scikit-learn matplotlib seaborn tqdm joblib"),
    @("PyTorch", "torch"),
    @("ONNX", "onnx onnxruntime"),
    @("TensorFlow", "tensorflow"),
    @("ONNX-TF bridge", "onnx-tf tensorflow-probability")
)

foreach ($pkg in $packages) {
    Write-Host "    Installing $($pkg[0])..." -ForegroundColor Gray
    $pkgList = $pkg[1] -split " "
    pip install @pkgList --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install $($pkg[0])" -ForegroundColor Red
        exit 1
    }
}
Write-Host "    All dependencies installed!" -ForegroundColor Green

# ── Step 5: Create directories ───────────────────────────────
Write-Host ""
Write-Host "[5] Creating output directories..." -ForegroundColor Yellow
foreach ($dir in @("data", "models", "docs", "esp32")) {
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
}
Write-Host "    Directories ready!" -ForegroundColor Green

# ── Step 6: Run each pipeline step ───────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Running Pipeline" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Set-Location (Join-Path $ProjectRoot "python")

$steps = @(
    @("1/5", "Preprocessing EIS data",        "preprocess.py"),
    @("2/5", "Training MLP + CNN1D models",    "train_model.py"),
    @("3/5", "Converting and quantizing INT8", "convert_quantize.py"),
    @("4/5", "Evaluating FP32 vs INT8",        "evaluate.py"),
    @("5/5", "Generating ESP32 C array",       "convert_to_c_array.py")
)

foreach ($step in $steps) {
    Write-Host ""
    Write-Host "[Step $($step[0])] $($step[1])..." -ForegroundColor Yellow
    python $step[2]
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] $($step[2]) failed!" -ForegroundColor Red
        Set-Location $ProjectRoot
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "    Done!" -ForegroundColor Green
}

Set-Location $ProjectRoot

# ── Done ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  PIPELINE COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Results:" -ForegroundColor White
Write-Host "  - Trained models:   models\" -ForegroundColor Gray
Write-Host "  - Evaluation plots: docs\" -ForegroundColor Gray
Write-Host "  - ESP32 C header:   esp32\model_data.h" -ForegroundColor Gray
Write-Host "  - Scaler params:    models\scaler_params.h" -ForegroundColor Gray
Write-Host ""
Read-Host "Press Enter to exit"
