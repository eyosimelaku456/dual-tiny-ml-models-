@echo off
REM ============================================================
REM setup_and_run.bat
REM Creates a Python3 virtual environment and runs full pipeline
REM Double-click this file OR run from Command Prompt
REM ============================================================

echo.
echo ============================================================
echo   TinyML Battery SOH - Setup and Run
echo ============================================================
echo.

REM ── Step 0: Check Python3 ────────────────────────────────────
echo [0] Checking Python3...
python3 --version 2>nul
IF ERRORLEVEL 1 (
    echo     python3 not found, trying python...
    python --version 2>nul
    IF ERRORLEVEL 1 (
        echo [ERROR] Python not found! Install Python 3.8+ and try again.
        pause
        exit /b 1
    )
    SET PYTHON=python
) ELSE (
    SET PYTHON=python3
)
echo     Using: %PYTHON%
echo.

REM ── Step 1: Create virtual environment ───────────────────────
echo [1] Creating virtual environment (venv)...
IF EXIST venv (
    echo     venv already exists, skipping creation.
) ELSE (
    %PYTHON% -m venv venv
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to create venv!
        pause
        exit /b 1
    )
    echo     venv created successfully.
)
echo.

REM ── Step 2: Activate venv ────────────────────────────────────
echo [2] Activating venv...
call venv\Scripts\activate.bat
IF ERRORLEVEL 1 (
    echo [ERROR] Failed to activate venv!
    pause
    exit /b 1
)
echo     venv activated.
echo.

REM ── Step 3: Upgrade pip ──────────────────────────────────────
echo [3] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo.

REM ── Step 4: Install requirements ─────────────────────────────
echo [4] Installing dependencies (this may take a few minutes)...
echo     Installing core packages first...
pip install numpy pandas scikit-learn matplotlib seaborn tqdm joblib --quiet
echo     Installing PyTorch...
pip install torch --quiet
echo     Installing ONNX...
pip install onnx onnxruntime --quiet
echo     Installing TensorFlow...
pip install tensorflow --quiet
echo     Installing onnx-tf...
pip install onnx-tf tensorflow-probability --quiet
echo.
echo     All dependencies installed!
echo.

REM ── Step 5: Create output directories ────────────────────────
echo [5] Creating output directories...
IF NOT EXIST data   mkdir data
IF NOT EXIST models mkdir models
IF NOT EXIST docs   mkdir docs
IF NOT EXIST esp32  mkdir esp32
echo     Directories ready.
echo.

REM ── Step 6: Run pipeline ─────────────────────────────────────
echo ============================================================
echo   Running Pipeline
echo ============================================================
echo.

cd python

echo [Step 1/5] Preprocessing EIS data...
python preprocess.py
IF ERRORLEVEL 1 ( echo [ERROR] preprocess.py failed! & cd .. & pause & exit /b 1 )
echo.

echo [Step 2/5] Training MLP + CNN1D models...
python train_model.py
IF ERRORLEVEL 1 ( echo [ERROR] train_model.py failed! & cd .. & pause & exit /b 1 )
echo.

echo [Step 3/5] Converting and quantizing to INT8...
python convert_quantize.py
IF ERRORLEVEL 1 ( echo [ERROR] convert_quantize.py failed! & cd .. & pause & exit /b 1 )
echo.

echo [Step 4/5] Evaluating FP32 vs INT8...
python evaluate.py
IF ERRORLEVEL 1 ( echo [ERROR] evaluate.py failed! & cd .. & pause & exit /b 1 )
echo.

echo [Step 5/5] Generating ESP32 C array header...
python convert_to_c_array.py
IF ERRORLEVEL 1 ( echo [ERROR] convert_to_c_array.py failed! & cd .. & pause & exit /b 1 )
echo.

cd ..

REM ── Done ─────────────────────────────────────────────────────
echo ============================================================
echo   PIPELINE COMPLETE!
echo ============================================================
echo.
echo   Results:
echo   - Trained models:   models\
echo   - Evaluation plots: docs\
echo   - ESP32 firmware:   esp32\model_data.h
echo   - Scaler params:    models\scaler_params.h
echo.
echo   Next step: Flash esp32\ to your ESP32 using PlatformIO.
echo.
pause
