# TinyML Battery State of Health (SOH) Estimator

This project implements an end-to-end Machine Learning pipeline designed to estimate the **State of Health (SOH)** of a battery using **Electrochemical Impedance Spectroscopy (EIS)** data. The ultimate goal is to deploy the trained model onto an **ESP32 microcontroller** (Xtensa LX6, 520 KB RAM).

Because the target device has strict memory constraints, this project utilizes **TinyML principles**, emphasizing model optimization, static shapes, and INT8 post-training quantization to fit the model within a strict <100 KB footprint.

---

## 📂 Architecture & Directory Structure

The architecture is split into a Python-based ML training pipeline and a C++ based ESP32 firmware deployment.

- **`python/`**: Contains the 5-step sequential ML pipeline scripts (`preprocess.py`, `train_model.py`, `convert_quantize.py`, `evaluate.py`, `convert_to_c_array.py`).
- **`esp32/`**: Contains the PlatformIO-based C++ firmware and auto-generated C-headers.
- **`Dataset-model/`**: Contains the raw EIS CSV dataset.
- **`models/`**: Stores PyTorch weights, ONNX exports, TFLite binaries, and metadata.
- **`docs/`**: Stores evaluation metrics, scatter plots, and residual analysis.

---

## 🛠️ Requirements & Setup

This project requires a Linux environment (or WSL on Windows) to compile the machine learning quantization tools properly.

1. Ensure you have **Python 3.12** and **WSL (Ubuntu)** installed.
2. Create and activate a virtual environment:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the ML Pipeline

To execute the entire machine learning pipeline sequentially, run:
```bash
python run_pipeline.py
```

### What happens during execution?
1. **Preprocessing (`preprocess.py`)**: Extracts the 40 real impedance features, normalizes them using a `StandardScaler`, and outputs a C-header (`scaler_params.h`) so the ESP32 can identically normalize live data.
2. **Training (`train_model.py`)**: Trains an MLP and a 1D CNN in PyTorch, exporting them to ONNX format.
3. **Quantization (`convert_quantize.py`)**: Uses `onnx2tf` to translate the ONNX graphs to TensorFlow, and then performs **INT8 Post-Training Quantization** using a representative dataset to shrink the models by up to 70%.
4. **Evaluation (`evaluate.py`)**: Compares the FP32 vs INT8 models, generating plots and accuracy metrics (MAE, RMSE, R²) in the `docs/` folder.
5. **C-Array Generation (`convert_to_c_array.py`)**: Converts the best `.tflite` model into a raw C byte array (`model_data.h`) for the microcontroller.

---

## 🔌 Flashing the ESP32 Firmware

The embedded firmware utilizes the **TensorFlow Lite for Microcontrollers (TFLM)** library and is configured via **PlatformIO**.

1. Install the **PlatformIO IDE** extension in VSCode.
2. Open the `esp32/` folder specifically inside VSCode (`File > Open Folder...`).
3. Connect your ESP32 board via USB.
4. Click the **Checkmark (Build)** icon in the bottom PlatformIO toolbar to download the TensorFlow Lite libraries and compile the code.
5. Click the **Right Arrow (Upload)** icon to flash the firmware to your board.
6. Open the **Serial Monitor** (Plug icon) to view live SOH predictions running entirely on the edge!

---

### ESP32 Memory Management
The firmware is designed to be highly memory efficient. The ESP32 allocates a static **Tensor Arena** of just 100 KB in the DRAM. The live sensor readings are automatically normalized, quantized to `int8`, passed through the AI interpreter, and dequantized back into a human-readable Health Percentage without ever needing a cloud connection.
# dual-tiny-ml-models-
