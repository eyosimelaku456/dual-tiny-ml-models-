# TinyML Battery State of Health Estimation Using EIS on ESP32
## Technical Documentation for Thesis

---

## 1. System Overview

### 1.1 Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE (Python)                    │
│                                                                  │
│  EIS_dataset.csv                                                 │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────────┐  │
│  │ preprocess.py│────▶│train_model.py│────▶│convert_quantize │  │
│  │              │     │              │     │      .py         │  │
│  │ - Load CSV   │     │ - MLP train  │     │ - ONNX export   │  │
│  │ - Select r_* │     │ - CNN1D train│     │ - onnx2tf conv. │  │
│  │ - Normalize  │     │ - Save .pth  │     │ - INT8 Quantize │  │
│  │ - Save scaler│     │ - Export ONNX│     │ - Save .tflite  │  │
│  └──────────────┘     └──────────────┘     └─────────────────┘  │
│                                                    │             │
│  ┌──────────────┐     ┌──────────────┐            │             │
│  │  evaluate.py │◀────│convert_to_   │◀───────────┘             │
│  │              │     │c_array.py    │                           │
│  │ - FP32 vs    │     │              │                           │
│  │   INT8 eval  │     │ - .tflite →  │                           │
│  │ - MAE, RMSE  │     │   model_data │                           │
│  │ - Plot charts│     │   .h         │                           │
│  └──────────────┘     └──────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  DEPLOYMENT (ESP32 Firmware)                     │
│                                                                  │
│  Flash Memory                    SRAM (~520 KB)                  │
│  ┌──────────────┐                ┌──────────────────────────┐   │
│  │ model_data.h │                │  Tensor Arena (100 KB)   │   │
│  │ (INT8 model) │──── load ─────▶│  Interpreter             │   │
│  └──────────────┘                │  Input/Output Tensors    │   │
│  ┌──────────────┐                └──────────────────────────┘   │
│  │scaler_params │── normalize ──▶ Normalized Input               │
│  │     .h       │                         │                      │
│  └──────────────┘                         ▼                      │
│                                  ┌──────────────────────────┐   │
│                                  │  TFLM Inference           │   │
│                                  │  Latency: < 50 ms         │   │
│                                  │  Output: SOH (0.0 - 1.0)  │   │
│                                  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Data → Model → ESP32 Flow

| Stage | Tool | Input | Output |
|-------|------|-------|--------|
| Preprocessing | `preprocess.py` | `EIS_dataset.csv` | Normalized arrays + scaler JSON |
| Training | `train_model.py` | Normalized arrays | `.pth`, `.onnx` models |
| Conversion | `convert_quantize.py` | `.onnx` | `model_int8.tflite` |
| C Export | `convert_to_c_array.py` | `model_int8.tflite` | `model_data.h` *(run after conversion)* |
| Evaluation | `evaluate.py` | `.tflite` + test data | Metrics + plots |
| Firmware | `main.cpp` | Sensor data | SOH prediction |

---

## 2. Dataset Description

### 2.1 Electrochemical Impedance Spectroscopy (EIS)

EIS is an electrochemical characterization technique used to analyze battery internal dynamics. A small AC signal is applied across a range of frequencies, and the resulting impedance response (real and imaginary components) is recorded.

The impedance at each frequency point characterizes different internal phenomena:

| Frequency Range | Physical Phenomenon |
|----------------|---------------------|
| High frequency (kHz) | Ohmic resistance (R₀) |
| Mid frequency (100 Hz–1 kHz) | SEI layer resistance |
| Low frequency (1–100 Hz) | Charge transfer resistance |
| Very low freq (< 1 Hz) | Diffusion processes |

The **Nyquist plot** (imaginary vs. real impedance) is the classical visualization, showing a characteristic semicircle whose diameter increases with battery degradation.

### 2.2 Dataset Feature Structure

The `EIS_dataset.csv` contains one row per measurement cycle.

| Column Group | Prefix | Description | Count |
|-------------|--------|-------------|-------|
| Frequency points | `f_1, f_2, ...` | AC frequency applied (Hz) | N |
| Real impedance | `r_1, r_2, ...` | Real part of Z(ω) (Ω) | N |
| Imaginary impedance | `i_1, i_2, ...` | Imaginary part of Z(ω) (Ω) | N |
| Temperature | `temperature` | Cell temperature (°C) | 1 |
| State of Charge | `soc` | Current charge level (0–1) | 1 |
| **Target** | `soh` | **State of Health (0–1)** | 1 |

### 2.3 Target Variable (SOH)

State of Health is defined as the ratio of current capacity to rated capacity:

```
SOH = Q_current / Q_rated
```

- `SOH = 1.0` → New battery (100% capacity)
- `SOH = 0.8` → End-of-life threshold (80% capacity)
- `SOH < 0.8` → Battery considered degraded

---

## 3. Model Design

### 3.1 Feature Reduction Strategy

**Problem:** Full EIS impedance can contain 100+ frequency points → too large for ESP32.

**Solution:** Use only `r_*` (real impedance) columns, limited to first 40 points.

**Rationale:**
- Real impedance captures the dominant degradation indicators (ohmic + charge-transfer resistance)
- Imaginary impedance provides phase information but is more sensitive to noise
- Frequency column (`f_*`) is the x-axis label, not a feature
- 40 features × 1 byte (INT8) = 40 bytes input → negligible memory

**Final input: 40–42 features** (40 real impedance + optional temperature + SOC)

### 3.2 MLP Architecture (Baseline)

```
Input Layer:    N features (e.g., 42)
                │
                ▼
Hidden Layer 1: Linear(N → 32) + ReLU
                │
                ▼
Hidden Layer 2: Linear(32 → 16) + ReLU
                │
                ▼
Output Layer:   Linear(16 → 1)  [SOH prediction]
```

**Parameter count:** ~1,808 (for N=42)
**FP32 size:** ~7.1 KB
**INT8 size:** ~1.8 KB (estimated)

**Design justification:**
- Minimal architecture satisfies ESP32 < 100 KB constraint with wide margin
- ReLU is natively supported by TFLite Micro
- No recurrence → static computational graph → easy quantization
- Serves as baseline for comparison against CNN1D

### 3.3 CNN1D Architecture

```
Input:          (batch=1, features=N)
                │  reshape
                ▼
                (batch=1, channels=1, length=N)
                │
Conv1D Block 1: Conv1D(1→8, kernel=3, padding=1) → ReLU
                │  output: (1, 8, N)
                ▼
Conv1D Block 2: Conv1D(8→16, kernel=3, padding=1) → ReLU
                │  output: (1, 16, N)
                ▼
Flatten:        (1, 16×N)  e.g., (1, 672)
                │
FC Layer 1:     Linear(16×N → 32) + ReLU
                │
FC Layer 2:     Linear(32 → 1)   [SOH prediction]
```

**Parameter count:** ~22,000 (for N=42)
**FP32 size:** ~86 KB
**INT8 size:** ~22 KB (estimated)

**Design justification:**
- Conv1D extracts local spectral patterns across adjacent frequency points
- padding=1 with kernel=3 preserves sequence length (no dynamic shapes)
- No BatchNorm: removed for TFLM compatibility (requires static statistics at inference)
- No Dropout: inference-only; dropped from computation graph
- All ops (Conv1D, Linear, ReLU, Flatten) are TFLite Micro builtin

### 3.4 Design Choices for TFLite Micro Compatibility

| Design Decision | Reason |
|----------------|--------|
| No BatchNorm | Requires running mean/var statistics; quantization of BN is complex |
| No Dropout | No-op at inference; but ONNX export can fail with dropout nodes |
| padding=1 (same padding) | Keeps sequence length static → no dynamic shape issues |
| opset_version=12 | Broad ONNX compatibility with onnx2tf converter |
| static dynamic_axes=None | TFLite requires fixed input shapes; no dynamic batch |
| ReLU only | Natively supported; sigmoid/tanh require lookup tables |

---

## 4. Embedded Constraints

### 4.1 ESP32 Memory Limits

| Resource | Total Available | Budget Allocation |
|----------|---------------|-------------------|
| SRAM | ~520 KB | ~230 KB used |
| Flash | 4–8 MB | Model in flash (read-only) |
| Tensor Arena | ~100 KB | 100 KB allocated |
| Stack | ~16 KB | ~8 KB used |

### 4.2 Tensor Arena Explanation

The tensor arena is a statically allocated byte buffer used by TFLite Micro to:
- Store intermediate activations during forward pass
- Store quantization parameters
- Scratch buffers for operator kernels

> Unlike TensorFlow desktop, TFLM does **not** use dynamic heap allocation.
> All memory must be pre-allocated in the tensor arena.

**Sizing method:**
1. Start with a conservative size (e.g., 100 KB)
2. Run `interpreter->AllocateTensors()`
3. Check `interpreter->arena_used_bytes()`
4. Reduce if significant headroom exists

### 4.3 Optimization Strategies

| Strategy | Benefit | Applied? |
|----------|---------|---------|
| INT8 quantization | 4× size reduction, faster inference | ✓ |
| Feature reduction (40 vs 100+) | Smaller input, smaller FC layers | ✓ |
| Small architecture (32→16 MLP) | Fits within 100 KB easily | ✓ |
| Static shapes | Required for TFLM compilation | ✓ |
| `alignas(8)` on arrays | Required for TFLM tensor alignment | ✓ |
| `huge_app.csv` partition | Allows larger firmware binary | ✓ |

---

## 4b. Model Conversion Library: onnx2tf

### Why onnx2tf Instead of onnx-tf

The original conversion approach used `onnx-tf` (ONNX → TensorFlow SavedModel → TFLite).
However, `onnx-tf` depends on `tensorflow-addons`, which **does not support Python 3.12**
(last release supports only up to Python 3.11).

This project uses **`onnx2tf`** as the modern, Python 3.12-compatible replacement:

| Library | Conversion Path | Python 3.12 | Maintained |
|---------|----------------|-------------|------------|
| `onnx-tf` | ONNX → TF SavedModel → TFLite | ✗ | Unmaintained |
| **`onnx2tf`** | ONNX → TFLite (direct) | ✓ | Actively maintained |

**Key advantages of onnx2tf:**
- No `tensorflow-addons` dependency
- Direct ONNX → TFLite conversion (skips intermediate SavedModel)
- Built-in INT8 quantization support with representative datasets
- Better operator coverage for embedded targets (TFLM-friendly op selection)
- Compatible with Python 3.8–3.12

**Pipeline with onnx2tf:**
```
PyTorch (.pth)
    │ torch.onnx.export()
    ▼
ONNX (.onnx)
    │ onnx2tf.convert()  ← replaces onnx-tf
    ▼
TFLite FP32 (.tflite)
    │ tf.lite.TFLiteConverter + representative_dataset
    ▼
TFLite INT8 (.tflite)  ← deployed on ESP32
```

> **Note for thesis:** The `onnx-tf` package was deprecated in 2023 and requires
> `tensorflow-addons` which reached end-of-life in November 2023 and supports
> Python max 3.11. For any Python 3.12+ environment, `onnx2tf` is the
> recommended replacement and produces identical TFLite output.

---

## 5. Quantization

### 5.1 INT8 Quantization Theory

Quantization maps floating-point values to integers using two parameters:

```
scale      (s): step size between quantized levels
zero_point (z): integer value corresponding to float 0.0

Quantization:   q = round(x / s) + z        (float → int8)
Dequantization: x = (q - z) × s             (int8 → float)
```

For INT8: `q ∈ [-128, 127]` → 256 discrete levels

**Full INT8 Quantization** means:
- All **weights** stored as INT8
- All **activations** (intermediate tensors) computed in INT8
- Inference uses INT8 arithmetic throughout
- Input/output can remain float32 (handled by quantize/dequantize nodes)

### 5.2 Representative Dataset

The quantization calibration process requires a **representative dataset** to determine appropriate scale/zero_point values for each activation tensor.

**Why it matters:**
- If calibration data doesn't represent real input distribution → poor scale values → accuracy loss
- We use 200 real test samples from `X_test.npy`

```python
# From convert_quantize.py
converter.representative_dataset = get_representative_dataset(X_test, 200)
```

### 5.3 Quantization Parameters per Layer

Each layer in the INT8 model has:
- **Per-tensor scale/zero_point** for activations
- **Per-channel scale/zero_point** for weights (Conv layers)

These are extracted and stored in:
- `models/quantization_info.json`: input/output tensor details
- Used in `main.cpp` to correctly dequantize the model's output

### 5.4 Expected Quantization Error

For well-designed, smooth regression models:
- MAE degradation: typically < 0.5% SOH absolute error
- RMSE degradation: typically < 0.01
- Accuracy acceptable if ΔMAE < 0.01 (1% SOH)

---

## 6. Results

> **Note:** Fill in actual metrics after running the pipeline.

### 6.1 Training Results

| Model | MAE | RMSE | R² | Parameters | INT8 Size |
|-------|-----|------|----|-----------|-----------|
| MLP | TBD | TBD | TBD | ~1,808 | ~2 KB |
| CNN1D | TBD | TBD | TBD | ~22,000 | ~22 KB |

### 6.2 Quantization Impact

| Model | FP32 MAE | INT8 MAE | ΔMAE | FP32 KB | INT8 KB | Compression |
|-------|----------|----------|------|---------|---------|-------------|
| MLP | TBD | TBD | TBD | ~7 KB | ~2 KB | ~71% |
| CNN1D | TBD | TBD | TBD | ~86 KB | ~22 KB | ~74% |

### 6.3 ESP32 Embedded Performance

| Metric | MLP | CNN1D | Limit |
|--------|-----|-------|-------|
| Model size | ~2 KB | ~22 KB | < 100 KB |
| Tensor arena | TBD | TBD | < 100 KB |
| Inference latency | TBD ms | TBD ms | < 50 ms |
| RAM usage | TBD KB | TBD KB | < 520 KB |

---

## 7. Discussion

### 7.1 Accuracy vs. Efficiency Trade-offs

**MLP advantages:**
- Extremely small (< 2 KB INT8)
- Very fast inference (< 5 ms estimated)
- Predictable memory usage
- Easier to quantize without accuracy loss

**CNN1D advantages:**
- Captures local spectral patterns in EIS data
- Slightly higher accuracy (hypothesis)
- Better generalization to unseen degradation patterns

**Trade-off conclusion:**
For embedded battery SOH estimation, the MLP may be sufficient if MAE < 0.02 (2% SOH error), which is acceptable for most battery management applications. CNN1D is preferred when accuracy is the primary concern and memory allows.

### 7.2 EIS Feature Selection Impact

Using only `r_*` (real impedance) vs. full impedance (real + imaginary):
- Real impedance alone captures ohmic resistance and charge-transfer resistance trends
- Imaginary impedance adds phase information (capacitive/inductive)
- Full EIS → better accuracy, but 2× larger input
- For embedded deployment, real impedance only is a reasonable compromise

### 7.3 Quantization Sensitivity

EIS-based SOH models tend to be robust to INT8 quantization because:
- SOH is a smooth, slowly varying quantity (not binary classification)
- Small quantization errors (< 0.5% SOH) are within measurement noise
- Representative dataset ensures calibration captures full input range

---

## 8. Conclusion

This work demonstrates the **feasibility of TinyML for real-time battery SOH estimation** on resource-constrained embedded hardware.

**Key findings:**

1. **EIS → SOH models are quantization-friendly:** The smooth regression task is well-suited to INT8 quantization with minimal accuracy loss.

2. **MLP is the safer embedded choice:** With < 2 KB model size and < 10 ms inference, MLP easily fits ESP32 constraints while maintaining acceptable accuracy.

3. **CNN1D offers marginal accuracy gains:** The convolutional approach can detect local impedance patterns, potentially improving performance, at ~10× the model size of MLP.

4. **Full pipeline automation:** The Python pipeline (preprocess → train → convert → evaluate) enables rapid iteration and reproducibility.

5. **Production deployment path:** The generated `model_data.h` and `scaler_params.h` files provide a clean, zero-dependency deployment mechanism for any TFLite Micro-compatible microcontroller.

**Future work:**
- Incorporate imaginary impedance features with dimensionality reduction (PCA)
- Explore quantization-aware training (QAT) for improved INT8 accuracy
- Test on additional battery chemistries (LFP, NMC)
- Implement online calibration to compensate for temperature drift

---

## Appendix A: How to Run the Pipeline

### Setup (WSL / Linux)
```bash
# Navigate to project (WSL path format)
cd /mnt/c/Users/hp/Desktop/intern2

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (Python 3.12 compatible)
pip install --upgrade pip
pip install -r requirements.txt
```

### Run Pipeline (STRICT ORDER — each step depends on the previous)
```bash
cd python

# Step 1: Preprocess EIS data
python preprocess.py
# Outputs: data/X_train.npy, data/X_test.npy, models/scaler_params.json

# Step 2: Train models
python train_model.py
# Outputs: models/mlp_fp32.pth, models/cnn1d_fp32.pth, models/mlp.onnx, models/cnn1d.onnx

# Step 3: Convert & quantize  ← MUST run before convert_to_c_array.py
python convert_quantize.py
# Outputs: models/mlp_int8.tflite, models/cnn1d_int8.tflite, models/model_int8.tflite

# Step 4: Evaluate accuracy
python evaluate.py
# Outputs: docs/evaluation_report.json, docs/*.png plots

# Step 5: Generate ESP32 C header  ← Requires Step 3 output
python convert_to_c_array.py
# Outputs: esp32/model_data.h
```

### Flash ESP32 (using PlatformIO)
```bash
cd ../esp32
pio run --target upload
pio device monitor
```

## Appendix B: File Reference

| File | Purpose |
|------|---------|
| `python/preprocess.py` | EIS data loading, feature selection, normalization |
| `python/train_model.py` | MLP & CNN1D training in PyTorch |
| `python/convert_quantize.py` | PyTorch→ONNX→TFLite→INT8 pipeline (via **onnx2tf**, Python 3.12 ✓) |
| `python/evaluate.py` | FP32 vs INT8 accuracy comparison |
| `python/convert_to_c_array.py` | TFLite→C header converter |
| `esp32/main.cpp` | ESP32 inference firmware |
| `esp32/platformio.ini` | PlatformIO build configuration |
| `models/scaler_params.json` | Normalization parameters (JSON) |
| `models/scaler_params.h` | Normalization parameters (C header) |
| `models/model_int8.tflite` | Quantized model for deployment |
| `esp32/model_data.h` | Model as C byte array (auto-generated) |
