"""
=============================================================================
convert_quantize.py - Model Conversion & INT8 Quantization Pipeline
=============================================================================
Author  : TinyML Battery SOH Pipeline
Purpose : Convert trained PyTorch models through the full pipeline:
          PyTorch (ONNX) → TF SavedModel (via onnx2tf) → TFLite FP32 + INT8

Pipeline:
    1. ONNX → TF SavedModel  (via onnx2tf — Python 3.12 / TF 2.16 compatible)
    2. SavedModel → TFLite FP32  (baseline, for accuracy comparison)
    3. SavedModel → TFLite INT8  (full post-training quantization with
                                   real EIS calibration samples)

Outputs:
    - models/mlp_fp32.tflite
    - models/mlp_int8.tflite        ← Deploy this on ESP32
    - models/cnn1d_fp32.tflite
    - models/cnn1d_int8.tflite
    - models/model_int8.tflite      ← Best model copied here
    - models/quantization_info.json
=============================================================================
"""

import os
import json
import shutil
import tempfile
import numpy as np

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
DATA_DIR   = r"../data"
MODELS_DIR = r"../models"

# Number of representative samples for INT8 calibration
N_REPRESENTATIVE = 200

os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# STEP 1: ONNX → TF SavedModel  (via onnx2tf)
# ─────────────────────────────────────────────────────────────

def find_saved_model_dir(base_dir: str) -> str:
    """
    Locate the SavedModel directory produced by onnx2tf.

    onnx2tf may write the SavedModel directly into base_dir or
    into a subdirectory named after the model.  We search both.
    """
    # Case 1: SavedModel is directly in base_dir
    if os.path.exists(os.path.join(base_dir, "saved_model.pb")):
        return base_dir

    # Case 2: SavedModel is one level deep
    for entry in os.listdir(base_dir):
        candidate = os.path.join(base_dir, entry)
        if os.path.isdir(candidate):
            if os.path.exists(os.path.join(candidate, "saved_model.pb")):
                return candidate

    raise FileNotFoundError(
        f"No SavedModel (saved_model.pb) found under: {base_dir}\n"
        f"Contents: {os.listdir(base_dir)}"
    )


def onnx_to_saved_model(onnx_path: str, output_dir: str) -> str:
    """
    Convert ONNX model to TF SavedModel via onnx2tf.

    Returns path to the SavedModel directory.
    """
    import onnx2tf

    print(f"[INFO] onnx2tf: {onnx_path} → SavedModel...")
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=output_dir,
        non_verbose=True,
        copy_onnx_input_output_names_to_tflite=True,
    )
    sm_dir = find_saved_model_dir(output_dir)
    print(f"[INFO] SavedModel found at: {sm_dir}")
    return sm_dir


# ─────────────────────────────────────────────────────────────
# STEP 2a: SavedModel → FP32 TFLite
# ─────────────────────────────────────────────────────────────

def saved_model_to_fp32_tflite(sm_dir: str, output_path: str) -> int:
    """Convert TF SavedModel to FP32 TFLite. Returns size in bytes."""
    import tensorflow as tf

    print(f"[INFO] Converting SavedModel → FP32 TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size = len(tflite_model)
    print(f"[INFO] FP32 TFLite saved → {output_path} ({size / 1024:.2f} KB)")
    return size


# ─────────────────────────────────────────────────────────────
# STEP 2b: SavedModel → INT8 TFLite (full post-training quantization)
# ─────────────────────────────────────────────────────────────

def saved_model_to_int8_tflite(sm_dir: str, output_path: str,
                                X_test: np.ndarray) -> int:
    """
    Convert TF SavedModel to fully INT8-quantized TFLite.

    Quantization strategy:
      - Weights    : INT8 (per-channel for Conv, per-tensor for Dense)
      - Activations: INT8 (calibrated with real EIS test samples)
      - Input type : FLOAT32 (firmware sends normalized floats directly)
      - Output type: FLOAT32 (firmware reads SOH as float directly)

    Returns size in bytes.
    """
    import tensorflow as tf

    print(f"[INFO] Converting SavedModel → INT8 TFLite (with calibration)...")

    # Build calibration samples from real EIS test data
    n = min(N_REPRESENTATIVE, len(X_test))
    indices = np.random.choice(len(X_test), n, replace=False)
    calib_samples = X_test[indices].astype(np.float32)

    def representative_dataset():
        for sample in calib_samples:
            yield [sample.reshape(1, -1)]   # shape: (1, input_size)

    converter = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    converter.optimizations             = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset    = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type      = tf.float32
    converter.inference_output_type     = tf.float32

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size = len(tflite_model)
    print(f"[INFO] INT8 TFLite saved → {output_path} ({size / 1024:.2f} KB)")
    return size


# ─────────────────────────────────────────────────────────────
# STEP 3: INSPECT QUANTIZATION PARAMETERS
# ─────────────────────────────────────────────────────────────

def inspect_tflite(tflite_path: str) -> dict:
    """
    Extract input/output tensor details including quantization scale & zero_point.

    The output tensor's scale and zero_point are needed in ESP32 firmware
    to dequantize INT8 output back to a float SOH value:
        soh = (int8_output - zero_point) * scale
    """
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    info = {
        "input": {
            "shape"     : inp["shape"].tolist(),
            "dtype"     : str(inp["dtype"]),
            "scale"     : float(inp["quantization"][0]),
            "zero_point": int(inp["quantization"][1]),
        },
        "output": {
            "shape"     : out["shape"].tolist(),
            "dtype"     : str(out["dtype"]),
            "scale"     : float(out["quantization"][0]),
            "zero_point": int(out["quantization"][1]),
        },
    }

    print(f"    Input  → dtype: {info['input']['dtype']}, "
          f"scale: {info['input']['scale']:.6f}, "
          f"zp: {info['input']['zero_point']}")
    print(f"    Output → dtype: {info['output']['dtype']}, "
          f"scale: {info['output']['scale']:.6f}, "
          f"zp: {info['output']['zero_point']}")
    return info


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE FOR ONE MODEL
# ─────────────────────────────────────────────────────────────

def convert_model(model_name: str, X_test: np.ndarray) -> dict:
    """
    Run the full ONNX → SavedModel → FP32/INT8 TFLite pipeline for one model.
    """
    print(f"\n{'='*55}")
    print(f"  Converting: {model_name.upper()}")
    print(f"{'='*55}")

    onnx_path = os.path.join(MODELS_DIR, f"{model_name}.onnx")
    fp32_path = os.path.join(MODELS_DIR, f"{model_name}_fp32.tflite")
    int8_path = os.path.join(MODELS_DIR, f"{model_name}_int8.tflite")

    if not os.path.exists(onnx_path):
        print(f"[WARN] ONNX not found: {onnx_path}. Skipping {model_name}.")
        return {}

    tmp_dir = tempfile.mkdtemp(prefix="onnx2tf_")
    try:
        # Step 1: ONNX → SavedModel
        sm_dir = onnx_to_saved_model(onnx_path, tmp_dir)

        # Step 2a: SavedModel → FP32 TFLite
        print("\n  [Pass 1] FP32 TFLite...")
        fp32_bytes = saved_model_to_fp32_tflite(sm_dir, fp32_path)
        fp32_info  = inspect_tflite(fp32_path)

        # Step 2b: SavedModel → INT8 TFLite
        print("\n  [Pass 2] INT8 TFLite...")
        int8_bytes = saved_model_to_int8_tflite(sm_dir, int8_path, X_test)
        int8_info  = inspect_tflite(int8_path)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    compression = (1 - int8_bytes / fp32_bytes) * 100
    esp32_ok    = int8_bytes < 100 * 1024   # < 100 KB

    print(f"\n  Size: {fp32_bytes/1024:.2f} KB (FP32) → "
          f"{int8_bytes/1024:.2f} KB (INT8) "
          f"[{compression:.1f}% reduction]")
    print(f"  ESP32 fit (<100 KB): {'✓ YES' if esp32_ok else '✗ NO'}")

    return {
        "fp32_kb" : fp32_bytes / 1024,
        "int8_kb" : int8_bytes / 1024,
        "esp32_ok": esp32_ok,
        "fp32_info": fp32_info,
        "int8_info": int8_info,
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  EIS Battery SOH - Conversion & Quantization Pipeline")
    print("=" * 60)

    # Load calibration data from preprocessed test set
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy")).astype(np.float32)
    print(f"[INFO] Calibration data: {X_test.shape} "
          f"({N_REPRESENTATIVE} samples used for INT8 calibration)")

    quant_results = {}

    # Convert both models
    for model_name in ["mlp", "cnn1d"]:
        result = convert_model(model_name, X_test)
        quant_results[model_name] = result

    # ── Select best model for ESP32 ───────────────────────────
    # Prefer CNN1D (better accuracy) if it fits in <100 KB; else MLP
    selected = None
    for candidate in ["cnn1d", "mlp"]:
        r = quant_results.get(candidate, {})
        if r.get("esp32_ok", False):
            selected = candidate
            print(f"\n[INFO] ✓ Selected for ESP32: {candidate.upper()}")
            break

    if selected is None:
        selected = "mlp"
        print("[WARN] No model fits <100 KB! Forcing MLP as fallback.")

    quant_results["selected_for_esp32"] = selected

    # Copy selected INT8 model as canonical deployment file
    src = os.path.join(MODELS_DIR, f"{selected}_int8.tflite")
    dst = os.path.join(MODELS_DIR, "model_int8.tflite")
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"[INFO] Deployment model copied → {dst}")

    # Save full quantization report
    report_path = os.path.join(MODELS_DIR, "quantization_info.json")
    with open(report_path, "w") as f:
        json.dump(quant_results, f, indent=2)
    print(f"[INFO] Quantization report → {report_path}")

    # ── Summary table ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CONVERSION SUMMARY")
    print("=" * 60)
    print(f"{'Model':<10} {'FP32 KB':>10} {'INT8 KB':>10} {'ESP32 OK':>12}")
    print("-" * 48)
    for mn, mr in quant_results.items():
        if isinstance(mr, dict) and "fp32_kb" in mr:
            ok = "✓" if mr["esp32_ok"] else "✗"
            print(f"{mn:<10} {mr['fp32_kb']:>10.2f} {mr['int8_kb']:>10.2f} {ok:>12}")

    print(f"\n[✓] Conversion complete! Deploy: models/model_int8.tflite")
    print(f"    Next: run convert_to_c_array.py to generate esp32/model_data.h")
