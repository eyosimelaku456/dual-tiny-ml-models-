"""
=============================================================================
evaluate.py - Model Evaluation & Quantization Impact Analysis
=============================================================================
Author  : TinyML Battery SOH Pipeline
Purpose : Validate TFLite models (FP32 and INT8) against ground truth.
          Quantify accuracy loss from quantization.
          Generate evaluation plots and thesis-ready metrics report.

Outputs:
    - docs/evaluation_report.json
    - docs/fp32_vs_int8_comparison.png
    - docs/prediction_scatter.png
    - docs/residuals_plot.png
=============================================================================
"""

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
DATA_DIR   = r"../data"
MODELS_DIR = r"../models"
DOCS_DIR   = r"../docs"

os.makedirs(DOCS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


# ─────────────────────────────────────────────────────────────
# TFLITE INFERENCE
# ─────────────────────────────────────────────────────────────

def run_tflite_inference(model_path: str, X: np.ndarray) -> tuple:
    """
    Run full inference on dataset using TFLite interpreter.

    Returns:
        (predictions, avg_latency_ms)
    """
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    preds = []
    latencies = []

    for sample in X:
        inp = sample.reshape(1, -1).astype(np.float32)

        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]["index"])
        t1 = time.perf_counter()

        preds.append(float(pred.ravel()[0]))
        latencies.append((t1 - t0) * 1000)  # ms

    return np.array(preds), np.mean(latencies)


# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────

def plot_fp32_vs_int8(y_true, fp32_preds, int8_preds, model_name):
    """Scatter plot: FP32 vs INT8 predictions overlaid on ground truth."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name.upper()} — FP32 vs INT8 Comparison", fontsize=14, fontweight="bold")

    # Left: Predicted vs Actual
    axes[0].scatter(y_true, fp32_preds, alpha=0.4, s=10, color="#4F8EF7", label="FP32")
    axes[0].scatter(y_true, int8_preds, alpha=0.4, s=10, color="#F7744F", label="INT8")
    mn, mx = y_true.min(), y_true.max()
    axes[0].plot([mn, mx], [mn, mx], "k--", lw=1.5, label="Ideal")
    axes[0].set_xlabel("Ground Truth SOH")
    axes[0].set_ylabel("Predicted SOH")
    axes[0].set_title("Predicted vs Ground Truth")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: Difference between FP32 and INT8 predictions
    diff = np.abs(fp32_preds - int8_preds)
    axes[1].hist(diff, bins=50, color="#A04FF7", edgecolor="white", alpha=0.8)
    axes[1].axvline(diff.mean(), color="red", linestyle="--",
                    label=f"Mean diff: {diff.mean():.4f}")
    axes[1].set_xlabel("|FP32 - INT8| Prediction Difference")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Quantization Error Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(DOCS_DIR, f"{model_name}_fp32_vs_int8.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Comparison plot saved → {path}")


def plot_residuals(y_true, preds, model_name, variant="INT8"):
    """Residual plot for model evaluation."""
    residuals = y_true - preds

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{model_name.upper()} [{variant}] — Residual Analysis",
                 fontsize=13, fontweight="bold")

    axes[0].scatter(preds, residuals, alpha=0.3, s=8, color="#4FF7A0")
    axes[0].axhline(0, color="red", linestyle="--", lw=1.5)
    axes[0].set_xlabel("Predicted SOH")
    axes[0].set_ylabel("Residual (True - Pred)")
    axes[0].set_title("Residuals vs Predicted")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=50, color="#F7A04F", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(DOCS_DIR, f"{model_name}_{variant.lower()}_residuals.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Residual plot saved → {path}")


def plot_model_comparison(results: dict):
    """Bar chart comparing MLP vs CNN1D on MAE and RMSE."""
    models = [m for m in results if isinstance(results[m], dict) and "int8" in results[m]]
    mae_vals  = [results[m]["int8"]["mae"]  for m in models]
    rmse_vals = [results[m]["int8"]["rmse"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, mae_vals,  width, label="MAE",  color="#4F8EF7", alpha=0.85)
    bars2 = ax.bar(x + width/2, rmse_vals, width, label="RMSE", color="#F7744F", alpha=0.85)

    ax.set_xlabel("Model")
    ax.set_ylabel("Error")
    ax.set_title("Model Comparison — MLP vs CCN1D Professor's Model (INT8 Quantized)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.4f}",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.4f}",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")

    plt.tight_layout()
    path = os.path.join(DOCS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Model comparison plot → {path}")


# ─────────────────────────────────────────────────────────────
# EVALUATE ONE MODEL (FP32 + INT8)
# ─────────────────────────────────────────────────────────────

def evaluate_model(model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate a model's FP32 and INT8 TFLite variants."""
    print(f"\n{'='*50}")
    print(f"  Evaluating: {model_name.upper()}")
    print(f"{'='*50}")

    fp32_path = os.path.join(MODELS_DIR, f"{model_name}_fp32.tflite")
    int8_path = os.path.join(MODELS_DIR, f"{model_name}_int8.tflite")

    result = {}

    for variant, path in [("fp32", fp32_path), ("int8", int8_path)]:
        if not os.path.exists(path):
            print(f"[WARN] {path} not found. Skipping {variant}.")
            continue

        print(f"\n  ── {variant.upper()} ──")
        preds, avg_lat = run_tflite_inference(path, X_test)

        m = {
            "mae"           : mae(y_test, preds),
            "rmse"          : rmse(y_test, preds),
            "r2"            : r2(y_test, preds),
            "avg_latency_ms": avg_lat,
        }
        result[variant] = m

        print(f"    MAE     : {m['mae']:.4f}")
        print(f"    RMSE    : {m['rmse']:.4f}")
        print(f"    R²      : {m['r2']:.4f}")
        print(f"    Latency : {m['avg_latency_ms']:.3f} ms/sample (Python)")

        if variant == "int8":
            plot_residuals(y_test, preds, model_name, "INT8")

        # Store predictions for comparison plot
        if variant == "fp32":
            result["_fp32_preds"] = preds
        else:
            result["_int8_preds"] = preds

    # FP32 vs INT8 comparison
    if "_fp32_preds" in result and "_int8_preds" in result:
        plot_fp32_vs_int8(y_test, result["_fp32_preds"], result["_int8_preds"], model_name)

        # Quantization accuracy loss
        if "fp32" in result and "int8" in result:
            mae_loss  = result["int8"]["mae"]  - result["fp32"]["mae"]
            rmse_loss = result["int8"]["rmse"] - result["fp32"]["rmse"]
            result["quantization_impact"] = {
                "mae_delta" : mae_loss,
                "rmse_delta": rmse_loss,
                "acceptable": abs(mae_loss) < 0.01  # <1% SOH degradation target
            }
            print(f"\n  Quantization Impact:")
            print(f"    ΔMAE  : {mae_loss:+.4f}")
            print(f"    ΔRMSE : {rmse_loss:+.4f}")
            ok = "✓ Acceptable" if result["quantization_impact"]["acceptable"] else "✗ Above threshold"
            print(f"    Status: {ok}")

    # Clean up internal arrays from JSON serialization
    result.pop("_fp32_preds", None)
    result.pop("_int8_preds", None)

    return result


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  EIS Battery SOH - Evaluation Pipeline")
    print("=" * 60)

    # Load test data
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy")).astype(np.float32)
    print(f"[INFO] Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    all_results = {}

    # Evaluate each model
    for model_name in ["mlp", "cnn1d"]:
        fp32_path = os.path.join(MODELS_DIR, f"{model_name}_fp32.tflite")
        int8_path = os.path.join(MODELS_DIR, f"{model_name}_int8.tflite")
        if os.path.exists(fp32_path) or os.path.exists(int8_path):
            all_results[model_name] = evaluate_model(model_name, X_test, y_test)

    # Cross-model comparison plot
    if len(all_results) > 1:
        plot_model_comparison(all_results)

    # Load quantization info from conversion step
    q_info_path = os.path.join(MODELS_DIR, "quantization_info.json")
    if os.path.exists(q_info_path):
        with open(q_info_path) as f:
            quant_info = json.load(f)
        all_results["model_sizes"] = {
            k: {"fp32_kb": v.get("fp32_kb"), "int8_kb": v.get("int8_kb")}
            for k, v in quant_info.items()
            if isinstance(v, dict) and "fp32_kb" in v
        }

    # Save full evaluation report
    report_path = os.path.join(DOCS_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[INFO] Evaluation report saved → {report_path}")

    # ── Final Summary ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  EVALUATION SUMMARY (INT8 Models)")
    print("=" * 65)
    print(f"{'Model':<10} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Lat(ms)':>10} {'ΔMAE':>10}")
    print("-" * 65)
    for mn, mr in all_results.items():
        if not isinstance(mr, dict) or "int8" not in mr:
            continue
        m = mr["int8"]
        delta = mr.get("quantization_impact", {}).get("mae_delta", float("nan"))
        print(f"{mn.upper():<10} {m['mae']:>8.4f} {m['rmse']:>8.4f} "
              f"{m['r2']:>8.4f} {m['avg_latency_ms']:>10.3f} {delta:>10.4f}")

    print(f"\n[✓] Evaluation complete! Reports in: {DOCS_DIR}")
