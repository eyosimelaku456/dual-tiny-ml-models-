"""
=============================================================================
preprocess.py - EIS Dataset Preprocessing for Battery SOH Estimation
=============================================================================
Author  : TinyML Battery SOH Pipeline
Purpose : Load, clean, reduce, and normalize EIS features for model training
          and embedded (ESP32) deployment.

Dataset Structure Expected:
    Columns: f_1..f_n, r_1..r_n, i_1..i_n, [temperature], [soc], soh

Output:
    - data/X_train.npy, X_test.npy, y_train.npy, y_test.npy
    - models/scaler_params.json   (mean & std for ESP32 firmware)
    - models/feature_config.json  (input size, selected feature names)
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
DATASET_PATH  = r"../Dataset-model/Dataset-model/EIS_dataset.csv"
OUTPUT_DIR    = r"../data"
MODELS_DIR    = r"../models"

# Feature reduction: use only r_* (real impedance) columns
# Limit to first N_REAL_FEATURES for ESP32 memory constraints
N_REAL_FEATURES = 40        # Max real impedance features to keep

# Include auxiliary features if present in dataset
USE_TEMPERATURE = True      # Include temperature column if available
USE_SOC         = True      # Include SOC column if available

# Train/test split
TEST_SIZE       = 0.2
RANDOM_STATE    = 42

# ─────────────────────────────────────────────────────────────
# SETUP DIRECTORIES
# ─────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_dataset(path: str) -> pd.DataFrame:
    """Load EIS CSV dataset and report basic statistics."""
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Raw shape: {df.shape}")
    print(f"[INFO] Columns sample: {list(df.columns[:10])} ...")
    print(f"[INFO] Null values total: {df.isnull().sum().sum()}")
    return df


def identify_columns(df: pd.DataFrame):
    """
    Identify and categorize columns by prefix.

    Expected prefixes:
        f_* → frequency points (not used as features, they are the x-axis)
        r_* → real part of impedance  (PRIMARY features)
        i_* → imaginary part of impedance
        temperature → thermal measurement
        soc         → State of Charge
        soh         → State of Health (TARGET)
    """
    all_cols = list(df.columns)

    f_cols   = sorted([c for c in all_cols if c.startswith('f_')],
                      key=lambda x: int(x.split('_')[1]))
    r_cols   = sorted([c for c in all_cols if c.startswith('r_')],
                      key=lambda x: int(x.split('_')[1]))
    i_cols   = sorted([c for c in all_cols if c.startswith('i_')],
                      key=lambda x: int(x.split('_')[1]))
    temp_col = [c for c in all_cols if 'temperature' in c.lower() or c.lower() == 'temp']
    soc_col  = [c for c in all_cols if c.lower() == 'soc']
    soh_col  = [c for c in all_cols if c.lower() == 'soh']

    print(f"\n[INFO] Column breakdown:")
    print(f"  f_* (frequency)  : {len(f_cols)} columns")
    print(f"  r_* (real imp.)  : {len(r_cols)} columns")
    print(f"  i_* (imag. imp.) : {len(i_cols)} columns")
    print(f"  temperature      : {temp_col}")
    print(f"  soc              : {soc_col}")
    print(f"  soh (target)     : {soh_col}")

    return f_cols, r_cols, i_cols, temp_col, soc_col, soh_col


def select_features(df, r_cols, i_cols, temp_col, soc_col) -> tuple:
    """
    Select and reduce features for embedded deployment.

    Strategy: Use only r_* (real impedance) columns, first N_REAL_FEATURES.
    This reduces dimensionality while preserving the most informative
    EIS features (real impedance captures resistance-dominated behaviour).

    Optionally appends temperature and SOC as auxiliary features.
    """
    # Limit real impedance to first N_REAL_FEATURES
    selected_r = r_cols[:N_REAL_FEATURES]
    feature_cols = list(selected_r)

    # Append auxiliary features if available and requested
    if USE_TEMPERATURE and temp_col:
        feature_cols += temp_col[:1]
        print(f"[INFO] Added temperature feature: {temp_col[0]}")

    if USE_SOC and soc_col:
        feature_cols += soc_col[:1]
        print(f"[INFO] Added SOC feature: {soc_col[0]}")

    print(f"\n[INFO] Selected {len(feature_cols)} features for model input:")
    print(f"  → {len(selected_r)} real impedance (r_1 .. r_{N_REAL_FEATURES})")
    if USE_TEMPERATURE and temp_col:
        print(f"  → 1 temperature")
    if USE_SOC and soc_col:
        print(f"  → 1 SOC")

    return feature_cols


def preprocess(df, feature_cols, soh_col) -> dict:
    """
    Normalize features with StandardScaler and split into train/test sets.

    Returns:
        Dict with X_train, X_test, y_train, y_test (all numpy float32)
        and the fitted scaler for parameter extraction.
    """
    # Drop rows with NaN in selected columns
    subset = df[feature_cols + soh_col].dropna()
    print(f"\n[INFO] Rows after NaN drop: {len(subset)} (from {len(df)})")

    X = subset[feature_cols].values.astype(np.float32)
    y = subset[soh_col].values.astype(np.float32).ravel()

    print(f"[INFO] X shape: {X.shape}")
    print(f"[INFO] y shape: {y.shape}")
    print(f"[INFO] SOH range: [{y.min():.4f}, {y.max():.4f}]")

    # ── Normalize features ──────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # ── Train / Test Split ──────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print(f"\n[INFO] Train set: {X_train.shape[0]} samples")
    print(f"[INFO] Test  set: {X_test.shape[0]} samples")

    return {
        "X_train": X_train,
        "X_test" : X_test,
        "y_train": y_train,
        "y_test" : y_test,
        "scaler" : scaler,
    }


def save_scaler_params(scaler: StandardScaler, feature_cols: list):
    """
    Save normalization parameters (mean, std) to JSON.

    CRITICAL: These exact values must be replicated in the ESP32 C++ firmware
    to correctly normalize live sensor inputs before inference.
    """
    params = {
        "feature_names" : feature_cols,
        "n_features"    : len(feature_cols),
        "mean"          : scaler.mean_.tolist(),
        "std"           : scaler.scale_.tolist(),
    }
    out_path = os.path.join(MODELS_DIR, "scaler_params.json")
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\n[INFO] Scaler parameters saved → {out_path}")

    # Also generate a C header snippet for easy ESP32 integration
    c_header = generate_c_scaler_header(feature_cols, scaler.mean_, scaler.scale_)
    c_path = os.path.join(MODELS_DIR, "scaler_params.h")
    with open(c_path, "w") as f:
        f.write(c_header)
    print(f"[INFO] C scaler header saved  → {c_path}")


def generate_c_scaler_header(feature_cols, mean, std) -> str:
    """Generate a C header file with scaler parameters for ESP32 firmware."""
    n = len(feature_cols)
    mean_str = ", ".join(f"{v:.8f}f" for v in mean)
    std_str  = ", ".join(f"{v:.8f}f" for v in std)

    header = f"""// ============================================================
// scaler_params.h - Auto-generated by preprocess.py
// DO NOT EDIT MANUALLY
// Normalization parameters for EIS → SOH inference on ESP32
// ============================================================
#ifndef SCALER_PARAMS_H
#define SCALER_PARAMS_H

#define INPUT_SIZE {n}

// Feature mean values (from StandardScaler)
static const float FEATURE_MEAN[INPUT_SIZE] = {{
    {mean_str}
}};

// Feature standard deviation values (from StandardScaler)
static const float FEATURE_STD[INPUT_SIZE] = {{
    {std_str}
}};

// Normalize a single input value:
// normalized = (raw_value - mean) / std
inline float normalize_feature(float raw, int feature_idx) {{
    return (raw - FEATURE_MEAN[feature_idx]) / FEATURE_STD[feature_idx];
}}

#endif // SCALER_PARAMS_H
"""
    return header


def save_feature_config(feature_cols):
    """Save feature metadata for use by other pipeline scripts."""
    config = {
        "input_size"    : len(feature_cols),
        "feature_names" : feature_cols,
        "n_real_features": N_REAL_FEATURES,
        "use_temperature": USE_TEMPERATURE,
        "use_soc"       : USE_SOC,
    }
    out_path = os.path.join(MODELS_DIR, "feature_config.json")
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] Feature config saved  → {out_path}")


def save_arrays(data: dict):
    """Save train/test numpy arrays to disk."""
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), data["X_train"])
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),  data["X_test"])
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), data["y_train"])
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),  data["y_test"])
    print(f"\n[INFO] Arrays saved to: {OUTPUT_DIR}/")
    print(f"  X_train.npy : {data['X_train'].shape}")
    print(f"  X_test.npy  : {data['X_test'].shape}")
    print(f"  y_train.npy : {data['y_train'].shape}")
    print(f"  y_test.npy  : {data['y_test'].shape}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  EIS Battery SOH - Data Preprocessing Pipeline")
    print("=" * 60)

    # 1. Load
    df = load_dataset(DATASET_PATH)

    # 2. Identify columns
    f_cols, r_cols, i_cols, temp_col, soc_col, soh_col = identify_columns(df)

    if not soh_col:
        raise ValueError("[ERROR] 'soh' column not found in dataset!")
    if not r_cols:
        raise ValueError("[ERROR] No 'r_*' columns found in dataset!")

    # 3. Select features
    feature_cols = select_features(df, r_cols, i_cols, temp_col, soc_col)

    # 4. Preprocess
    data = preprocess(df, feature_cols, soh_col)

    # 5. Save scaler parameters (→ JSON + C header for ESP32)
    save_scaler_params(data["scaler"], feature_cols)

    # 6. Save feature config
    save_feature_config(feature_cols)

    # 7. Save arrays
    save_arrays(data)

    print("\n[✓] Preprocessing complete!")
    print(f"    INPUT_SIZE = {len(feature_cols)} features")
    print(f"    → {N_REAL_FEATURES} real impedance + extras")
    print(f"    Ready for model training.\n")
