import os
import json
import struct
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tempfile
import shutil

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATASET_PATH = os.path.join(PROJECT_ROOT, "Dataset-model", "Dataset-model", "Battery_RUL_with_ID.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ESP32_DIR = os.path.join(PROJECT_ROOT, "esp32")
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ESP32_DIR, exist_ok=True)

# 1. LOAD AND PREPROCESS DATA
print("[1] Loading RUL Dataset...")
df = pd.read_csv(DATASET_PATH)

feature_cols = [
    'Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 
    'Max. Voltage Dischar. (V)', 'Min. Voltage Charg. (V)', 
    'Time at 4.15V (s)', 'Time constant current (s)', 'Charging time (s)'
]
target_col = 'RUL'

df = df.dropna(subset=feature_cols + [target_col])
X = df[feature_cols].values.astype(np.float32)
y = df[target_col].values.astype(np.float32)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save Scaler Params for C++
mean_str = ", ".join(f"{v:.8f}f" for v in scaler.mean_)
std_str = ", ".join(f"{v:.8f}f" for v in scaler.scale_)
c_header = f"""// Auto-generated scaler for RUL
#ifndef SCALER_PARAMS_RUL_H
#define SCALER_PARAMS_RUL_H
#define INPUT_SIZE_RUL {len(feature_cols)}
static const float FEATURE_MEAN_RUL[INPUT_SIZE_RUL] = {{{mean_str}}};
static const float FEATURE_STD_RUL[INPUT_SIZE_RUL] = {{{std_str}}};
inline float normalize_feature_rul(float raw, int idx) {{
    return (raw - FEATURE_MEAN_RUL[idx]) / FEATURE_STD_RUL[idx];
}}
#endif
"""
with open(os.path.join(MODELS_DIR, "scaler_params_rul.h"), "w") as f:
    f.write(c_header)

print(f"Features: {len(feature_cols)}, Train: {len(X_train)}, Test: {len(X_test)}")

# 2. TRAIN MODEL
class RUL_MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

print("[2] Training RUL MLP...")
model = RUL_MLP(len(feature_cols))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(1, EPOCHS+1):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    preds = model(torch.tensor(X_test))
    mae = torch.mean(torch.abs(preds - torch.tensor(y_test)))
print(f"Training Complete! Test MAE: {mae.item():.2f} cycles")

# 3. EXPORT TO ONNX
onnx_path = os.path.join(MODELS_DIR, "rul_mlp.onnx")
dummy_input = torch.zeros(1, len(feature_cols))
torch.onnx.export(model, dummy_input, onnx_path, input_names=["input"], output_names=["output"], dynamic_axes=None)
print(f"[3] Exported ONNX to {onnx_path}")

# 4. CONVERT TO INT8 TFLITE
import onnx2tf
import tensorflow as tf

tmp_dir = tempfile.mkdtemp()
print("[4] Converting to TFLite INT8...")
onnx2tf.convert(input_onnx_file_path=onnx_path, output_folder_path=tmp_dir, non_verbose=True, copy_onnx_input_output_names_to_tflite=True)

def representative_dataset():
    for sample in X_test[:200]:
        yield [sample.reshape(1, -1)]

converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()
tflite_path = os.path.join(MODELS_DIR, "rul_int8.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
shutil.rmtree(tmp_dir, ignore_errors=True)
print(f"Saved INT8 TFLite ({len(tflite_model)/1024:.2f} KB) to {tflite_path}")

# 5. GENERATE C ARRAY
print("[5] Generating ESP32 Header...")
hex_lines = []
for i in range(0, len(tflite_model), 16):
    chunk = tflite_model[i:i+16]
    hex_lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk))
hex_array = ",\n".join(hex_lines)

header = f"""// Auto-generated RUL Model
#ifndef MODEL_DATA_RUL_H
#define MODEL_DATA_RUL_H
alignas(8) const unsigned char g_model_data_rul[] = {{
{hex_array}
}};
const unsigned int g_model_data_rul_len = {len(tflite_model)};
#endif
"""
with open(os.path.join(ESP32_DIR, "model_data_rul.h"), "w") as f:
    f.write(header)
print("[✓] RUL Pipeline Complete! model_data_rul.h generated.")
