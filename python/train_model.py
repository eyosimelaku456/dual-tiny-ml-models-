"""
=============================================================================
train_model.py - MLP & CCN1D Model Training for Battery SOH Estimation
=============================================================================
Author  : TinyML Battery SOH Pipeline
Purpose : Train two PyTorch models (MLP baseline + professor's CCN1D) on
          preprocessed EIS impedance data to predict State of Health (SOH).

Models:
    1. MLP   - Lightweight fully-connected network (baseline)
    2. CCN1D - Professor's 1D CNN with BatchNorm + Dropout (primary model)

Outputs:
    - models/mlp_fp32.pth       (MLP PyTorch weights)
    - models/cnn1d_fp32.pth     (CCN1D PyTorch weights)
    - models/mlp.onnx           (MLP ONNX export)
    - models/cnn1d.onnx         (CCN1D ONNX export)
    - models/training_results.json
=============================================================================
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
DATA_DIR    = r"../data"
MODELS_DIR  = r"../models"
DOCS_DIR    = r"../docs"

BATCH_SIZE  = 64
EPOCHS      = 100
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR,   exist_ok=True)

# Load feature config
with open(os.path.join(MODELS_DIR, "feature_config.json")) as f:
    cfg = json.load(f)
INPUT_SIZE = cfg["input_size"]

print(f"[INFO] Device   : {DEVICE}")
print(f"[INFO] Input size: {INPUT_SIZE} features")


# ─────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    Lightweight Multi-Layer Perceptron for SOH regression.

    Architecture (ESP32-friendly):
        Input(N) → Linear(32) → ReLU → Linear(16) → ReLU → Linear(1)

    Design rationale:
        - Small parameter count (~1-2 KB INT8)
        - No dynamic ops → fully TFLite Micro compatible
        - ReLU activation → natively supported by TFLM
    """
    def __init__(self, input_size: int):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CCN1D(nn.Module):
    """
    Professor's CCN1D architecture for SOH regression on EIS data.

    Faithfully implements the provided CCN1D.py with the following
    adaptations for embedded deployment:

      - Input: (batch, input_size) reshaped internally to (batch, 1, input_size)
      - BatchNorm: present during training for stability; folds into Conv
        weights automatically during ONNX export (do_constant_folding=True)
      - Dropout: active during training only; removed at inference via .eval()
      - Output: global mean pooling over sequence dimension → scalar SOH per sample

    Architecture (num_layers=3, hidden_channels=32):
        Conv1d(1 → 32) → BN → ReLU → Dropout
        Conv1d(32 → 16) → BN → ReLU → Dropout
        Conv1d(16 →  8) → BN → ReLU → Dropout
        Conv1d( 8 →  1)  [output layer]
        GlobalMeanPool  → scalar SOH

    TFLite Micro Compatibility:
        ✓ BatchNorm folded into Conv weights at export (transparent to TFLM)
        ✓ Dropout identity at inference
        ✓ Static input shape (42 features)
        ✓ Ops: Conv1D, MeanPool, ReLU — all TFLM-supported
    """
    def __init__(self,
                 input_size: int,
                 hidden_channels: int = 32,
                 num_layers: int = 3,
                 kernel_size: int = 3,
                 dropout: float = 0.1,
                 output_size: int = 1):
        super(CCN1D, self).__init__()
        self.input_size = input_size

        self.conv_layers       = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()

        # First conv layer: 1 → hidden_channels
        self.conv_layers.append(
            nn.Conv1d(1, hidden_channels, kernel_size, padding=kernel_size // 2)
        )
        self.batch_norm_layers.append(nn.BatchNorm1d(hidden_channels))

        # Progressive channel-halving layers
        for i in range(1, num_layers):
            in_ch  = hidden_channels // (2 ** (i - 1))
            out_ch = hidden_channels // (2 ** i)
            self.conv_layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
            )
            self.batch_norm_layers.append(nn.BatchNorm1d(out_ch))

        # Output projection: final_channels → 1
        final_ch = hidden_channels // (2 ** (num_layers - 1))
        self.output_layer = nn.Conv1d(
            final_ch, output_size, kernel_size, padding=kernel_size // 2
        )

        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_size)  — flat EIS feature vector
        Returns:
            (batch,) — predicted SOH scalar
        """
        x = x.unsqueeze(1)   # (batch, 1, input_size)

        for conv, bn in zip(self.conv_layers, self.batch_norm_layers):
            x = self.relu(bn(conv(x)))
            x = self.dropout(x)

        x = self.output_layer(x)   # (batch, 1, input_size)
        x = x.mean(dim=-1)         # (batch, 1)  — global mean pool
        return x.squeeze(-1)       # (batch,)


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────

def load_data():
    """Load preprocessed numpy arrays and create DataLoaders."""
    X_train = torch.tensor(np.load(os.path.join(DATA_DIR, "X_train.npy")))
    X_test  = torch.tensor(np.load(os.path.join(DATA_DIR, "X_test.npy")))
    y_train = torch.tensor(np.load(os.path.join(DATA_DIR, "y_train.npy")))
    y_test  = torch.tensor(np.load(os.path.join(DATA_DIR, "y_test.npy")))

    train_ds = TensorDataset(X_train, y_train)
    test_ds  = TensorDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    print(f"[INFO] Train samples: {len(train_ds)}")
    print(f"[INFO] Test  samples: {len(test_ds)}")
    return train_loader, test_loader, X_test, y_test


# ─────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    """Run one training epoch, return average loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Evaluate model on loader, return loss, MAE, RMSE."""
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        total_loss += loss.item() * X_batch.size(0)
        all_preds.append(preds.cpu())
        all_targets.append(y_batch.cpu())

    preds   = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    mae  = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    return total_loss / len(loader.dataset), mae, rmse


def train_model(model, name, train_loader, test_loader):
    """Full training loop with early-stop patience and LR scheduling."""
    print(f"\n{'='*50}")
    print(f"  Training: {name}")
    print(f"{'='*50}")

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}
    best_val_loss = float("inf")
    patience_counter = 0
    PATIENCE = 20

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_mae, val_rmse = evaluate(model, test_loader, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{EPOCHS} | "
                  f"Train MSE: {tr_loss:.6f} | "
                  f"Val MAE: {val_mae:.4f} | "
                  f"Val RMSE: {val_rmse:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  [Early Stop] No improvement for {PATIENCE} epochs.")
                break

    # Restore best weights
    model.load_state_dict(best_state)
    _, final_mae, final_rmse = evaluate(model, test_loader, criterion)
    print(f"\n  ✓ Final → MAE: {final_mae:.4f} | RMSE: {final_rmse:.4f}")

    return model, history, {"mae": final_mae, "rmse": final_rmse}


# ─────────────────────────────────────────────────────────────
# EXPORT FUNCTIONS
# ─────────────────────────────────────────────────────────────

def save_pytorch(model, name):
    """Save PyTorch state dict."""
    path = os.path.join(MODELS_DIR, f"{name}_fp32.pth")
    torch.save(model.state_dict(), path)
    print(f"[INFO] Saved PyTorch model → {path}")


def export_onnx(model, name, input_size):
    """
    Export model to ONNX format with static input shape.

    CRITICAL for TFLite conversion:
        - opset_version=12 (broad compatibility)
        - dynamic_axes=None (static shapes required for TFLM)
        - input_names and output_names must match conversion script

    NOTE: Model is moved to CPU for export — PyTorch 2.x ONNX exporter
    cannot handle mixed CPU/CUDA device propagation during graph tracing.
    """
    model = model.cpu()   # Must be on CPU for ONNX graph tracing
    model.eval()
    dummy = torch.zeros(1, input_size)   # CPU tensor matches CPU model
    path  = os.path.join(MODELS_DIR, f"{name}.onnx")

    torch.onnx.export(
        model,
        dummy,
        path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        # Static shapes: critical for embedded deployment
        dynamic_axes=None,
    )
    print(f"[INFO] Exported ONNX → {path}")
    return path


def plot_training(history, name):
    """Save training curves plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train MSE", color="#4F8EF7")
    axes[0].plot(history["val_loss"],   label="Val MSE",   color="#F7744F")
    axes[0].set_title(f"{name} - Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_mae"],  label="MAE",  color="#4FF7A0")
    axes[1].plot(history["val_rmse"], label="RMSE", color="#F74F8E")
    axes[1].set_title(f"{name} - Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(DOCS_DIR, f"{name}_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Training curves saved → {path}")


def count_parameters(model):
    """Count trainable parameters and estimate model sizes."""
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fp32_kb  = n_params * 4 / 1024
    int8_kb  = n_params * 1 / 1024
    return n_params, fp32_kb, int8_kb


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  EIS Battery SOH - Model Training Pipeline")
    print("=" * 60)

    # Load data
    train_loader, test_loader, X_test, y_test = load_data()

    results = {}

    # ── 1. Train MLP ─────────────────────────────────────────
    mlp = MLP(INPUT_SIZE)
    n, fp32, int8 = count_parameters(mlp)
    print(f"\n[MLP Architecture]")
    print(f"  Parameters : {n:,}")
    print(f"  FP32 size  : {fp32:.2f} KB")
    print(f"  INT8 size  : {int8:.2f} KB (estimated)")
    print(mlp)

    mlp, mlp_hist, mlp_metrics = train_model(mlp, "MLP", train_loader, test_loader)
    save_pytorch(mlp, "mlp")
    export_onnx(mlp, "mlp", INPUT_SIZE)
    plot_training(mlp_hist, "MLP")
    results["mlp"] = {**mlp_metrics, "n_params": n, "fp32_kb": fp32, "int8_kb": int8}

    # ── 2. Train CCN1D (Professor's architecture) ─────────────
    # Parameters: hidden_channels=32, num_layers=3, kernel=3, dropout=0.1
    # Channel progression: 1 → 32 → 16 → 8 → 1 (global mean pool)
    cnn = CCN1D(INPUT_SIZE, hidden_channels=32, num_layers=3, kernel_size=3, dropout=0.1)
    n, fp32, int8 = count_parameters(cnn)
    print(f"\n[CCN1D Architecture — Professor's Model]")
    print(f"  hidden_channels : 32  |  num_layers : 3  |  dropout : 0.1")
    print(f"  Parameters      : {n:,}")
    print(f"  FP32 size       : {fp32:.2f} KB")
    print(f"  INT8 size       : {int8:.2f} KB (estimated)")
    print(cnn)

    cnn, cnn_hist, cnn_metrics = train_model(cnn, "CNN1D", train_loader, test_loader)
    save_pytorch(cnn, "cnn1d")
    export_onnx(cnn, "cnn1d", INPUT_SIZE)
    plot_training(cnn_hist, "CNN1D")
    results["cnn1d"] = {**cnn_metrics, "n_params": n, "fp32_kb": fp32, "int8_kb": int8}

    # ── 3. Save Results ───────────────────────────────────────
    out_path = os.path.join(MODELS_DIR, "training_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Training results saved → {out_path}")

    # ── 4. Summary ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    print(f"{'Model':<10} {'MAE':>10} {'RMSE':>10} {'Params':>10} {'INT8 KB':>10}")
    print("-" * 55)
    for model_name, m in results.items():
        print(f"{model_name:<10} {m['mae']:>10.4f} {m['rmse']:>10.4f} "
              f"{m['n_params']:>10,} {m['int8_kb']:>10.2f}")

    print("\n[✓] Training complete! Models saved to:", MODELS_DIR)

    # ── 5. Embedded feasibility check ────────────────────────
    print("\n[INFO] ESP32 Feasibility Check:")
    for model_name, m in results.items():
        kb = m["int8_kb"]
        ok = "✓" if kb < 100 else "✗"
        print(f"  {ok} {model_name}: {kb:.2f} KB (limit: 100 KB)")
