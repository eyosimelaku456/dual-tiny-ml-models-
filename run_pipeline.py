"""
=============================================================================
run_pipeline.py - Full Pipeline Runner
=============================================================================
Run this script to execute all pipeline steps in sequence.
Equivalent to running each script individually in order.
=============================================================================
"""
import subprocess
import sys
import os

PYTHON = sys.executable
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "python")

steps = [
    ("Step 1: Preprocessing",       "preprocess.py"),
    ("Step 2: Model Training",       "train_model.py"),
    ("Step 3: Convert & Quantize",   "convert_quantize.py"),
    ("Step 4: Evaluate",             "evaluate.py"),
    ("Step 5: Generate C Array",     "convert_to_c_array.py"),
]

print("=" * 60)
print("  TinyML Battery SOH - Full Pipeline Runner")
print("=" * 60)

for name, script in steps:
    print(f"\n{'─'*50}")
    print(f"  ▶ {name}")
    print(f"{'─'*50}")
    script_path = os.path.join(SCRIPTS_DIR, script)
    result = subprocess.run(
        [PYTHON, script_path],
        cwd=SCRIPTS_DIR,
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] {name} failed (exit code {result.returncode})")
        print("  Fix the error above, then re-run from this step.")
        sys.exit(result.returncode)
    print(f"  ✓ {name} complete!")

print("\n" + "=" * 60)
print("  ✓ Full pipeline complete!")
print("  → ESP32 files ready in: esp32/")
print("  → Docs & plots ready in: docs/")
print("=" * 60)
