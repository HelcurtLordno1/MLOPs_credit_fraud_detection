"""
Quick script to prepare training data for hyperparameter tuning
"""
import sys
from pathlib import Path
import os

# Set proper paths
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data" / "processed"

print(f"Project Root: {PROJECT_ROOT}")
print(f"Data Dir: {DATA_DIR}")

# Check if data exists
if not (DATA_DIR / "train.parquet").exists():
    print("❌ Train data not found. Running data pipeline...")
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.fraud_detection.data.pipeline import prepare_data
    prepare_data()
else:
    print("✅ Train data already exists, skipping preparation")

# Verify all files exist
files_needed = ["train.parquet", "val.parquet", "test.parquet"]
for f in files_needed:
    fpath = DATA_DIR / f
    if fpath.exists():
        print(f"   ✅ {f}: {fpath.stat().st_size / 1e6:.2f} MB")
    else:
        print(f"   ❌ {f}: NOT FOUND")

print("\nData preparation complete!")
