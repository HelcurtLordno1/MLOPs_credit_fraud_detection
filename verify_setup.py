#!/usr/bin/env python
"""Quick verification script for MLOps fraud detection pipeline."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 70)
print("🔍 MLOPS FRAUD DETECTION - MODULE VERIFICATION")
print("=" * 70)

# Test 1: Import data modules
print("\n✓ Testing data modules...")
try:
    from fraud_detection.data.schema import validate_dataset, SOURCE_COLUMNS
    from fraud_detection.data.features import engineer_all_features, get_feature_names
    from fraud_detection.data.pipeline import create_train_val_test_splits, load_splits
    print("  ✅ Data modules imported successfully")
except ImportError as e:
    print(f"  ❌ Data module import failed: {e}")
    sys.exit(1)

# Test 2: Import modeling modules
print("\n✓ Testing modeling modules...")
try:
    from fraud_detection.modeling.train import train_logistic_regression, train_lightgbm, prepare_features
    from fraud_detection.modeling.evaluate import (
        compute_metrics, evaluate_model, analyze_bias_variance,
        get_classification_report, get_confusion_matrix
    )
    print("  ✅ Modeling modules imported successfully")
except ImportError as e:
    print(f"  ❌ Modeling module import failed: {e}")
    sys.exit(1)

# Test 3: Import monitoring modules
print("\n✓ Testing monitoring modules...")
try:
    from fraud_detection.monitoring.drift import (
        detect_distribution_drift, detect_feature_drift, detect_target_drift
    )
    from fraud_detection.monitoring.promotion import ModelPromoter
    print("  ✅ Monitoring modules imported successfully")
except ImportError as e:
    print(f"  ❌ Monitoring module import failed: {e}")
    sys.exit(1)

# Test 4: Check configuration files
print("\n✓ Testing configuration files...")
try:
    import yaml
    with open("configs/data.yaml") as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/train.yaml") as f:
        train_cfg = yaml.safe_load(f)
    print(f"  ✅ Configuration files loaded")
    print(f"     - data.yaml: train_csv={data_cfg.get('data', {}).get('processed_dir')}")
    print(f"     - train.yaml: experiment={train_cfg.get('experiment', {}).get('name')}")
except Exception as e:
    print(f"  ❌ Configuration loading failed: {e}")
    sys.exit(1)

# Test 5: Check feature categories
print("\n✓ Testing feature engineering...")
try:
    features = get_feature_names()
    total_features = sum(len(v) for v in features.values())
    print(f"  ✅ Feature categories defined:")
    for category, feats in features.items():
        print(f"     - {category:20s}: {len(feats):2d} features")
    print(f"  Total features: {total_features}")
except Exception as e:
    print(f"  ❌ Feature engineering test failed: {e}")
    sys.exit(1)

# Test 6: Check data files
print("\n✓ Testing data files...")
root = Path(".")
train_file = root / "fraudTrain.csv"
test_file = root / "fraudTest.csv"

if train_file.exists():
    print(f"  ✅ fraudTrain.csv found ({train_file.stat().st_size / 1e6:.1f} MB)")
else:
    print(f"  ⚠️  fraudTrain.csv not found - needed for Day 1")

if test_file.exists():
    print(f"  ✅ fraudTest.csv found ({test_file.stat().st_size / 1e6:.1f} MB)")
else:
    print(f"  ⚠️  fraudTest.csv not found - needed for Day 1")

print("\n" + "=" * 70)
print("✅ VERIFICATION COMPLETE - All modules ready for execution!")
print("=" * 70)

print("\n📅 Next Steps:")
print("  1. Run Day 1 notebook: jupyter notebook notebook/Day1_Data_Pipeline.ipynb")
print("  2. Run Day 2 notebook: jupyter notebook notebook/Day2_Model_Training_BiasVariance.ipynb")
print("  3. Run Day 3 notebook: jupyter notebook notebook/Day3_Evaluation_Promotion.ipynb")

print("\n📊 Expected outputs after running all notebooks:")
print("  - data/processed/train.parquet")
print("  - data/processed/validation.parquet")
print("  - data/processed/test.parquet")
print("  - models/trained/best_model.joblib")
print("  - reports/metrics/day1_data_summary.json")
print("  - reports/metrics/train_metrics.json")
print("  - reports/metrics/day3_evaluation_report.json")
print("  - Multiple visualization PNGs in reports/")

print("\n" + "=" * 70)
