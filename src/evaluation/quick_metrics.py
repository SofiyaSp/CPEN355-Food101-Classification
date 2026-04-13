"""
Quick script to compute test accuracy and F1-score from prediction CSVs
"""
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

# Root path
REPO_ROOT = Path(__file__).parent.parent.parent

# CSV paths
baseline_csv = REPO_ROOT / "baseline_predictions_apr12_lr_003.csv"
efficientnet_csv = REPO_ROOT / "efficientnet_predictions.csv"

print("=" * 70)
print("QUICK METRICS COMPUTATION FROM PREDICTION CSVs")
print("=" * 70)

# ===== BASELINE MODEL =====
print("\n📊 BASELINE CNN (ResNet18)")
print("-" * 70)

try:
    df_baseline = pd.read_csv(baseline_csv)
    print(f"Loaded {len(df_baseline)} predictions from {baseline_csv.name}")
    print(f"Columns: {df_baseline.columns.tolist()}")
    
    # Inspect first few rows
    print("\nFirst 3 rows:")
    print(df_baseline.head(3))
    
    # Try to identify column names
    col_names = df_baseline.columns.tolist()
    true_col = None
    pred_col = None
    
    # Common column name patterns
    for col in ['label', 'true_label', 'true_class', 'actual_label', 'ground_truth']:
        if col in col_names:
            true_col = col
            break
    
    for col in ['predicted_class', 'prediction', 'predicted_label', 'pred_class', 'predicted']:
        if col in col_names:
            pred_col = col
            break
    
    if true_col and pred_col:
        true_labels = df_baseline[true_col].values
        predictions = df_baseline[pred_col].values
        
        # Calculate metrics
        acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
        
        print(f"\n✅ BASELINE RESULTS:")
        print(f"   Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"   Macro-averaged F1-Score: {f1:.4f}")
    else:
        print(f"❌ Error: Could not identify label columns")
        print(f"   True label column candidates: {[c for c in col_names if 'label' in c.lower() or 'truth' in c.lower()]}")
        print(f"   Pred column candidates: {[c for c in col_names if 'pred' in c.lower()]}")
        
except FileNotFoundError:
    print(f"❌ File not found: {baseline_csv}")
except Exception as e:
    print(f"❌ Error: {e}")

# ===== EFFICIENTNET MODEL =====
print("\n" + "=" * 70)
print("📊 EFFICIENTNET-B0 (Transfer Learning)")
print("-" * 70)

try:
    df_eff = pd.read_csv(efficientnet_csv)
    print(f"Loaded {len(df_eff)} predictions from {efficientnet_csv.name}")
    print(f"Columns: {df_eff.columns.tolist()}")
    
    # Inspect first few rows
    print("\nFirst 3 rows:")
    print(df_eff.head(3))
    
    # Try to identify column names
    col_names = df_eff.columns.tolist()
    true_col = None
    pred_col = None
    
    # Common column name patterns
    for col in ['label', 'true_label', 'true_class', 'actual_label', 'ground_truth']:
        if col in col_names:
            true_col = col
            break
    
    for col in ['predicted_class', 'prediction', 'predicted_label', 'pred_class', 'predicted']:
        if col in col_names:
            pred_col = col
            break
    
    if true_col and pred_col:
        true_labels = df_eff[true_col].values
        predictions = df_eff[pred_col].values
        
        # Calculate metrics
        acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
        
        print(f"\n✅ EFFICIENTNET RESULTS:")
        print(f"   Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"   Macro-averaged F1-Score: {f1:.4f}")
    else:
        print(f"❌ Error: Could not identify label columns")
        print(f"   True label column candidates: {[c for c in col_names if 'label' in c.lower() or 'truth' in c.lower()]}")
        print(f"   Pred column candidates: {[c for c in col_names if 'pred' in c.lower()]}")
        
except FileNotFoundError:
    print(f"❌ File not found: {efficientnet_csv}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 70)
print("✓ Done!")
print("=" * 70)
