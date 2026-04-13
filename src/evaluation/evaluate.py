"""
Evaluation Script for Food-101 Classification Models
Member 3: Evaluation & Analysis

Loads prediction CSVs from both models and computes:
- Accuracy, precision, recall, F1-score (macro-averaged)
- Confusion matrices and misclassification analysis
- Comparison visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Set up paths
REPO_ROOT = Path(__file__).parent.parent.parent
EVAL_OUTPUT_DIR = Path(__file__).parent / "results"
EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data configuration
NUM_CLASSES = 101


def load_predictions_csv(csv_path, model_name="Model"):
    """Load predictions from CSV file."""
    if not Path(csv_path).exists():
        print(f"❌ File not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} predictions from {csv_path.name}")
        return df
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None


def extract_labels_and_predictions(df, model_name="Model"):
    """Extract true labels and predictions from DataFrame."""
    col_names = df.columns.tolist()
    
    # Find true label column (case-insensitive)
    true_col = None
    for col in col_names:
        col_lower = col.lower()
        if any(x in col_lower for x in ['true', 'label', 'actual', 'ground']):
            true_col = col
            break
    
    # Find predicted label column (case-insensitive)
    pred_col = None
    for col in col_names:
        col_lower = col.lower()
        if any(x in col_lower for x in ['pred', 'predicted']):
            pred_col = col
            break
    
    if true_col is None or pred_col is None:
        print(f"❌ Could not identify label columns in {model_name}")
        print(f"   Available columns: {col_names}")
        return None, None
    
    true_labels = df[true_col].values
    predictions = df[pred_col].values
    
    # Convert to int if necessary
    true_labels = np.array([int(x) if isinstance(x, (int, float, np.integer)) else x for x in true_labels])
    predictions = np.array([int(x) if isinstance(x, (int, float, np.integer)) else x for x in predictions])
    
    return true_labels, predictions


def compute_metrics(predictions, labels):
    """Compute accuracy, precision, recall, F1-score (macro-averaged)."""
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro', zero_division=0)
    recall = recall_score(labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def generate_confusion_matrix(predictions, labels, model_name="model"):
    """Generate and save confusion matrix."""
    cm = confusion_matrix(labels, predictions)
    
    # Save as PNG
    fig, ax = plt.subplots(figsize=(14, 14))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_title(f'{model_name.upper()} - Confusion Matrix (101×101)', fontsize=14, pad=15)
    ax.set_xlabel('Predicted Class', fontsize=11)
    ax.set_ylabel('True Class', fontsize=11)
    plt.colorbar(im, ax=ax, label='Count')
    
    plt.tight_layout()
    cm_fig_path = EVAL_OUTPUT_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(cm_fig_path, dpi=100, bbox_inches='tight')
    print(f"  ✓ Saved confusion matrix to {cm_fig_path}")
    plt.close()
    
    # Save as CSV
    cm_csv_path = EVAL_OUTPUT_DIR / f"{model_name}_confusion_matrix.csv"
    pd.DataFrame(cm).to_csv(cm_csv_path, index=False)
    
    return cm


def find_top_misclassifications(predictions, labels, model_name="Model", top_n=15):
    """Find and save top misclassification pairs."""
    cm = confusion_matrix(labels, predictions)
    np.fill_diagonal(cm, 0)  # Ignore correct predictions
    
    misclass_list = []
    for true_idx in range(len(cm)):
        for pred_idx in range(len(cm)):
            if cm[true_idx, pred_idx] > 0:
                misclass_list.append({
                    'true_class_idx': true_idx,
                    'predicted_class_idx': pred_idx,
                    'count': int(cm[true_idx, pred_idx])
                })
    
    misclass_list = sorted(misclass_list, key=lambda x: x['count'], reverse=True)
    
    print(f"\n  Top {min(top_n, len(misclass_list))} misclassifications:")
    for i, item in enumerate(misclass_list[:top_n], 1):
        print(f"    {i:2d}. Class {item['true_class_idx']:3d} → {item['predicted_class_idx']:3d} ({item['count']:3d} times)")
    
    # Save to CSV
    misclass_df = pd.DataFrame(misclass_list[:top_n])
    misclass_csv_path = EVAL_OUTPUT_DIR / f"{model_name}_top_misclassifications.csv"
    misclass_df.to_csv(misclass_csv_path, index=False)
    
    return misclass_list[:top_n]


def plot_metrics_comparison(baseline_metrics, efficientnet_metrics):
    """Create side-by-side comparison plot."""
    models = ['Baseline CNN', 'EfficientNet-B0']
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    baseline_vals = [baseline_metrics[m] for m in metrics]
    efficientnet_vals = [efficientnet_metrics[m] for m in metrics]
    
    ax.bar(x - width/2, baseline_vals, width, label='Baseline CNN', alpha=0.8)
    ax.bar(x + width/2, efficientnet_vals, width, label='EfficientNet-B0', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: Evaluation Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (b, e) in enumerate(zip(baseline_vals, efficientnet_vals)):
        ax.text(i - width/2, b + 0.02, f'{b:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, e + 0.02, f'{e:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    comp_path = EVAL_OUTPUT_DIR / "model_comparison.png"
    plt.savefig(comp_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved comparison plot to {comp_path}")
    plt.close()


def main():
    """Main evaluation pipeline from CSV predictions."""
    print("=" * 75)
    print("FOOD-101 CLASSIFICATION - EVALUATION FROM PREDICTION CSVs")
    print("=" * 75)
    
    # CSV file paths
    baseline_csv = REPO_ROOT / "member_1_code" / "models" / "baseline_predictions_apr12_lr_003.csv"
    efficientnet_csv = REPO_ROOT / "member_1_code" / "models" / "efficientnet_predictions.csv"
    
    print(f"\nLooking for prediction CSVs:")
    print(f"  - Baseline: {baseline_csv}")
    print(f"  - EfficientNet: {efficientnet_csv}")
    
    model_results = {}
    
    # ===== BASELINE MODEL =====
    print("\n" + "=" * 75)
    print("BASELINE CNN (ResNet18 - trained from scratch)")
    print("=" * 75)
    
    baseline_df = load_predictions_csv(baseline_csv, "Baseline CNN")
    if baseline_df is not None:
        baseline_labels, baseline_preds = extract_labels_and_predictions(baseline_df, "Baseline CNN")
        if baseline_labels is not None:
            baseline_metrics = compute_metrics(baseline_preds, baseline_labels)
            
            print(f"\nResults:")
            print(f"  Accuracy:  {baseline_metrics['accuracy']:.4f} ({baseline_metrics['accuracy']*100:.2f}%)")
            print(f"  Precision: {baseline_metrics['precision']:.4f} (macro-averaged)")
            print(f"  Recall:    {baseline_metrics['recall']:.4f} (macro-averaged)")
            print(f"  F1-Score:  {baseline_metrics['f1_score']:.4f} (macro-averaged)")
            
            model_results['Baseline CNN'] = {
                'metrics': baseline_metrics,
                'predictions': baseline_preds,
                'labels': baseline_labels
            }
    
    # ===== EFFICIENTNET MODEL =====
    print("\n" + "=" * 75)
    print("EFFICIENTNET-B0 (Transfer Learning - frozen backbone)")
    print("=" * 75)
    
    efficientnet_df = load_predictions_csv(efficientnet_csv, "EfficientNet-B0")
    if efficientnet_df is not None:
        efficientnet_labels, efficientnet_preds = extract_labels_and_predictions(efficientnet_df, "EfficientNet-B0")
        if efficientnet_labels is not None:
            efficientnet_metrics = compute_metrics(efficientnet_preds, efficientnet_labels)
            
            print(f"\nResults:")
            print(f"  Accuracy:  {efficientnet_metrics['accuracy']:.4f} ({efficientnet_metrics['accuracy']*100:.2f}%)")
            print(f"  Precision: {efficientnet_metrics['precision']:.4f} (macro-averaged)")
            print(f"  Recall:    {efficientnet_metrics['recall']:.4f} (macro-averaged)")
            print(f"  F1-Score:  {efficientnet_metrics['f1_score']:.4f} (macro-averaged)")
            
            model_results['EfficientNet-B0'] = {
                'metrics': efficientnet_metrics,
                'predictions': efficientnet_preds,
                'labels': efficientnet_labels
            }
    
    # ===== GENERATE OUTPUTS =====
    if len(model_results) > 0:
        print("\n" + "=" * 75)
        print("GENERATING EVALUATION REPORTS & VISUALIZATIONS")
        print("=" * 75)
        
        # Summary table
        print("\nMetrics Summary Table:")
        summary_data = []
        for model_name, results in model_results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{results['metrics']['accuracy']:.4f}",
                'Precision': f"{results['metrics']['precision']:.4f}",
                'Recall': f"{results['metrics']['recall']:.4f}",
                'F1-Score': f"{results['metrics']['f1_score']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))
        
        # Save summary
        summary_csv_path = EVAL_OUTPUT_DIR / "evaluation_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        
        # Generate visualizations for each model
        for model_name, results in model_results.items():
            print(f"\n  Evaluating {model_name}...")
            generate_confusion_matrix(results['predictions'], results['labels'], model_name.replace(' ', '_').lower())
            find_top_misclassifications(results['predictions'], results['labels'], model_name.replace(' ', '_').lower())
        
        # Comparison plot (if both models)
        if len(model_results) == 2:
            print(f"\n  Creating comparison visualization...")
            baseline_metrics = model_results['Baseline CNN']['metrics']
            efficientnet_metrics = model_results['EfficientNet-B0']['metrics']
            plot_metrics_comparison(baseline_metrics, efficientnet_metrics)
        
        print(f"\n✓ Summary saved to {summary_csv_path}")
        print(f"✓ All results saved to: {EVAL_OUTPUT_DIR}\n")
    else:
        print("\n❌ No models could be evaluated. Check CSV file paths.")


if __name__ == "__main__":
    main()
