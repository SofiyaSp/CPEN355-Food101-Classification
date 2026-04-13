"""Generate publication-quality figures for the report."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import confusion_matrix
from pathlib import Path

EVAL_OUTPUT_DIR = Path(__file__).parent / "results"

def generate_pipeline_diagram():
    """Generate the data processing pipeline architecture diagram."""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Define stages
    stages = [
        ("Raw Images\n(Variable resolution)", "#e1f5ff"),
        ("Preprocessing\n(Resize 224×224,\nNormalize ImageNet)", "#fff9c4"),
        ("Data Augmentation\n(Flips, Crops,\nColor Jitter)", "#fff9c4"),
        ("Feature Extraction\nEfficientNet-B0\n(ImageNet pretrained,\nFrozen backbone)", "#c8e6c9"),
        ("Classification Head\n(Linear: 768→101)", "#ffccbc"),
        ("Output\n(101 Classes)", "#f8bbd0"),
    ]
    
    # Draw boxes
    box_width = 2.2
    box_height = 1.2
    y_center = 0.5
    x_positions = np.linspace(0.3, 14, len(stages))
    
    boxes = []
    for i, (label, color) in enumerate(stages):
        x = x_positions[i]
        box = FancyBboxPatch(
            (x - box_width/2, y_center - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            ec='black', fc=color, linewidth=2.5, edgecolor='#333333'
        )
        ax.add_patch(box)
        ax.text(x, y_center, label, ha='center', va='center',
                fontsize=13, fontweight='bold', color='#333333')
        boxes.append((x, y_center))
    
    # Draw arrows between boxes
    for i in range(len(boxes) - 1):
        x1, y1 = boxes[i]
        x2, y2 = boxes[i + 1]
        arrow = FancyArrowPatch(
            (x1 + box_width/2 + 0.05, y1),
            (x2 - box_width/2 - 0.05, y2),
            arrowstyle='->', mutation_scale=30, linewidth=2.5,
            color='#333333'
        )
        ax.add_patch(arrow)
    
    ax.set_xlim(-0.5, 15)
    ax.set_ylim(-0.3, 1.3)
    ax.axis('off')
    
    plt.tight_layout()
    output_path = EVAL_OUTPUT_DIR / "pipeline_diagram.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved pipeline diagram to {output_path}")
    plt.close()


def regenerate_confusion_matrix_large():
    """Regenerate confusion matrix with larger text for readability in PDF."""
    # Load predictions
    baseline_csv = Path(__file__).parent.parent.parent / "member_1_code" / "models" / "baseline_predictions_apr12_lr_003.csv"
    efficient_csv = Path(__file__).parent.parent.parent / "member_1_code" / "models" / "efficientnet_predictions.csv"
    
    for csv_file, model_name in [(baseline_csv, "baseline_cnn"), (efficient_csv, "efficientnet-b0")]:
        df = pd.read_csv(csv_file)
        
        # Handle column name variations
        true_col = next((col for col in df.columns if 'true' in col.lower() or 'label' in col.lower() and 'predict' not in col.lower()), None)
        pred_col = next((col for col in df.columns if 'predict' in col.lower() or col.lower() == 'label'), None)
        
        if true_col is None or pred_col is None:
            print(f"Could not find label columns in {csv_file}")
            continue
        
        labels = df[true_col].values
        predictions = df[pred_col].values
        
        cm = confusion_matrix(labels, predictions)
        
        # Create larger figure for better readability
        fig, ax = plt.subplots(figsize=(18, 16))
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        
        # Larger fonts
        ax.set_title(f'{model_name.upper().replace("_", "-")} - Confusion Matrix (101×101)', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Class', fontsize=15, fontweight='bold')
        ax.set_ylabel('True Class', fontsize=15, fontweight='bold')
        
        # Colorbar with larger text
        cbar = plt.colorbar(im, ax=ax, label='Count')
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('Count', fontsize=13, fontweight='bold')
        
        # Larger tick labels (but remove for 101x101 to avoid cluttering)
        ax.tick_params(axis='both', which='major', labelsize=0)
        
        plt.tight_layout()
        cm_fig_path = EVAL_OUTPUT_DIR / f"{model_name}_confusion_matrix.png"
        plt.savefig(cm_fig_path, dpi=150, bbox_inches='tight')
        print(f"✓ Regenerated {model_name} confusion matrix (larger text) at {cm_fig_path}")
        plt.close()


if __name__ == "__main__":
    print("Generating figures for report...\n")
    generate_pipeline_diagram()
    regenerate_confusion_matrix_large()
    print("\nAll figures generated successfully!")
