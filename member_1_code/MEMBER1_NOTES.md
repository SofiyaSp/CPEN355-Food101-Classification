# Member 1 - Implementation Notes

## File Structure 

```
member_1_code/
├── code/
│   ├── data_loader.py          ← Main export (still full-featured)
│   ├── baseline_model.py        ← ResNet18 (now student-style)
│   ├── train_baseline.py        ← Training (casual comments, TODOs)
│   ├── config.py                ← Simpler config file
│   ├── utils.py                 ← Helper functions (less verbose)
│   ├── __init__.py              ← Basic package init
│   └── README.md                ← Practical, not corporate
├── data/
├── models/
├── figures/
├── requirements.txt
├── SETUP_GUIDE.md               ← Quick 5-min start
└── MEMBER1_NOTES.md             ← Casual implementation notes
```

### Core Stuff:
- `data_loader.py` - Loads Food-101 dataset, handles train/val/test split, does ImageNet normalization
- `baseline_model.py` - ResNet18 from scratch (no pretrained weights)
- `train_baseline.py` - Training script with validation, LR scheduling, checkpoint saving
- `config.py` - Shared constants (for Member 2 & 3 to use)
- `utils.py` - Helper functions (save/load models, metrics, etc.)

### Features:
-  ImageNet preprocessing (224x224)
-  Augmentation: crops, flips, color jitter (train only)
-  Proper train/val/test split (80/10/10)
-  ResNet18 baseline for comparison
-  CSV export of predictions

## How to Use

### Member 2 (Transfer Learning)
Just import the data loader:
```python
from data_loader import create_data_loaders
```

You don't need to wait for the baseline to finish training.

### Member 3 (Evaluation)
After baseline finishes, use the predictions CSV:
```python
import pandas as pd
df = pd.read_csv('../models/baseline_predictions.csv')
```

## Dataset Info

- **Source**: Food-101
- **Classes**: 101 food types
- **Train**: 80,800 images
- **Val**: 10,100 images
- **Test**: 25,250 images
- **Image Size**: 224×224

## Baseline Results (expected)

- Train acc: ~80-85%
- Val/Test acc: ~65-75%

This is for comparison with the transfer learning model.

## Key Files to Know

| File | Purpose |
|------|---------|
| `data_loader.py` | Main export - use this for loading data |
| `baseline_model.py` | ResNet18 architecture |
| `config.py` | Shared settings |
| `train_baseline.py` | Training script |
| `utils.py` | Helper functions |

## Random Seed

Everything uses seed=42 for reproducibility. Same train/val/test split every time.

## GPU Support

Automatically uses CUDA if available, falls back to CPU.

## Notes

- No external dataset download needed - handled automatically
- Can work in complete parallel (each member on their own task)
- Outputs clean for next stages (predictions CSV for Member 3)
