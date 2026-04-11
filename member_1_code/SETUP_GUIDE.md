# Quick Start - Member 1

## 5-Minute Setup

```bash
cd member_1_code
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Test It Works

```bash
cd code
python data_loader.py --download
```

## Train Baseline (Optional)
If you have GPU and ~2-3 hours:
```bash
python train_baseline.py \
    --data-root ../data \
    --download \
    --batch-size 128 \
    --num-workers 8 \
    --num-epochs 30 \
    --early-stop-patience 5 \
    --output-dir ../models
```

Google Colab (save checkpoints to Drive):
```bash
python member_1_code/code/train_baseline.py \
    --data-root /content/food_data \
    --download \
    --batch-size 128 \
    --num-workers 8 \
    --num-epochs 30 \
    --early-stop-patience 5 \
    --output-dir /content/drive/MyDrive/CPEN355/models
```

Outputs:
- `models/baseline_resnet18_best.pth` (trained model)
- `models/baseline_predictions.csv` (for Member 3)

## Use in Your Code

Member 2 (Transfer Learning):
```python
from data_loader import create_data_loaders

train_loader, val_loader, test_loader, info = create_data_loaders(
    data_root='../data'
)
```

Member 3 (Evaluation):
```python
import pandas as pd
pred = pd.read_csv('../models/baseline_predictions.csv')
```

## Common Issues

| Problem | Fix |
|---------|-----|
| Out of memory | `--batch-size 32` |
| Download fails | Download manually from Kaggle, extract to `data/` |
| Import errors | Make sure you're in `code/` directory |

## Summary

✓ Data loaders ready  
✓ Baseline model ready  
✓ Can work in parallel (Member 2 doesn't need to wait)  
✓ Predictions exported for Member 3
