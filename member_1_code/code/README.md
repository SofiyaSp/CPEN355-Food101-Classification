# CPEN 355 - Food Classification (Member 1)

## What This Does

Data pipeline and baseline model for the food classification project.

- Loads Food-101 dataset (101 food classes, 1000 images each)
- Preprocesses images (224x224, ImageNet normalization)
- Splits into train/val/test (80/10/10)
- Implements ResNet18 baseline (from scratch)
- Outputs predictions for evaluation

## Setup

```bash
# 1. Virtual env
python -m venv venv
venv\Scripts\activate  # Windows

# 2. Install packages
pip install -r requirements.txt

# 3. Test data loader
cd code
python data_loader.py --download

# 4. (Optional) Train baseline - takes ~2-3 hours
python train_baseline.py --download --num-epochs 50
```

## Files

- `data_loader.py` - Main export. Use this to load data in your code
- `baseline_model.py` - ResNet18 architecture
- `train_baseline.py` - Training script
- `config.py` - Shared constants for team
- `utils.py` - Helper functions

## Using the Data Loaders

```python
from data_loader import create_data_loaders, get_transforms

# Create all loaders
train_loader, val_loader, test_loader, info = create_data_loaders(
    data_root='../data',
    batch_size=32
)

# Or just get the transforms
train_transform = get_transforms(split='train')
val_transform = get_transforms(split='val')
```

## Data

- **Sources**: Food-101 from Kaggle
- **Train**: 80,800 images
- **Val**: 10,100 images
- **Test**: 25,250 images
- **Classes**: 101 food types

## Preprocessing

Training data:
- Resize to 224×224
- Random crop
- Random horizontal flip (50%)
- Color jitter (brightness/contrast/saturation/hue)
- ImageNet normalization

Val/Test data:
- Resize to 224×224
- Center crop
- ImageNet normalization (no augmentation)

## Baseline Model

- ResNet18 (no pretrained weights)
- 11.2M parameters
- 101 output classes
- Expected test accuracy: 65-75%

## Output Files

After training baseline:
- `models/baseline_resnet18_best.pth` - Best model weights
- `models/baseline_predictions.csv` - Test predictions (for Member 3)

CSV format: `true_label, predicted_label, confidence`

## For Member 2

Just import from data_loader.py - you can work in parallel without waiting for baseline training to finish.

```python
from data_loader import create_data_loaders
```

## For Member 3

Once baseline training is done, grab the predictions CSV and use it for:
- Confusion matrix
- Precision/recall/F1 calculations
- Error analysis

## Issues?

- CUDA out of memory → reduce batch size: `--batch-size 32`
- Download fails → manually download from Kaggle and extract to `data/`
- Import errors → make sure you're in the `code/` directory

## Notes

- Random seed fixed at 42 for reproducibility
- GPU support automatic (falls back to CPU)
- All imports from torchvision, PyTorch, scipy (no custom dependencies)
