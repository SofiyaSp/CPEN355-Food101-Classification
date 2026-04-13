# Config for CPEN 355 food classification project
# Shared constants - edit here if you change anything

from pathlib import Path
import torch

# Directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / 'data'
MODELS_ROOT = PROJECT_ROOT / 'models'
FIGURES_ROOT = PROJECT_ROOT / 'figures'

# Create dirs
DATA_ROOT.mkdir(parents=True, exist_ok=True)
MODELS_ROOT.mkdir(parents=True, exist_ok=True)
FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

# Dataset stuff
NUM_CLASSES = 101
IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 32

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Splits
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

# Hyperparameters for baseline
BASELINE_EPOCHS = 50
BASELINE_LR = 0.01
BASELINE_MOMENTUM = 0.9
BASELINE_WEIGHT_DECAY = 5e-4
BASELINE_BATCH_SIZE = 64

# For EfficientNet (Member 2)
EFFICIENTNET_EPOCHS = 50
EFFICIENTNET_LR = 0.001
EFFICIENTNET_BATCH_SIZE = 64
EFFICIENTNET_UNFREEZE_EPOCH = 15

# Augmentation settings
AUGMENTATION = {
    'random_crop': True,
    'horizontal_flip': True,
    'flip_prob': 0.5,
    'color_jitter': True,
    'jitter_brightness': 0.2,
    'jitter_contrast': 0.2,
    'jitter_saturation': 0.2,
    'jitter_hue': 0.1,
}

# Model file paths
BASELINE_MODEL_BEST = MODELS_ROOT / 'baseline_resnet18_best.pth'
BASELINE_PREDICTIONS_CSV = MODELS_ROOT / 'baseline_predictions.csv'
EFFICIENTNET_MODEL_BEST = MODELS_ROOT / 'efficientnet_best.pth'
EFFICIENTNET_PREDICTIONS_CSV = MODELS_ROOT / 'efficientnet_predictions.csv'

# Device & workers
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4
RANDOM_SEED = 42

# Metrics to calculate
METRICS = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']


def print_config():
    """Print current settings"""
    print("=" * 50)
    print("PROJECT CONFIG")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Device: {DEVICE}")
    print(f"Num Classes: {NUM_CLASSES}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Random Seed: {RANDOM_SEED}")
    print("=" * 50)


if __name__ == '__main__':
    print_config()
