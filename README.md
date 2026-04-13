# CPEN 355 Final Project: Food-101 Image Classification

## Team Members
- Jasia Azreen (90739129) - Baseline CNN & Data Pipeline
- Sofiya Spolitak (69202497) - EfficientNet Transfer Learning
- John Song (12837472) - Evaluation Metrics & Report

## Overview
This repository contains the code for our CPEN 355 final project. We implemented an automated image classification system using the Food-101 dataset to categorize diverse food images into 101 distinct meal classes. Our approach utilizes a two-phase transfer learning strategy via an EfficientNet-B0 backbone, which is compared against a baseline ResNet18 trained from scratch.

## Project Structure
- `member_1_code/code/` - Core training directory containing data loaders (`data_loader.py`), the baseline training script (`train_baseline.py`), and the advanced EfficientNet fine-tuning script (`train.py`).
- `member_1_code/models/` - Directory where trained `.pth` weights and prediction CSVs are saved.
- `src/evaluation/` - Contains scripts to compute final metrics (`evaluate.py`) and generate the confusion matrix (`generate_figures.py`).
- `data/` - Target directory for the Food-101 dataset downloads.

## Dataset & Preprocessing
The dataset used is the official Food-101 Dataset (101 food classes, 1000 images each). 
You do not need to download this manually! Our PyTorch `create_data_loaders` function is configured with `download=True`, so it will automatically download and extract the dataset into the `data/` directory on the first run.

**Data Augmentation Pipeline:**
- **Training:** Resize to 224×224, random crop, random horizontal flip (50%), color jitter (brightness, contrast, saturation, hue), and ImageNet normalization.
- **Val/Test:** Resize to 224×224, center crop, and ImageNet normalization (no augmentation).

## Installation
Ensure you have Python 3.10+ installed. Create a virtual environment and install the dependencies:
## Selected Baseline Model

The **April 12 ResNet18 baseline model** has been selected as the final baseline for this project:
- **Model file**: `member_1_code/models/baseline_resnet18_best_apr12_lr_003.pth`
- **Configuration**: Learning Rate = 0.003
- **Prediction CSV**: `member_1_code/models/baseline_predictions_apr12_lr_003.csv`

## How to Reproduce Results
We strongly recommend running this code in an environment with GPU acceleration (such as Google Colab with a T4 GPU). The random seed is fixed at 42 across all scripts for full reproducibility.

### 1a. Test the Pre-trained April 12 Baseline Model
To evaluate the pre-trained April 12 baseline model on test data:
```bash
cd member_1_code/code
python train_baseline.py \
  --data-root ../data \
  --download \
  --batch-size 128 \
  --num-workers 8 \
  --checkpoint-path ../models/baseline_resnet18_best_apr12_lr_003.pth \
  --predict-only \
  --output-csv baseline_predictions_apr12_lr_003_test.csv
```

This will load the pre-trained model, skip training, and generate test predictions.

### 1b. Train a New Baseline ResNet18 (Optional)
To train the baseline model from scratch for 60 epochs, ensure you are in the root directory and run:
```bash
python member_1_code/code/train_baseline.py \
  --data-root data \
  --download \
  --batch-size 128 \
  --num-workers 8 \
  --num-epochs 60 \
  --early-stop-patience 5 \
  --output-dir member_1_code/models
```

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

## How to Reproduce Results
We strongly recommend running this code in an environment with GPU acceleration (such as Google Colab with a T4 GPU). The random seed is fixed at 42 across all scripts for full reproducibility.

### 1. Train the Baseline ResNet18
To train the baseline model from scratch for 60 epochs, ensure you are in the root directory and run:
```bash
python member_1_code/code/train_baseline.py \
  --data-root data \
  --download \
  --batch-size 128 \
  --num-workers 8 \
  --num-epochs 60 \
  --early-stop-patience 5 \
  --output-dir member_1_code/models
```

### 2. Train the EfficientNet-B0
To run our advanced two-phase transfer learning pipeline, execute our main training script. This automatically handles Phase 1 (training the classification head) and Phase 2 (unfreezing the top 3 feature blocks with learning rate decay) for a total of 50 epochs:
```bash
python member_1_code/code/train.py
```
*(This script relies on `member_1_code/code/config.py` for hyperparameters and saves `efficientnet_predictions.csv` directly into the `member_1_code/models/` directory).*

### 3. Evaluate the Results
Once both models have generated their prediction CSVs in the models folder, run the evaluation script to calculate the final metrics (Accuracy, Macro-F1, Precision, Recall) and generate the confusion matrix:
```bash
python src/evaluation/evaluate.py
```

## Troubleshooting
- **CUDA out of memory:** If your GPU runs out of VRAM, reduce the batch size in the training command (e.g., `--batch-size 32` or update `config.py`).
- **Download fails:** If the Kaggle API rate limits the auto-download, manually download the dataset from Kaggle and extract it to the `data/` folder.