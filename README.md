# CPEN 355 Final Project: Food-101 Image Classification

## Team Members

## Overview
This repository contains the code for our CPEN 355 final project. We implemented an automated image classification system using the Food-101 dataset to categorize diverse food images into 101 distinct meal classes. Our approach utilizes transfer learning via an EfficientNet backbone compared against a baseline simple CNN. 

## Dataset Preparation
To reproduce these results, you must first obtain the Kaggle Food-101 dataset:
1. Download the dataset from [Kaggle's Food-101 page](https://www.kaggle.com/datasets/dansbecker/food-101).
2. Extract the dataset and place it inside a folder named `data/raw/` at the root of this project.

## Installation
Ensure you have Python 3.10+ installed. Install the necessary dependencies using pip:
```bash
pip install -r requirements.txt
```

## Selected Baseline Model

The **April 12 ResNet18 baseline model** has been selected as the final baseline for this project:
- **Model file**: `member_1_code/models/baseline_resnet18_best_apr12_lr_003.pth`
- **Configuration**: Learning Rate = 0.003
- **Test Accuracy**: Saved predictions in `member_1_code/models/baseline_predictions_apr12_lr_003.csv`

## How to Run the Code
We recommend running this code in an environment with GPU acceleration (such as Google Colab).

### 1. Test the April 12 Baseline Model

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

This will:
- Load the pre-trained April 12 model
- Skip training and go directly to evaluation
- Generate predictions on the test set
- Save predictions to `member_1_code/models/baseline_predictions_apr12_lr_003_test.csv`
- Report the test accuracy

### 2. Train a New Baseline Model (Optional)

To train a fresh baseline ResNet18 model from scratch:

```bash
cd member_1_code/code
python train_baseline.py \
  --data-root ../data \
  --download \
  --batch-size 128 \
  --num-workers 8 \
  --num-epochs 30 \
  --early-stop-patience 5 \
  --lr 0.003 \
  --output-dir ../models
```

On Google Colab (save checkpoints to Drive):
```bash
python member_1_code/code/train_baseline.py \
  --data-root /content/food_data \
  --download \
  --batch-size 128 \
  --num-workers 8 \
  --num-epochs 30 \
  --early-stop-patience 5 \
  --lr 0.003 \
  --output-dir /content/drive/MyDrive/CPEN355/models
```

2. **Legacy script (root src):** To run the older combined training loop for both baseline CNN and EfficientNet, execute:
   ```bash
   python src/train.py
   ```
   *Note: This path is legacy and may not match member_1_code outputs.*

3. **Evaluate the Results:** To generate the evaluation metrics (Accuracy, Macro-F1, Precision, Recall) and output the confusion matrix, execute:
   ```bash
   python src/evaluate.py
   ```
