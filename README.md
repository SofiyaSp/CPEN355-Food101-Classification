# CPEN 355 Final Project: Food-101 Image Classification

## Team Members
- Jazia Azreen (90739129)
- John Song (12837472)
- Sofiya Spolitak (69202497)

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

## How to Run the Code
We recommend running this code in an environment with GPU acceleration (such as Google Colab).

1. **Train the Models:** To run the training loop for both the baseline CNN and the EfficientNet model, execute:
   ```bash
   python src/train.py
   ```
   *Note: Model weights will be saved to the `outputs/saved_models/` directory.*

2. **Evaluate the Results:** To generate the evaluation metrics (Accuracy, Macro-F1, Precision, Recall) and output the confusion matrix, execute:
   ```bash
   python src/evaluate.py
   ```