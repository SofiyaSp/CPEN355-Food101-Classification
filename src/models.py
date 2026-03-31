# src/models.py
import torch.nn as nn
from torchvision import models

def build_efficientnet(num_classes=101):
    """
    Member 2: Core ML Role
    Loads pre-trained EfficientNet-B0, freezes backbone, and replaces the classification head.
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    
    # Freeze the convolutional base
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the classification head for Food-101
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model

class SimpleCNN(nn.Module):
    """
    Member 1: Baseline Model
    A simple 3-layer CNN trained from scratch to compare against EfficientNet.
    """
    def __init__(self, num_classes=101):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # Image size goes from 224 -> 112 -> 56 -> 28
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x