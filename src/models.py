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
    Member 1: Baseline Model (Upgraded for real datasets)
    Includes BatchNorm, Dropout, and Adaptive Pooling to prevent OOM crashes and overfitting.
    """
    def __init__(self, num_classes=101):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Size: 112x112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Size: 56x56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Size: 28x28
            
            # Forces the feature map to 7x7, preventing massive memory spikes in the Linear layer
            nn.AdaptiveAvgPool2d((7, 7)) 
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5), # Randomly zeroes 50% of neurons to prevent overfitting
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x