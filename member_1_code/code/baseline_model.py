# Baseline Model - ResNet18 from scratch
# Used for comparison with transfer learning approach

import torch
import torch.nn as nn
import torchvision.models as models


class BaselineResNet18(nn.Module):
    # ResNet18 for Food-101 classification
    # Trained from scratch (no pretrained weights)
    
    def __init__(self, num_classes=101, pretrained=False):
        super(BaselineResNet18, self).__init__()
        
        self.num_classes = num_classes
        
        # Load ResNet18 - no pretrained weights for baseline
        self.model = models.resnet18(pretrained=pretrained) # models is a module from torchvision that provides pre-defined architectures for popular convolutional neural networks, including ResNet18. 
        
        # Replace the final fully connected layer
        num_features = self.model.fc.in_features # Get the number of input features to the final layer
        self.model.fc = nn.Linear(num_features, num_classes) # Replace the final layer with a new one that has num_classes outputs
        #Linear layer is a fully connected layer that takes the output from the previous layer (num_features) and produces an output of size num_classes, which corresponds to the number of classes in the Food-101 dataset. 
        #This allows the model to make predictions for each class based on the features extracted by the earlier layers of ResNet18.
        
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)
    
    def get_name(self):
        """Get model name for logging."""
        return "ResNet18-Baseline"


def create_baseline_model(num_classes=101, device='cpu'):
    """
    Create and return a baseline ResNet18 model.
    
    Args:
        num_classes (int): Number of output classes
        device (str): Device to allocate model to ('cpu' or 'cuda')
        
    Returns:
        nn.Module: Baseline ResNet18 model
    """
    model = BaselineResNet18(num_classes=num_classes, pretrained=False)
    model = model.to(device)
    return model


if __name__ == "__main__":
    import argparse
    
    # Create an argument parser to allow testing the baseline model from the command line. This will enable users to specify parameters such as the number of classes and the device to use (CPU or GPU) when running the script. The parser will handle parsing these arguments and making them accessible in the code for creating and testing the baseline model.
    parser = argparse.ArgumentParser(description='Test baseline model')
    parser.add_argument('--num-classes', type=int, default=101,
                        help='Number of classes')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    args = parser.parse_args()
    
    print("Creating baseline ResNet18 model...")
    model = create_baseline_model(
        num_classes=args.num_classes,
        device=args.device
    ) 
    
    print(f"Model: {model.get_name()}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(4, 3, 224, 224).to(args.device) # Create a dummy input tensor with the shape (batch_size, channels, height, width) that matches the expected input size for ResNet18. In this case, we use a batch size of 4 and an image size of 224x224 with 3 color channels (RGB). This will allow us to test the forward pass of the model and verify that it produces the expected output shape.
    model.eval() # Set the model to evaluation mode, which is important for certain layers like dropout and batch normalization that behave differently during training and evaluation. In evaluation mode, these layers will use their learned parameters without applying any randomization, ensuring that the output is consistent and suitable for inference.
    with torch.no_grad(): # This context manager is used to disable gradient calculations during the forward pass. Since we are only testing the model and not performing any training or backpropagation, we don't need to compute gradients. Using torch.no_grad() can save memory and improve performance during inference.
        output = model(dummy_input) # Pass the dummy input through the model to get the output logits. This will allow us to verify that the model is producing outputs of the correct shape and that the forward pass is working as expected. The output will contain the raw predictions (logits) for each class, which can be further processed to obtain probabilities or predicted classes if needed.
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output logits (first sample): {output[0][:5]}")
