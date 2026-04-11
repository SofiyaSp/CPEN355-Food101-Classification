# Food-101 Dataset Loader & Preprocessing
# Member 1 - Data Engineering & Baseline Model
#
# Handles dataset loading, preprocessing, train/val/test splits
# and data augmentations

import os
import shutil
import torch
import torchvision.transforms as transforms
from torchvision.datasets import Food101
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np


# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406] # Standard mean for ImageNet pretraining
IMAGENET_STD = [0.229, 0.224, 0.225] # Standard standard deviation for ImageNet pretraining
# The 3 elements int he list correspond to the 3 channels of the image (R, G, B) and are used to normalize the pixel values of the images to have a mean of 0 and a standard deviation of 1
# the numbers 0.485, 0.456, and 0.406 represent the mean pixel values for the red, green, and blue channels of the images in the ImageNet dataset, respectively. The numbers 0.229, 0.224, and 0.225 represent the standard deviation of the pixel values for the red, green, and blue channels, respectively. 
# These values are used to normalize the pixel values of the images during preprocessing, ensuring that the input data is in a similar range to what the pretrained models expect, which can improve performance when fine-tuning on the Food-101 dataset.

# Default image size
DEFAULT_IMAGE_SIZE = 224 # Standard size for ResNet and EfficientNet models
NUM_CLASSES = 101 # Number of classes in Food-101


def get_transforms(split='train', image_size=DEFAULT_IMAGE_SIZE):
    # Returns appropriate transforms based on split
    # training data gets augmentation, val/test only gets normalization
    if split == 'train':
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            ),
        ])
    else:
        # Validation/Test transforms - no augmentation, just resize and normalize
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            ),
        ])
    
    return transform


class Food101Dataset(torch.utils.data.Dataset):
    # Custom wrapper for Food-101 with flexible split handling
    
    def __init__(self, root, split='train', image_size=DEFAULT_IMAGE_SIZE, 
                 download=False, transform=None):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.download = download
        
        # Load the official Food-101 dataset
        # Note: Food101 from torchvision uses 'train' and 'test' splits
        # We'll use train for training and val, then test for final evaluation
        is_train = (split == 'train')
        split_name = 'train' if is_train else 'test'

        try:
            self.dataset = Food101(
                root=root,
                split=split_name,
                download=download,
                transform=None  # We'll apply our custom transform
            )
        except RuntimeError as e:
            # Handle interrupted/corrupted download by clearing artifacts and retrying once.
            if download and 'File not found or corrupted' in str(e):
                archive_path = os.path.join(root, 'food-101.tar.gz')
                extract_path = os.path.join(root, 'food-101')

                if os.path.exists(archive_path):
                    os.remove(archive_path)
                if os.path.exists(extract_path):
                    shutil.rmtree(extract_path)

                print('Detected corrupted Food-101 download. Retrying with a clean archive...')
                self.dataset = Food101(
                    root=root,
                    split=split_name,
                    download=True,
                    transform=None
                )
            else:
                raise
        
        # Get default transform if none provided
        # get_transforms will return different transforms based on the split (train/val/test)
        # a transform is a function that takes in an image and applies a series of operations to it, such as resizing, cropping, flipping, color jittering, converting to tensor, and normalizing.
        # The specific operations applied depend on whether the split is 'train' (which includes data augmentation) or 'val/test' (which only includes resizing and normalization).
        if transform is None:
            transform = get_transforms(split=split, image_size=image_size)
        self.transform = transform
    
    # The __len__ method returns the total number of samples in the dataset, which is determined by the length of the underlying Food-101 dataset.
    def __len__(self):
        return len(self.dataset)
    
    # The __getitem__ method retrieves an image and its corresponding label from the underlying Food-101 dataset using the provided index (idx). 
    # If a transform is defined, it applies the transform to the image before returning it along with the label.
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def create_data_loaders(
    data_root,
    batch_size=32,
    num_workers=4,
    image_size=DEFAULT_IMAGE_SIZE,
    train_ratio=0.8,
    val_ratio=0.1,
    download=False
):
    # Creates train/val/test loaders for Food-101
    # Train: 80% of official train, Val: 10%, Test: official test set
    
    # Load official train and test sets
    train_dataset_full = Food101Dataset(
        root=data_root,
        split='train',
        image_size=image_size,
        download=download,
        transform=get_transforms(split='train', image_size=image_size)
    )
    
    test_dataset = Food101Dataset(
        root=data_root,
        split='test',
        image_size=image_size,
        download=False,
        transform=get_transforms(split='test', image_size=image_size)
    )
    
    # Split official train set into train and val
    total_train = len(train_dataset_full)
    train_size = int(total_train * train_ratio)
    val_size = total_train - train_size
    
    # random_split is a function from PyTorch that randomly splits a dataset into non-overlapping new datasets of given lengths.
    # here, it takes the full training dataset (train_dataset_full) and splits it into two subsets: one for training (train_dataset) and one for validation (val_dataset). 
    train_dataset, val_dataset = random_split(
        train_dataset_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create DataLoaders
    common_loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        # pin_memory speeds host->GPU transfer for CUDA training.
        'pin_memory': True,
    }

    if num_workers > 0:
        # Keep workers alive between epochs and prefetch batches to hide input-pipeline latency.
        common_loader_kwargs['persistent_workers'] = True
        # Queue extra batches ahead of time so the GPU waits less on data.
        common_loader_kwargs['prefetch_factor'] = 4

    train_loader = DataLoader(
        train_dataset,
        shuffle=True, # shuffle training data for better generalization
        **common_loader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False, # no need to shuffle validation data as we won't be training on it
        **common_loader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        shuffle=False, # no need to shuffle test data as we won't be training on it
        **common_loader_kwargs
    )
    
    # Dataset info for logging
    dataset_info = {
        'num_classes': NUM_CLASSES,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': len(test_dataset),
        'image_size': image_size,
        'batch_size': batch_size,
    }
    
    print(f"Dataset Information:")
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Number of classes: {NUM_CLASSES}")
    print(f"  Image size: {image_size}x{image_size}")
    
    return train_loader, val_loader, test_loader, dataset_info


if __name__ == "__main__":
    import argparse # argparse is a Python module that provides a way to handle command-line arguments. It allows you to define what arguments your program requires, parse those arguments from the command line, and access them in your code. In this context, argparse is used to create a command-line interface for testing the data loaders, allowing users to specify parameters such as the data root directory, batch size, and whether to download the dataset if needed.
    
    parser = argparse.ArgumentParser(description='Test data loaders') # This line creates an ArgumentParser object, which is used to define and parse command-line arguments. The description parameter provides a brief description of the program, which will be displayed when the user runs the program with the --help flag. In this case, the description is 'Test data loaders', indicating that this script is intended for testing the data loading functionality.
    parser.add_argument('--data-root', type=str, default='./data',
                        help='Path to Food-101 dataset') # The --data-root argument allows the user to specify the path to the Food-101 dataset. If not provided, it defaults to './data'. This is where the dataset will be loaded from or downloaded to if the --download flag is used.
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size') # The --batch-size argument allows the user to specify the batch size for the data loaders. If not provided, it defaults to 32. This determines how many samples will be loaded in each batch during training or evaluation.
    parser.add_argument('--download', action='store_true',
                        help='Download dataset if needed') # The --download argument allows the user to download the dataset if it is not already present in the specified data root directory.
    args = parser.parse_args() # This line parses the command-line arguments provided by the user and stores them in the args variable. The arguments can then be accessed using args.data_root, args.batch_size, and args.download in the code.
    
    print("Creating data loaders...")
    # The create_data_loaders function is called with the specified parameters to create the training, validation, and test data loaders for the Food-101 dataset. The function will handle loading the dataset, applying the necessary transformations, and splitting it into the appropriate subsets for training, validation, and testing. The resulting data loaders and dataset information are stored in the variables train_loader, val_loader, test_loader, and info, respectively.
    # a data_loader is a PyTorch DataLoader object that provides an iterable over the dataset, allowing you to efficiently load and batch the data during training or evaluation. The resulting data loaders can then be used in the training loop or evaluation process to access the data in batches.
    train_loader, val_loader, test_loader, info = create_data_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        download=args.download
    )
    
    print("\nTesting train loader...")
    # Iterate through a few batches of the training loader and print their shapes to verify everything is working correctly
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"  Batch {batch_idx}: Images shape {images.shape}, Labels shape {labels.shape}")
        if batch_idx == 2: # Just check the first 3 batches to avoid too much output
            break
    
    print("\nData loaders ready!")
