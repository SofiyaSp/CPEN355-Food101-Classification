# Training script for baseline ResNet18
# TODO: Add cross-validation if time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from pathlib import Path
import csv

from data_loader import create_data_loaders
from baseline_model import create_baseline_model


class BaselineTrainer:
    # Handles training and validation loop
    
    def __init__(self, model, device, output_dir='./models'):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_accuracy = 0.0
        self.train_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader, optimizer, criterion):
        # One full pass through training data
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training') # tqdm is a Python library that provides a progress bar for loops. This allows you to visually track the progress of the training loop and see how much of the training data has been processed at any given time.
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = criterion(outputs, labels) #criterion is a loss function
            
            optimizer.zero_grad() # Clear the gradients of all optimized parameters before performing the backward pass. 
            loss.backward()
            optimizer.step() # Update the model parameters based on the computed gradients
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1) # Get the predicted class by finding the index of the maximum logit for each sample in the batch
            correct += predicted.eq(labels).sum().item() # Compare the predicted classes with the true labels and count how many predictions are correct.
            total += labels.size(0)
            
            # Update the progress bar with the current average loss and accuracy for the epoch.
            pbar.set_postfix({
                'loss': total_loss / len(pbar),
                'acc': correct / total if total > 0 else 0.0
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        # Check performance on val set
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validating')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix({
                    'loss': total_loss / len(pbar),
                    'acc': correct / total if total > 0 else 0.0
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=50, lr=0.001):
        # Main training loop
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        print(f"\nStarting baseline training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            self.val_accuracies.append(val_acc)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.save_checkpoint(epoch, 'best')
                print(f'Saved best model with val accuracy: {val_acc:.4f}')
            
            # Learning rate scheduling
            scheduler.step(val_acc)
        
        print(f"\nTraining complete! Best validation accuracy: {self.best_val_accuracy:.4f}")
        return self.train_losses, self.val_accuracies
    
    def save_checkpoint(self, epoch, tag='latest'):
        # Save model weights
        checkpoint_path = self.output_dir / f'baseline_resnet18_{tag}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
        }, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')
    
    def load_checkpoint(self, checkpoint_path):
        # Load saved weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded checkpoint from {checkpoint_path}')
    
    def generate_predictions(self, test_loader, output_csv='baseline_predictions.csv'):
        # Run model on test set and save predictions to CSV
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        print("\nGenerating predictions on test set...")
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Predicting'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                confidences, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_confidences.extend(torch.nn.functional.softmax(outputs, dim=1).max(1).values.cpu().numpy().tolist())
        
        # Save predictions to CSV
        output_path = self.output_dir / output_csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['true_label', 'predicted_label', 'confidence'])
            for true, pred, conf in zip(all_labels, all_predictions, all_confidences):
                writer.writerow([true, pred, f'{conf:.4f}'])
        
        print(f"Predictions saved to {output_path}")
        
        # Calculate accuracy
        accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_predictions)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        return all_predictions, all_labels, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train baseline ResNet18 model')
    parser.add_argument('--data-root', type=str, default='./data',
                        help='Path to Food-101 dataset')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--download', action='store_true',
                        help='Download dataset if needed')
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download
    )
    
    # Create model
    print("Creating baseline model...")
    model = create_baseline_model(num_classes=101, device=device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = BaselineTrainer(model, device, output_dir='./models')
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.num_epochs,
        lr=args.lr
    )
    
    # Generate predictions on test set
    trainer.generate_predictions(test_loader, output_csv='baseline_predictions.csv')
    
    print("\nBaseline training complete!")


if __name__ == '__main__':
    main()
