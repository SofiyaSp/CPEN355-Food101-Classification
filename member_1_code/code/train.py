# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Import YOUR model architecture
from models import build_efficientnet

# Import MEMBER 1's Data Loader and Constants
from data_loader import create_data_loaders
import config

# ==========================================
# TRAINING LOGIC (Member 2's Domain)
# ==========================================
def train_model(model, train_loader, val_loader, optimizer, device, epochs=3, save_name="best_model"):
    criterion = nn.CrossEntropyLoss()
    
    # Change 1: Use ReduceLROnPlateau instead of StepLR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_val_acc = 0.0 
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            pbar.set_postfix({'loss': loss.item()})
            
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / total
        print(f"--> Epoch {epoch+1} Summary | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        # Change 2: Step the scheduler using the validation accuracy!
        scheduler.step(val_acc)
        
        # SAVE THE BEST MODEL
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = config.MODELS_ROOT / f"{save_name}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model to {save_path} with accuracy: {best_val_acc:.2f}%\n")


def generate_outputs(model, test_loader, device, output_csv_name="efficientnet_predictions.csv"):
    model.eval()
    all_preds, all_labels = [], []
    
    print("\nGenerating final test set predictions for Member 3...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Predicting'):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # CRITICAL: Save using config.MODELS_ROOT
    output_path = config.MODELS_ROOT / output_csv_name
    df = pd.DataFrame({"true_label": all_labels, "predicted_label": all_preds})
    df.to_csv(output_path, index=False)
    print(f"Predictions successfully saved to {output_path}")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # 2. Load the REAL Data using Member 1's function
    print("Loading real Food-101 Dataset...")
    train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
        data_root=config.DATA_ROOT,
        batch_size=config.EFFICIENTNET_BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        download=True # Safely downloads the data if you are in Colab and don't have it yet
    )

    # 3. Create the EfficientNet Model
    print("\n--- Initializing EfficientNet ---")
    eff_model = build_efficientnet(num_classes=config.NUM_CLASSES).to(device)
    
    # 4. Train just the Classification Head
    print(f"\n--- Phase 1: Training EfficientNet (Head Only) for {config.EFFICIENTNET_UNFREEZE_EPOCH} epochs ---")
    optimizer_eff = optim.Adam(eff_model.classifier.parameters(), lr=config.EFFICIENTNET_LR)
    train_model(
        eff_model, train_loader, val_loader, optimizer_eff, device, 
        epochs=config.EFFICIENTNET_UNFREEZE_EPOCH, 
        save_name="efficientnet_head_best"
    )
    
    # 5. Fine-Tune the Deeper Layers
    print("\n--- Phase 2: Fine-Tuning EfficientNet (Deeper Layers) ---")
    
    # Reload the absolute best weights from the Head training phase before we unfreeze
    head_best_path = config.MODELS_ROOT / "efficientnet_head_best.pth"
    eff_model.load_state_dict(torch.load(head_best_path, map_location=device))
    
    # Unfreeze the last block
    for param in eff_model.features[7].parameters():
        param.requires_grad = True
        
    # Use a much smaller learning rate for fine-tuning
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, eff_model.parameters()), lr=1e-5)
    
    # Train for the remaining epochs
    remaining_epochs = config.EFFICIENTNET_EPOCHS - config.EFFICIENTNET_UNFREEZE_EPOCH
    train_model(
        eff_model, train_loader, val_loader, optimizer_ft, device, 
        epochs=remaining_epochs, 
        save_name="efficientnet_best"
    )
    
    # 6. Generate Outputs for Member 3 on the TEST loader (Not the validation loader!)
    
    # Reload the absolute best weights from fine-tuning
    final_best_path = config.MODELS_ROOT / "efficientnet_best.pth"
    eff_model.load_state_dict(torch.load(final_best_path, map_location=device))
    
    generate_outputs(eff_model, test_loader, device, output_csv_name="efficientnet_predictions.csv")