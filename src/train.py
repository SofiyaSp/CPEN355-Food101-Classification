# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Import the models from our other file
from models import build_efficientnet, SimpleCNN

# ==========================================
# TEMPORARY DUMMY DATA (Member 1 will replace this)
# Eventually, you will just do: from dataset import get_dataloaders
# ==========================================
def get_dataloaders(batch_size=32, num_classes=101):
    print("Loading Dummy Data...")
    X_train = torch.randn(batch_size * 5, 3, 224, 224)
    y_train = torch.randint(0, num_classes, (batch_size * 5,))
    X_val = torch.randn(batch_size * 2, 3, 224, 224)
    y_val = torch.randint(0, num_classes, (batch_size * 2,))
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# ==========================================
# TRAINING LOGIC (Member 2's Domain)
# ==========================================
def train_model(model, train_loader, val_loader, optimizer, device, epochs=3, save_name="best_model"):
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    
    best_val_acc = 0.0 # Track the best accuracy
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # ... (keep your existing training loop code here) ...
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        scheduler.step()
        
        # ... (keep your existing validation loop code here) ...
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        # SAVE THE BEST MODEL
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{save_name}.pth")
            print(f"--> Saved new best model with accuracy: {best_val_acc:.2f}%")

def generate_outputs(model, val_loader, device, model_name):
    torch.save(model.state_dict(), f"{model_name}.pth")
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    df = pd.DataFrame({"True_Label": all_labels, "Predicted_Label": all_preds})
    df.to_csv(f"{model_name}_predictions.csv", index=False)
    print(f"Saved {model_name}.pth and {model_name}_predictions.csv\n")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    train_loader, val_loader = get_dataloaders()

    # --- 1. Train Baseline CNN ---
    print("--- Training Baseline CNN ---")
    baseline_model = SimpleCNN().to(device)
    optimizer_base = optim.Adam(baseline_model.parameters(), lr=1e-3)
    train_model(baseline_model, train_loader, val_loader, optimizer_base, device, epochs=3, save_name="baseline_cnn_best")
    generate_outputs(baseline_model, val_loader, device, "baseline_cnn_final")

    # --- 2. Train EfficientNet ---
    print("--- Training EfficientNet (Head Only) ---")
    eff_model = build_efficientnet().to(device)
    optimizer_eff = optim.Adam(eff_model.classifier.parameters(), lr=1e-3)
    train_model(eff_model, train_loader, val_loader, optimizer_eff, device, epochs=3, save_name="efficientnet_head_best")
    
    print("--- Fine-Tuning EfficientNet (Deeper Layers) ---")
    for param in eff_model.features[7].parameters():
        param.requires_grad = True
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, eff_model.parameters()), lr=1e-5)
    train_model(eff_model, train_loader, val_loader, optimizer_ft, device, epochs=2, save_name="efficientnet_finetuned_best")
    generate_outputs(eff_model, val_loader, device, "efficientnet_final")