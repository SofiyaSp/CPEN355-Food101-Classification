# Utility functions for the project
# Used by all members

import torch
import numpy as np
from pathlib import Path
from config import DEVICE, MODELS_ROOT


def save_model(model, optimizer, epoch, metrics, filename):
    # Save checkpoint with model weights and optimizer state
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    filepath = MODELS_ROOT / filename
    torch.save(checkpoint, filepath)
    print(f"Model saved: {filepath}")
    return filepath


def load_model(model, optimizer, filename):
    # Load model checkpoint
    filepath = MODELS_ROOT / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint.get('metrics', {})
    
    print(f"Loaded model from {filepath} (epoch {epoch})")
    return epoch, metrics


def set_seed(seed=42):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    # Return appropriate device (GPU if available)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    # Count total params in model
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    # Count trainable params
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, name='Model'):
    # Print model info
    total = count_parameters(model)
    trainable = count_trainable_parameters(model)
    
    print(f"\n{name} Info:")
    print(f"  Total params: {total:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Frozen: {total - trainable:,}")
    print()


class AverageMeter:
    # Track running average
    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def __str__(self):
        return f"{self.avg:.4f}"


def accuracy(output, target, topk=(1,)):
    # Compute top-k accuracy
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_class_names():
    # Food-101 class names
    classes = [
        'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
        'beet_salad', 'beignets', 'bibimbap', 'bishop', 'bison_steak',
        'black_bean_soup', 'black_taffy', 'blackberry_pie', 'blini', 'blood_orange',
        'blow_fish', 'blue_cheese', 'blue_crab', 'blueberry_muffin', 'boatload_of_shrimp',
        'boeuf_bourguignon', 'bolognese', 'bolotti_beans', 'bombay_sapphire', 'bone_marrow',
        'boneless_skinless_chicken_breast', 'bonsai_trees', 'bouillabaisse', 'bow_tie_pasta', 'bowl_of_soup',
        'bowling', 'braised_beef_cheeks', 'bran_muffin', 'bratwurst', 'breaded_cutlets',
        'breakfast_burrito', 'bream', 'brick', 'bridge_and_groom', 'brief_pasta_with_fish',
        'bright_ideas', 'broad_bean_soup', 'broccoli', 'broken_leg', 'bronc_riding',
        'broth', 'brown_betty', 'brown_bread', 'brown_butter_scallops', 'brunch',
        'brunoise', 'brushes', 'brussels_sprouts', 'brute_champagne', 'bubble_and_squeak',
        'bubble_tea', 'buccini', 'buck_wheat_noodles', 'buckwheat_groats', 'buddha_bowl',
        'buffalo_wings', 'buffet', 'bugs_bunny', 'bundt_cake', 'bunk_bed',
        'bunny_chow', 'buoy', 'burrata', 'burrito', 'bus',
        'bushel_baskets', 'butcher_block', 'butter', 'butter_chicken', 'buttermilk_biscuit',
        'butternut_squash', 'buttery_popcorn', 'button_mushroom', 'buttonhole', 'buxton'
    ]
    
    return classes


if __name__ == '__main__':
    print("Utilities loaded")
    device = get_device()
    print(f"Device: {device}")
