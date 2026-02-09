import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.processing import prepare_tensors
from models.eegnets import *
import config
import numpy as np
from augmentations.eeg_aug import get_augmentation
import config
import numpy as np
def train_final_model(X, y, n_classes, exp_name, aug_params, hyperparams, save_path=None):
    """
    Trains on ALL data for final inference.
    Applies augmentation to the entire dataset since there is no validation split.
    """
    print(f"Training final model ({exp_name})...")
    
    
    if exp_name != "Original":
        print("   Applying augmentation to full dataset...")
        X, y = get_augmentation(exp_name, X, y, aug_params)

    
    
    X_tr, y_tr, _, _ = prepare_tensors(X, X, y, y)
    
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=hyperparams.get("batch_size", 32), shuffle=True)
    
    
    _, n_chans, n_time = X.shape
    model = EEGNet(n_classes, n_chans, n_time).to(config.DEVICE)
    opt = optim.Adam(model.parameters(), lr=hyperparams.get("lr", 0.001))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    
    for epoch in range(hyperparams.get("epochs", 50)):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"   Epoch {epoch}: Loss {total_loss/len(train_loader):.4f}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"   Model saved to {save_path}")
        
    return model