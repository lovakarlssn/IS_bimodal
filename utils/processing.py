
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import config


def prepare_tensors(X_tr_raw, X_val_raw, y_tr_raw, y_val_raw):
    N_tr, Ch, T = X_tr_raw.shape
    N_val = X_val_raw.shape[0]
    scaler = StandardScaler()
    
    X_tr_flat = X_tr_raw.reshape(N_tr, -1)
    X_val_flat = X_val_raw.reshape(N_val, -1)
    
    
    X_tr_sc = scaler.fit_transform(X_tr_flat).reshape(N_tr, 1, Ch, T)
    X_val_sc = scaler.transform(X_val_flat).reshape(N_val, 1, Ch, T)
    
    return (torch.FloatTensor(X_tr_sc).to(config.DEVICE), 
            torch.LongTensor(y_tr_raw).to(config.DEVICE),
            torch.FloatTensor(X_val_sc).to(config.DEVICE),
            torch.LongTensor(y_val_raw).to(config.DEVICE))
    
def prepare_tensors(X_tr_raw, X_val_raw, y_tr_raw, y_val_raw):
    """Per-Trial Scaling: Each trial is normalized independently to prevent leakage."""

    X_tr = torch.FloatTensor(X_tr_raw)
    X_val = torch.FloatTensor(X_val_raw)
    mean_tr = X_tr.mean(dim=(1, 2), keepdim=True)
    std_tr = X_tr.std(dim=(1, 2), keepdim=True)
    X_tr = (X_tr - mean_tr) / (std_tr + 1e-6) 
    
    
    mean_val = X_val.mean(dim=(1, 2), keepdim=True)
    std_val = X_val.std(dim=(1, 2), keepdim=True)
    X_val = (X_val - mean_val) / (std_val + 1e-6)

    
    if X_tr.ndim == 3:
        X_tr = X_tr.unsqueeze(1)
        X_val = X_val.unsqueeze(1)
        
    return (X_tr.to(config.DEVICE), 
            torch.LongTensor(y_tr_raw).to(config.DEVICE),
            X_val.to(config.DEVICE), 
            torch.LongTensor(y_val_raw).to(config.DEVICE))
def get_groups(n_samples, base_size=320):
    """
    Generates groups to keep original and augmented trials together.
    Assumes data is stacked: [Original ... | Aug_1 ... | Aug_2 ...]
    """
    
    if n_samples == base_size:
        return np.arange(base_size)
    
    
    n_copies = n_samples 
    
    
    
    return np.tile(np.arange(base_size), n_copies)[:n_samples]