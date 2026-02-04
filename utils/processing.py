
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import config

def get_groups(n_samples):
    factor = n_samples // config.BASE_TRIALS
    return np.repeat(np.arange(config.BASE_TRIALS), factor)[:n_samples]

def prepare_tensors(X_tr_raw, X_val_raw, y_tr_raw, y_val_raw):
    N_tr, Ch, T = X_tr_raw.shape
    N_val = X_val_raw.shape[0]
    scaler = StandardScaler()
    
    X_tr_flat = X_tr_raw.reshape(N_tr, -1)
    X_val_flat = X_val_raw.reshape(N_val, -1)
    
    # LEAKAGE PREVENTION: Fit only on Train
    X_tr_sc = scaler.fit_transform(X_tr_flat).reshape(N_tr, 1, Ch, T)
    X_val_sc = scaler.transform(X_val_flat).reshape(N_val, 1, Ch, T)
    
    return (torch.FloatTensor(X_tr_sc).to(config.DEVICE), 
            torch.LongTensor(y_tr_raw).to(config.DEVICE),
            torch.FloatTensor(X_val_sc).to(config.DEVICE),
            torch.LongTensor(y_val_raw).to(config.DEVICE))
    
    
def get_groups(n_samples, base_size=320):
    """
    If we have 640 samples (Original + Aug), 
    groups are [0..319, 0..319] to keep siblings together.
    """
    if n_samples == base_size:
        return np.arange(base_size)
    
    factor = n_samples // base_size
    return np.repeat(np.arange(base_size), factor)[:n_samples]