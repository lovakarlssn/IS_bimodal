# loaders/fmri_loader.py
import os
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder

# Add root to path for config access
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def load_fmri_dataset(mode="loso", subject_id=None, normalize=True, verbose=False):
    """
    Robustly loads fMRI Beta Maps.
    Handles NaNs, Infs, and zero-variance trials automatically.
    """
    base_path = "./data"
    
    x_path = os.path.join(base_path, "X_fMRI.npy")
    y_path = os.path.join(base_path, "y_fMRI.npy")
    g_path = os.path.join(base_path, "groups_fMRI.npy")

    # 1. Check Files
    if not os.path.exists(x_path):
        if verbose: print(f"[Error] fMRI data not found at {x_path}")
        return None, None, None, 0

    # 2. Load Data
    X_all = np.load(x_path) 
    y_all = np.load(y_path)
    groups_all = np.load(g_path)
    trial_ids_all = np.arange(len(X_all))


    if mode == "single":
        if subject_id is None:
            raise ValueError("subject_id must be provided when mode='single'")
        
        indices = np.where(groups_all == int(subject_id))[0]
        if len(indices) == 0:
            return None, None, None, 0
            
        X = X_all[indices]
        y = y_all[indices]
        groups = groups_all[indices]
        trial_ids = trial_ids_all[indices]
    else:
        X, y, groups = X_all, y_all, groups_all
        trial_ids = trial_ids_all
    if normalize:
        # 4. Robust Z-Score Normalization
        # Calculate Mean/Std per sample (Trial)
        # axes=(1, 2, 3, 4) collapses (C, D, H, W)
        means = np.mean(X, axis=(1, 2, 3, 4), keepdims=True)
        stds = np.std(X, axis=(1, 2, 3, 4), keepdims=True)

        stds[stds == 0] = 1.0 
        
        X = (X - means) / stds

    
        if verbose: print(f"   [Info] Post-normalization check: NaNs={np.isnan(X).sum()}, Infs={np.isinf(X).sum()}")

    # 5. Encode Labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    n_classes = len(le.classes_)

    metadata = {
        "subject_ids": groups,
        "trial_ids": trial_ids,
        "classes": le.classes_
    }

    if verbose:
        print(f"fMRI Data Loaded & Scaled.")
        print(f"   Shape: {X.shape}")
        print(f"   Range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"   Mean: {X.mean():.3f}, Std: {X.std():.3f}")
        print(f"   Classes: {n_classes}")

    return X, y, metadata, n_classes

if __name__ == "__main__":
    # Test it
    X, y, meta, n = load_fmri_dataset(mode="single", subject_id="01", verbose=True)