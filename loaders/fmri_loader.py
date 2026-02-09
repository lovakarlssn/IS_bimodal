import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_fmri_dataset(mode="loso", subject_id=None, normalize=True, verbose=False):
    """
    Robustly loads fMRI Beta Maps and handles normalization and metadata.
    """
    # Robustly determine data path relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__)) # .../loaders
    project_root = os.path.dirname(current_dir)              # .../
    base_path = os.path.join(project_root, "data")
    
    x_path = os.path.join(base_path, "X_fMRI.npy")
    y_path = os.path.join(base_path, "y_fMRI.npy")
    g_path = os.path.join(base_path, "groups_fMRI.npy")

    if not os.path.exists(x_path):
        if verbose: print(f"[Error] fMRI Data not found at: {x_path}")
        return None, None, None, 0

    X_all = np.load(x_path)
    y_all = np.load(y_path)
    groups_all = np.load(g_path)
    trial_ids_all = np.arange(len(X_all))

    if mode == "single":
        if subject_id is None:
            raise ValueError("subject_id must be provided when mode='single'")
        indices = np.where(groups_all == int(subject_id))[0]
        if len(indices) == 0:
            if verbose: print(f"[Error] No data found for subject {subject_id}")
            return None, None, None, 0
        X, y, groups, trial_ids = X_all[indices], y_all[indices], groups_all[indices], trial_ids_all[indices]
    else:
        X, y, groups, trial_ids = X_all, y_all, groups_all, trial_ids_all

    if normalize:
        means = np.mean(X, axis=(1, 2, 3, 4), keepdims=True)
        stds = np.std(X, axis=(1, 2, 3, 4), keepdims=True)
        stds[stds == 0] = 1.0
        X = (X - means) / stds

    le = LabelEncoder()
    y = le.fit_transform(y)
    n_classes = len(le.classes_)

    metadata = {
        "subject_ids": groups,
        "trial_ids": trial_ids,
        "classes": le.classes_
    }

    if verbose:
        print(f"fMRI Loaded: {X.shape}, Classes: {n_classes}")

    return X, y, metadata, n_classes