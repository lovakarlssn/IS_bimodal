import os
import numpy as np
import config
from sklearn.preprocessing import LabelEncoder

def load_eeg_dataset(mode="loso", subject_id=None, verbose=False):
    """
    Loads EEG datasets with automated Z-score normalization and channel padding/cropping.
    """
    # Robustly determine data path relative to this file
    # This ensures it works from notebooks, experiments, or root
    current_dir = os.path.dirname(os.path.abspath(__file__)) # .../loaders
    project_root = os.path.dirname(current_dir)              # .../
    base_path = os.path.join(project_root, "data")

    x_path = os.path.join(base_path, "X_EEG.npy")
    y_path = os.path.join(base_path, "y_EEG.npy")
    g_path = os.path.join(base_path, "groups_EEG.npy")

    if not os.path.exists(x_path):
        if verbose: print(f"[Error] EEG Data not found at: {x_path}")
        return None, None, None, 0

    X_all = np.load(x_path)
    y_all = np.load(y_path)
    groups_all = np.load(g_path)
    trial_ids_all = np.arange(len(X_all))

    means = np.mean(X_all, axis=-1, keepdims=True)
    stds = np.std(X_all, axis=-1, keepdims=True) + 1e-8
    X_all = (X_all - means) / stds

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

    le = LabelEncoder()
    y = le.fit_transform(y)
    n_classes = len(le.classes_)

    target_chans = config.TARGET_CHANS
    if X.shape[1] < target_chans:
        X = np.pad(X, ((0, 0), (0, target_chans - X.shape[1]), (0, 0)), mode='constant')
    elif X.shape[1] > target_chans:
        X = X[:, :target_chans, :]

    metadata = {"subject_ids": groups, "trial_ids": trial_ids}

    if verbose:
        print(f"EEG Loaded: {X.shape}, Range: [{X.min():.2f}, {X.max():.2f}]")

    return X, y, metadata, n_classes