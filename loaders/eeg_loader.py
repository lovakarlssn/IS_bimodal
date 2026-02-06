import os
import numpy as np
import config
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_eeg_dataset(mode="loso", subject_id=None, verbose=False):
    base_path = "../data"
    x_path = os.path.join(base_path, "X_EEG.npy")
    y_path = os.path.join(base_path, "y_EEG.npy")
    g_path = os.path.join(base_path, "groups_EEG.npy")

    if not os.path.exists(x_path):
        return None, None, None, 0

    X_all = np.load(x_path)
    y_all = np.load(y_path)
    groups_all = np.load(g_path)
    trial_ids_all = np.arange(len(X_all))

    # --- Z-Score Normalization ---
    means = np.mean(X_all, axis=-1, keepdims=True)
    stds = np.std(X_all, axis=-1, keepdims=True) + 1e-8
    X_all = (X_all - means) / stds

    if mode == "single":
        indices = np.where(groups_all == int(subject_id))[0]
        if len(indices) == 0: return None, None, None, 0
            
        X = X_all[indices]
        y = y_all[indices]
        groups = groups_all[indices]
        trial_ids = trial_ids_all[indices]
    else:
        X, y, groups = X_all, y_all, groups_all
        trial_ids = trial_ids_all

    le = LabelEncoder()
    y = le.fit_transform(y)
    n_classes = len(le.classes_)

    target_chans = config.TARGET_CHANS
    if X.shape[1] < target_chans:
        X = np.pad(X, ((0, 0), (0, target_chans - X.shape[1]), (0, 0)), mode='constant')
    elif X.shape[1] > target_chans:
        X = X[:, :target_chans, :]

    metadata = {
        "subject_ids": groups,
        "trial_ids": trial_ids
    }

    if verbose:
        print(f"Data Loaded & Scaled. Range: [{X.min():.2f}, {X.max():.2f}]")

    return X, y, metadata, n_classes