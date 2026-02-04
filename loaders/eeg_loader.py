import os
import numpy as np
from scipy.signal import resample
from sklearn.preprocessing import LabelEncoder
import config

def load_eeg_dataset(name, target_fs=config.TARGET_FS, target_chans=config.TARGET_CHANS, target_samples=None):
    x_path = f"../data/X_{name}.npy"
    y_path = f"../data/y_{name}.npy"
    
    if not os.path.exists(x_path):
        print(f"File {x_path} not found.")
        return None, None, None

    X = np.load(x_path) 
    y = np.load(y_path)
    
    orig_fs = 5000 if "sync" in name.lower() else 512

    # 1. Resampling logic (Standardizing Time)
    if target_samples:
        X = resample(X, target_samples, axis=2)
    elif orig_fs != target_fs:
        new_t = int(X.shape[2] * (target_fs / orig_fs))
        X = resample(X, new_t, axis=2)

    # 2. Channel Padding to 64
    if X.shape[1] < target_chans:
        diff = target_chans - X.shape[1]
        X = np.pad(X, ((0, 0), (0, diff), (0, 0)), mode='constant')
    elif X.shape[1] > target_chans:
        X = X[:, :target_chans, :]
    
    le = LabelEncoder()
    return X, le.fit_transform(y), len(le.classes_)