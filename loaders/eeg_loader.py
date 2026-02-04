import os
import numpy as np
from scipy.signal import resample
from sklearn.preprocessing import LabelEncoder
from config import TARGET_FS, TARGET_CHANS

def load_eeg_dataset(name, target_fs=TARGET_FS, target_chans=TARGET_CHANS, target_samples=None):
    x_path = os.path.join("./data", f"X_{name}.npy")
    y_path = os.path.join("./data", f"y_{name}.npy")
    
    if not os.path.exists(x_path):
        return None, None, None

    X = np.load(x_path) 
    y = np.load(y_path)
    
    # Rate Detection
    orig_fs = 5000 if "sync" in name.lower() else 512

    # Resampling Logic
    current_time_points = X.shape[2]
    if target_samples is not None:
        if current_time_points != target_samples:
            X = resample(X, target_samples, axis=2)
    elif orig_fs != target_fs:
        new_time_points = int(current_time_points * (target_fs / orig_fs))
        X = resample(X, new_time_points, axis=2)
    
    # Channel Padding
    if X.shape[1] < target_chans:
        diff = target_chans - X.shape[1]
        X = np.pad(X, ((0, 0), (0, diff), (0, 0)), mode='constant', constant_values=0)
    elif X.shape[1] > target_chans:
        X = X[:, :target_chans, :]
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, y_enc, len(le.classes_)