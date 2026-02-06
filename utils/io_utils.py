# utils/io_utils.py
import os
import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# Add parent directory to path to find utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from loaders.dataset import load_eeg_dataset
    from utils.io_utils import save_experiment_results
    import config
except ImportError:
    print("WARNING: Imports failed. Check structure.")

class FeatureExtractor:
    def __init__(self, sfreq=512):
        self.sfreq = sfreq
        self.bands = [(4, 8), (8, 13), (13, 30), (30, 100)]

    def _compute_band_power(self, signal, low, high):
        freqs = np.fft.rfftfreq(signal.shape[-1], d=1/self.sfreq)
        fft_vals = np.abs(np.fft.rfft(signal, axis=-1)) ** 2
        idx = np.logical_and(freqs >= low, freqs <= high)
        if np.sum(idx) == 0: return np.zeros(signal.shape[:-1])
        return np.mean(fft_vals[..., idx], axis=-1)

    def fit(self, X, y=None): return self

    def transform(self, X):
        print(f"Extracting features for {X.shape[0]} samples...")
        mean = np.mean(X, axis=-1)
        std = np.std(X, axis=-1)
        ptp = np.ptp(X, axis=-1)
        features = [mean, std, ptp]
        for (low, high) in self.bands:
            features.append(self._compute_band_power(X, low, high))
        return np.concatenate(features, axis=1)

def run_ml_experiment(model_type="RF", mode="single", subject_id=1, verbose=True):
    print(f"\nLoading Data (Mode: {mode})...")
    X, y, metadata, n_classes = load_eeg_dataset(mode=mode, subject_id=subject_id)
    
    if X is None: return 0.0, 0.0

    groups = metadata["subject_ids"]

    if model_type == "SVM":
        clf = SVC(kernel='rbf', C=1.0, class_weight='balanced')
    elif model_type == "RF":
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced')
    elif model_type == "LDA":
        clf = LinearDiscriminantAnalysis()
    else:
        raise ValueError("Model must be SVM, RF, or LDA")

    pipeline = Pipeline([
        ('features', FeatureExtractor(sfreq=512)), 
        ('scaler', StandardScaler()),              
        ('classifier', clf)
    ])

    accuracies, f1_scores = [], []
    
    if mode == "loso":
        print(f"\n>>> Starting {model_type} Benchmark (LOSO)...")
        splitter = GroupKFold(n_splits=len(np.unique(groups)))
        cv_split = splitter.split(X, y, groups)
    else: 
        print(f"\n>>> Starting {model_type} Benchmark (Single Subject {subject_id})...")
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_split = splitter.split(X, y)

    for fold, (train_idx, val_idx) in enumerate(cv_split):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        accuracies.append(acc)
        f1_scores.append(f1)
        
        if verbose: print(f"[Fold {fold+1}] Acc={acc:.2%} | F1={f1:.4f}")

    mean_acc = np.mean(accuracies)
    mean_f1 = np.mean(f1_scores)
    
    print(f"FINAL RESULTS ({model_type}): Mean Acc: {mean_acc:.2%}")
    
    # Use centralized util
    save_experiment_results({
        "model": model_type,
        "mode": mode,
        "mean_accuracy": mean_acc,
        "mean_f1": mean_f1,
        "fold_accuracies": accuracies
    }, folder="../results")
    
    return mean_acc, mean_f1