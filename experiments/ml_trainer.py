# experiments/ml_trainer.py
import os
import numpy as np
import json
import datetime
import sys

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

try:
    from loaders.eeg_loader import load_eeg_dataset
    from augmentations.apply_aug import get_augmentation
    import config
except ImportError as e:
    print(f"WARNING: Import failed - {e}")

class FeatureExtractor:
    """
    Extracts efficient features from raw EEG (Channels x Time).
    """
    def __init__(self, sfreq=512):
        self.sfreq = sfreq
        # Band ranges: Theta (4-8), Alpha (8-13), Beta (13-30), Gamma (30-100)
        self.bands = [(4, 8), (8, 13), (13, 30), (30, 100)]

    def _compute_band_power(self, signal, low, high):
        # Simple FFT-based power
        freqs = np.fft.rfftfreq(signal.shape[-1], d=1/self.sfreq)
        fft_vals = np.abs(np.fft.rfft(signal, axis=-1)) ** 2
        idx = np.logical_and(freqs >= low, freqs <= high)
        
        if np.sum(idx) == 0:
            return np.zeros(signal.shape[:-1])
            
        return np.mean(fft_vals[..., idx], axis=-1)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X shape: (N_samples, N_channels, N_time)
        # 1. Time Domain Stats
        mean = np.mean(X, axis=-1)
        std = np.std(X, axis=-1)
        ptp = np.ptp(X, axis=-1) # Peak-to-peak amplitude
        
        features = [mean, std, ptp]
        
        # 2. Frequency Domain Power
        for (low, high) in self.bands:
            band_pow = self._compute_band_power(X, low, high)
            features.append(band_pow)
            
        # Concatenate features along channel axis (flattening channels)
        X_feat = np.concatenate(features, axis=1)
        return X_feat

def run_ml_experiment(model_type="RF", mode="single", subject_id=1, aug_name="Original", verbose=True):
    """
    Main function to run ML baselines with Augmentations.
    """
    # 1. Load Data
    if verbose: print(f"\nLoading Data (Mode: {mode})...")
    X, y, metadata, n_classes = load_eeg_dataset(mode=mode, subject_id=subject_id)
    
    if X is None:
        print("Error: Dataset not found.")
        return

    groups = metadata["subject_ids"]

    # 2. Define Model Classifier
    if model_type == "SVM":
        clf = SVC(kernel='rbf', C=1.0, class_weight='balanced')
    elif model_type == "RF":
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced')
    elif model_type == "LDA":
        clf = LinearDiscriminantAnalysis()
    else:
        raise ValueError("Model must be SVM, RF, or LDA")

    # Note: FeatureExtractor is NOT in the pipeline because we need to augment Raw data FIRST
    # We will manually extract features after augmentation loop
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),              
        ('classifier', clf)
    ])
    
    extractor = FeatureExtractor(sfreq=config.TARGET_FS)

    # 3. Define Cross-Validation Strategy
    accuracies = []
    f1_scores = []
    
    if mode == "loso":
        if verbose: print(f"\n>>> Starting {model_type} with {aug_name} (LOSO)...")
        splitter = GroupKFold(n_splits=len(np.unique(groups)))
        cv_split = splitter.split(X, y, groups)
    else:
        if verbose: print(f"\n>>> Starting {model_type} with {aug_name} (Single Subject {subject_id})...")
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_split = splitter.split(X, y)

    # 4. Training Loop
    for fold, (train_idx, val_idx) in enumerate(cv_split):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # --- A. Apply Augmentation (Only to Training Data) ---
        if aug_name != "Original":
            # get_augmentation returns (Original + Augmented)
            X_train, y_train = get_augmentation(
                modality="EEG", 
                name=aug_name, 
                X=X_train, 
                y=y_train, 
                params=config.AUG_PARAMS
            )
        
        # --- B. Extract Features ---
        # We transform Raw Time-Series -> Tabular Features
        X_train_feat = extractor.transform(X_train)
        X_val_feat = extractor.transform(X_val)
        
        # --- C. Train & Predict ---
        pipeline.fit(X_train_feat, y_train)
        y_pred = pipeline.predict(X_val_feat)
        
        # Metrics
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        
        accuracies.append(acc)
        f1_scores.append(f1)
        
        if verbose:
            label = f"Subject {groups[val_idx][0]}" if mode == "loso" else f"Fold {fold+1}"
            print(f"[{label}] Acc={acc:.2%} | F1={f1:.4f} | Train Size={len(y_train)}")

    # 5. Final Results
    mean_acc = np.mean(accuracies)
    mean_f1 = np.mean(f1_scores)
    
    print("\n" + "="*30)
    print(f"RESULTS ({model_type} + {aug_name})")
    print(f"Mean Acc: {mean_acc:.2%} (+/- {np.std(accuracies):.2%})")
    print("="*30)
    
    save_results(model_type, aug_name, mean_acc, mean_f1, accuracies, mode)

def save_results(model, aug, acc, f1, fold_accs, mode):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    res_dir = config.RESULTS_DIR
    os.makedirs(res_dir, exist_ok=True)
    
    data = {
        "model": model,
        "augmentation": aug,
        "mode": mode,
        "mean_accuracy": acc,
        "mean_f1": f1,
        "fold_accuracies": fold_accs
    }
    
    # Simple naming to avoid overwriting
    filename = f"ml_{model}_{aug}_{mode}_{timestamp}.json"
    with open(os.path.join(res_dir, filename), "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    # Example usage
    run_ml_experiment(model_type="SVM", mode="single", subject_id=1, aug_name="ChannelsDropout", verbose=True)