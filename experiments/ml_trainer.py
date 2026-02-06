# experiments/ml_trainer.py
import os
import numpy as np
import json
import datetime
import sys
sys.path.append('..')
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# Attempt to import your loader. 
try:
    from loaders.eeg_loader import load_eeg_dataset
    import config
except ImportError:
    print("WARNING: Could not import 'load_eeg_dataset'. Please ensure loaders/dataset.py exists.")

class FeatureExtractor:
    """
    extracts efficient features from raw EEG (Channels x Time) 
    to make ML models (SVM/RF) viable.
    
    Extracts:
    1. Time-domain: Mean, Std, Peak-to-Peak (PtP)
    2. Freq-domain: Power in Theta, Alpha, Beta, Gamma bands
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
        
        # Handle empty bands
        if np.sum(idx) == 0:
            return np.zeros(signal.shape[:-1])
            
        return np.mean(fft_vals[..., idx], axis=-1)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X shape: (N_samples, N_channels, N_time)
        print(f"Extracting features for {X.shape[0]} samples...")
        
        # 1. Time Domain Stats
        mean = np.mean(X, axis=-1)
        std = np.std(X, axis=-1)
        ptp = np.ptp(X, axis=-1) # Peak-to-peak amplitude
        
        features = [mean, std, ptp]
        
        # 2. Frequency Domain Power
        for (low, high) in self.bands:
            band_pow = self._compute_band_power(X, low, high)
            features.append(band_pow)
            
       
        X_feat = np.concatenate(features, axis=1)
        
        print(f" -> Feature shape: {X_feat.shape}")
        return X_feat

def run_ml_experiment(model_type="RF", mode="single", subject_id=1, verbose=True):
    """
    Main function to run ML baselines using LOSO or Single Subject CV.
    """
    # 1. Load Data
    print(f"\nLoading Data (Mode: {mode})...")
    # Pass mode and subject_id correctly to the loader
    X, y, metadata, n_classes = load_eeg_dataset(mode=mode, subject_id=subject_id)
    
    if X is None:
        print("Error: Dataset not found.")
        return

    groups = metadata["subject_ids"]

    # 2. Define Model Pipeline
    if model_type == "SVM":
        clf = SVC(kernel='rbf', C=1.0, class_weight='balanced')
    elif model_type == "RF":
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced')
    elif model_type == "LDA":
        clf = LinearDiscriminantAnalysis()
    else:
        raise ValueError("Model must be SVM, RF, or LDA")

    pipeline = Pipeline([
        ('features', FeatureExtractor(sfreq=512)), # 512 Hz as requested
        ('scaler', StandardScaler()),              
        ('classifier', clf)
    ])

    # 3. Define Cross-Validation Strategy
    accuracies = []
    f1_scores = []
    
    if mode == "loso":
        print(f"\n>>> Starting {model_type} Benchmark (LOSO)...")
        # GroupKFold ensures we test on unseen subjects
        splitter = GroupKFold(n_splits=len(np.unique(groups)))
        cv_split = splitter.split(X, y, groups)
        
    else: # mode == "single"
        print(f"\n>>> Starting {model_type} Benchmark (Single Subject {subject_id})...")
        # StratifiedKFold is better for single subject class balance
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_split = splitter.split(X, y) # Groups ignored here

    # 4. Training Loop
    for fold, (train_idx, val_idx) in enumerate(cv_split):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Log info
        if mode == "loso":
            test_subj = groups[val_idx][0]
            info_str = f"Subject {test_subj}"
        else:
            info_str = f"Fold {fold+1}"
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_val)
        
        # Metrics
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        
        accuracies.append(acc)
        f1_scores.append(f1)
        
        if verbose:
            print(f"[{info_str}] Acc={acc:.2%} | F1={f1:.4f}")

    # 5. Final Results
    mean_acc = np.mean(accuracies)
    mean_f1 = np.mean(f1_scores)
    
    print("\n" + "="*30)
    print(f"FINAL RESULTS ({model_type} - {mode})")
    print(f"Mean Accuracy: {mean_acc:.2%} (+/- {np.std(accuracies):.2%})")
    print(f"Mean Macro-F1: {mean_f1:.4f}")
    print("="*30)
    
    # Optional: Save results
    save_results(model_type, mean_acc, mean_f1, accuracies, mode)

def save_results(model, acc, f1, fold_accs, mode):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    res_dir = "../results"
    os.makedirs(res_dir, exist_ok=True)
    
    data = {
        "model": model,
        "mode": mode,
        "mean_accuracy": acc,
        "mean_f1": f1,
        "fold_accuracies": fold_accs
    }
    
    with open(f"{res_dir}/ml_{model}_{mode}_{timestamp}.json", "w") as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to {res_dir}")

if __name__ == "__main__":
    # Example 1: Run SVM in LOSO mode (all subjects)
    run_ml_experiment(model_type="SVM", mode="single", subject_id=1)
    
    # Example 2: Run RF on Single Subject (e.g., ID 1)
    # run_ml_experiment(model_type="RF", mode="single", subject_id=1)