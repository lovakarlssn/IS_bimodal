# experiments/trainer.py
import sys
import os
import numpy as np

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from loaders.eeg_loader import load_eeg_dataset
from loaders.fmri_loader import load_fmri_dataset
from experiments.cross_val import run_experiment
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, GroupKFold

def run_training_session(
    modality=None,
    model_name=None,
    data_mode=None,
    subject_id=None,
    aug_name=None,
    aug_params=None,
    hyperparams=None,
    arch_config=None,
    verbose=True
):
    """
    Master function to run a training session.
    Prioritizes passed arguments; falls back to config.py if None.
    """
    
    # --- 1. CONFIG FALLBACK LOGIC ---
    modality = modality if modality is not None else config.MODALITY
    data_mode = data_mode if data_mode is not None else config.DATA_MODE
    subject_id = subject_id if subject_id is not None else config.SUBJECT_ID
    
    # Default to first model in config list if not specified
    if model_name is None:
        model_name = config.DL_MODELS[0] if config.DL_MODELS else "EEGNet"
        
    # Default to "Original" if not specified
    aug_name = aug_name if aug_name is not None else "Original"
    aug_params = aug_params if aug_params is not None else config.AUG_PARAMS
    
    hyperparams = hyperparams if hyperparams is not None else config.HYPERPARAMS
    arch_config = arch_config if arch_config is not None else {}

    if verbose:
        print("="*40)
        print(f"STARTING RUN: {modality} | {model_name} | {data_mode}")
        if data_mode == "single": print(f"Subject: {subject_id}")
        print(f"Augmentation: {aug_name}")
        print(f"Hyperparams: {hyperparams}")
        if arch_config: print(f"Arch Config: {arch_config}")
        print("="*40)

    # --- 2. DATA LOADING ---
    if modality == "EEG":
        X, y, meta, n_classes = load_eeg_dataset(mode=data_mode, subject_id=subject_id, verbose=verbose)
    elif modality == "fMRI":
        X, y, meta, n_classes = load_fmri_dataset(mode=data_mode, subject_id=subject_id, verbose=verbose)
    else:
        raise ValueError(f"Unknown modality: {modality}")

    if X is None:
        print("[Error] Data loading failed.")
        return 0.0

    groups = meta["subject_ids"]

    # --- 3. CROSS-VALIDATION STRATEGY ---
    if data_mode == "loso":
        # Leave-One-Subject-Out
        if verbose: print("Strategy: LOSO (GroupKFold/LOGO)")
        # GroupKFold is generally safer than LeaveOneGroupOut if groups are fragmented, 
        # but LeaveOneGroupOut is standard for LOSO.
        cv_splitter = LeaveOneGroupOut()
    else:
        # Single Subject or simple mixed batch
        if verbose: print("Strategy: Stratified K-Fold")
        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # StratifiedKFold doesn't use groups, but we pass them to run_experiment for API consistency

    # --- 4. EXECUTE RUN ---
    avg_acc = run_experiment(
        modality=modality,
        X=X,
        y=y,
        groups=groups,
        n_classes=n_classes,
        cv_splitter=cv_splitter,
        aug_name=aug_name,
        aug_params=aug_params,
        hyperparams=hyperparams,
        arch_config=arch_config,
        model_name=model_name,
        verbose=verbose
    )

    return avg_acc


