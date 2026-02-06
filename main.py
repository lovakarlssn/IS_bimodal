import argparse
import copy
import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.model_selection import StratifiedKFold, GroupKFold
from utils.io_utils import plot_history
from loaders.eeg_loader import load_eeg_dataset
from experiments.cross_val import run_cv_experiment
from experiments.ml_trainer import run_ml_experiment
from utils.io_utils import save_experiment_results
import config 
import json
import datetime
import traceback

def run_experiment(exp_name, mode="single", subject_id=1, hyperparams=None, aug_params=None):
    if hyperparams is None:
        hyperparams = {
            "batch_size": config.BATCH_SIZE,
            "lr": config.LEARNING_RATE,
            "epochs": config.EPOCHS,
            "weight_decay": config.WEIGHT_DECAY
        }
        
    if aug_params is None:
        aug_params = copy.deepcopy(config.AUG_PARAMS)

    print(f"\n--- Starting: {exp_name} | Mode: {mode} | Sub: {subject_id} ---")

    # X, y, metadata (contains subject_ids and trial_ids), n_classes
    X, y, metadata, n_classes = load_eeg_dataset(mode=mode, subject_id=subject_id)
    
    if X is None: 
        print("Experiment Aborted: No data found.")
        return None, 0.0

    # CROSS-VALIDATION STRATEGY
    if mode == "single":
        # Within-subject: use trial_ids to ensure index isolation
        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        groups = metadata['trial_ids'] 
    else:
        # LOSO: use subject_ids to ensure subject isolation
        cv_splitter = LeaveOneGroupOut()
        groups = metadata['subject_ids']

    history, mean_acc = run_cv_experiment(
        X=X, 
        y=y, 
        groups=groups, 
        n_classes=n_classes, 
        cv_splitter=cv_splitter, 
        exp_name=exp_name,         
        aug_params=aug_params,
        hyperparams=hyperparams,
        verbose=True
    )
    
    print(f"FINAL RESULT [{exp_name}]: {mean_acc:.2%}")
    return history, mean_acc


def get_aug_config(exp_name, lookup_dict):
    if exp_name == "Original": return None
    
    # Map friendly names to config keys
    key_map = {
        "ChannelsDropout": "channels_dropout",
        "FreqSurrogate": "freq_surrogate",
        "SmoothTimeMask": "smooth_time_mask",
        "TimeReverse": "time_reverse",
        "SignFlip": "sign_flip"
    }
    
    key = key_map.get(exp_name)
    if not key or key not in lookup_dict:
        raise ValueError(f"Augmentation '{exp_name}' not found in AUG_PARAMS")
        
    return {key: lookup_dict[key]}

# ==============================================================================
#   MAIN RUNNER
# ==============================================================================
def run_master_experiments(
    exp_mode=config.EXP_MODE,
    data_mode=config.DATA_MODE,
    subject_id=config.SUBJECT_ID,
    dl_models=config.DL_MODELS,
    ml_models=config.ML_MODELS,
    experiments=config.EXPERIMENTS,   
    hyperparams=config.HYPERPARAMS,   
    aug_params=config.AUG_PARAMS,     
    filename_prefix=None
):
    
    # 1. Setup Filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    prefix = f"{filename_prefix}_" if filename_prefix else ""
    dl_res_file = f"{prefix}DL_Results_{timestamp}.json"
    ml_res_file = f"{prefix}ML_Results_{timestamp}.json"

    # 2. Load Data
    print(f"\n[MASTER] Loading Data ({data_mode})...")
    X, y, metadata, n_classes = load_eeg_dataset(mode=data_mode, subject_id=subject_id)
    
    if X is None:
        print("Error: Data not found.")
        return

    # 3. Setup Splitter (FIXED)
    if data_mode == "loso":
        groups = metadata["subject_ids"]
        # Use LeaveOneGroupOut for strict Subject-independent testing
        cv_splitter = LeaveOneGroupOut()
    else:
        groups = metadata["trial_ids"]
        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ---------------------------------------------------------
    # PART A: MACHINE LEARNING (SVM, RF)
    # ---------------------------------------------------------
    if exp_mode in ["ML", "BOTH"]:
        print(f"\n=== ML EXPERIMENTS -> {ml_res_file} ===")
        ml_results = []
        
        for model_name in ml_models:
            run_id = f"ML_{model_name}_{data_mode}"
            print(f">>> RUNNING: {run_id}")
            
            try:
                # Note: You need to update run_ml_experiment in ml_trainer.py 
                # to accept the 'cv_splitter' object if you want it to use this exact split,
                # otherwise it likely recreates its own internally. 
                # For consistency, I recommend passing it:
                acc, f1 = run_ml_experiment(
                    model_type=model_name, 
                    mode=data_mode, 
                    subject_id=subject_id, 
                    verbose=False
                    # Ideally pass cv_splitter here too
                )
                
                res = {
                    "run_id": run_id, "type": "ML", "model": model_name,
                    "aug": "Original", "accuracy": acc, "f1": f1
                }
                ml_results.append(res)
                save_experiment_results(ml_results, filename=ml_res_file, folder=config.RESULTS_DIR)
                print(f"   [DONE] Acc: {acc:.2%}")
            except Exception:
                print(f"!!! CRASH in {run_id}")
                traceback.print_exc()

    # ---------------------------------------------------------
    # PART B: DEEP LEARNING (EEGNet, Transformer)
    # ---------------------------------------------------------
    if exp_mode in ["DL", "BOTH"]:
        print(f"\n=== DL EXPERIMENTS -> {dl_res_file} ===")
        dl_results = []

        for model_name in dl_models:
            for exp_name in experiments:
                
                run_id = f"DL_{model_name}_{exp_name}_LR{hyperparams['lr']}"
                print(f">>> RUNNING: {run_id}")
                
                try:
                    current_aug_config = get_aug_config(exp_name, aug_params)
                    
                    history, mean_acc = run_cv_experiment(
                        X, y, groups, n_classes, cv_splitter,
                        exp_name=exp_name,
                        aug_params=current_aug_config,
                        hyperparams=hyperparams, 
                        model_name=model_name,
                        verbose=False
                    )
                    
                    res = {
                        "run_id": run_id, "type": "DL", "model": model_name,
                        "aug": exp_name, "accuracy": mean_acc,
                        **hyperparams
                    }
                    dl_results.append(res)
                    save_experiment_results(dl_results, filename=dl_res_file, folder=config.RESULTS_DIR)
                    print(f"   [DONE] Acc: {mean_acc:.2%}")

                except Exception:
                    print(f"!!! CRASH in {run_id}")
                    traceback.print_exc()

    print(f"\n[MASTER] Finished. Results in {config.RESULTS_DIR}")

if __name__ == "__main__":
    run_master_experiments()