import argparse
import copy
import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

import config
from loaders.eeg_loader import load_eeg_dataset
from experiments.cross_val import run_cv_experiment
from utils.visualization import plot_history


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="Original")
    parser.add_argument("--mode", type=str, default="single")
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    
    args = parser.parse_args()

    cli_hyperparams = {
        "batch_size": config.BATCH_SIZE,
        "lr": args.lr,
        "epochs": args.epochs,
        "weight_decay": config.WEIGHT_DECAY
    }

    exps_to_run = config.EXPERIMENTS if args.exp.lower() == "all" else [args.exp]
    results_summary = {}

    for current_exp in exps_to_run:
        hist, acc = run_experiment(
            current_exp, 
            mode=args.mode, 
            subject_id=args.sub, 
            hyperparams=cli_hyperparams
        )
        results_summary[current_exp] = acc

    print("\n" + "="*30)
    print("ALL EXPERIMENTS COMPLETE")
    for name, score in sorted(results_summary.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:15}: {score:.2%}")
    print("="*30)