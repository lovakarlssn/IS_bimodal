import argparse
import config
from loaders.eeg_loader import load_eeg_dataset
from augmentations.eeg_aug import get_augmentation
from experiments.cross_val import run_cv_experiment
from utils.processing import get_groups
import numpy as np

def run_experiment(exp_name):
    X, y, n_classes = load_eeg_dataset("Original")
    if X is None: return
    
    params = config.AUG_PARAMS
    # Apply Augmentation
    X_aug, y_aug = get_augmentation(exp_name, X, y, params)
    groups = get_groups(len(y_aug))
    
    # Run Engine
    history, mean_acc = run_cv_experiment(X_aug, y_aug, groups, n_classes, exp_name)
    print(f"FINAL RESULT {exp_name}: {mean_acc:.2%}")
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="Original")
    args = parser.parse_args()
    run_pipeline(args.exp)