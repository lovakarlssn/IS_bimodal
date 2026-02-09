import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
from sklearn.metrics import f1_score, cohen_kappa_score
from collections import defaultdict
import config
from augmentations.apply_aug import get_augmentation
from utils.get_model import get_model

class MultiModalDataset(Dataset):
    def __init__(self, modality, X, y, aug_name, aug_params, multiplier=1, p_aug=0.5):
        self.modality = modality
        self.X, self.y = X, y
        self.aug_name, self.aug_params = aug_name, aug_params
        self.multiplier, self.p_aug = multiplier, p_aug
        self.num_original = len(X)

    def __len__(self):
        return self.num_original * self.multiplier

    def __getitem__(self, idx):
        real_idx = idx % self.num_original
        # COPY is crucial here to prevent modifying the original dataset in memory
        x, label = self.X[real_idx].copy(), self.y[real_idx]
        
        # Apply Augmentation
        if self.aug_name not in ["Original", "None"] and np.random.rand() < self.p_aug:
            x_pair, _ = get_augmentation(self.modality, self.aug_name, x[np.newaxis, ...], [label], self.aug_params)
            # get_augmentation returns [Original, Augmented]; take the augmented one (index 1)
            x = x_pair[1]
        
        x_tensor = torch.tensor(x, dtype=torch.float32)
        
        # --- DIMENSION CHECKING ---
        if self.modality == "EEG":
            # EEGNet expects (1, Channels, Time)
            if x_tensor.ndim == 2:
                x_tensor = x_tensor.unsqueeze(0) 
                
        elif self.modality == "fMRI":
            # 3DCNN expects (1, Depth, Height, Width)
            if x_tensor.ndim == 3:
                x_tensor = x_tensor.unsqueeze(0)
            
        return x_tensor, torch.tensor(label, dtype=torch.long)

def run_experiment(modality, X, y, groups, n_classes, cv_splitter, aug_name, aug_params, hyperparams, arch_config, model_name, verbose=False):
    """
    Generalized CV experiment runner.
    Tracks Kappa and F1-Score for model selection.
    """
    best_val_metrics = []
    
    # Global history to store metrics across all folds for averaging
    # Structure: metric_name -> list of lists (folds x epochs)
    global_history = defaultdict(list)
    
    # --- 1. LOGGING SETUP ---
    timestamp = datetime.datetime.now().strftime('%d_%H%M')
    base_log_dir = os.path.join("runs", f"{modality}_{model_name}", aug_name, timestamp)
    
    if verbose:
        print(f"Logging metrics to: {base_log_dir}")

    # --- CROSS VALIDATION LOOP ---
    for fold, (t_idx, v_idx) in enumerate(cv_splitter.split(X, y, groups=groups)):
        if verbose: print(f"\n--- Fold {fold+1} ---")
        
        # Local history for this specific fold
        fold_history = defaultdict(list)
        
        # Create a separate writer for each fold
        writer = SummaryWriter(log_dir=os.path.join(base_log_dir, f"fold_{fold+1}"))

        # 2. Prepare Data Loaders
        train_ds = MultiModalDataset(
            modality, X[t_idx], y[t_idx], aug_name, aug_params, 
            multiplier=hyperparams.get("multiplier", 1)
        )
        train_loader = DataLoader(
            train_ds, batch_size=hyperparams.get("batch_size", 32), shuffle=True
        )
        
        # Validation Data
        X_v = torch.tensor(X[v_idx], dtype=torch.float32)
        if modality == "EEG":
            if X_v.ndim == 3: X_v = X_v.unsqueeze(1)
        elif modality == "fMRI":
            if X_v.ndim == 4: X_v = X_v.unsqueeze(1)
            
        val_loader = DataLoader(
            TensorDataset(X_v, torch.tensor(y[v_idx], dtype=torch.long)), 
            batch_size=hyperparams.get("batch_size", 32)
        )

        # 3. Model Setup
        model = get_model(modality, model_name, n_classes, X[0].shape, arch_config=arch_config).to(config.DEVICE)
        opt = optim.AdamW(model.parameters(), lr=hyperparams.get("lr", 1e-4), weight_decay=hyperparams.get("weight_decay", 1e-4))
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        fold_best_kappa = -1.0
        fold_best_acc = 0.0
        
        # --- EPOCH LOOP ---
        for epoch in range(hyperparams.get("epochs", 50)):
            
            # A. Training
            model.train()
            tr_loss_sum, tr_correct, tr_total = 0.0, 0, 0
            
            for xb, yb in train_loader:
                xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
                opt.zero_grad()
                logits = model(xb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()
                
                tr_loss_sum += loss.item() * xb.size(0)
                tr_correct += (logits.argmax(1) == yb).sum().item()
                tr_total += yb.size(0)
                
            train_loss = tr_loss_sum / tr_total
            train_acc = tr_correct / tr_total
            
            # B. Validation
            model.eval()
            val_loss_sum = 0.0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for xv, yv in val_loader:
                    xv, yv = xv.to(config.DEVICE), yv.to(config.DEVICE)
                    logits = model(xv)
                    loss = crit(logits, yv)
                    
                    val_loss_sum += loss.item() * xv.size(0)
                    preds = logits.argmax(1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(yv.cpu().numpy())
            
            val_loss = val_loss_sum / len(all_targets)
            
            # C. Robust Metrics (F1 & Kappa)
            val_acc = np.mean(np.array(all_preds) == np.array(all_targets))
            val_f1 = f1_score(all_targets, all_preds, average='macro')
            val_kappa = cohen_kappa_score(all_targets, all_preds)
            
            # D. Logging & Storage
            metrics_dict = {
                'Loss/Train': train_loss,
                'Loss/Val': val_loss,
                'Accuracy/Train': train_acc,
                'Accuracy/Val': val_acc,
                'Metrics/F1_Macro': val_f1,
                'Metrics/Kappa': val_kappa
            }
            
            for key, val in metrics_dict.items():
                writer.add_scalar(key, val, epoch)
                fold_history[key].append(val)
            
            # E. Track Best (Based on Kappa)
            if val_kappa > fold_best_kappa:
                fold_best_kappa = val_kappa
                fold_best_acc = val_acc 
        
        # Append this fold's history to global history
        for key, val_list in fold_history.items():
            global_history[key].append(val_list)

        best_val_metrics.append(fold_best_acc)
        if verbose: 
            print(f"    Best Val Kappa: {fold_best_kappa:.4f} (Acc: {fold_best_acc:.2%})")

        writer.close()

    # --- 4. MEAN SUMMARY LOGGING ---
    print("Generating Mean Summary Log...")
    mean_writer = SummaryWriter(log_dir=os.path.join(base_log_dir, "mean"))
    
    n_epochs = hyperparams.get("epochs", 50)
    
    for epoch in range(n_epochs):
        for metric_name, folds_data in global_history.items():
            # Collect values for this specific epoch across all folds
            # Check length to handle potential early stopping differences
            epoch_values = [fold[epoch] for fold in folds_data if len(fold) > epoch]
            
            if epoch_values:
                mean_val = np.mean(epoch_values)
                mean_writer.add_scalar(metric_name, mean_val, epoch)
                
    mean_writer.close()
    
    return np.mean(best_val_metrics)