import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from loaders.eeg_loader import load_eeg_dataset
from utils import get_groups, prepare_tensors
from models.eegnets import CompactEEGNet
from config import *
from torch import nn, optim
from tqdm import tqdm
def train_experiment(name):
    # Load and potentially augment
    X_raw, y_raw, n_classes = load_eeg_dataset(name)
    # (Optional: call augment functions here based on 'name')
    
    N, n_chans, n_time = X_raw.shape
    groups = get_groups(name, len(y_raw))
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize Main History (To store ALL folds)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    fold_accuracies = []

    print("--- 5-fold Cross-Validation ---")

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_raw, y_raw, groups=groups)):
        # ... [Data Splitting & Scaling Code is Fine] ...
        X_tr_raw, y_tr_raw = X_raw[train_idx], y_raw[train_idx]
        X_val_raw, y_val_raw = X_raw[val_idx], y_raw[val_idx]
        X_tr, y_tr, scaler = prepare_tensors(X_tr_raw, y_tr_raw, fit_scaler=True)
        X_val, y_val, _    = prepare_tensors(X_val_raw, y_val_raw, scaler=scaler)
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
        
        model = CompactEEGNet(nb_classes=n_classes, input_time=n_time, input_chans=n_chans).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_fold_acc = 0.0
        patience, count = 5, 0
        
        # Temporary lists for THIS fold
        fold_t_l, fold_v_l, fold_t_a, fold_v_a = [], [], [], []
        
        with tqdm(total=EPOCHS, desc=f"Fold {fold+1}", leave=False) as pbar:
            for epoch in range(EPOCHS):
                # --- TRAIN LOOP ---
                model.train()
                t_loss, t_corr, t_tot = 0, 0, 0
                for x_b, y_b in train_loader:
                    optimizer.zero_grad()
                    out = model(x_b)
                    loss = criterion(out, y_b)
                    loss.backward()
                    optimizer.step()
                    
                    t_loss += loss.item()
                    t_corr += (out.argmax(1) == y_b).sum().item()
                    t_tot += y_b.size(0)
                
                # --- VAL LOOP ---
                model.eval()
                v_loss, v_corr, v_tot = 0, 0, 0
                with torch.no_grad():
                    for x_b, y_b in val_loader:
                        out = model(x_b)
                        v_loss += criterion(out, y_b).item()
                        v_corr += (out.argmax(1) == y_b).sum().item()
                        v_tot += y_b.size(0)
                
                # --- CALCULATE METRICS ---
                t_a = t_corr/t_tot if t_tot > 0 else 0
                v_a = v_corr/v_tot if v_tot > 0 else 0
                avg_t_loss = t_loss/len(train_loader)
                avg_v_loss = v_loss/len(val_loader)
                
                # --- LOGGING (Missing part added here) ---
                fold_t_l.append(avg_t_loss)
                fold_v_l.append(avg_v_loss)
                fold_t_a.append(t_a)
                fold_v_a.append(v_a)
                
                pbar.set_postfix({'Val': f"{v_a:.1%}"})
                pbar.update(1)
                
                # --- EARLY STOPPING ---
                if v_a > best_fold_acc:
                    best_fold_acc = v_a
                    count = 0
                else:
                    count += 1
                if count >= patience: break
        
        # --- SAVE FOLD HISTORY TO MAIN DICT ---
        history['train_loss'].append(fold_t_l)
        history['val_loss'].append(fold_v_l)
        history['train_acc'].append(fold_t_a)
        history['val_acc'].append(fold_v_a)
        fold_accuracies.append(best_fold_acc)

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"   --> CV Result: {mean_acc:.2%} (+/- {std_acc:.2%})")
    
    return mean_acc, std_acc, history