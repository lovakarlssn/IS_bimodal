# experiments/cross_val.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
import json
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmentations.eeg_aug import get_augmentation
from models.transformer import SpectroTemporalTransformer
from models.eegnets import *
import config

class EEGStochasticDataset(Dataset):
    def __init__(self, X, y, exp_name, aug_params, multiplier=4, p_aug=0.5):
        self.X = X
        self.y = y
        self.exp_name = exp_name
        self.aug_params = aug_params
        self.multiplier = multiplier
        self.p_aug = p_aug
        self.num_original = len(X)

    def __len__(self):
        return self.num_original * self.multiplier

    def __getitem__(self, idx):
        real_idx = idx % self.num_original
        x = self.X[real_idx].copy()
        label = self.y[real_idx]

        if self.exp_name not in ["Original", "Non_Augmented"]:
            if np.random.rand() < self.p_aug:
                x_pair, _ = get_augmentation(self.exp_name, x[np.newaxis, ...], [label], self.aug_params)
                x = x_pair[1] 

        x_tensor = torch.tensor(x, dtype=torch.float32)
        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(0)
            
        return x_tensor, torch.tensor(label, dtype=torch.long)

def run_cv_experiment(X, y, groups, n_classes, cv_splitter, exp_name, aug_params, hyperparams, model_name="EEGNet", verbose=False):
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    best_accs = []
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create descriptive folder name
    aug_str = str(aug_params).replace("{", "").replace("}", "").replace("'", "").replace(":", "-").replace(" ", "")[:30]
    folder_name = f"{model_name}_{exp_name}_{timestamp}_{aug_str}"
    log_dir = os.path.join(project_root, "runs", folder_name)
    writer = SummaryWriter(log_dir=log_dir)
    
    BS = hyperparams.get("batch_size", 32)
    LR = hyperparams.get("lr", 1e-4)
    EPOCHS = hyperparams.get("epochs", 50)
    W_DECAY = hyperparams.get("weight_decay", 1e-4)
    DATA_MULTIPLIER = hyperparams.get("data_multiplier", 4) 
    
    for fold, (t_idx, v_idx) in enumerate(cv_splitter.split(X, y, groups=groups)):
        if verbose: print(f"\n>>> Starting Fold {fold+1}")

        X_tr_raw, X_val_raw = X[t_idx], X[v_idx]
        y_tr_raw, y_val_raw = y[t_idx], y[v_idx]

        train_loader = DataLoader(
            EEGStochasticDataset(X_tr_raw, y_tr_raw, exp_name, aug_params, multiplier=DATA_MULTIPLIER),
            batch_size=BS, shuffle=True
        )

        X_val_t = torch.tensor(X_val_raw, dtype=torch.float32).unsqueeze(1)
        y_val_t = torch.tensor(y_val_raw, dtype=torch.long)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BS, shuffle=False)

        n_chans, n_time = X_tr_raw.shape[-2], X_tr_raw.shape[-1]
        
        # Model Selection
        if model_name == "EEGNet":
            model = EEGNet(nb_classes=n_classes, Chans=n_chans, Samples=n_time).to(config.DEVICE)
        elif model_name == "DeepConvNet":
            model = DeepConvNet(nb_classes=n_classes, Chans=n_chans, Samples=n_time).to(config.DEVICE)
        elif model_name == "ShallowConvNet":
            model = ShallowConvNet(nb_classes=n_classes, Chans=n_chans, Samples=n_time).to(config.DEVICE)
        elif model_name == "CustomEEGNet":
            model = CustomEEGNet(nb_classes=n_classes, Chans=n_chans, Samples=n_time).to(config.DEVICE)
        elif model_name == "SpectroTemporalTransformer":
            model = SpectroTemporalTransformer(
                nb_classes=n_classes, 
                Chans=n_chans, 
                Samples=n_time, 
                sfreq=config.TARGET_FS
            ).to(config.DEVICE)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=W_DECAY)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        f_best_acc = 0.0

        for epoch in range(EPOCHS):
            model.train()
            tr_a, tr_c = 0.0, 0
            
            for xb, yb in train_loader:
                xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
                opt.zero_grad()
                out = model(xb)
                loss = crit(out, yb)
                loss.backward()
                opt.step()
                tr_a += (out.argmax(1) == yb).sum().item(); tr_c += yb.size(0)
            
            # Validation
            model.eval()
            v_a, v_c = 0.0, 0
            with torch.no_grad():
                for xb_v, yb_v in val_loader:
                    xb_v, yb_v = xb_v.to(config.DEVICE), yb_v.to(config.DEVICE)
                    out_v = model(xb_v)
                    v_a += (out_v.argmax(1) == yb_v).sum().item(); v_c += yb_v.size(0)

            cur_acc = v_a/v_c
            if cur_acc > f_best_acc: f_best_acc = cur_acc
            
            # Record per-epoch metric
            if fold == 0: # Only record history detailed for first fold to save memory in return
                history['train_acc'].append(tr_a/tr_c)
                history['val_acc'].append(cur_acc)

        best_accs.append(f_best_acc)
        
        if verbose: print(f" -> Fold {fold+1} Finished. Best Val Acc: {f_best_acc:.2%}")

    # Metadata saving
    with open(os.path.join(log_dir, "metadata.json"), "w") as f:
        json.dump({"aug_params": aug_params, "hyperparams": hyperparams, "best_accs": best_accs}, f, indent=4)

    writer.close()
    return history, np.mean(best_accs)

