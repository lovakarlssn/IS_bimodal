import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import datetime
import json
from augmentations.eeg_aug import get_augmentation
from models.eegnets import *
import config

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

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
        # 1. Map virtual index back to the real data index
        real_idx = idx % self.num_original
        
        # Grab original sample: shape (Channels, Time)
        x = self.X[real_idx].copy()
        label = self.y[real_idx]

        # 2. Apply Augmentation Logic
        if self.exp_name not in ["Original", "Non_Augmented"]:
            if np.random.rand() < self.p_aug:
                # get_augmentation expects a batch (N, C, T)
                # We wrap x in [np.newaxis] to make it (1, 64, 1537)
                # It returns: (np.concatenate([orig, aug]), np.concatenate([y, y]))
                x_pair, _ = get_augmentation(self.exp_name, x[np.newaxis, ...], [label], self.aug_params)
                
                # The first element [0] is the original, the second [1] is the augmented
                x = x_pair[1] 

        # 3. Ensure final shape is (1, Channels, Time) for the Conv2d layer
        # If x is (64, 1537), unsqueeze(0) makes it (1, 64, 1537)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(0)
            
        y_tensor = torch.tensor(label, dtype=torch.long)
        
        return x_tensor, y_tensor

def run_cv_experiment(X, y, groups, n_classes, cv_splitter, exp_name, aug_params, hyperparams, model_name="EEGNet", verbose=False):
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    best_accs = []
    
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        project_root = os.getcwd() 
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    aug_id = str(aug_params).replace("{", "").replace("}", "").replace("'", "").replace(":", "-").replace(" ", "")[:40]
    folder_name = f"{model_name}_{exp_name}_{timestamp}_{aug_id}"
    log_dir = os.path.join(project_root, "runs", folder_name)
    writer = SummaryWriter(log_dir=log_dir)
    
    BS = hyperparams.get("batch_size", 32)
    LR = hyperparams.get("lr", 1e-4)
    EPOCHS = hyperparams.get("epochs", 50)
    W_DECAY = hyperparams.get("weight_decay", 1e-4)
    DATA_MULTIPLIER = hyperparams.get("data_multiplier", 4) 
    
    num_folds = cv_splitter.get_n_splits()
    fold_tr_acc_mat = np.zeros((num_folds, EPOCHS))
    fold_val_acc_mat = np.zeros((num_folds, EPOCHS))
    fold_tr_loss_mat = np.zeros((num_folds, EPOCHS))
    fold_val_loss_mat = np.zeros((num_folds, EPOCHS))

    for fold, (t_idx, v_idx) in enumerate(cv_splitter.split(X, y, groups=groups)):
        if verbose: print(f"\n>>> Starting Fold {fold+1}/{num_folds}")

        X_tr_raw, X_val_raw = X[t_idx], X[v_idx]
        y_tr_raw, y_val_raw = y[t_idx], y[v_idx]

        if fold == 0 and verbose:
            print(f"--- DATA DEBUG ---")
            print(f"X Mean: {np.mean(X_tr_raw):.4f}, Std: {np.std(X_tr_raw):.4f}")
            print(f"Range: [{np.min(X_tr_raw):.4f}, {np.max(X_tr_raw):.4f}]")
            print("------------------")

        # Training set with "lying" multiplier and stochastic coin-flip
        train_loader = DataLoader(
            EEGStochasticDataset(X_tr_raw, y_tr_raw, exp_name, aug_params, multiplier=DATA_MULTIPLIER),
            batch_size=BS, shuffle=True
        )

        # Validation set remains static and original
        X_val_t = torch.tensor(X_val_raw, dtype=torch.float32).unsqueeze(1)
        y_val_t = torch.tensor(y_val_raw, dtype=torch.long)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BS, shuffle=False)

        n_chans, n_time = X_tr_raw.shape[-2], X_tr_raw.shape[-1]
        
        # Select Model
        if model_name == "EEGNet":
            model = EEGNet(nb_classes=n_classes, Chans=n_chans, Samples=n_time, kernLength=256).to(config.DEVICE)
        elif model_name == "DeepConvNet":
            model = DeepConvNet(nb_classes=n_classes, Chans=n_chans, Samples=n_time).to(config.DEVICE)
        elif model_name == "ShallowConvNet":
            model = ShallowConvNet(nb_classes=n_classes, Chans=n_chans, Samples=n_time).to(config.DEVICE)
        elif model_name == "CustomEEGNet":
            model = CustomEEGNet(nb_classes=n_classes, Chans=n_chans, Samples=n_time).to(config.DEVICE)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        
        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=W_DECAY)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)

        f_hist = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
        f_best_acc = 0.0

        for epoch in range(EPOCHS):
            model.train()
            tr_l, tr_a, tr_c = 0.0, 0.0, 0
            
            for xb, yb in train_loader:
                xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
                opt.zero_grad()
                out = model(xb)
                loss = crit(out, yb)
                loss.backward()
                opt.step()
                
                tr_l += loss.item() * yb.size(0); tr_a += (out.argmax(1) == yb).sum().item(); tr_c += yb.size(0)
            
            # Validation
            model.eval()
            v_l, v_a, v_c = 0.0, 0.0, 0
            with torch.no_grad():
                for xb_v, yb_v in val_loader:
                    xb_v, yb_v = xb_v.to(config.DEVICE), yb_v.to(config.DEVICE)
                    out_v = model(xb_v)
                    v_l += crit(out_v, yb_v).item() * yb_v.size(0)
                    v_a += (out_v.argmax(1) == yb_v).sum().item()
                    v_c += yb_v.size(0)

            fold_tr_loss_mat[fold, epoch], fold_tr_acc_mat[fold, epoch] = tr_l/tr_c, tr_a/tr_c
            fold_val_loss_mat[fold, epoch], fold_val_acc_mat[fold, epoch] = v_l/v_c, v_a/v_c
            f_hist['train_loss'].append(tr_l/tr_c); f_hist['train_acc'].append(tr_a/tr_c)
            f_hist['val_loss'].append(v_l/v_c); f_hist['val_acc'].append(v_a/v_c)

            if (v_a/v_c) > f_best_acc: f_best_acc = v_a/v_c

        best_accs.append(f_best_acc)
        for k in history: history[k].append(f_hist[k])
        print(f" -> Fold {fold+1} Finished. Best Val Acc: {f_best_acc:.2%}")

    # Aggregated Logging
    for ep in range(EPOCHS):
        writer.add_scalar('Accuracy_Mean/Train', np.mean(fold_tr_acc_mat[:, ep]), ep)
        writer.add_scalar('Accuracy_Mean/Val', np.mean(fold_val_acc_mat[:, ep]), ep)
        writer.add_scalar('Loss_Mean/Train', np.mean(fold_tr_loss_mat[:, ep]), ep)
        writer.add_scalar('Loss_Mean/Val', np.mean(fold_val_loss_mat[:, ep]), ep)

    hparam_dict = {"exp_name": exp_name, "model": model_name, "lr": LR, "mult": DATA_MULTIPLIER}
    metric_dict = {"hparam/mean_cv_acc": np.mean(best_accs), "hparam/std_cv_acc": np.std(best_accs)}
    writer.add_hparams(hparam_dict, metric_dict)
    
    with open(os.path.join(log_dir, "metadata.json"), "w") as f:
        json.dump({"aug_params": aug_params, "hyperparams": hyperparams, "best_accs": best_accs}, f, indent=4)

    writer.close()
    return history, np.mean(best_accs)