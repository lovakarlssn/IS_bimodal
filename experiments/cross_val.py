import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from utils.processing import prepare_tensors
from models.eegnets import CompactEEGNet
from tqdm import tqdm
import config
import numpy as np

def run_cv_experiment(X, y, groups, n_classes, label):
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    best_accs = []
    _, n_chans, n_time = X.shape

    for fold, (t_idx, v_idx) in enumerate(skf.split(X, y, groups=groups)):
        X_tr_raw, X_val_raw = X[t_idx], X[v_idx]
        y_tr_raw, y_val_raw = y[t_idx], y[v_idx]

        X_tr, y_tr, X_val, y_val = prepare_tensors(X_tr_raw, X_val_raw, y_tr_raw, y_val_raw)
        
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config.BATCH_SIZE, shuffle=False)

        model = CompactEEGNet(n_classes, n_chans, n_time).to(config.DEVICE)
        opt = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)

        f_h = {'ta':[], 'va':[], 'tl':[], 'vl':[]}
        for epoch in range(config.EPOCHS):
            model.train()
            tl, ta, tt = 0, 0, 0
            for xb, yb in train_loader:
                opt.zero_grad(); out = model(xb); loss = crit(out, yb)
                loss.backward(); opt.step()
                tl += loss.item(); ta += (out.argmax(1)==yb).sum().item(); tt += yb.size(0)
            
            model.eval()
            vl, va, vt = 0, 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    out = model(xb); vl += crit(out, yb).item()
                    va += (out.argmax(1)==yb).sum().item(); vt += yb.size(0)
            
            f_h['ta'].append(ta/tt); f_h['va'].append(va/vt)
            f_h['tl'].append(tl/len(train_loader)); f_h['vl'].append(vl/len(val_loader))

        history['train_acc'].append(f_h['ta']); history['val_acc'].append(f_h['va'])
        history['train_loss'].append(f_h['tl']); history['val_loss'].append(f_h['vl'])
        best_accs.append(max(f_h['va']))
        print(f"[{label}] Fold {fold+1} Best Acc: {best_accs[-1]:.2%}")
        
    return history, np.mean(best_accs)