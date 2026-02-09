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

        self.X = X
        self.y = y
        self.aug_name = aug_name
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

        should_augment = False

        if self.multiplier > 1:

            if idx >= self.num_original:
                should_augment = True
        else:

            if np.random.rand() < self.p_aug:
                should_augment = True

        if should_augment and self.aug_name not in ["Original", "None"]:
            try:

                x_batch = x[np.newaxis, ...]
                y_batch = np.array([label])

                x_pair, _ = get_augmentation(
                    self.modality, self.aug_name, x_batch, y_batch, self.aug_params
                )

                x = x_pair[1]

            except Exception as e:

                print(
                    f"Warning: Augmentation {self.aug_name} failed on sample {real_idx}: {e}"
                )

        x_tensor = torch.tensor(x, dtype=torch.float32)

        if self.modality == "EEG":

            if x_tensor.ndim == 2:
                x_tensor = x_tensor.unsqueeze(0)

        elif self.modality == "fMRI":

            if x_tensor.ndim == 3:
                x_tensor = x_tensor.unsqueeze(0)

        return x_tensor, torch.tensor(label, dtype=torch.long)


def run_experiment(
    modality,
    X,
    y,
    groups,
    n_classes,
    cv_splitter,
    aug_name,
    aug_params,
    hyperparams,
    arch_config,
    model_name,
    verbose=False,
):
    """
    Generalized CV experiment runner.
    Tracks Kappa and F1-Score for model selection.
    """
    best_val_metrics = []

    global_history = defaultdict(list)

    timestamp = datetime.datetime.now().strftime("%d_%H%M")
    base_log_dir = os.path.join("runs", f"{modality}_{model_name}", aug_name, timestamp)

    if verbose:
        print(f"Logging metrics to: {base_log_dir}")

    for fold, (t_idx, v_idx) in enumerate(cv_splitter.split(X, y, groups=groups)):
        if verbose:
            print(f"\n--- Fold {fold+1} ---")

        fold_history = defaultdict(list)

        writer = SummaryWriter(log_dir=os.path.join(base_log_dir, f"fold_{fold+1}"))

        train_ds = MultiModalDataset(
            modality,
            X[t_idx],
            y[t_idx],
            aug_name,
            aug_params,
            multiplier=hyperparams.get("multiplier", 1),
        )
        train_loader = DataLoader(
            train_ds, batch_size=hyperparams.get("batch_size", 32), shuffle=True
        )

        X_v = torch.tensor(X[v_idx], dtype=torch.float32)

        if modality == "EEG":

            if X_v.ndim == 3:
                X_v = X_v.unsqueeze(1)
        elif modality == "fMRI":

            if X_v.ndim == 4:
                X_v = X_v.unsqueeze(1)

        val_loader = DataLoader(
            TensorDataset(X_v, torch.tensor(y[v_idx], dtype=torch.long)),
            batch_size=hyperparams.get("batch_size", 32),
            shuffle=False,
        )

        dummy_input_shape = train_ds[0][0].shape

        model = get_model(
            modality, model_name, n_classes, dummy_input_shape, arch_config=arch_config
        ).to(config.DEVICE)
        opt = optim.AdamW(
            model.parameters(),
            lr=hyperparams.get("lr", 1e-4),
            weight_decay=hyperparams.get("weight_decay", 1e-4),
        )
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)

        fold_best_kappa = -1.0
        fold_best_acc = 0.0

        for epoch in range(hyperparams.get("epochs", 50)):

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

            val_acc = np.mean(np.array(all_preds) == np.array(all_targets))
            val_f1 = f1_score(all_targets, all_preds, average="macro")
            val_kappa = cohen_kappa_score(all_targets, all_preds)

            metrics_dict = {
                "Loss/Train": train_loss,
                "Loss/Val": val_loss,
                "Accuracy/Train": train_acc,
                "Accuracy/Val": val_acc,
                "Metrics/F1_Macro": val_f1,
                "Metrics/Kappa": val_kappa,
            }

            for key, val in metrics_dict.items():
                writer.add_scalar(key, val, epoch)
                fold_history[key].append(val)

            if val_kappa > fold_best_kappa:
                fold_best_kappa = val_kappa
                fold_best_acc = val_acc

        for key, val_list in fold_history.items():
            global_history[key].append(val_list)

        best_val_metrics.append(fold_best_acc)
        if verbose:
            print(
                f"    Best Val Kappa: {fold_best_kappa:.4f} (Acc: {fold_best_acc:.2%})"
            )

        writer.close()

    mean_writer = SummaryWriter(log_dir=os.path.join(base_log_dir, "mean"))

    n_epochs = hyperparams.get("epochs", 50)

    for epoch in range(n_epochs):
        for metric_name, folds_data in global_history.items():

            epoch_values = [fold[epoch] for fold in folds_data if len(fold) > epoch]

            if epoch_values:
                mean_val = np.mean(epoch_values)
                mean_writer.add_scalar(metric_name, mean_val, epoch)

    mean_writer.close()

    return np.mean(best_val_metrics)
