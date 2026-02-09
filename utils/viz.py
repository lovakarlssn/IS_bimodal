import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import numpy as np
import json
import os
import datetime
import torch


def plot_history(history, title="Cross-Validation Training History"):
    """
    Plots training and validation metrics with shaded error bars.
    Robust to Early Stopping (jagged fold lengths).
    """
    
    def get_padded_matrix(key):
        data = history[key]
        max_len = max(len(f) for f in data)
        return np.array([f + [np.nan] * (max_len - len(f)) for f in data])

    tr_loss = get_padded_matrix('train_loss')
    va_loss = get_padded_matrix('val_loss')
    tr_acc  = get_padded_matrix('train_acc')
    va_acc  = get_padded_matrix('val_acc')

    max_epochs = tr_loss.shape[1]
    x_axis = np.arange(1, max_epochs + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    metrics = [
        ('Loss', tr_loss, va_loss, 'upper right'),
        ('Accuracy', tr_acc, va_acc, 'lower right')
    ]

    for i, (label, train_data, val_data, loc) in enumerate(metrics):
        ax = axes[i]
        
        t_mean, t_std = np.nanmean(train_data, axis=0), np.nanstd(train_data, axis=0)
        v_mean, v_std = np.nanmean(val_data, axis=0), np.nanstd(val_data, axis=0)

        ax.plot(x_axis, t_mean, label=f'Train {label}', color='#1f77b4', linestyle='--')
        ax.fill_between(x_axis, t_mean - t_std, t_mean + t_std, color='#1f77b4', alpha=0.1)

        ax.plot(x_axis, v_mean, label=f'Val {label}', color='#ff7f0e', linewidth=2)
        ax.fill_between(x_axis, v_mean - v_std, v_mean + v_std, color='#ff7f0e', alpha=0.2)

        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'Mean {label} across Folds', fontsize=14)
        ax.legend(loc=loc)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    
def save_experiment_results(results_dict, filename=None, folder="../results"):
    """
    Saves experiment results to a JSON file.
    """
    os.makedirs(folder, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = results_dict.get('model', 'UnknownModel')
        filename = f"{model_name}_{timestamp}.json"
    
    # Helper to convert non-serializable numpy types
    def convert_numpy(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    file_path = os.path.join(folder, filename)
    
    try:
        with open(file_path, "w") as f:
            json.dump(results_dict, f, indent=4, default=convert_numpy)
        print(f"[IO] Results saved to {file_path}")
    except Exception as e:
        print(f"[IO] Error saving results: {e}")
        
        
def plot_tuning_results(results):
    """
    Plots hyperparameter tuning results compared against methods.
    """
    plt.figure(figsize=(12, 7))
    
    methods = sorted(list(set([r['method'] for r in results])))
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for i, method in enumerate(methods):
        method_data = [r for r in results if r['method'] == method]
        method_data.sort(key=lambda x: str(x['params']))
        
        x_labels = [str(r['params']) for r in method_data]
        y_values = [r['accuracy'] for r in method_data]
        
        plt.plot(x_labels, y_values, marker='s', markersize=8, 
                 label=method, color=colors[i], linewidth=2.5, alpha=0.8)

    plt.ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Hyperparameter Magnitude / Type', fontsize=12, fontweight='bold')
    plt.title('Augmentation Hyperparameter Tuning', fontsize=15, pad=20)
    
    plt.xticks(rotation=30, ha='right')
    plt.legend(title="Augmentation Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.show()


# --- SALIENCY / RESPONSIBILITY TOOLS ---

def compute_saliency(model, x_tensor, y_tensor, device="cpu"):
    """
    Computes the Saliency Map (Gradient of output w.r.t input).
    """
    model.eval()
    x_tensor = x_tensor.to(device)
    y_tensor = y_tensor.to(device)
    
    # We need gradients w.r.t input, not weights
    x_tensor.requires_grad_()
    
    # Forward pass
    outputs = model(x_tensor)
    
    # Get score for the target class
    if y_tensor.dim() == 0: y_tensor = y_tensor.unsqueeze(0)
    score = outputs.gather(1, y_tensor.view(-1, 1)).squeeze()
    
    # Backward pass to get gradients
    if score.dim() == 0:
        score.backward()
    else:
        torch.sum(score).backward()
    
    # Saliency is the magnitude of the gradient
    saliency = x_tensor.grad.abs()
    return saliency.detach().cpu().numpy()


def plot_saliency_eeg(signal, saliency, sample_rate=512, title="Model Responsibility (Saliency)", zoom_to_hotspot=False):
    """
    Plots the EEG signal colored by how 'responsible' each point was for the prediction.
    Red/Hot = High Importance, Blue/Cold = Low Importance.
    """
    time = np.arange(len(signal)) / sample_rate
    
    # Normalize saliency for coloring (0 to 1)
    s_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    # Create line segments for multicolor plot
    points = np.array([time, signal]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Create a LineCollection
    lc = mcoll.LineCollection(segments, cmap='turbo', norm=plt.Normalize(0, 1))
    lc.set_array(s_norm)
    lc.set_linewidth(1.5)
    ax.add_collection(lc)
    
    ax.set_xlim(time.min(), time.max())
    ax.set_ylim(signal.min(), signal.max())
    
    # Add colorbar
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label("Importance (Gradient Magnitude)")
    
    # Automatic Cropping (Zoom) logic
    if zoom_to_hotspot:
        threshold = np.percentile(s_norm, 80) # Top 20% importance
        high_imp_indices = np.where(s_norm > threshold)[0]
        
        if len(high_imp_indices) > 0:
            pad_samples = int(0.5 * sample_rate)
            start_idx = max(0, high_imp_indices.min() - pad_samples)
            end_idx = min(len(time)-1, high_imp_indices.max() + pad_samples)
            
            ax.set_xlim(time[start_idx], time[end_idx])
            title += " (Auto-Zoomed)"

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    
    return fig


# --- COMPARISON PLOTS ---
    
def plot_eeg_comparison(original, augmented, title="Augmentation", sample_rate=512, channel_idx=0, time_range=None):
    """
    Plots a specific channel of the Original vs Augmented EEG signal.
    """
    time = np.arange(original.shape[-1]) / sample_rate
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Plot Original
    axes[0].plot(time, original[channel_idx], color='black', alpha=0.8, linewidth=1)
    axes[0].set_title(f"Original Signal (Channel {channel_idx})")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    
    # Plot Augmented
    axes[1].plot(time, augmented[channel_idx], color='teal', alpha=0.8, linewidth=1)
    axes[1].set_title(f"Augmented: {title} (Channel {channel_idx})")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    
    if time_range is not None:
        axes[1].set_xlim(time_range)
    
    plt.tight_layout()
    return fig

def plot_fmri_comparison(original, augmented, title="Augmentation", slice_idx=None):
    """
    Plots a central slice of the Original vs Augmented fMRI volume.
    Shared vmin/vmax ensures intensity changes are actually visible.
    """
    if slice_idx is None:
        slice_idx = original.shape[1] // 2
        
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Determine common scale so brightness changes are visible
    vmin = min(original.min(), augmented.min())
    vmax = max(original.max(), augmented.max())
    
    # Original
    im1 = axes[0].imshow(original[0, slice_idx, :, :], cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Original (Slice {slice_idx})")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Augmented
    im2 = axes[1].imshow(augmented[0, slice_idx, :, :], cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Augmented: {title}")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig

def save_plot(fig, filename, results_dir="../results/plots", show=False):
    """
    Saves the figure to disk.
    If show=True, displays it in the notebook.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    fig.savefig(path)
    print(f"Plot saved to {path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)