# utils/io_utils.py
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import datetime
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
    Handles creating directories and converting numpy types automatically.
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
    
    Args:
        results (list of dicts): [{'method': str, 'params': any, 'accuracy': float}]
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