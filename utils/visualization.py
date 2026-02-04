# utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_cv_history(history, experiment_name, save_dir="./results"):
    """
    Plots training and validation metrics with standard deviation shadows.
    """
    epochs = range(1, len(history['t_acc'][0]) + 1)
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Helper to plot mean + shadow
    def add_metric_plot(ax, train_data, val_data, title, ylabel):
        t_mean = np.mean(train_data, axis=0)
        t_std = np.std(train_data, axis=0)
        v_mean = np.mean(val_data, axis=0)
        v_std = np.std(val_data, axis=0)
        
        ax.plot(epochs, t_mean, label='Train', color='blue')
        ax.fill_between(epochs, t_mean-t_std, t_mean+t_std, alpha=0.1, color='blue')
        
        ax.plot(epochs, v_mean, label='Val', color='red')
        ax.fill_between(epochs, v_mean-v_std, v_mean+v_std, alpha=0.1, color='red')
        
        ax.set_title(title)
        ax.set_xlabel('Epochs')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    add_metric_plot(ax1, history['t_acc'], history['v_acc'], f'{experiment_name} Accuracy', 'Accuracy')
    add_metric_plot(ax2, history['t_loss'], history['v_loss'], f'{experiment_name} Loss', 'Cross Entropy')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{experiment_name}_metrics.png")
    plt.show()