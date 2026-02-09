# augmentations/eeg_aug.py
import numpy as np

# --- SPATIAL DOMAIN ---
def channels_dropout(X, p_drop=0.2):
    """Randomly sets entire channels to zero."""
    X_aug = X.copy()
    N, Ch, T = X_aug.shape
    for i in range(N):
        mask = np.random.binomial(1, 1 - p_drop, size=(Ch, 1))
        X_aug[i] = X_aug[i] * mask
    return X_aug

# --- FREQUENCY DOMAIN ---
def freq_surrogate(X, phase_noise_max=0.5):
    """Applies phase noise in the frequency domain to augment signals."""
    X_aug = np.zeros_like(X)
    N, Ch, T = X.shape
    for i in range(N):
        f_transform = np.fft.rfft(X[i], axis=-1) 
        n_freqs = f_transform.shape[-1]
        delta_phi = np.random.uniform(0, phase_noise_max, size=(1, n_freqs))
        perturbation = np.exp(1j * delta_phi)
        new_f_transform = f_transform * perturbation
        # Added .real to ensure no complex residual gradients crash the model
        X_aug[i] = np.fft.irfft(new_f_transform, n=T, axis=-1).real
    return X_aug

# --- TIME DOMAIN ---
def time_reverse(X):
    """Flips the signal along the time axis."""
    return np.flip(X, axis=-1)

def smooth_time_mask(X, mask_len_samples=100):
    """Zeroes out a continuous block of time samples."""
    X_aug = X.copy()
    N, Ch, T = X_aug.shape
    for i in range(N):
        t_start = np.random.randint(0, T - mask_len_samples)
        t_end = t_start + mask_len_samples
        mask = np.ones(T)
        mask[t_start : t_end] = 0
        X_aug[i] = X_aug[i] * mask[np.newaxis, :]
    return X_aug
