# augmentations/eeg_aug.py
import numpy as np

# --- 1. SPATIAL DOMAIN ---
def channels_dropout(X, p_drop=0.2):
    X_aug = X.copy()
    N, Ch, T = X_aug.shape
    for i in range(N):
        mask = np.random.binomial(1, 1 - p_drop, size=(Ch, 1))
        X_aug[i] = X_aug[i] * mask
    return X_aug

# --- 2. FREQUENCY DOMAIN (The Winner: FT Surrogate) ---
def freq_surrogate(X, phase_noise_max=0.5):
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

# --- 3. TIME DOMAIN ---
def time_reverse(X):
    return np.flip(X, axis=-1)

def smooth_time_mask(X, mask_len_samples=100):
    X_aug = X.copy()
    N, Ch, T = X_aug.shape
    for i in range(N):
        t_start = np.random.randint(0, T - mask_len_samples)
        t_end = t_start + mask_len_samples
        mask = np.ones(T)
        mask[t_start : t_end] = 0
        X_aug[i] = X_aug[i] * mask[np.newaxis, :]
    return X_aug

def get_augmentation(name, X, y, params):
    if name == "ChannelsDropout":
        X_aug = channels_dropout(X, **params.get("channels_dropout", {}))
    elif name == "FTSurrogate":
        X_aug = freq_surrogate(X, **params.get("freq_surrogate", {}))
    elif name == "TimeReverse":
        X_aug = time_reverse(X)
    elif name == "SmoothTimeMask":
        X_aug = smooth_time_mask(X, **params.get("smooth_time_mask", {}))
    else:
        return X, y
    return np.concatenate([X, X_aug]), np.concatenate([y, y])