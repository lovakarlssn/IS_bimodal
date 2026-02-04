import numpy as np

def spatial_shuffle(X, n_swaps=1):
    X_aug = X.copy()
    N, Ch, T = X_aug.shape
    for i in range(N):
        for _ in range(n_swaps):
            a, b = np.random.choice(Ch, 2, replace=False)
            X_aug[i, a, :], X_aug[i, b, :] = X_aug[i, b, :].copy(), X_aug[i, a, :].copy()
    return X_aug

def time_slice(X, y, slice_len=716, n_slices=2, stride=None):
    N, Ch, T = X.shape
    X_slices, y_slices = [], []
    for i in range(N):
        if stride is not None:
            starts = np.arange(0, T - slice_len + 1, stride)
            for s in starts:
                X_slices.append(X[i, :, s : s + slice_len])
                y_slices.append(y[i])
        else:
            for _ in range(n_slices):
                s = np.random.randint(0, T - slice_len + 1)
                X_slices.append(X[i, :, s : s + slice_len])
                y_slices.append(y[i])
    return np.array(X_slices), np.array(y_slices)

def freq_surrogate(X, phase_noise_std=0.5):
    X_aug = np.zeros_like(X)
    N, Ch, T = X.shape
    for i in range(N):
        for c in range(Ch):
            f_transform = np.fft.rfft(X[i, c, :])
            magnitudes = np.abs(f_transform)
            phases = np.angle(f_transform)
            noise = np.random.normal(0, phase_noise_std, size=phases.shape)
            new_f_transform = magnitudes * np.exp(1j * (phases + noise))
            X_aug[i, c, :] = np.fft.irfft(new_f_transform, n=T)
    return X_aug

def get_augmentation(name, X, y, params):
    """Factory to return (X_total, y_total) for consistent comparison."""
    if name == "Spatial_Shuffle":
        X_aug = spatial_shuffle(X, **params["spatial_shuffle"])
        return np.concatenate([X, X_aug]), np.concatenate([y, y])
    elif name == "Freq_Surrogate":
        X_aug = freq_surrogate(X, **params["freq_surrogate"])
        return np.concatenate([X, X_aug]), np.concatenate([y, y])
    elif name == "Time_Slicing":
        return time_slice(X, y, **params["time_slice"])
    return X, y # Original Baseline