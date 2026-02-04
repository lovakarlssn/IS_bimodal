# augmentations/eeg_augment.py
import numpy as np

def spatial_shuffle(X, n_swaps=1):
    """
    Shuffles channels in the EEG data by swapping pairs of channels for each trial.
    
    Args:
        X: np.ndarray of shape (N, Ch, T)
        n_swaps: Number of channel swaps to perform per trial
        
    Returns:
        X_aug: Shuffled copy of X
    """
    X_aug = X.copy()
    N, Ch, T = X_aug.shape
    
    for i in range(N):
        for _ in range(n_swaps):
            # Pick two distinct channels to swap
            a, b = np.random.choice(Ch, 2, replace=False)
            
            # Swap the rows (channels) for this trial
            temp = X_aug[i, a, :].copy()
            X_aug[i, a, :] = X_aug[i, b, :]
            X_aug[i, b, :] = temp
            
    return X_aug

def time_slice(X, y, slice_len=716, n_slices=2, stride=None):
    """
    Extracts temporal crops from EEG data.
    
    Args:
        X: (N, Ch, T) array.
        y: (N,) array.
        slice_len: Length of each output window (in samples).
        n_slices: Number of random windows to extract (only used if stride is None).
        stride: If provided, uses a fixed sliding window. 
                e.g., stride=slice_len // 2 for 50% overlap.
    
    Returns:
        X_slices: (N_new, Ch, slice_len)
        y_slices: (N_new,)
    """
    N, Ch, T = X.shape
    X_slices, y_slices = [], []
    
    # Validation
    if slice_len > T:
        raise ValueError(f"slice_len ({slice_len}) cannot be larger than input length ({T})")

    for i in range(N):
        if stride is not None:
            # --- Sliding Windows (Deterministic) ---
            # Generate start indices: 0, stride, 2*stride, ... until we hit the end
            starts = np.arange(0, T - slice_len + 1, stride)
            
            for s in starts:
                crop = X[i, :, s : s + slice_len]
                X_slices.append(crop)
                y_slices.append(y[i])
                
        else:
            # --- Random Cropping (Augmentation) ---
            for _ in range(n_slices):
                # Random start index
                s = np.random.randint(0, T - slice_len + 1)
                
                crop = X[i, :, s : s + slice_len]
                X_slices.append(crop)
                y_slices.append(y[i])
                
    return np.array(X_slices), np.array(y_slices)

def freq_surrogate(X, phase_noise_std=0.5):
    """
    Applies Frequency Shift Surrogate augmentation.
    Modifies the phase of the signal in the frequency domain while preserving the power spectrum.
    
    Args:
        X: (N, Ch, T) array
        phase_noise_std: Standard deviation of the phase noise (0 to 2*pi range usually)
        
    Returns:
        X_aug: (N, Ch, T) array with phase perturbations
    """
    X_aug = np.zeros_like(X)
    N, Ch, T = X.shape
    
    for i in range(N):
        for c in range(Ch):
            # 1. Compute FFT (Real FFT since EEG is real-valued)
            f_transform = np.fft.rfft(X[i, c, :])
            
            # 2. Extract Magnitude and Phase
            magnitudes = np.abs(f_transform)
            phases = np.angle(f_transform)
            
            # 3. Add random noise to the phase
            # We generate noise for all freq bins, then apply it
            noise = np.random.normal(0, phase_noise_std, size=phases.shape)
            new_phases = phases + noise
            
            # 4. Reconstruct the signal: Mag * exp(j * new_phase)
            # Ensure the DC component (index 0) phase is unmodified to keep signal real/centered correctly
            # (Though rfft usually handles this, it's safer to keep DC phase 0 or Pi)
            new_f_transform = magnitudes * np.exp(1j * new_phases)
            
            # 5. Inverse FFT
            X_aug[i, c, :] = np.fft.irfft(new_f_transform, n=T)
            
    return X_aug