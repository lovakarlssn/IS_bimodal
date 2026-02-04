import numpy as np
import mne
import os
import glob
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
BASE_PATH = "Data_sub01"  
SAVE_DIR = "Augmented_Data_EEG_01"

# --- THE GOLDEN PARAMETERS ---
CROP_LEN = 1.4  # Seconds
STRIDE   = 0.2  # Seconds
# -----------------------------

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ==========================================
# 1. LOAD DATA
# ==========================================
def load_eeg_data(subject_id="sub-01", task="words"):
    eeg_dir = os.path.join(BASE_PATH, "Async_data_preproc_new", subject_id, "ses-eeg", "eeg")
    search_pattern = os.path.join(eeg_dir, f"*{subject_id}*desc-{task}_epo.fif")
    
    files = glob.glob(search_pattern)
    if not files:
        raise FileNotFoundError(f"No files found for {task}")
        
    print(f"Reading: {os.path.basename(files[0])}")
    epochs = mne.read_epochs(files[0], preload=True, verbose=False)
    
    
    return epochs.get_data(copy=True), epochs.events[:, -1], epochs.info['sfreq']

# ==========================================
# 2. AUGMENTATION FUNCTIONS
# ==========================================
def aug_time_window_slicing(X, y, sfreq, crop_len=1.5, stride=0.25):
    print(f"Applying Time Slicing (Crop={crop_len}s, Stride={stride}s)...")
    n_epochs, n_ch, n_time = X.shape
    win_samples = int(crop_len * sfreq)
    stride_samples = int(stride * sfreq)
    
    X_new, y_new = [], []
    start_indices = range(0, n_time - win_samples + 1, stride_samples)
    
    for i in range(n_epochs):
        for start in start_indices:
            crop = X[i, :, start : start + win_samples]
            X_new.append(crop)
            y_new.append(y[i])
            
    crops_per_trial = len(start_indices)
    print(f"   -> Generated {crops_per_trial} crops per trial.")
    return np.array(X_new), np.array(y_new), crops_per_trial

def aug_freq_ft_surrogate(X, y, multiplier=1, phase_noise_magnitude=0.5):
    print(f"Applying Freq Surrogate (x{multiplier} augmentations)...")
    X_total = [X]
    y_total = [y]
    
    for m in range(multiplier):
        print(f"   -> Generating batch {m+1}/{multiplier}...")
        X_aug = np.zeros_like(X)
        for i in range(len(X)):
            for ch in range(X.shape[1]):
                f_pos = np.fft.rfft(X[i, ch, :])
                magnitudes = np.abs(f_pos)
                max_mag = np.max(magnitudes)
                mask = magnitudes > (0.001 * max_mag)
                rand_phase = np.random.uniform(-np.pi, np.pi, len(f_pos)) * phase_noise_magnitude
                rotator = np.ones(len(f_pos), dtype=complex)
                rotator[mask] = np.exp(1j * rand_phase[mask])
                rotator[0] = 1.0 
                f_shifted = f_pos * rotator
                X_aug[i, ch, :] = np.fft.irfft(f_shifted, n=X.shape[2])
        X_total.append(X_aug)
        y_total.append(y)
    
    return np.concatenate(X_total), np.concatenate(y_total)

def aug_spatial_channel_shuffle(X, y, multiplier=1, p=0.1):
    print(f"Applying Spatial Shuffle (x{multiplier} augmentations)...")
    X_total = [X]
    y_total = [y]
    n_ch = X.shape[1]
    n_swap = int(n_ch * p)
    
    for m in range(multiplier):
        print(f"   -> Generating batch {m+1}/{multiplier}...")
        X_aug = X.copy()
        for i in range(len(X)):
            indices = np.random.choice(n_ch, n_swap, replace=False)
            shuffled_indices = np.random.permutation(indices)
            X_aug[i, indices, :] = X_aug[i, shuffled_indices, :]
        X_total.append(X_aug)
        y_total.append(y)
        
    return np.concatenate(X_total), np.concatenate(y_total)

# ==========================================
# 3. MAIN
# ==========================================
if __name__ == "__main__":
    X_orig, y_orig, sfreq = load_eeg_data()

    # 1. Time Slicing sets the baseline count
    X_time, y_time, crops_per_trial = aug_time_window_slicing(
        X_orig, y_orig, sfreq, crop_len=CROP_LEN, stride=STRIDE
    )
    
    # 2. Calculate balancing multiplier
    # Time Slicing = N * crops
    # Others = N * (1 + multiplier)
    # multiplier = crops - 1
    aug_multiplier = crops_per_trial - 1
    
    print(f"\n--- BALANCING ---")
    print(f"Time Slicing Factor: {crops_per_trial}x")
    print(f"Target Multiplier for others: {aug_multiplier}x (+ 1 Original)")

    # 3. Generate balanced datasets
    X_freq, y_freq = aug_freq_ft_surrogate(X_orig, y_orig, multiplier=aug_multiplier)
    X_spat, y_spat = aug_spatial_channel_shuffle(X_orig, y_orig, multiplier=aug_multiplier)

    # 4. Save
    datasets = [
        ("Original", X_orig, y_orig),
        ("Time_Slicing", X_time, y_time),
        ("Freq_Surrogate", X_freq, y_freq),
        ("Spatial_Shuffle", X_spat, y_spat)
    ]

    print(f"\n--- SAVING TO {SAVE_DIR} ---")
    for name, X_d, y_d in datasets:
        np.save(os.path.join(SAVE_DIR, f"X_{name}.npy"), X_d)
        np.save(os.path.join(SAVE_DIR, f"y_{name}.npy"), y_d)
        print(f"Saved {name}: X{X_d.shape} y{y_d.shape}")

    print("\nDone!")