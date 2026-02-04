import numpy as np
import mne
import os
import glob
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
# Adjusted base path for Sync data structure
BASE_PATH = "Data_sub01"  
SAVE_DIR = "Augmented_Data_EEG_Sync" # Saving to a new folder to avoid mix-ups

# --- THE GOLDEN PARAMETERS ---
# (Kept the same unless your Sync trials are shorter than 2s)
CROP_LEN = 1.4  
STRIDE   = 0.2  
# -----------------------------

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ==========================================
# 1. LOAD DATA (UPDATED FOR SYNC / MULTI-SESSION)
# ==========================================
def load_eeg_data(subject_id="sub-03", task="words"):
    """
    Loads sessions, handles channel mismatches by keeping only common channels,
    and concatenates them.
    """
    search_pattern = os.path.join(
        BASE_PATH, 
        "Sync_data_preproc_new", 
        subject_id, 
        "ses-*", 
        "eeg", 
        f"*{subject_id}*desc-{task}_epo.fif.gz"
    )
    
    files = sorted(glob.glob(search_pattern))
    if not files:
        raise FileNotFoundError(f"No files found for {subject_id}")
        
    print(f"Found {len(files)} session file(s) for {subject_id}.")
    epoch_list = []
    
    # 1. Load all files
    for f in files:
        print(f"  -> Loading: {os.path.basename(f)}")
        epo = mne.read_epochs(f, preload=True, verbose=False)
        epoch_list.append(epo)
    
    # 2. Fix Channel Mismatch (Intersection)
    if len(epoch_list) > 1:
        # Start with channels from the first file
        common_chans = set(epoch_list[0].info['ch_names'])
        
        # Intersect with all other files
        for epo in epoch_list[1:]:
            common_chans = common_chans.intersection(set(epo.info['ch_names']))
            
        common_chans = list(common_chans)
        print(f"  -> Found {len(common_chans)} common channels. Dropping non-matching ones...")
        
        # Prune all epochs to this common set
        for i in range(len(epoch_list)):
            epoch_list[i] = epoch_list[i].pick(common_chans)
            
        # 3. Concatenate
        print("  -> Concatenating sessions...")
        epochs = mne.concatenate_epochs(epoch_list)
    else:
        epochs = epoch_list[0]
    
    print(f"Total Combined Trials: {len(epochs)}")
    
    return epochs.get_data(copy=True), epochs.events[:, -1], epochs.info['sfreq']
# ==========================================
# 2. AUGMENTATION FUNCTIONS (UNCHANGED)
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
    # Change subject_id to "sub-03" or "sub-01" as needed
    X_orig, y_orig, sfreq = load_eeg_data(subject_id="sub-03")

    # 4. Save
    datasets = [
        ("SYNC", X_orig, y_orig),

    ]

    print(f"\n--- SAVING TO {SAVE_DIR} ---")
    for name, X_d, y_d in datasets:
        np.save(os.path.join(SAVE_DIR, f"X_{name}.npy"), X_d)
        np.save(os.path.join(SAVE_DIR, f"y_{name}.npy"), y_d)
        print(f"Saved {name}: X{X_d.shape} y{y_d.shape}")

    print("\nDone!")