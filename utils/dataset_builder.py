import os
import glob
import mne
import numpy as np
import pandas as pd
import nibabel as nib
import sys
from sklearn.preprocessing import LabelEncoder

# Fix path to access config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# --- CONFIGURATION ---
EEG_RAW_PATH = r"D:\OneDrive - Luleå University of Technology\År 5\EX\Data_sub01\Async_data_preproc_new"
FMRI_RAW_PATH = r"D:\OneDrive - Luleå University of Technology\År 5\EX\Data_sub01\beta_images_activation_async"
OUTPUT_PATH = "./data"
FMRI_SESSIONS = ["single_trial_sess1", "single_trial_sess2"]

# Global Class Mapping
MASTER_CLASSES = sorted(['child', 'daughter', 'father', 'four', 'six', 'ten', 'three', 'wife'])

def load_subject_eeg(subject_id, task="words"):
    print(f"   [EEG] Looking for subject {subject_id}...")
    search_pattern = os.path.join(EEG_RAW_PATH, f"sub-{subject_id}", "ses-eeg", "eeg", f"*desc-{task}_epo.fif")
    found_files = glob.glob(search_pattern, recursive=True)
    
    if not found_files:
        print(f"   [EEG Warning] No .fif file found.")
        return None, None

    file_path = found_files[0]
    try:
        epochs = mne.read_epochs(file_path, preload=True, verbose=False)
    except Exception as e:
        print(f"   [EEG Error] Failed to load {file_path}: {e}")
        return None, None

    X = epochs.get_data(copy=True) 
    y_indices = epochs.events[:, -1]
    
    # Map integers to strings
    if epochs.event_id:
        id_to_class = {v: k for k, v in epochs.event_id.items()}
        y_strings = []
        for i in y_indices:
            raw = id_to_class.get(i, "unknown").lower()
            # Strip prefix (social/father -> father)
            clean = raw.split('/')[-1] if '/' in raw else raw
            y_strings.append(clean)
    else:
        y_strings = [str(i) for i in y_indices]

    if epochs.info['sfreq'] != config.TARGET_FS:
        print(f"      Resampling {epochs.info['sfreq']} -> {config.TARGET_FS}Hz...")
        epochs.resample(config.TARGET_FS)
        X = epochs.get_data() 

    return X, np.array(y_strings)

def load_subject_fmri(subject_id):
    print(f"   [fMRI] Looking for subject {subject_id}...")
    
    X_list = []
    y_list = []

    # Iterate both sessions
    for sess_num in [1, 2]:
        # Dynamic CSV path for session 1 or 2
        csv_name = f"beta_labels_subject{int(subject_id)}_session{sess_num}.csv"
        # Try finding the CSV
        csv_path = os.path.join(FMRI_RAW_PATH, csv_name)
        if not os.path.exists(csv_path):
             # Try alternative naming if 'subject01' vs 'subject1'
            csv_name_alt = f"beta_labels_subject{subject_id}_session{sess_num}.csv"
            csv_path = os.path.join(FMRI_RAW_PATH, csv_name_alt)
        
        if not os.path.exists(csv_path):
            print(f"      [Warning] CSV not found: {csv_name}. Skipping session.")
            continue

        print(f"      -> Loading Session {sess_num} from {csv_name}...")
        df = pd.read_csv(csv_path)

        for idx, row in df.iterrows():
            filename = row['file']
            
            # Find file in session folders
            found_path = None
            for sess_folder in FMRI_SESSIONS:
                p = os.path.join(FMRI_RAW_PATH, sess_folder, filename)
                if os.path.exists(p):
                    found_path = p
                    break
            
            if found_path:
                try:
                    img = nib.load(found_path)
                    data = img.get_fdata(dtype=np.float32)
                    
                    # --- CRITICAL FIX: CLEAN NANS HERE ---
                    # Replace NaN with 0.0 (Background)
                    data = np.nan_to_num(data, nan=0.0)
                    
                    X_list.append(data)
                    y_list.append(row['class'].lower())
                except Exception as e:
                    print(f"         [Error] {filename}: {e}")

    if not X_list:
        return None, None

    # Stack
    X_arr = np.array(X_list)
    X_arr = np.expand_dims(X_arr, axis=1) # (N, 1, D, H, W)
    
    return X_arr, np.array(y_list)

def verify_saved_data(le):
    print("\n" + "="*60)
    print("               DATA VERIFICATION REPORT")
    print("="*60)
    
    def check_array(name, data, labels):
        print(f"[{name}]")
        print(f"  Shape:     {data.shape}")
        
        # Check NaNs
        n_nans = np.isnan(data).sum()
        if n_nans > 0:
            print(f"  ⚠️ ALERT: Found {n_nans} NaN values!")
        else:
            print(f"  Sanity:    No NaNs found. ✅")
            
        print(f"  Range:     [{np.min(data):.3f}, {np.max(data):.3f}]")
        print(f"  Labels:    {labels[:5]} -> {le.inverse_transform(labels[:5])}")
        print("-" * 60)

    # EEG
    if os.path.exists(os.path.join(OUTPUT_PATH, "X_EEG.npy")):
        check_array("EEG", np.load(os.path.join(OUTPUT_PATH, "X_EEG.npy")), np.load(os.path.join(OUTPUT_PATH, "y_EEG.npy")))

    # fMRI
    if os.path.exists(os.path.join(OUTPUT_PATH, "X_fMRI.npy")):
        check_array("fMRI", np.load(os.path.join(OUTPUT_PATH, "X_fMRI.npy")), np.load(os.path.join(OUTPUT_PATH, "y_fMRI.npy")))
    
    print("="*60 + "\n")

def build_dataset(subjects=["01"], build_eeg=True, build_fmri=True):
    if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)
    print(f"\n[Info] Building Dataset for subjects: {subjects}")

    le = LabelEncoder()
    le.fit(MASTER_CLASSES)
    np.save(os.path.join(OUTPUT_PATH, "label_classes.npy"), le.classes_)

    if build_eeg:
        X_list, y_list, g_list = [], [], []
        for sub in subjects:
            X, y_str = load_subject_eeg(sub)
            if X is not None:
                # Encode and filter errors
                y_enc = [le.transform([l])[0] if l in le.classes_ else -1 for l in y_str]
                y_enc = np.array(y_enc)
                mask = y_enc != -1
                
                X_list.append(X[mask])
                y_list.append(y_enc[mask])
                g_list.append(np.full(mask.sum(), int(sub)))
        
        if X_list:
            np.save(os.path.join(OUTPUT_PATH, "X_EEG.npy"), np.concatenate(X_list))
            np.save(os.path.join(OUTPUT_PATH, "y_EEG.npy"), np.concatenate(y_list))
            np.save(os.path.join(OUTPUT_PATH, "groups_EEG.npy"), np.concatenate(g_list))

    if build_fmri:
        X_list, y_list, g_list = [], [], []
        for sub in subjects:
            X, y_str = load_subject_fmri(sub)
            if X is not None:
                y_enc = le.transform(y_str)
                X_list.append(X)
                y_list.append(y_enc)
                g_list.append(np.full(len(y_enc), int(sub)))
        
        if X_list:
            np.save(os.path.join(OUTPUT_PATH, "X_fMRI.npy"), np.concatenate(X_list))
            np.save(os.path.join(OUTPUT_PATH, "y_fMRI.npy"), np.concatenate(y_list))
            np.save(os.path.join(OUTPUT_PATH, "groups_fMRI.npy"), np.concatenate(g_list))

    verify_saved_data(le)

if __name__ == "__main__":
    build_dataset(subjects=["01"], build_eeg=False, build_fmri=True)