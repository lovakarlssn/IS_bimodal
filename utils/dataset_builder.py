import os
import glob
import mne
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


RAW_DATA_PATH = "D:\\OneDrive - Luleå University of Technology\\År 5\\EX\\Data_sub01\\Async_data_preproc_new" 
OUTPUT_PATH = "./data"


def load_subject_raw(subject_id, task="words"):
    """
    Helper: Loads ONE subject's .fif file from disk.
    """
    
    print(f"Looking for subject {subject_id} data in path: {RAW_DATA_PATH}" )
    
    search_pattern = os.path.join(RAW_DATA_PATH, f"sub-{subject_id}", "ses-eeg", "eeg", f"*desc-{task}_epo.fif")
    print(f"   Searching with pattern: {search_pattern}")
    found_files = glob.glob(search_pattern, recursive=True)
    
    if not found_files:
        print(f"   [Warning] No .fif file found for Subject {subject_id}")
        return None, None

    file_path = found_files[0]
    print(f"   -> Loading: {os.path.basename(file_path)}")

    try:
        epochs = mne.read_epochs(file_path, preload=True, verbose=False)
    except Exception as e:
        print(f"   [Error] Failed to load {file_path}: {e}")
        return None, None

    X = epochs.get_data(copy=True) 
    y = epochs.events[:, -1]       

    
    if epochs.info['sfreq'] != config.TARGET_FS:
        print(f"      Resampling {epochs.info['sfreq']} -> {config.TARGET_FS}Hz...")
        epochs.resample(config.TARGET_FS)
        X = epochs.get_data() 

    return X, y

def build_dataset(subjects=["01", "02", "03", "05"]):
    """
    Creates ONE Master Dataset containing all subjects.
    Saves: X_EEG_Master.npy, y_EEG_Master.npy, groups_EEG_Master.npy
    """
    if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)

    print(f"\n[Info] Building Master Dataset for subjects: {subjects}")
    
    X_list = []
    y_list = []
    groups_list = []

    for sub_id in subjects:
        print(f"Processing Subject {sub_id}...")
        X, y = load_subject_raw(sub_id)
        
        if X is None: 
            continue

        X_list.append(X)
        y_list.append(y)
        
        
        
        groups_list.append(np.full(len(y), int(sub_id)))

    if not X_list:
        print("[Error] No data loaded. Check paths.")
        return

    
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    groups_all = np.concatenate(groups_list, axis=0)

    
    np.save(os.path.join(OUTPUT_PATH, "X_EEG.npy"), X_all)
    np.save(os.path.join(OUTPUT_PATH, "y_EEG.npy"), y_all)
    np.save(os.path.join(OUTPUT_PATH, "groups_EEG.npy"), groups_all)

    print(f"\n[Success] Saved Master Dataset.")
    print(f"   Total Trials: {len(y_all)}")
    print(f"   Subjects included: {np.unique(groups_all)}")
    print(f"   Shape: {X_all.shape}")

if __name__ == "__main__":
    
    build_dataset(subjects=["01", "02", "03", "05"])