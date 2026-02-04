import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import pickle
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
import mne
from scipy.signal import resample
import pywt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nilearn.image import resample_img
import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_eeg_trials(epochs_or_file, events_tsv, include_labels=None, skip_first=0):
    if isinstance(epochs_or_file, str):
        epochs = mne.read_epochs(epochs_or_file, preload=True, verbose=False)
    else:
        epochs = epochs_or_file.copy()
    events = pd.read_csv(events_tsv, sep='\t')
    if skip_first > 0:
        events = events.iloc[skip_first:].reset_index(drop=True)
    if include_labels is None:
        include_labels = [l.lower() for l in events['trial_type'].unique() if l.lower() not in ['rest','fixation']]

    trials, labels, onsets = [], [], []
    for idx, row in events.iterrows():
        lab = str(row['trial_type']).lower()
        if lab not in include_labels or idx >= len(epochs): continue
        data = epochs[idx].get_data()[0]
        trials.append(data)
        labels.append(lab)
        onsets.append(row['onset'])

    return trials, labels, np.array(onsets)


# -----------------------------
#  Trial Alignment
# -----------------------------
def align_trials_async(fmri_trials, fmri_labels, eeg_trials, eeg_labels):
    pairs = []
    eeg_index_by_label = {}
    for j, lab in enumerate(eeg_labels):
        eeg_index_by_label.setdefault(lab, []).append(j)
    fmri_aligned, eeg_aligned, labels = [], [], []
    for i, lab in enumerate(fmri_labels):
        pool = eeg_index_by_label.get(lab, [])
        if pool:
            j = pool.pop(0)
            fmri_aligned.append(fmri_trials[i])
            eeg_aligned.append(eeg_trials[j])
            labels.append(lab)
    print(f"[INFO] Aligned {len(labels)} trials asynchronously")
    return fmri_aligned, eeg_aligned, labels



class BimodalDataset(Dataset):
    def __init__(self, fmri, eeg, labels, label_encoder=None):
        self.fmri = fmri
        self.eeg = eeg

        if label_encoder is None:
            self.encoder = LabelEncoder()
            self.labels = self.encoder.fit_transform(labels)
        else:
            self.encoder = label_encoder
            self.labels = self.encoder.transform(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "fmri": torch.tensor(self.fmri[i], dtype=torch.float32),
            "eeg": torch.tensor(self.eeg[i], dtype=torch.float32),
            "label": torch.tensor(self.labels[i], dtype=torch.long)
        }

    def inverse_labels(self, encoded):
        """Return original labels from encoded integers"""
        return self.encoder.inverse_transform(encoded)
    
    