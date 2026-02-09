import torch

# --- SYSTEM ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "./results"

# --- EXPERIMENT CONTROL ---
MODALITY = "EEG"           # "EEG" or "fMRI"
EXP_MODE = "BOTH"          # "DL", "ML", or "BOTH"
DATA_MODE = "loso"         # "loso" or "single"
SUBJECT_ID = 1             # Only for single mode

# --- DATASET ---
TARGET_FS = 512
TARGET_CHANS = 64
BASE_TRIALS = 320 

# --- DATA PATHS ---
DATA_DIR = "./data"

# --- MODELS TO RUN (Default List) ---
DL_MODELS = ["SpectroTemporalTransformer", "EEGNet"] 
ML_MODELS = ["SVM", "RF"]

# --- AUGMENTATIONS TO RUN (Default List) ---
# The script will run these one by one.
EXPERIMENTS = ["Original", "ChannelsDropout", "FTSurrogate", "TimeReverse", "SmoothTimeMask"]

# --- HYPERPARAMETERS (Single Configuration) ---
# NO LISTS HERE. Just the specific values for this run.
HYPERPARAMS = {
    "batch_size": 32,
    "lr": 1e-4,
    "epochs": 20,
    "weight_decay": 1e-4,
    "data_multiplier": 4 
}

# --- AUGMENTATION PARAMETERS ---
# Specific settings for the augmentations when they run
AUG_PARAMS = {
    "channels_dropout": { "p_drop": 0.4 },
    "freq_surrogate": { "phase_noise_max": 6.28 },
    "smooth_time_mask": { "mask_len_samples": 300 },
    "time_reverse": { "active": True },
    "sign_flip": { "active": True }
}