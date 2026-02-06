import torch

# Device & Training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20

# EEG Architecture Specs
TARGET_FS = 512
TARGET_CHANS = 64
BASE_TRIALS = 320  # Total trials in original Async dataset
DATA_MULTIPLIER = 2
EXPERIMENTS = ["Original", "ChannelsDropout", "FTSurrogate", "TimeReverse", "SmoothTimeMask"]

HYPERPARAMS = {"batch_size": BATCH_SIZE,
               "lr": LEARNING_RATE,
               "epochs": EPOCHS,
               "weight_decay": WEIGHT_DECAY,
               "data_multiplier": DATA_MULTIPLIER
               }

AUG_PARAMS = {
    "channels_dropout": {
        "p_drop": 0.4   
    },
    "freq_surrogate": {
        "phase_noise_max": 6.28  
    },
    "smooth_time_mask": {
        "mask_len_samples": 300  
    }
}