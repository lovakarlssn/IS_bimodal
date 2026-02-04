import torch

# Device & Training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50

# EEG Architecture Specs
TARGET_FS = 512
TARGET_CHANS = 64
BASE_TRIALS = 320  # Total trials in original Async dataset

# Augmentation Tuning (Tweak these for your thesis comparison)
AUG_PARAMS = {
    "spatial_shuffle": {"n_swaps": 2},
    "time_slice": {"slice_len": 716, "n_slices": 2, "stride": None},
    "freq_surrogate": {"phase_noise_std": 0.5}
}