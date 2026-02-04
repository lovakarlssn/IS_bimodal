# augmentations/__init__.py
from .eeg_augment import spatial_shuffle, freq_surrogate, time_slice
from .fmri_augment import spatial_smoothing, intensity_normalization

# Now in main.py you can just do:
# from augmentations import spatial_shuffle