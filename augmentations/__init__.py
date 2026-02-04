# augmentations/__init__.py
from .eeg_aug import spatial_shuffle, freq_surrogate, time_slice, get_augmentation
from .fmri_aug import spatial_smoothing, intensity_normalization

# Now in main.py you can just do:
# from augmentations import spatial_shuffle