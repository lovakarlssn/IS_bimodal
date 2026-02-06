# augmentations/__init__.py
# augmentations/__init__.py
from .eeg_aug import (
    channels_dropout, 
    freq_surrogate, 
    time_reverse, 
    smooth_time_mask, 
    get_augmentation
)