# augmentations/__init__.py

from .eeg_aug import (
    channels_dropout, 
    freq_surrogate, 
    time_reverse, 
    smooth_time_mask
)

# fMRI Augmentations (Using the updated names)
from .fmri_aug import (
    add_gaussian_noise,   
    random_contrast,      
    random_affine,
    elastic_transform,
    cutout_volume,
    intra_class_mixup
)

# Unified Entry Point
from .apply_aug import get_augmentation