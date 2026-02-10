# augmentations/__init__.py
from .eeg_aug import channels_dropout, freq_surrogate, time_reverse, smooth_time_mask
from .fmri_aug import spatial_noise, intensity_scale
from .apply_aug import get_augmentation