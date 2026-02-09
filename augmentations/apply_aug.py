# augmentations/apply_aug.py
import numpy as np
from .eeg_aug import channels_dropout, freq_surrogate, time_reverse, smooth_time_mask
from .fmri_aug import spatial_noise, intensity_scale

def get_augmentation(modality, name, X, y, params):
    """
    Unified entry point for data augmentation. 
    Concatenates augmented data to the original batch.
    """
    if params is None: params = {}

    if modality == "EEG":
        if name == "ChannelsDropout":
            X_aug = channels_dropout(X, **params.get("channels_dropout", {}))
        elif name == "FTSurrogate":
            X_aug = freq_surrogate(X, **params.get("freq_surrogate", {}))
        elif name == "TimeReverse":
            X_aug = time_reverse(X)
        elif name == "SmoothTimeMask":
            X_aug = smooth_time_mask(X, **params.get("smooth_time_mask", {}))
        else: 
            return X, y # Return original if no valid aug name
    
    elif modality == "fMRI":
        if name == "SpatialNoise":
            X_aug = spatial_noise(X, **params.get("spatial_noise", {}))
        elif name == "IntensityScale":
            X_aug = intensity_scale(X, **params.get("intensity_scale", {}))
        else: 
            return X, y
    
    else:
        return X, y

    # Return concatenated dataset (Original + Augmented)
    return np.concatenate([X, X_aug]), np.concatenate([y, y])