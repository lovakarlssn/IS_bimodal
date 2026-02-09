# augmentations/fmri_aug.py
import numpy as np

def spatial_noise(X, noise_level=0.2):
    """
    Adds Gaussian noise to fMRI volumes.
    noise_level: Standard deviation of noise relative to signal std (e.g. 0.2 = 20% of signal std).
    """
    # Calculate noise sigma relative to the actual signal's standard deviation
    sigma = np.std(X) * noise_level
    if sigma == 0: sigma = 1.0 # Safety for flat signals
    
    return X + np.random.normal(0, sigma, X.shape)

def intensity_scale(X, scale_range=(0.7, 1.3)):
    """
    Randomly scales voxel intensities.
    Default range increased to +/- 30% for better visibility.
    """
    factor = np.random.uniform(scale_range[0], scale_range[1])
    return X * factor