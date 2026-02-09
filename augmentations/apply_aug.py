import numpy as np
from .eeg_aug import channels_dropout, freq_surrogate, time_reverse, smooth_time_mask
from .fmri_aug import (
    add_gaussian_noise, 
    random_contrast, 
    random_affine, 
    elastic_transform, 
    cutout_volume,
    intra_class_mixup
)
def get_augmentation(modality, name, X, y, params):
    X_aug = np.zeros_like(X)
    
    # Loop through every sample in the batch
    for i in range(len(X)):
        # Squeeze out the batch and channel dims to get (79, 95, 79)
        # We'll put them back after
        img_3d = np.squeeze(X[i]) 
        
        if name == "SpatialNoise":
            aug_img = add_gaussian_noise(img_3d, params['noise_level'])
        elif name == "RandomAffine":
            aug_img = random_affine(img_3d)
        elif name == "ElasticTransform":
            aug_img = elastic_transform(img_3d)
        elif name == "ROIMasking":
            aug_img = cutout_volume(img_3d)
        else:
            # For things like IntensityScale that work fine on arrays
            aug_img = random_contrast(img_3d)
            
        # Reshape back to (1, 79, 95, 79) and store
        X_aug[i] = aug_img[np.newaxis, ...]

    # Special case for Mixup which NEEDS the whole batch at once
    if name == "IntraClassMixup":
        X_aug, y_aug_labels = intra_class_mixup(X, y)
    else:
        y_aug_labels = y.copy()

    return np.concatenate([X, X_aug]), np.concatenate([y, y_aug_labels])