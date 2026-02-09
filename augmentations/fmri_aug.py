import numpy as np
from scipy import ndimage


def add_gaussian_noise(X, noise_level=0.1):
    """
    Adds Gaussian noise to the 3D Beta Map.
    """
    # Calculate std only on non-zero brain voxels to avoid under-scaling
    sigma = np.std(X[X != 0]) * noise_level 
    if sigma == 0: sigma = 1.0
    
    noise = np.random.normal(0, sigma, X.shape)
    return X + noise

def random_contrast(X, contrast_range=(0.8, 1.2)):
    """
    Non-linear intensity scaling.
    Scales deviations from the mean rather than just multiplying.
    """
    factor = np.random.uniform(contrast_range[0], contrast_range[1])
    mean = np.mean(X)
    return (X - mean) * factor + mean

def random_affine(X, max_angle=2, max_shift=1):
    """
    Applies VERY small random rotations and translations.
    Constraint: fMRI data is in MNI space. Large rotations invalidate the anatomy.
    We only allow subtle jitter to simulate registration noise.
    """
    # 1. Rotation (Restricted to very small angles, e.g., < 2 degrees)
    angle_x = np.random.uniform(-max_angle, max_angle)
    angle_y = np.random.uniform(-max_angle, max_angle)
    angle_z = np.random.uniform(-max_angle, max_angle)
    
    X_aug = ndimage.rotate(X, angle_x, axes=(0, 1), reshape=False, mode='nearest')
    X_aug = ndimage.rotate(X_aug, angle_y, axes=(1, 2), reshape=False, mode='nearest')
    X_aug = ndimage.rotate(X_aug, angle_z, axes=(0, 2), reshape=False, mode='nearest')
    
    # 2. Translation (Sub-voxel or single voxel shifts)
    shift_x = np.random.randint(-max_shift, max_shift + 1)
    shift_y = np.random.randint(-max_shift, max_shift + 1)
    shift_z = np.random.randint(-max_shift, max_shift + 1)
    
    X_aug = ndimage.shift(X_aug, shift=[shift_x, shift_y, shift_z], mode='nearest')
    
    return X_aug

def elastic_transform(X, alpha=8, sigma=3):
    """
    Applies small non-linear deformations (Anatomical Jitter).
    Basis: Brains have natural anatomical variability that affine transforms can't capture.
    This morphs the grid slightly using random displacement fields.
    
    Args:
        alpha: Scaling factor for deformation intensity.
        sigma: Smoothing factor (elasticity).
    """
    random_state = np.random.RandomState(None)
    shape = X.shape
    
    # Generate random displacement fields
    dx = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    
    # Create meshgrid
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    
    # Map coordinates
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1)), np.reshape(z+dz, (-1, 1))
    
    # Interpolate
    X_aug = ndimage.map_coordinates(X, indices, order=1, mode='nearest').reshape(shape)
    return X_aug

def cutout_volume(X, patch_size=4, num_cutouts=3):
    """
    Sets random small 3D cubic blocks of the volume to zero (ROI Masking).
    Default patch_size=4 (4x4x4 voxels) as requested.
    Forces model to learn distributed patterns.
    """
    X_aug = X.copy()
    x_len, y_len, z_len = X.shape
    
    for _ in range(num_cutouts):
        # Random center
        cx = np.random.randint(0, x_len)
        cy = np.random.randint(0, y_len)
        cz = np.random.randint(0, z_len)
        
        # Calculate limits with clipping
        x1 = np.clip(cx - patch_size // 2, 0, x_len)
        x2 = np.clip(cx + patch_size // 2, 0, x_len)
        y1 = np.clip(cy - patch_size // 2, 0, y_len)
        y2 = np.clip(cy + patch_size // 2, 0, y_len)
        z1 = np.clip(cz - patch_size // 2, 0, z_len)
        z2 = np.clip(cz + patch_size // 2, 0, z_len)
        
        X_aug[x1:x2, y1:y2, z1:z2] = 0
        
    return X_aug

def intra_class_mixup(X, y, alpha=0.2):
    """
    Linearly interpolates between two images of the SAME class.
    Formula: C = lambda * A + (1 - lambda) * B
    """
    X_aug = np.zeros_like(X)
    
    for i in range(len(X)):
        # 1. Find all indices that belong to the same class as item 'i'
        current_class = y[i]
        candidates = np.where(y == current_class)[0]
        
        # 2. Exclude self if possible (unless it's the only one)
        if len(candidates) > 1:
            candidates = candidates[candidates != i]
            
        # 3. Pick a random partner
        idx_j = np.random.choice(candidates)
        
        # 4. Generate Mixup Lambda
        # Using Beta distribution is standard, or uniform if preferred
        lam = np.random.beta(alpha, alpha) 
        
        # 5. Mix
        X_aug[i] = lam * X[i] + (1 - lam) * X[idx_j]
        
    # Labels remain the same since we mixed same-class items
    return X_aug, y