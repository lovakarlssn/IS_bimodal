from models.eeg_models import EEGNet, SpectroTemporalTransformer
from models.fmri_models import Simple3DCNN

def get_model(modality, model_name, n_classes, input_shape, arch_config=None):
    """
    Factory function to instantiate models with custom architecture configurations.
    Handles variable input shapes (C, T) vs (1, C, T).
    """
    if arch_config is None:
        arch_config = {}

    if modality == "EEG":
        
        if len(input_shape) == 3:
            C, T = input_shape[1], input_shape[2]
        elif len(input_shape) == 2:
            C, T = input_shape[0], input_shape[1]
        else:
            raise ValueError(f"Unexpected EEG input shape: {input_shape}")

        if model_name == "EEGNet":
            return EEGNet(
                nb_classes=n_classes, 
                Chans=C, 
                Samples=T,
                **arch_config
            )
        elif model_name == "Transformer":
            return SpectroTemporalTransformer(
                nb_classes=n_classes, 
                Chans=C, 
                Samples=T,
                **arch_config
            )
    
    elif modality == "fMRI":
        if model_name == "Simple3DCNN":
            return Simple3DCNN(
                nb_classes=n_classes,
                **arch_config
            )
            
    raise ValueError(f"Unknown model {model_name} for modality {modality}")