from models.eeg_models import EEGNet, SpectroTemporalTransformer
from models.fmri_models import Simple3DCNN

def get_model(modality, model_name, n_classes, input_shape, arch_config=None):
    """
    Factory function to instantiate models with custom architecture configurations.
    """
    if arch_config is None:
        arch_config = {}

    if modality == "EEG":
        if model_name == "EEGNet":
            return EEGNet(
                nb_classes=n_classes, 
                Chans=input_shape[0], 
                Samples=input_shape[1],
                **arch_config
            )
        elif model_name == "Transformer":
            return SpectroTemporalTransformer(
                nb_classes=n_classes, 
                Chans=input_shape[0], 
                Samples=input_shape[1],
                **arch_config
            )
    
    elif modality == "fMRI":
        if model_name == "Simple3DCNN":
            return Simple3DCNN(
                nb_classes=n_classes,
                **arch_config
            )
            
    raise ValueError(f"Unknown model {model_name} for modality {modality}")