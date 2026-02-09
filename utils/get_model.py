# utils/get_model.py
import models.transformer as transformer
import models.eegnets as eegnets 
import models.fmri_models as fmri_models



def get_model(model_name, n_classes, input_shape):
    if model_name == "Transformer":
        return transformer.SpectroTemporalTransformer(nb_classes=n_classes, input_shape=input_shape)
    elif model_name == "EEGNet":
        return eegnets.EEGNet(nb_classes=n_classes, Chans=input_shape[0], Samples=input_shape[1])
    elif model_name == "CustomEEGNet":
        return eegnets.CustomEEGNet(nb_classes=n_classes, Chans=input_shape[0], Samples=input_shape[1])
    elif model_name == "SimpleFMRI":
        pass
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    