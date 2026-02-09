import torch
import torch.nn as nn

class Simple3DCNN(nn.Module):
    """
    Flexible 3D CNN for fMRI volume classification.
    """
    def __init__(self, nb_classes, base_filters=16, kernel_size=3):
        super(Simple3DCNN, self).__init__()
        
        padding = kernel_size // 2
        
        self.features = nn.Sequential(
            nn.Conv3d(1, base_filters, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(base_filters, base_filters * 2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(base_filters * 2),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.AdaptiveAvgPool3d((4, 4, 4))
        )
        
        self.classifier = nn.Linear(base_filters * 2 * 4 * 4 * 4, nb_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))