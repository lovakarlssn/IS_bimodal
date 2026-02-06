import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class EEGNet(nn.Module):
    """
    Pytorch Implementation of EEGNet (Lawhern et al., 2018)
    Matches the 'v4' architecture from the official Keras code.
    
    Args:
        nb_classes (int): Number of classification classes.
        Chans (int): Number of channels in your EEG data.
        Samples (int): Number of time points in your EEG data.
        dropoutRate (float): Dropout fraction (default 0.25).
        kernLength (int): Length of temporal convolution (default 64).
        F1 (int): Number of temporal filters (default 8).
        D (int): Number of spatial filters per temporal filter (default 2).
        F2 (int): Number of pointwise filters (default 16).
    """
    def __init__(self, nb_classes, Chans=64, Samples=128, 
                 dropoutRate=0.25, kernLength=64, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        
        # Block 1: Temporal Convolution
        # Keras: Conv2D(F1, (1, kernLength), padding='same')
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength//2), bias=False),
            nn.BatchNorm2d(F1, momentum=0.01)
        )
        
        # Block 2: Depthwise Spatial Convolution
        # Keras: DepthwiseConv2D((Chans, 1), depth_multiplier=D)
        # In PyTorch, we use groups=F1 to achieve depthwise convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )
        
        # Block 3: Separable Convolution
        # Keras: SeparableConv2D = Depthwise Conv + Pointwise Conv
        self.block3 = nn.Sequential(
            # 3a. Depthwise (Spatial)
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            # 3b. Pointwise (1x1)
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )
        
        # Auto-calculate classifier size using a dummy forward pass
        # This fixes the "Linear layer shape mismatch" errors automatically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, Chans, Samples)
            x = self.block1(dummy)
            x = self.block2(x)
            x = self.block3(x)
            self.flatten_dim = x.flatten(1).shape[1]
            
        self.classifier = nn.Linear(self.flatten_dim, nb_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(1)
        return self.classifier(x)


class DeepConvNet(nn.Module):
    """
    DeepConvNet (Schirrmeister et al., 2017)
    A deeper, standard CNN often used as a baseline for EEG.
    """
    def __init__(self, nb_classes, Chans=64, Samples=128, dropoutRate=0.5):
        super(DeepConvNet, self).__init__()
        
        # Block 1: Temporal + Spatial
        self.conv_time = nn.Conv2d(1, 25, (1, 10), bias=False)
        self.conv_spat = nn.Conv2d(25, 25, (Chans, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(25, momentum=0.01)
        self.pool1 = nn.MaxPool2d((1, 3))
        self.drop1 = nn.Dropout(dropoutRate)
        
        # Block 2-4: Standard Conv Blocks
        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, (1, 10), bias=False),
            nn.BatchNorm2d(50, momentum=0.01),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(dropoutRate)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, (1, 10), bias=False),
            nn.BatchNorm2d(100, momentum=0.01),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(dropoutRate)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, (1, 10), bias=False),
            nn.BatchNorm2d(200, momentum=0.01),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(dropoutRate)
        )
        
        # Auto-calculate size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, Chans, Samples)
            x = self.conv_time(dummy)
            x = self.conv_spat(x)
            x = self.drop1(self.pool1(self.bn1(x)))
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            self.flatten_dim = x.flatten(1).shape[1]

        self.classifier = nn.Linear(self.flatten_dim, nb_classes)

    def forward(self, x):
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.drop1(self.pool1(self.bn1(x)))
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.flatten(1)
        return self.classifier(x)

class CustomEEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128, 
                 dropoutRate=0.25, kernLength=256, F1=8, D=2, F2=16):
        super(CustomEEGNet, self).__init__()
        
        # Block 1: Temporal Convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength//2), bias=False),
            nn.BatchNorm2d(F1, momentum=0.01)
        )
        
        # Block 2: Depthwise Spatial Convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )
        
        # Block 3: Separable Convolution
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, Chans, Samples)
            x = self.block1(dummy)
            x = self.block2(x)
            x = self.block3(x)
            self.flatten_dim = x.flatten(1).shape[1]
            
        self.classifier = nn.Linear(self.flatten_dim, nb_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(1)
        return self.classifier(x)
    
    

class ShallowConvNet(nn.Module):
    """
    ShallowConvNet (Schirrmeister et al., 2017).
    Specifically designed for capturing power spectral features.
    
    Args:
        nb_classes (int): Number of classification classes.
        Chans (int): Number of channels.
        Samples (int): Number of time points.
        dropoutRate (float): Dropout fraction.
    """
    def __init__(self, nb_classes, Chans=64, Samples=128, dropoutRate=0.5):
        super(ShallowConvNet, self).__init__()
        
        # 1. Temporal Convolution: captures frequency info
        # Filter size (1, 25) is standard, but can be increased for 512Hz
        self.conv_time = nn.Conv2d(1, 40, (1, 25), bias=False)
        
        # 2. Spatial Convolution: captures cross-channel relationships
        self.conv_spat = nn.Conv2d(40, 40, (Chans, 1), bias=False)
        
        self.bn1 = nn.BatchNorm2d(40, momentum=0.1)
        
        # 3. Square-Pool-Log Activation Sequence
        # This part mimics the power calculation of the EEG signal
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.drop = nn.Dropout(dropoutRate)
        
        # Auto-calculate the flatten dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 1, Chans, Samples)
            x = self.conv_time(dummy)
            x = self.conv_spat(x)
            x = self.bn1(x)
            x = x**2 # Square
            x = self.pool(x)
            x = torch.log(torch.clamp(x, min=1e-6)) # Log
            self.flatten_dim = x.flatten(1).shape[1]

        self.classifier = nn.Linear(self.flatten_dim, nb_classes)

    def forward(self, x):
        # Layer 1: Conv + Spatial Conv
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn1(x)
        
        # Layer 2: Square -> Pool -> Log
        # This extracts the log-variance (power) of the signal
        x = x**2 
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        
        # Layer 3: Flatten + Classify
        x = self.drop(x)
        x = x.flatten(1)
        return self.classifier(x)
    
    
