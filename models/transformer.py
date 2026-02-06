import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MorletWaveletBlock(nn.Module):
    """
    Learned or Fixed Morlet Wavelet Transform implemented as a Conv1d layer.
    Replicates the 'Input Wavelet Maps' step from Milyani & Attar (2025).
    """
    def __init__(self, c_in, freqs, sfreq=250, n_cycles=3.0, trainable=False):
        super(MorletWaveletBlock, self).__init__()
        self.freqs = freqs
        self.n_freqs = len(freqs)
        
        # We process Real and Imaginary parts separately to compute power
        self.conv_real = nn.ModuleList()
        self.conv_imag = nn.ModuleList()
        
        for f in freqs:
            # Create Morlet wavelet kernel
            sigma = n_cycles / (2.0 * np.pi * f)
            t_axis = np.arange(-4*sigma, 4*sigma, 1/sfreq)
            envelope = np.exp(-0.5 * (t_axis / sigma)**2)
            
            wavelet_real = np.cos(2 * np.pi * f * t_axis) * envelope
            wavelet_imag = np.sin(2 * np.pi * f * t_axis) * envelope
            
            # Normalize
            norm = 1.0 / np.sqrt(0.5 * np.sum(t_axis**2))
            wavelet_real *= norm
            wavelet_imag *= norm
            
            kernel_size = len(t_axis)
            
            # Initialize Conv1d weights
            # Shape: (Out, In/Groups, Kernel) -> (Channels, 1, Kernel) with Groups=Channels
            w_real = torch.tensor(wavelet_real, dtype=torch.float32).view(1, 1, -1).repeat(c_in, 1, 1)
            w_imag = torch.tensor(wavelet_imag, dtype=torch.float32).view(1, 1, -1).repeat(c_in, 1, 1)
            
            # Depthwise convolution (groups=c_in) treats each channel independently
            cr = nn.Conv1d(c_in, c_in, kernel_size, stride=1, padding=kernel_size//2, groups=c_in, bias=False)
            ci = nn.Conv1d(c_in, c_in, kernel_size, stride=1, padding=kernel_size//2, groups=c_in, bias=False)
            
            cr.weight.data = w_real
            ci.weight.data = w_imag
            
            # The paper implies fixed decomposition, but making it trainable is an option
            cr.weight.requires_grad = trainable
            ci.weight.requires_grad = trainable
            
            self.conv_real.append(cr)
            self.conv_imag.append(ci)

    def forward(self, x):
        # Input: (Batch, Channels, Time)
        out_bands = []
        for i in range(self.n_freqs):
            real = self.conv_real[i](x)
            imag = self.conv_imag[i](x)
            # Power calculation
            power = real**2 + imag**2
            out_bands.append(power.unsqueeze(2)) # (Batch, Channels, 1, Time)
            
        return torch.cat(out_bands, dim=2) # (Batch, Channels, Freqs, Time)

class SpectroTemporalTransformer(nn.Module):
    """
    Spectro-Temporal Transformer adapted for 1500+ timepoints.
    Based on Milyani & Attar (2025)
    
    Architecture:
    1. Wavelet Transform (5 bands)
    2. Spatial Pooling (Channels -> 37) 
    3. Tokenization (Flatten Freq & Time)
    4. Transformer Encoder (4 blocks, 8 heads) 
    5. Classification
    """
    def __init__(self, nb_classes, Chans=64, Samples=1537, sfreq=250):
        super(SpectroTemporalTransformer, self).__init__()
        
        # 1. Wavelet Features
        # Centers approx: Theta, Alpha, Low Beta, High Beta, Gamma
        self.freqs = [6, 10, 20, 30, 45] 
        self.wavelet = MorletWaveletBlock(Chans, self.freqs, sfreq=sfreq)
        
        # 2. Downsampling 
        # Paper reduced 513 -> 129 (Factor ~4). 
        self.pool_factor = 4
        self.time_pool = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        
        # 3. Spatial Pooling
        # Paper reduced 73 -> 37 channels. We reduce Chans -> 37.
        target_chans = 37
        self.spatial_pool = nn.Linear(Chans, target_chans)
        
        # 4. Transformer Setup
        # Token dim is target_chans (37). 
        # Since 37 is prime and not divisible by 8 (heads), we project to 128 units.
        self.d_model = 128
        self.input_proj = nn.Linear(target_chans, self.d_model)
        
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=8, # [cite: 167]
            dim_feedforward=256, 
            dropout=0.2, # Paper used 0.5 dropout generally
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=4) 
        
        # Positional Encoding
        # Max sequence length: Freqs (5) * (Samples // 4)
        max_len = len(self.freqs) * (Samples // self.pool_factor) + 50 # +buffer
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, self.d_model))
        
        # 5. Classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.d_model, nb_classes)

    def forward(self, x):
        # x input from StochasticDataset: (Batch, 1, Channels, Time)
        if x.dim() == 4:
            x = x.squeeze(1) # -> (Batch, Channels, Time)
            
        # --- 1. Wavelet Extraction ---
        # Out: (Batch, Channels, Freqs, Time)
        x = self.wavelet(x) 
        
        # --- 2. Downsampling ---
        # Pooling over Time dimension (last dim)
        x = self.time_pool(x) 
        
        # --- 3. Spatial Pooling ---
        # Permute to (Batch, Freqs, Time, Channels) for Linear layer
        x = x.permute(0, 2, 3, 1) 
        x = self.spatial_pool(x) # -> (Batch, Freqs, Time, 37)
        x = F.elu(x) # Activation not specified in paper text, but standard for deep layers
        
        # --- 4. Tokenization ---
        # Flatten Freqs and Time into Sequence
        B, F, T, C = x.shape
        x = x.reshape(B, F*T, C) # (Batch, SeqLen, 37)
        
        # Project to 128 dim for multi-head attention
        x = self.input_proj(x) # (Batch, SeqLen, 128)
        
        # Add Positional Encoding
        seq_len = x.shape[1]
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # --- 5. Transformer ---
        x = self.transformer(x)
        
        # --- 6. Classification ---
        # Global Average Pooling over Sequence dimension
        x = x.permute(0, 2, 1) # (Batch, 128, SeqLen)
        x = self.global_pool(x).squeeze(-1) # (Batch, 128)
        
        return self.classifier(x)