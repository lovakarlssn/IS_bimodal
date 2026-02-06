import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F  # Global 'F' for functions

class MorletWaveletBlock(nn.Module):
    """
    Learned or Fixed Morlet Wavelet Transform implemented as a Conv1d layer.
    """
    def __init__(self, c_in, freqs, sfreq=250, n_cycles=3.0, trainable=False):
        super(MorletWaveletBlock, self).__init__()
        self.freqs = freqs
        self.n_freqs = len(freqs)
        
        self.conv_real = nn.ModuleList()
        self.conv_imag = nn.ModuleList()
        
        for f in freqs:
            sigma = n_cycles / (2.0 * np.pi * f)
            t_axis = np.arange(-4*sigma, 4*sigma, 1/sfreq)
            envelope = np.exp(-0.5 * (t_axis / sigma)**2)
            
            wavelet_real = np.cos(2 * np.pi * f * t_axis) * envelope
            wavelet_imag = np.sin(2 * np.pi * f * t_axis) * envelope
            
            norm = 1.0 / np.sqrt(0.5 * np.sum(t_axis**2))
            wavelet_real *= norm
            wavelet_imag *= norm
            
            kernel_size = len(t_axis)
            
            w_real = torch.tensor(wavelet_real, dtype=torch.float32).view(1, 1, -1).repeat(c_in, 1, 1)
            w_imag = torch.tensor(wavelet_imag, dtype=torch.float32).view(1, 1, -1).repeat(c_in, 1, 1)
            
            cr = nn.Conv1d(c_in, c_in, kernel_size, stride=1, padding=kernel_size//2, groups=c_in, bias=False)
            ci = nn.Conv1d(c_in, c_in, kernel_size, stride=1, padding=kernel_size//2, groups=c_in, bias=False)
            
            cr.weight.data = w_real
            ci.weight.data = w_imag
            cr.weight.requires_grad = trainable
            ci.weight.requires_grad = trainable
            
            self.conv_real.append(cr)
            self.conv_imag.append(ci)

    def forward(self, x):
        out_bands = []
        for i in range(self.n_freqs):
            real = self.conv_real[i](x)
            imag = self.conv_imag[i](x)
            power = real**2 + imag**2
            out_bands.append(power.unsqueeze(2)) 
            
        return torch.cat(out_bands, dim=2) 

class SpectroTemporalTransformer(nn.Module):
    """
    Spectro-Temporal Transformer adapted for 1500+ timepoints.
    """
    def __init__(self, nb_classes, Chans=64, Samples=1537, sfreq=250):
        super(SpectroTemporalTransformer, self).__init__()
        
        # 1. Wavelet Features
        self.freqs = [6, 10, 20, 30, 45] 
        self.wavelet = MorletWaveletBlock(Chans, self.freqs, sfreq=sfreq)
        
        # 2. Downsampling 
        self.pool_factor = 4
        self.time_pool = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        
        # 3. Spatial Pooling
        target_chans = 37
        self.spatial_pool = nn.Linear(Chans, target_chans)
        
        # 4. Transformer Setup
        self.d_model = 128
        self.input_proj = nn.Linear(target_chans, self.d_model)
        
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=8, 
            dim_feedforward=256, 
            dropout=0.2, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=4) 
        
        # Positional Encoding
        max_len = len(self.freqs) * (Samples // self.pool_factor) + 50
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, self.d_model))
        
        # 5. Classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.d_model, nb_classes)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
            
        # --- 1. Wavelet Extraction ---
        x = self.wavelet(x) 
        
        # --- 2. Downsampling ---
        x = self.time_pool(x) 
        
        # --- 3. Spatial Pooling ---
        x = x.permute(0, 2, 3, 1) 
        x = self.spatial_pool(x) 
        x = F.elu(x) # <--- This works now because we don't shadow F below
        
        # --- 4. Tokenization ---
        # FIXED: Changed variable name from 'F' to 'n_freqs'
        B, n_freqs, n_time, n_feats = x.shape
        x = x.reshape(B, n_freqs * n_time, n_feats) 
        
        # Project to 128 dim
        x = self.input_proj(x) 
        
        # Add Positional Encoding
        seq_len = x.shape[1]
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # --- 5. Transformer ---
        x = self.transformer(x)
        
        # --- 6. Classification ---
        x = x.permute(0, 2, 1) 
        x = self.global_pool(x).squeeze(-1)
        
        return self.classifier(x)