import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config

class EEGNet(nn.Module):
    """
    Standard EEGNet implementation with flexible architectural parameters.
    """
    def __init__(self, nb_classes, Chans=64, Samples=128, 
                 dropoutRate=0.25, kernLength=64, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength//2), bias=False),
            nn.BatchNorm2d(F1, momentum=0.01)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )
        
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
        return self.classifier(x.flatten(1))

class MorletWaveletBlock(nn.Module):
    """Learned or Fixed Morlet Wavelet Transform via Conv1d."""
    def __init__(self, c_in, freqs, sfreq=config.TARGET_FS, n_cycles=3.0, trainable=False):
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
            wavelet_real *= norm; wavelet_imag *= norm
            kernel_size = len(t_axis)
            
            cr = nn.Conv1d(c_in, c_in, kernel_size, stride=1, padding=kernel_size//2, groups=c_in, bias=False)
            ci = nn.Conv1d(c_in, c_in, kernel_size, stride=1, padding=kernel_size//2, groups=c_in, bias=False)
            
            cr.weight.data = torch.tensor(wavelet_real, dtype=torch.float32).view(1, 1, -1).repeat(c_in, 1, 1)
            ci.weight.data = torch.tensor(wavelet_imag, dtype=torch.float32).view(1, 1, -1).repeat(c_in, 1, 1)
            cr.weight.requires_grad = trainable; ci.weight.requires_grad = trainable
            
            self.conv_real.append(cr)
            self.conv_imag.append(ci)

    def forward(self, x):
        out_bands = []
        for i in range(self.n_freqs):
            real, imag = self.conv_real[i](x), self.conv_imag[i](x)
            out_bands.append((real**2 + imag**2).unsqueeze(2))
        return torch.cat(out_bands, dim=2)

class SpectroTemporalTransformer(nn.Module):
    """
    Spectro-Temporal Transformer with flexible wavelet and transformer configurations.
    """
    def __init__(self, nb_classes, Chans=64, Samples=1537, sfreq=config.TARGET_FS,
                 freqs=[6, 10, 20, 30, 45], d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=256, dropout=0.2):
        super(SpectroTemporalTransformer, self).__init__()
        
        self.freqs = freqs
        self.wavelet = MorletWaveletBlock(Chans, self.freqs, sfreq=sfreq)
        self.time_pool = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        
        # Reduced spatial dimension before transformer projection
        self.spatial_pool = nn.Linear(Chans, 37)
        self.d_model = d_model
        self.input_proj = nn.Linear(37, self.d_model)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout, 
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 2000, self.d_model))
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.d_model, nb_classes)

    def forward(self, x):
        if x.dim() == 4: x = x.squeeze(1)
        x = self.wavelet(x)
        x = self.time_pool(x)
        x = self.spatial_pool(x.permute(0, 2, 3, 1))
        x = F.elu(x)
        
        B, n_freqs, n_time, n_feats = x.shape
        x = self.input_proj(x.reshape(B, n_freqs * n_time, n_feats))
        x = x + self.pos_embedding[:, :x.shape[1], :]
        
        x = self.transformer(x).permute(0, 2, 1)
        return self.classifier(self.global_pool(x).squeeze(-1))