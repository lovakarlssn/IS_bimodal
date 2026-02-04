import torch
import torch.nn as nn

class CompactEEGNet(nn.Module):
    def __init__(self, nb_classes, input_chans=64, input_time=1537):
        super().__init__()
        self.F1, self.D, self.F2 = 8, 2, 16
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(self.F1),
            nn.Conv2d(self.F1, self.F1*self.D, (input_chans, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1*self.D),
            nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.25)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1*self.D, self.F1*self.D, (1, 16), groups=self.F1*self.D, padding=(0, 8), bias=False),
            nn.Conv2d(self.F1*self.D, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(0.25)
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_chans, input_time)
            self.feature_dim = self.block2(self.block1(dummy)).numel()
        self.classifier = nn.Linear(self.feature_dim, nb_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x.view(x.size(0), -1))