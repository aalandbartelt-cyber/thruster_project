import torch.nn as nn

class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=3, padding=1, dilation=1), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=4, dilation=4), nn.ReLU()
        )
        self.fc = nn.Linear(256, 1)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.net(x)
        return self.fc(out.transpose(1, 2))