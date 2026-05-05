import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=64, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, 1)
        
    def forward(self, x):
        x = x.transpose(1, 2) 
        c = self.relu(self.conv(x))
        c = c.transpose(1, 2) 
        out, _ = self.lstm(c)
        return self.fc(out)