import torch.nn as nn

class TSTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(3, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True, dim_feedforward=256)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.input_proj(x)
        out = self.transformer(x)
        return self.fc(out)