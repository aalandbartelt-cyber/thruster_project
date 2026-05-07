import torch.nn as nn

class Attention_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=256, num_layers=1, batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.fc = nn.Linear(256, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, attn_weights = self.attn(lstm_out, lstm_out, lstm_out)
        out = lstm_out + attn_out
        return self.fc(out), attn_weights
    


    