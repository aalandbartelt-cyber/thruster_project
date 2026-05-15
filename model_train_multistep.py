"""
model_train_multistep.py — Multi-step ahead prediction with Attention-LSTM
Predicts thrust & mfr at 4 future horizons: t+1, t+5, t+10, t+20
"""
import os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd, numpy as np
from data_pipeline_v2 import _build_features, THRUST_SCALE, MFR_MAX, INPUT_DIM

INPUT_LEN = 200; MAX_HORIZON = 20; SEQ_LEN = INPUT_LEN + MAX_HORIZON
HORIZONS = [1, 5, 10, 20]; HORIZON_WEIGHTS = [0.5, 0.3, 0.15, 0.05]
W_THRUST = 0.7; W_MFR = 0.3

class MultiStepDataset(Dataset):
    def __init__(self, md, dr, input_len=200, max_horizon=20, stride=100, file_filter=None):
        self.input_len = input_len; self.max_horizon = max_horizon
        self.seq_len = input_len + max_horizon; self.samples = []
        df = pd.read_csv(md); df['filename'] = df['filename'].str.strip()
        df = df[df['anomalous']==False]
        loaded = skipped = 0
        for _, r in df.iterrows():
            fn = r['filename']
            if file_filter is not None and fn not in file_filter: continue
            fp = os.path.join(dr, fn)
            if not os.path.exists(fp): continue
            try: d = pd.read_csv(fp)
            except: continue
            T = len(d)
            if T < self.seq_len: skipped += 1; continue
            for s in range(0, T - self.seq_len + 1, stride):
                xf, yf, _ = _build_features(d.iloc[s:s+self.seq_len], r)
                self.samples.append((xf[:input_len].astype(np.float32),
                                     yf.astype(np.float32)))
            loaded += 1
        print(f"  Loaded {loaded} files -> {len(self.samples)} windows (skipped {skipped} short)")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]; return torch.from_numpy(x), torch.from_numpy(y)

class DualOutputLSTMMultiStep(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=256, num_layers=2, n_heads=4, n_horizons=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(),
                                nn.Dropout(0.05), nn.Linear(128, n_horizons*2))
    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        attn_out, _ = self.attention(out, out, out)
        return self.fc(self.ln(out + attn_out))

def multistep_loss(pred, target):
    B, T, _ = pred.shape; total = torch.tensor(0.0, device=pred.device)
    mse = nn.MSELoss()
    for i, (h, w) in enumerate(zip(HORIZONS, HORIZON_WEIGHTS)):
        vl = T - h
        if vl <= 0: continue
        total += w * (W_THRUST * mse(pred[:,:vl,2*i], target[:,h:T,0]) +
                      W_MFR * mse(pred[:,:vl,2*i+1], target[:,h:T,1]))
    return total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\nMulti-step: horizons={HORIZONS} weights={HORIZON_WEIGHTS}")
    md = "data/metadata.csv"; dr = "data/dataset/dataset/train/"
    df = pd.read_csv(md); df['filename'] = df['filename'].str.strip()
    df = df[(df['anomalous']==False) & (df['sn']<=12)]
    files = df['filename'].tolist(); np.random.seed(42); np.random.shuffle(files)
    sp = int(len(files)*0.8)
    train_f = set(files[:sp]); val_f = set(files[sp:])
    print(f"Train: {len(train_f)} files, Val: {len(val_f)} files")
    print("Building train dataset...")
    train_ds = MultiStepDataset(md, dr, file_filter=train_f)
    print("Building val dataset...")
    val_ds = MultiStepDataset(md, dr, stride=200, file_filter=val_f)
    train_ld = DataLoader(train_ds, 64, True); val_ld = DataLoader(val_ds, 64, False)
    model = DualOutputLSTMMultiStep().to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    opt = optim.Adam(model.parameters(), 1e-3, weight_decay=2e-5)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.5, 5, min_lr=1e-5)
    os.makedirs("outputs/models/v2", exist_ok=True)
    best = float('inf'); no_imp = 0; pat = 20
    print(f"Training {100} epochs (early stop={pat})...")
    for ep in range(100):
        model.train(); tl = 0
        for x,y in train_ld: x,y=x.to(device),y.to(device); opt.zero_grad()
        l=multistep_loss(model(x),y); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tl+=l.item()
        model.eval(); vl=0
        with torch.no_grad():
            for x,y in val_ld: x,y=x.to(device),y.to(device); vl+=multistep_loss(model(x),y).item()
        tl/=len(train_ld); vl/=len(val_ld); sch.step(vl)
        if vl<best: best=vl; torch.save(model.state_dict(),"outputs/models/v2/dual_output_lstm_multistep.pth"); no_imp=0; print(f"  -> best (vl={vl:.6f})")
        else: no_imp+=1
        print(f"Epoch [{ep+1:3d}/100] train={tl:.6f} val={vl:.6f}")
        if no_imp>=pat: print(f"\nEarly stop @{ep+1}"); break
    print(f"\nBest val_loss={best:.6f}"); print("Done.")

if __name__=="__main__": main()
