"""
model_train_v2.py — 16维输入 + 双输出 LSTM 训练
M 负责，dev-m 分支
基于 v1_baseline/model_train.py 改写，核心变更：
  输入 3→16, 输出 1→2, 损失加权, Dataset 全量读入内存滑动窗口
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from data_pipeline_v2 import _build_features, THRUST_SCALE, MFR_MAX, INPUT_DIM, save_meta_info

# ---- Dataset ----
class ThrusterDataset(Dataset):
    def __init__(self, metadata_path, train_dir, seq_len=200, stride=100, file_filter=None):
        """
        file_filter: 可选 set of filenames，用于 train/val split；
                     如果为 None，则加载 metadata 中全部有效文件。
        """
        self.seq_len = seq_len
        self.samples = []

        df_meta = pd.read_csv(metadata_path)
        df_meta['filename'] = df_meta['filename'].str.strip()
        df_meta = df_meta[df_meta['anomalous'] == False]

        loaded = 0
        skipped_short = 0
        for _, row in df_meta.iterrows():
            fname = row['filename']
            if file_filter is not None and fname not in file_filter:
                continue
            fpath = os.path.join(train_dir, fname)
            if not os.path.exists(fpath):
                continue
            try:
                df = pd.read_csv(fpath)
                T = len(df)
                if T < seq_len:
                    skipped_short += 1
                    continue
                for start in range(0, T - seq_len + 1, stride):
                    x, y, _ = _build_features(df.iloc[start:start + seq_len], row)
                    self.samples.append((x, y))
                loaded += 1
            except Exception:
                continue

        print(f"  Loaded {loaded} files → {len(self.samples)} windows "
              f"(skipped {skipped_short} files < {seq_len} steps)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


# ---- 模型 ----
class DualOutputLSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        return self.fc(out)


# ---- 加权损失 ----
def dual_loss(pred, target, w_thrust=0.7, w_mfr=0.3):
    mse = nn.MSELoss()
    return w_thrust * mse(pred[:, :, 0], target[:, :, 0]) + \
           w_mfr    * mse(pred[:, :, 1], target[:, :, 1])


# ---- 主训练流程 ----
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    metadata_path = "data/metadata.csv"
    train_dir     = "data/dataset/dataset/train/"

    # -- Train/val split：取 80% 文件做训练，20% 做验证 --
    df_all = pd.read_csv(metadata_path)
    df_all['filename'] = df_all['filename'].str.strip()
    df_all = df_all[df_all['anomalous'] == False]
    df_all = df_all[df_all['sn'] <= 12]      # ground-test SN01-SN12 only; SN13-SN24 files not in train_dir
    all_files = df_all['filename'].tolist()
    np.random.seed(42)
    np.random.shuffle(all_files)
    split = int(len(all_files) * 0.8)
    train_files = set(all_files[:split])
    val_files   = set(all_files[split:])

    print(f"\nTrain files: {len(train_files)}, Val files: {len(val_files)}")

    # -- 构建 Dataset --
    print("\nBuilding train dataset...")
    train_dataset = ThrusterDataset(metadata_path, train_dir,
                                    seq_len=200, stride=100, file_filter=train_files)
    print("Building val dataset...")
    val_dataset = ThrusterDataset(metadata_path, train_dir,
                                  seq_len=200, stride=200, file_filter=val_files)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)

    # -- 模型、优化器 --
    model = DualOutputLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # -- 训练循环 --
    epochs = 100
    print(f"\nTraining {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        # --- train ---
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = dual_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # --- val ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += dual_loss(pred, y).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        print(f"Epoch [{epoch+1:3d}/{epochs}]  "
              f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    # -- 保存 --
    save_dir = "outputs/models/v2"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "dual_output_lstm_v2.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved → {save_path}")

    save_meta_info()

    # -- 快速 sanity check --
    print("\nSanity check:")
    model.eval()
    x_sample, y_sample = train_dataset[0]
    with torch.no_grad():
        pred = model(x_sample.unsqueeze(0).to(device))
    print(f"  input:  {x_sample.shape}  →  output: {pred.squeeze(0).shape}")
    print(f"  target thrust range: [{y_sample[:, 0].min():.3f}, {y_sample[:, 0].max():.3f}] "
          f"(x{THRUST_SCALE}=[{y_sample[:, 0].min()*THRUST_SCALE:.1f}, {y_sample[:, 0].max()*THRUST_SCALE:.1f}] N)")
    print(f"  target mfr    range: [{y_sample[:, 1].min():.3f}, {y_sample[:, 1].max():.3f}] "
          f"(x{MFR_MAX}=[{y_sample[:, 1].min()*MFR_MAX:.1f}, {y_sample[:, 1].max()*MFR_MAX:.1f}] mg/s)")


if __name__ == "__main__":
    main()
