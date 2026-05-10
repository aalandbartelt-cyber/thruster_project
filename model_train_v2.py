"""
model_train_v2.py — 17维输入 + SN embedding + 双输出 Attention-LSTM 训练
M 负责，dev-m 分支

v2.1 改进：
  - 加入 SN embedding (25 vocab × 8 dim)，区分不同推进器制造差异
  - 训练时 12% 概率 mask 为 UNKNOWN token，训练未见 SN 的泛化能力
  - 损失函数加入方差匹配项 (w_std=0.08)，缓解 MSE 过平滑
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from data_pipeline_v2 import (_build_features, THRUST_SCALE, MFR_MAX, INPUT_DIM,
                               SN_COUNT, UNKNOWN_SN_ID, SN_EMB_DIM, save_meta_info)

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
                    x, y, _, sn_id = _build_features(df.iloc[start:start + seq_len], row)
                    self.samples.append((x, y, sn_id))
                loaded += 1
            except Exception:
                continue

        print(f"  Loaded {loaded} files → {len(self.samples)} windows "
              f"(skipped {skipped_short} files < {seq_len} steps)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, sn_id = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y), sn_id


# ---- 模型 ----
class DualOutputLSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=256, num_layers=2, n_heads=4,
                 sn_vocab=SN_COUNT + 1, sn_emb_dim=SN_EMB_DIM):
        """
        sn_vocab: SN_COUNT + 1 = 25 (0–23 = SN01–SN24, 24 = UNKNOWN)
        sn_emb_dim: 8, SN embedding 拼接到每个时间步的特征上
        """
        super().__init__()
        self.sn_embedding = nn.Embedding(sn_vocab, sn_emb_dim)
        self.lstm = nn.LSTM(input_dim + sn_emb_dim, hidden_dim, num_layers,
                            batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads,
                                               batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 2),
        )

    def forward(self, x, sn_ids):
        # x: [B, T, 17], sn_ids: [B]
        sn_emb = self.sn_embedding(sn_ids)           # [B, sn_emb_dim]
        sn_emb = sn_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, T, sn_emb_dim]
        x = torch.cat([x, sn_emb], dim=-1)           # [B, T, 17+8]

        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)                    # [B, T, 256]
        attn_out, _ = self.attention(out, out, out)  # self-attention
        out = self.ln(out + attn_out)            # residual + norm
        return self.fc(out)                      # [B, T, 2]


# ---- 加权损失 + 方差匹配（缓解 MSE 过平滑） ----
def dual_loss(pred, target, w_thrust=0.7, w_mfr=0.3, w_std=0.08):
    mse = nn.MSELoss()
    mse_loss = (w_thrust * mse(pred[:, :, 0], target[:, :, 0]) +
                w_mfr    * mse(pred[:, :, 1], target[:, :, 1]))

    # 每个样本时间维度的标准差匹配 —— 惩罚 "预测太平滑"
    pred_std_thrust = pred[:, :, 0].std(dim=1)   # [B]
    true_std_thrust = target[:, :, 0].std(dim=1)  # [B]
    pred_std_mfr    = pred[:, :, 1].std(dim=1)    # [B]
    true_std_mfr    = target[:, :, 1].std(dim=1)  # [B]
    std_loss = mse(pred_std_thrust, true_std_thrust) + mse(pred_std_mfr, true_std_mfr)

    return mse_loss + w_std * std_loss


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
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

    # -- 训练循环 --
    epochs = 100
    sn_mask_prob = 0.12  # 12% 概率替换为 UNKNOWN，训练模型在未见 SN 上的泛化
    save_dir = "outputs/models/v2"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "dual_output_lstm_v2.pth")
    best_val_loss = float('inf')
    early_stop_patience = 20
    epochs_no_improve = 0
    print(f"\nTraining {epochs} epochs... (early stop patience={early_stop_patience}, "
          f"SN mask prob={sn_mask_prob})")
    for epoch in range(epochs):
        # --- train ---
        model.train()
        train_loss = 0
        for x, y, sn_ids in train_loader:
            x, y = x.to(device), y.to(device)
            sn_ids = sn_ids.to(device)

            # 随机将部分 SN 替换为 UNKNOWN，训练 embedding 24（未知推进器）
            mask = torch.rand(sn_ids.size(0), device=device) < sn_mask_prob
            sn_ids = sn_ids.clone()
            sn_ids[mask] = UNKNOWN_SN_ID

            optimizer.zero_grad()
            pred = model(x, sn_ids)
            loss = dual_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # --- val（不做 mask，使用真实 SN id）---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y, sn_ids in val_loader:
                x, y = x.to(device), y.to(device)
                sn_ids = sn_ids.to(device)
                pred = model(x, sn_ids)
                val_loss += dual_loss(pred, y).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
            print(f"  → best model saved (val_loss={val_loss:.6f})")
        else:
            epochs_no_improve += 1
        print(f"Epoch [{epoch+1:3d}/{epochs}]  "
              f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")
        if epochs_no_improve >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"(no improvement for {early_stop_patience} epochs)")
            break

    save_meta_info()

    # -- 快速 sanity check --
    print("\nSanity check:")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    x_sample, y_sample, sn_sample = train_dataset[0]
    with torch.no_grad():
        pred = model(x_sample.unsqueeze(0).to(device),
                     torch.tensor([sn_sample], device=device))
    print(f"  input:  {x_sample.shape}  + sn_id={sn_sample}  →  output: {pred.squeeze(0).shape}")
    print(f"  target thrust range: [{y_sample[:, 0].min():.3f}, {y_sample[:, 0].max():.3f}] "
          f"(x{THRUST_SCALE}=[{y_sample[:, 0].min()*THRUST_SCALE:.1f}, {y_sample[:, 0].max()*THRUST_SCALE:.1f}] N)")
    print(f"  target mfr    range: [{y_sample[:, 1].min():.3f}, {y_sample[:, 1].max():.3f}] "
          f"(x{MFR_MAX}=[{y_sample[:, 1].min()*MFR_MAX:.1f}, {y_sample[:, 1].max()*MFR_MAX:.1f}] mg/s)")


if __name__ == "__main__":
    main()
