import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class ThrusterDataset(Dataset):
    def __init__(self, metadata_path, train_dir, seq_length=200):
        self.train_dir = train_dir
        self.seq_length = seq_length
        
        # 读取并清理数据
        df_meta = pd.read_csv(metadata_path)
        df_meta['filename'] = df_meta['filename'].str.strip()
        df_meta = df_meta[df_meta['anomalous'] == False]
        
        # 【绝对防御】只有当文件真实存在时，才加入训练列表
        valid_rows = []
        for _, row in df_meta.iterrows():
            file_path = os.path.join(train_dir, row['filename'])
            if os.path.exists(file_path):
                valid_rows.append(row)
        
        self.data = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"成功找到并确认了 {len(self.data)} 个实际存在的训练文件。")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = os.path.join(self.train_dir, row['filename'])
        
        # 即使文件存在，万一读取时损坏，也用 try-except 保护
        try:
            df = pd.read_csv(file_path)
            u = df['ton'].values.astype(np.float32)
            p = np.full_like(u, row['test_pressure'] / 25.0)
            age = np.full_like(u, row['cumulated_on_time'] / 8.0)
            x = np.stack([u, p, age], axis=1)
            y = df['thrust'].values.astype(np.float32).reshape(-1, 1) / 4.0
            
            if len(x) > self.seq_length:
                x, y = x[:self.seq_length, :], y[:self.seq_length, :]
            else:
                pad_len = self.seq_length - len(x)
                x = np.pad(x, ((0, pad_len), (0, 0)), mode='constant')
                y = np.pad(y, ((0, pad_len), (0, 0)), mode='constant')
            return torch.from_numpy(x), torch.from_numpy(y)
            
        except Exception as e:
            # 遇到坏文件直接返回0矩阵，绝不崩溃
            return torch.zeros(self.seq_length, 3), torch.zeros(self.seq_length, 1)

class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 256, 2, batch_first=True)
        self.fc = nn.Linear(256, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n当前设备: {device}")
    
    metadata_path = "data/metadata.csv"
    train_dir = "data/dataset/dataset/train/"
    
    dataset = ThrusterDataset(metadata_path, train_dir)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = SimpleLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 损失函数使用均方误差
    criterion = nn.MSELoss()
    
    print(" 开始训练 (Epochs: 100) ...")
    model.train()
    for epoch in range(100):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/100] Loss: {total_loss/len(dataloader):.6f}")

    # ==========================================
    # 核心新增：训练结束后保存模型权重
    # ==========================================
    save_path = "thruster_lstm_v1.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n训练完成！模型已妥善保存至当前目录: {save_path}")

if __name__ == "__main__":
    main()

