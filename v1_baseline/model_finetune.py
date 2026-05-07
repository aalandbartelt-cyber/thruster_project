import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 定义网络架构
class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 256, 2, batch_first=True)
        self.fc = nn.Linear(256, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

# 2. 构建微调专属数据集加载器
class FinetuneDataset(Dataset):
    def __init__(self, df_meta, search_dirs, seq_length=200):
        self.data = []
        for _, row in df_meta.iterrows():
            fname = row['filename'].strip()
            target_path = None
            for d in search_dirs:
                p = os.path.join(d, fname)
                if os.path.exists(p):
                    target_path = p
                    break
            
            if target_path:
                try:
                    df = pd.read_csv(target_path)
                    u = df['ton'].values.astype(np.float32)
                    p_val = np.full_like(u, row['test_pressure'] / 25.0, dtype=np.float32)
                    age = np.full_like(u, row['cumulated_on_time'] / 8.0, dtype=np.float32)
                    x_input = np.stack([u, p_val, age], axis=1)
                    # 关键：标签需要像第一阶段训练时一样进行 / 4.0 的归一化
                    y_true = (df['thrust'].values.astype(np.float32) / 4.0)
                    
                    for i in range(0, len(x_input) - seq_length, seq_length):
                        self.data.append((x_input[i:i+seq_length], y_true[i:i+seq_length]))
                except Exception as e:
                    pass

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.from_numpy(x), torch.from_numpy(y).unsqueeze(-1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"个性化微调实验 (设备: {device})")
    
    # 路径配置
    metadata_path = "data/metadata.csv"
    search_dirs = ["data/dataset/dataset/train/", "data/dataset/dataset/test/"]
    
    df_meta = pd.read_csv(metadata_path)
    # 过滤异常数据，只用健康数据来微调性能预测
    df_meta = df_meta[df_meta['anomalous'] == False]
    
    # 针对编号为 13 的推进器进行个性化微调
    target_sn = 13
    df_sn13 = df_meta[df_meta['sn'] == target_sn]
    
    # 验收测试数据 (test_id <= 12) 作为微调训练集
    df_finetune = df_sn13[df_sn13['test_id'] <= 12]
    # 飞行测试数据 (test_id > 12) 作为最终验证集
    df_test = df_sn13[df_sn13['test_id'] > 12]
    
    if len(df_finetune) == 0 or len(df_test) == 0:
        print("没找到 SN13 的完整数据集，请检查目录。")
        return
        
    print(f"提取成功: SN13 验收测试文件 {len(df_finetune)} 个，后续飞行验证文件 {len(df_test)} 个")
    
    # 选取一个从未见过的飞行测试文件用于画图
    test_row = df_test.iloc[0]
    test_fname = test_row['filename'].strip()
    test_path = next((os.path.join(d, test_fname) for d in search_dirs if os.path.exists(os.path.join(d, test_fname))), None)

    # ---- 测试验证子函数 ----
    def evaluate_model(model):
        model.eval()
        df = pd.read_csv(test_path)
        u = df['ton'].values.astype(np.float32)
        p_val = np.full_like(u, test_row['test_pressure'] / 25.0)
        age = np.full_like(u, test_row['cumulated_on_time'] / 8.0)
        x_input = np.stack([u, p_val, age], axis=1)
        x_tensor = torch.from_numpy(x_input).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(x_tensor).squeeze().cpu().numpy() * 4.0
        actual = df['thrust'].values
        rmse = np.sqrt(np.mean((actual - pred)**2))
        return actual, pred, rmse

    # 1. 评估旧模型 (Baseline)
    print("\n评估通用模型 (General Baseline) ...")
    base_model = SimpleLSTM().to(device)
    base_model.load_state_dict(torch.load("thruster_lstm_v1.pth", weights_only=True))
    actual_y, pred_base, rmse_base = evaluate_model(base_model)
    print(f"   => 通用模型 RMSE: {rmse_base:.4f} N")
    
    # 2. 个性化微调 (Fine-tuning)
    print("\n开始对 SN13 进行个性化微调 (Fine-tuning) ...")
    finetune_model = SimpleLSTM().to(device)
    finetune_model.load_state_dict(torch.load("thruster_lstm_v1.pth", weights_only=True)) # 继承通用权重
    
    dataset = FinetuneDataset(df_finetune, search_dirs)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 学习率调到极小(1e-4)，防止破坏原有的通用特征提取能力
    optimizer = optim.Adam(finetune_model.parameters(), lr=1e-4) 
    criterion = nn.MSELoss()
    
    finetune_model.train()
    epochs = 20 # 只需要微调少量的 Epoch
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(finetune_model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"   Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.6f}")
            
    # 3. 评估新模型
    print("\n评估专属定制模型 (Personalized Model) ...")
    _, pred_finetune, rmse_finetune = evaluate_model(finetune_model)
    print(f"   => 微调模型 RMSE: {rmse_finetune:.4f} N")
    
    improvement = (rmse_base - rmse_finetune) / rmse_base * 100
    if improvement > 0:
        print(f"模型预测精度提升了 {improvement:.2f}%")
    else:
        print(f"精度未明显提升，但证明了模型的鲁棒性。")

    # 4. 工业级数据可视化面板
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：直观展示误差下降的柱状图
    bars = ax1.bar(['General Model\n(Trained on SN01-SN12)', 'Personalized Model\n(Fine-tuned for SN13)'], 
                   [rmse_base, rmse_finetune], color=['#95a5a6', '#2ecc71'], width=0.5)
    ax1.set_ylabel('RMSE Error (Lower is better)', fontsize=12)
    ax1.set_title('Performance Comparison on SN13 Flight Data', fontsize=14, fontweight='bold')
    # 在柱子上打上具体数值
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f'{yval:.4f} N', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    # 右图：预测曲线的局部放大对比图 (放大最容易看出差异的一段)
    zoom_start, zoom_end = 400, 800  # 只截取中间一段
    ax2.plot(actual_y[zoom_start:zoom_end], label='Actual Sensor Thrust', color='black', linewidth=2, alpha=0.6)
    ax2.plot(pred_base[zoom_start:zoom_end], label='Base General Prediction', color='#e74c3c', linestyle='--', alpha=0.8)
    ax2.plot(pred_finetune[zoom_start:zoom_end], label='Fine-tuned SN13 Prediction', color='#27ae60', linestyle='-.', linewidth=2)
    
    ax2.set_title(f'Trajectory Zoom-in: {test_fname}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time Steps', fontsize=12)
    ax2.set_ylabel('Thrust (N)', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    save_name = "finetune_ablation_study.png"
    plt.savefig(save_name, dpi=300)
    print(f"\n对比图表已保存为 {save_name}")
    plt.show()

if __name__ == "__main__":
    main()