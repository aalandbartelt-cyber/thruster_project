import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 256, 2, batch_first=True)
        self.fc = nn.Linear(256, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

def detect_anomaly():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前计算设备: {device}")
    
    # 1. 加载“健康”模型
    model = SimpleLSTM().to(device)
    model.load_state_dict(torch.load("thruster_lstm_v1.pth", weights_only=True))
    model.eval()
    
    # 2. 读取元数据，专门筛选出有异常(故障)的测试序列
    metadata_path = "data/metadata.csv"
    df_meta = pd.read_csv(metadata_path)
    df_meta['filename'] = df_meta['filename'].str.strip()
    
    # 筛选出 anomalous == True 的数据
    df_anom = df_meta[df_meta['anomalous'] == True]
    
    # 在目录中寻找第一个实际存在的异常文件
    search_dirs = ["data/dataset/dataset/train/", "data/dataset/dataset/test/"]
    target_path = None
    sample_row = None
    
    for _, row in df_anom.iterrows():
        fname = row['filename']
        for d in search_dirs:
            p = os.path.join(d, fname)
            if os.path.exists(p):
                target_path = p
                sample_row = row
                break
        if target_path:
            break
            
    if not target_path:
        print("未找到异常数据集文件，请检查 metadata.csv 或数据文件夹完整性！")
        return

    print(f"找到异常测试序列: {sample_row['filename']}")
    
    # 3. 数据预处理
    df = pd.read_csv(target_path)
    u = df['ton'].values.astype(np.float32)
    p = np.full_like(u, sample_row['test_pressure'] / 25.0)
    age = np.full_like(u, sample_row['cumulated_on_time'] / 8.0)
    
    x_input = np.stack([u, p, age], axis=1)
    x_tensor = torch.from_numpy(x_input).unsqueeze(0).to(device)
    
    # 4. 模型预测 (计算健康基准)
    with torch.no_grad():
        y_pred_tensor = model(x_tensor)
        y_pred = y_pred_tensor.squeeze().cpu().numpy() * 4.0
    
    y_actual = df['thrust'].values
    
    # 5. 核心逻辑：残差计算与阈值设定
    residual = np.abs(y_actual - y_pred)
    
    # 设定报警阈值 (这里设为 0.6 N)
    THRESHOLD = 0.6 
    
    # 生成布尔数组，表示每个时间步是否发生异常
    is_anomaly = residual > THRESHOLD

    # 6.双子图设计
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [2.5, 1]})
    
    # 上图：推力对比与红色警报区域
    ax1.plot(y_actual, label='Actual Thrust (Sensor Data)', color='blue', alpha=0.7)
    ax1.plot(y_pred, label='Predicted "Healthy" Thrust (AI Baseline)', color='green', linestyle='--', alpha=0.9)
    
    # 使用 fill_between 将异常区域背景标红
    ax1.fill_between(range(len(y_actual)), 0, max(y_actual)*1.1, 
                     where=is_anomaly, color='red', alpha=0.3, label='WARNING:ANOMALY DETECTED')
    
    ax1.set_title(f"Thruster Health Monitoring: {sample_row['filename']}", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Thrust (N)", fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # 下图：残差监控曲线
    ax2.plot(residual, label='Absolute Residual ($|Actual - Predicted|$)', color='orange')
    ax2.axhline(y=THRESHOLD, color='red', linestyle='-', linewidth=2, label=f'Safety Threshold ({THRESHOLD} N)')
    
    ax2.set_title("Real-time Residual Tracking", fontsize=12)
    ax2.set_xlabel("Time Steps", fontsize=12)
    ax2.set_ylabel("Error Margin (N)", fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    # 调整布局并保存
    plt.tight_layout()
    save_name = "anomaly_detection_dashboard.png"
    plt.savefig(save_name, dpi=300)
    print(f"✨ 故障检测面板已生成并保存为 {save_name}")
    plt.show()

if __name__ == "__main__":
    detect_anomaly()