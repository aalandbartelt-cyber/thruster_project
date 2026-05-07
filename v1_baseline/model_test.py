import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random

class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 256, 2, batch_first=True)
        self.fc = nn.Linear(256, 1)
        
    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        return self.fc(out)

def test_on_multiple_unseen_data(num_plot_samples=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载模型
    model = SimpleLSTM().to(device)
    model.load_state_dict(torch.load("thruster_lstm_v1.pth", weights_only=True))
    model.eval()
    
    # 2. 路径配置
    metadata_path = "data/metadata.csv"
    test_dir = "data/dataset/dataset/test/"
    
    # 3. 读取元数据并筛选
    df_meta = pd.read_csv(metadata_path)
    df_meta['filename'] = df_meta['filename'].str.strip()
    
    df_meta = df_meta[df_meta['anomalous'] == False]
    
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
    valid_meta = df_meta[df_meta['filename'].isin(test_files)]
    valid_files = valid_meta['filename'].tolist()
    
    if not valid_files:
        print("❌ 错误：在 test 目录下没找到符合条件的正常 CSV 文件！")
        return

    print(f"🚀 找到 {len(valid_files)} 个未见过的测试数据集。")
    print(f"⚡ 正在切换至原生 CUDA 引擎以处理超长工业时间序列...")
    
    all_rmse = []
    all_mae = []
    plot_data = [] 
    
    plot_indices = random.sample(range(len(valid_files)), min(num_plot_samples, len(valid_files)))
    
    # 4. 批量执行预测与指标计算
    for idx, target_filename in enumerate(valid_files):
        sample_row = valid_meta[valid_meta['filename'] == target_filename].iloc[0]
        file_path = os.path.join(test_dir, target_filename)
        
        # 数据预处理
        df = pd.read_csv(file_path)
        u = df['ton'].values.astype(np.float32)
        p = np.full_like(u, sample_row['test_pressure'] / 25.0, dtype=np.float32)
        age = np.full_like(u, sample_row['cumulated_on_time'] / 8.0, dtype=np.float32)
        
        x_input = np.stack([u, p, age], axis=1)
        
        # 强制内存连续拷贝 (双重保险)
        x_input_contig = np.ascontiguousarray(x_input)
        x_tensor = torch.from_numpy(x_input_contig).unsqueeze(0).to(device)
        
        # ==========================================
        # 终极修复：暂时关闭 cuDNN 限制，使用原生 CUDA 算子处理超长序列！
        # ==========================================
        with torch.no_grad():
            with torch.backends.cudnn.flags(enabled=False): 
                y_pred_tensor = model(x_tensor)
                y_pred = y_pred_tensor.squeeze().cpu().numpy() * 4.0 
        
        y_true = df['thrust'].values
        
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        mae = np.mean(np.abs(y_true - y_pred))
        
        all_rmse.append(rmse)
        all_mae.append(mae)
        
        if idx in plot_indices:
            plot_data.append({
                'filename': target_filename,
                'y_true': y_true,
                'y_pred': y_pred,
                'rmse': rmse,
                'mae': mae
            })
            
    # ================= 打印全局量化评估指标 =================
    mean_rmse = np.mean(all_rmse)
    mean_mae = np.mean(all_mae)
    print(f"\n✅ {len(valid_files)} 个测试集全部评估完成!")
    print(f"📊 【全局平均指标 (Global Metrics)】:")
    print(f"   - Global Mean RMSE: {mean_rmse:.4f} N")
    print(f"   - Global Mean MAE:  {mean_mae:.4f} N")
    
    # 5. 绘制 2x2 抽样对比网格图
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, data in enumerate(plot_data):
        ax = axes[i]
        ax.plot(data['y_true'], label='Actual Thrust', color='blue', alpha=0.6, linewidth=1.5)
        ax.plot(data['y_pred'], label='Predicted Thrust', color='red', linestyle='--', alpha=0.8)
        
        metrics_text = f"RMSE: {data['rmse']:.4f} N\nMAE: {data['mae']:.4f} N"
        ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes, fontsize=11,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ax.set_title(f"Test Target: {data['filename']}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Thrust (N)")
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.7)
        
    plt.suptitle(f"Multi-Scenario Test Evaluation (Global Mean RMSE: {mean_rmse:.4f} N)", fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    save_name = "multi_test_prediction_result.png"
    plt.savefig(save_name, dpi=300)
    print(f"\n✨ 批量测试完成！2x2 抽样对比图已保存为 {save_name}")
    plt.show()

if __name__ == "__main__":
    test_on_multiple_unseen_data()