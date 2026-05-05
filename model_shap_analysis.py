import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import warnings

# 忽略 SHAP 不认识 LSTM 的无害警告
warnings.filterwarnings("ignore", message="unrecognized nn.Module: LSTM")

class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 256, 2, batch_first=True)
        self.fc = nn.Linear(256, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

# 封装一个降维模型，专为 SHAP 解释用

class ShapWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    def forward(self, x):
        # 解释"输入特征如何影响整个序列的平均推力"
        out = self.base_model(x)
        return out.mean(dim=1)

def main():
    # Academic best practice for SHAP + PyTorch LSTM: Force CPU mode
    device = torch.device('cpu') 
    print("启动 SHAP 可解释性分析引擎 (CPU 模式)...")

    # 1. 加载模型与包装器
    base_model = SimpleLSTM().to(device)
    base_model.load_state_dict(torch.load("thruster_lstm_v1.pth", map_location=device, weights_only=True))
    
    # 用 train() 算梯度
    base_model.train() 
    wrapper_model = ShapWrapper(base_model).to(device)

    # 2.提取少量真实数据作为样本
    df_meta = pd.read_csv("data/metadata.csv")
    df_meta = df_meta[df_meta['anomalous'] == False].head(50) 
    
    data_list = []
    data_files = [] # 记录文件名以便核对
    train_dir = "data/dataset/dataset/train/"
    
    if not os.path.exists(train_dir):
        print("错误: 没找到训练数据文件夹，请检查 data/dataset/dataset/train/ 路径")
        return

    for _, row in df_meta.iterrows():
        try:
            df = pd.read_csv(os.path.join(train_dir, row['filename'].strip()))
            u = df['ton'].values.astype(np.float32)
            p = np.full_like(u, row['test_pressure'] / 25.0, dtype=np.float32)
            age = np.full_like(u, row['cumulated_on_time'] / 8.0, dtype=np.float32)
            x = np.stack([u, p, age], axis=1)
            # 截取前 200 步
            if len(x) >= 200:
                data_list.append(x[:200, :])
        except: pass

    if len(data_list) < 40:
        print("错误: 有效训练样本不足(小于40个)，无法计算，请增加 df_meta.head() 的数量")
        return

    data_tensor = torch.from_numpy(np.array(data_list)).to(device) # Shape: [50, 200, 3]
    
    background = data_tensor[:30] # 背景集
    test_samples = data_tensor[30:50] # 解释集
    
    print(f"数据加载完毕。背景集: {background.shape}, 解释集: {test_samples.shape}")
    print("正在计算 Shapley Values ")

    # 3. 初始化 SHAP DeepExplainer
    explainer = shap.DeepExplainer(wrapper_model, background)
    
    # 关闭 Additivity 校验以绕过严格的数学误差报错
    shap_values = explainer.shap_values(test_samples, check_additivity=False)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    print(f"SHAP 计算完毕。shap_values shape: {shap_values.shape}")
    print(f"   shap_values 数值范围: min={shap_values.min():.6f}, max={shap_values.max():.6f}, mean={shap_values.mean():.6f}")
    # 对 seq_len 维度求均值，再 squeeze 掉最后的输出维度
    shap_values_2d = np.abs(shap_values).mean(axis=1).squeeze()
    
    print(f"降维后 shap_values_2d shape: {shap_values_2d.shape}")
    print(f"   shap_values_2d 数值范围: min={shap_values_2d.min():.6f}, max={shap_values_2d.max():.6f}, mean={shap_values_2d.mean():.6f}")

    # 4. 计算每个特征的平均重要性
    feature_names = ['Valve State (ton)', 'Test Pressure', 'Aging (Cumulated Time)']
    mean_shap_per_feature = np.mean(shap_values_2d, axis=0)
    
    print(f"\n每个特征的平均 SHAP 值:")
    for i, fname in enumerate(feature_names):
        print(f"   {fname}: {mean_shap_per_feature[i]:.6f}")

    # 5. 绘制图表 1：手动柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(feature_names, mean_shap_per_feature, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Mean |SHAP value|', fontsize=12)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_title('SHAP Feature Importance for Thruster Prediction (Manual Bar Plot)', fontsize=14, fontweight='bold')
    
    # 在柱子上显示数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("shap_feature_importance_manual.png", dpi=300, bbox_inches='tight')
    print("\n手动柱状图已保存为: shap_feature_importance_manual.png")
    plt.show()

    # 6. 绘制图表 2：SHAP 官方 summary_plot（bar 类型）
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_2d, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance for Thruster Prediction (SHAP Summary Plot)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("shap_feature_importance_shap.png", dpi=300, bbox_inches='tight')
    print("SHAP summary_plot 已保存为: shap_feature_importance_shap.png")
    plt.show()

    # 7. 绘制图表 3：SHAP 官方 summary_plot（violin 类型，展示分布）
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values_2d, test_samples.cpu().numpy().mean(axis=1), 
                      feature_names=feature_names, plot_type="violin", show=False)
    plt.title("SHAP Feature Importance Distribution (Violin Plot)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("shap_feature_importance_violin.png", dpi=300, bbox_inches='tight')
    print("SHAP violin_plot 已保存为: shap_feature_importance_violin.png")
    plt.show()

    print("\n所有分析完成")

if __name__ == "__main__":
    main()
