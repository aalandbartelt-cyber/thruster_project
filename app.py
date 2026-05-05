import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 1. 页面基本设置与说明

st.set_page_config(page_title="航天推进器 AI 健康监控大屏", layout="wide")
class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 256, 2, batch_first=True)
        self.fc = nn.Linear(256, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleLSTM().to(device)
    model.load_state_dict(torch.load("thruster_lstm_v1.pth", map_location=device, weights_only=True))
    model.eval()
    return model, device

model, device = load_model()


# 2. 侧边栏：参数调节与故障注入

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg", width=150)
st.sidebar.title("推进器测试控制台")

st.sidebar.markdown("### 物理工况参数设定")
pressure_input = st.sidebar.slider("测试气压 (bars)", min_value=5.0, max_value=25.0, value=24.0, step=0.5)
age_input = st.sidebar.slider("老化时间 (小时)", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
valve_duration = st.sidebar.slider("阀门开启时长 (Time Steps)", min_value=50, max_value=150, value=100, step=10)

st.sidebar.markdown("---")
st.sidebar.markdown("### 异常检测测试")
st.sidebar.caption("由于缺乏真实飞行中的实时故障数据，我们在此处提供一个故障注入开关，用于验证 AI 的残差监控能力。")
# 故障注入开关
inject_fault = st.sidebar.toggle("启用模拟气阻故障 (Vapor Lock)")

# ==========================================
# 3. 数据生成与推理核心 (融合残差分析)
# ==========================================
st.title("航天单组元推进器 AI 健康监控大屏")
st.markdown("> **底层驱动**: `Baseline LSTM (256维隐藏层)` | **功能模块**: 基于残差分析的时间序列异常预警系统")

# 构建时间序列
seq_length = 250
ton_seq = np.zeros(seq_length)
start_step = 50
end_step = start_step + valve_duration
ton_seq[start_step:end_step] = 1.0  # 阀门开启区间

# 构造模型输入特征
u = ton_seq.astype(np.float32)
p = np.full_like(u, pressure_input / 25.0, dtype=np.float32)
age = np.full_like(u, age_input / 8.0, dtype=np.float32)

x_input = np.stack([u, p, age], axis=1)
x_tensor = torch.from_numpy(x_input).unsqueeze(0).to(device)

# AI 模型预测 (理论健康基准)
with torch.no_grad():
    y_pred = model(x_tensor).squeeze().cpu().numpy() * 4.0

# 模拟传感器回传的"真实推力" (加一点高斯白噪声让它看起来更真实)
y_actual = y_pred + np.random.normal(0, 0.05, seq_length)

# 如果开启了故障注入，强制在阀门开启的中后段拉低真实推力
fault_start, fault_end = start_step + int(valve_duration * 0.4), end_step - 10
if inject_fault:
    # 【修改】：采用"断崖式"掉流，直接掉落 60% 且至少掉落 0.8N，确保绝对击穿 0.6N 的阈值！
    drop_amount = np.maximum(y_pred[fault_start:fault_end] * 0.6, 0.8)
    y_actual[fault_start:fault_end] -= drop_amount

# 计算残差与异常状态 (同 model_anomaly_detect.py)
THRESHOLD = 0.6
residual = np.abs(y_actual - y_pred)
is_anomaly = residual > THRESHOLD

# ==========================================
# 4. 动态面板与图表渲染
# ==========================================
# 指标卡片
col1, col2, col3 = st.columns(3)
col1.metric("预测峰值推力", f"{np.max(y_pred):.2f} N")
col2.metric("当前最大残差", f"{np.max(residual):.2f} N")
if np.any(is_anomaly):
    col3.error("系统状态: 发生异常！")
    st.error(f" **警告：在残差分析中检测到推力偏离基准超过 {THRESHOLD}N高度疑似气阻现象,请立即检查催化床状态!**")
else:
    col3.success("系统状态: 健康运行")
    st.success(" **日志**：当前传感器数据与 AI 数字孪生模型匹配良好，推力输出正常。")

# 使用 Matplotlib 绘制带残差报警的图表 (与阶段二完美对应)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2.5, 1]})

# 上图：推力对比
ax1.plot(y_actual, label='Actual Sensor Thrust', color='#2980b9', linewidth=2, alpha=0.8)
ax1.plot(y_pred, label='AI Predicted Baseline', color='#27ae60', linestyle='--', linewidth=2)
# 把底部背景标灰表示阀门状态
ax1.fill_between(range(seq_length), 0, np.max(y_pred)*1.2, where=(ton_seq==1), color='gray', alpha=0.1, label='Valve Open')

if np.any(is_anomaly):
    ax1.fill_between(range(seq_length), 0, np.max(y_pred)*1.2, where=is_anomaly, color='red', alpha=0.2, label='ANOMALY ZONE')

ax1.set_title("Real-time Thrust Monitoring", fontsize=14, fontweight='bold')
ax1.set_ylabel("Thrust (N)")
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.6)

# 下图：残差监控
ax2.plot(residual, color='#e67e22', label='Absolute Residual')
ax2.axhline(y=THRESHOLD, color='red', linestyle='-', linewidth=1.5, label=f'Safety Threshold ({THRESHOLD}N)')
ax2.fill_between(range(seq_length), residual, THRESHOLD, where=is_anomaly, color='red', alpha=0.4)

ax2.set_xlabel("Time Steps")
ax2.set_ylabel("Residual (N)")
ax2.legend(loc='upper right')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
# 将 matplotlib 的图表传递给 Streamlit 显示
st.pyplot(fig)