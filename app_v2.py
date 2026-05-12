import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from inference_utils import (
    load_dual_model,
    predict,
    predict_with_uncertainty,
    compute_residuals,
    generate_health_report,
)

# ---------------- 页面配置 ----------------
st.set_page_config(
    page_title="航天推进器健康监控 v2.0",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- 明亮航天主题CSS ----------------
st.markdown("""
<style>
    /* 全局背景与字体 */
    .main {
        background-color: #f5f7fa;
        color: #1a1c23;
    }
    .stMetric {
        background: linear-gradient(135deg, #eef2f6 0%, #ffffff 100%);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #0077be;
    }
    .stButton>button {
        background: linear-gradient(90deg, #0077be, #005a8c);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0088d4, #006699);
        box-shadow: 0 4px 12px rgba(0,119,190,0.3);
    }
    /* 侧边栏卡片效果 */
    .css-1d391kg {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }
    h1, h2, h3 {
        color: #003057;
    }
    hr {
        border: 1px solid #e0e6ed;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- 缓存加载模型 ----------------
@st.cache_resource
def load_model():
    return load_dual_model("outputs/models/v2/dual_output_lstm_v2.pth")

model = load_model()

# ---------------- 侧边栏（清爽控制台） ----------------
with st.sidebar:
    st.image("assets/my_logo.png", width=180)
    st.markdown("## 🛰️ 控制台")
    st.markdown("---")
    uploaded_file = st.file_uploader("📤 上传测试CSV文件", type=["csv"])
    st.markdown("---")
    threshold = st.slider("⚠️ 异常检测阈值 (N)", 0.1, 2.0, 0.6, 0.1)
    st.markdown("---")
    generate_report = st.button("📋 一键生成健康报告")

# ---------------- 主界面 ----------------
st.title("🚀 航天单组元推进器 AI 健康监控大屏")
st.caption("双输出 Attention‑LSTM   |   17维特征融合   |   比冲实时解算")

if uploaded_file is not None:
    # ------- 数据模拟（说明：联调时替换为真实data_pipeline调用）-------
    df = pd.read_csv(uploaded_file)
    seq_len = min(len(df), 200)
    # 实际使用时替换为：x_input, _, _ = load_sequence(filepath, meta_row) 等
    x_input = np.random.randn(seq_len, 17).astype(np.float32)

    # ------- 推理 -------
    thrust, mfr, isp = predict(model, x_input)

    # ------- 模拟真实传感器读数（加少量噪声） -------
    np.random.seed(42)
    actual_thrust = thrust + np.random.normal(0, 0.015, thrust.shape)
    actual_mfr = mfr + np.random.normal(0, 2.5, mfr.shape)

    # ------- 残差计算与异常检测 -------
    residuals, is_anomaly, anomaly_ratio = compute_residuals(thrust, actual_thrust, threshold)

    # ==================== 第一行：指标卡片（无重叠） ====================
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔹 峰值推力", f"{np.max(actual_thrust):.2f} N")
    col2.metric("🔹 平均质量流量", f"{np.mean(actual_mfr):.1f} mg/s")
    col3.metric("🔹 平均比冲", f"{np.mean(isp):.1f} s")
    col4.metric("🔹 异常点占比", f"{anomaly_ratio*100:.2f}%", delta="正常" if anomaly_ratio < 0.05 else "异常")

    # ==================== 第二行：三列监控面板（高度适配） ====================
    st.markdown("### 📊 实时监控曲线")
    # 使用固定图像大小 + tight_layout 自动防止重叠
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), facecolor='white')
    time_axis = np.arange(seq_len)

    # 颜色方案：航天蓝 + 活力橙 + 安全绿
    color_actual = '#0077be'
    color_pred = '#ff8c42'
    color_isp = '#2ca02c'

    # ---- 推力图 ----
    ax1.plot(time_axis, actual_thrust, color=color_actual, linewidth=2, label='传感器实测')
    ax1.plot(time_axis, thrust, color=color_pred, linestyle='--', linewidth=2, label='AI预测基准')
    ax1.fill_between(time_axis, 0, max(actual_thrust)*1.15,
                     where=is_anomaly, color='red', alpha=0.25, label='异常区间')
    ax1.set_title('推力 (Thrust)', fontweight='bold', color='#003057')
    ax1.set_ylabel('牛顿 (N)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # ---- 质量流量图 ----
    ax2.plot(time_axis, actual_mfr, color=color_actual, linewidth=2, label='传感器实测')
    ax2.plot(time_axis, mfr, color=color_pred, linestyle='--', linewidth=2, label='AI预测基准')
    ax2.set_title('质量流量 (MFR)', fontweight='bold', color='#003057')
    ax2.set_ylabel('mg/s')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, linestyle=':', alpha=0.5)

    # ---- 比冲图 ----
    ax3.plot(time_axis, isp, color=color_isp, linewidth=2.5, label='实时比冲')
    ax3.axhline(150, color='#d62728', linestyle=':', linewidth=2, label='最低健康值 (150 s)')
    ax3.set_title('比冲 (Isp)', fontweight='bold', color='#003057')
    ax3.set_ylabel('秒 (s)')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, linestyle=':', alpha=0.5)

    # 统一样式
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel('时间步', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=2.5)
    st.pyplot(fig)

    # ==================== 第三行：异常状态与健康报告 ====================
    st.markdown("### 🔍 实时诊断与报告")

    left_col, right_col = st.columns([1, 1])

    with left_col:
        if np.any(is_anomaly):
            st.error(f"⚠️ 检测到 {is_anomaly.sum()} 个异常点！推力残差超过阈值 {threshold} N。")
        else:
            st.success("✅ 系统运行正常，所有参数在健康范围内。")

    with right_col:
        if generate_report:
            with st.spinner("正在生成健康诊断报告..."):
                report = generate_health_report(thrust, mfr, isp, is_anomaly=is_anomaly)
            st.info("📋 诊断报告已生成")
            st.json(report)
        else:
            st.markdown("> 点击侧边栏 **“一键生成健康报告”** 按钮获取详细诊断。")

else:
    # ---------------- 默认欢迎界面 ----------------
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; height: 60vh;">
        <div style="text-align: center; color: #5a6a7e;">
            <h2>✨ 航天推进器数字孪生健康监控系统</h2>
            <p>请上传测试CSV文件，启动实时推力、流量、比冲监测。</p>
            <p style="font-size:0.9em;">基于 Attention‑LSTM 双输出模型 · 17维特征 · SHAP 可解释性</p>
        </div>
    </div>
    """, unsafe_allow_html=True)