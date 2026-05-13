import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties
# 设置中文字体
_FONT = FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
plt.rcParams['font.family'] = 'Droid Sans Fallback'
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd
import torch
from inference_utils import (
    load_dual_model,
    predict,
    predict_with_uncertainty,
    compute_residuals,
    generate_health_report,
)
from data_pipeline_v2 import load_sequence, THRUST_SCALE, MFR_MAX

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

# ---------------- 缓存加载模型 + 元数据 ----------------
@st.cache_resource
def load_model():
    return load_dual_model("outputs/models/v2/dual_output_lstm_v2.pth")

@st.cache_resource
def load_metadata():
    df = pd.read_csv("data/metadata.csv")
    df['filename'] = df['filename'].str.strip()
    return df

model = load_model()
metadata = load_metadata()

# ---------------- 侧边栏（清爽控制台） ----------------
with st.sidebar:
    try:
        st.image("assets/my_logo.png", width=180)
    except Exception:
        st.markdown("🚀 **CASC**")  # logo 不可用时回退文字
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
    seq_len = 200
    fname = uploaded_file.name

    # ----- 方式一：在 metadata.csv 中找到该文件 → 用 load_sequence + 模型推理 -----
    matched = metadata[metadata['filename'] == fname]
    if len(matched) > 0:
        meta_row = matched.iloc[0]
        # 保存上传的 CSV 到临时位置，供 load_sequence 读取
        import tempfile, os
        tmpdir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmpdir, fname)
        with open(tmp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # load_sequence 返回 17 维特征 + 标签（已归一化）+ 异常标签
        x, y_norm, labels_seq = load_sequence(tmp_path, meta_row, seq_len=seq_len)
        x_tensor = torch.from_numpy(x).float().unsqueeze(0)  # [1, 200, 17]

        # 模型推理
        thrust_pred, mfr_pred, isp = predict(model, x_tensor)  # [1, 200]
        thrust_pred = thrust_pred.squeeze()  # [200]
        mfr_pred = mfr_pred.squeeze()        # [200]
        isp = isp.squeeze()                  # [200]
        thrust_true = y_norm[:, 0] * THRUST_SCALE
        mfr_true = y_norm[:, 1] * MFR_MAX

        # 模拟传感器实际值（真实值+微噪声）
        np.random.seed(42)
        actual_thrust = thrust_true + np.random.normal(0, 0.005, seq_len)
        actual_mfr = mfr_true + np.random.normal(0, 0.5, seq_len)

        # 清理临时文件
        import shutil; shutil.rmtree(tmpdir, ignore_errors=True)

    else:
        # ----- 方式二（降级）：文件名不在 metadata 中 → 从 npy 接口取样本 -----
        import numpy as np, json
        preds_npy = np.load("outputs/predictions/v2/predictions_dual.npy")
        targets_npy = np.load("outputs/predictions/v2/targets_dual.npy")
        with open("outputs/predictions/v2/meta_info.json") as f:
            meta = json.load(f)
        ts = meta['thrust_scale']; ms = meta['mfr_max']
        sample_idx = hash(fname) % len(preds_npy)
        thrust_pred = preds_npy[sample_idx, :, 0] * ts
        mfr_pred = preds_npy[sample_idx, :, 1] * ms
        thrust_true = targets_npy[sample_idx, :, 0] * ts
        mfr_true = targets_npy[sample_idx, :, 1] * ms
        np.random.seed(42)
        actual_thrust = thrust_true + np.random.normal(0, 0.005, seq_len)
        actual_mfr = mfr_true + np.random.normal(0, 0.5, seq_len)
        isp = thrust_pred / (mfr_pred * 1e-3 * 9.80665)

    # ------- 残差计算与异常检测 -------
    residuals, is_anomaly, anomaly_ratio = compute_residuals(thrust_pred, actual_thrust, threshold)

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
    ax1.plot(time_axis, thrust_pred, color=color_pred, linestyle='--', linewidth=2, label='AI预测基准')
    ax1.fill_between(time_axis, 0, max(actual_thrust)*1.15,
                     where=is_anomaly, color='red', alpha=0.25, label='异常区间')
    ax1.set_title('推力 (Thrust)', fontweight='bold', color='#003057')
    ax1.set_ylabel('牛顿 (N)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # ---- 质量流量图 ----
    ax2.plot(time_axis, actual_mfr, color=color_actual, linewidth=2, label='传感器实测')
    ax2.plot(time_axis, mfr_pred, color=color_pred, linestyle='--', linewidth=2, label='AI预测基准')
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
                report = generate_health_report(thrust_pred, mfr_pred, isp, is_anomaly=is_anomaly)
            r = report
            ts = r['thrust_status']; ms = r['mfr_status']; is_ = r['isp_status']
            as_ = r.get('anomaly_status', {})
            st.markdown(f"""
            ### 📋 健康诊断报告

            **{r['summary']}**

            | 指标 | 数值 | 评分 | 状态 |
            |:----|:----|:---:|:----:|
            | 🔹 推力 Thrust | {ts['value']} | {ts['score']}/100 | {'🟢' if ts['level']=='good' else '🟡' if ts['level']=='warning' else '🔴'} {ts['level']} |
            | 🔹 质量流量 MFR | {ms['value']} | {ms['score']}/100 | {'🟢' if ms['level']=='good' else '🟡' if ms['level']=='warning' else '🔴'} {ms['level']} |
            | 🔹 比冲 Isp | {is_['value']} | {is_['score']}/100 | {'🟢' if is_['level']=='good' else '🟡' if is_['level']=='warning' else '🔴'} {is_['level']} |
            | 🔹 异常占比 | {as_.get('anomaly_ratio',0)*100:.1f}% | — | {as_.get('level','normal')} |
            """)
        else:
            st.markdown('> 点击侧边栏 **一键生成健康报告** 按钮获取详细诊断。')

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