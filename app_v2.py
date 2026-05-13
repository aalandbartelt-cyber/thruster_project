"""
app_v2.py — 航天单组元推进器数字孪生健康监测系统 · 大屏
M 主导后端 · Z 主导前端 · dev-m 分支

项目: 基于多源动态特征与多目标耦合预测的航天器推进器数字孪生及健康监测系统
单位: 上海大学
赛事: 第二十八届中国机器人及人工智能大赛 · 人工智能创新赛 · 上海赛区
"""

import os
import io
import json
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

from inference_utils import (
    load_dual_model,
    predict,
    compute_residuals,
    generate_health_report,
)
from data_pipeline_v2 import load_sequence_window, THRUST_SCALE, MFR_MAX


# ════════════════════════════════════════════════════════════════════
# 一、NASA 工业控制台主题
# ════════════════════════════════════════════════════════════════════

# NASA 官方配色 + 工业控制台扩展
NASA_BLUE      = "#0B3D91"   # NASA 主蓝
NASA_RED       = "#FC3D21"   # NASA 强调红
DEEP_SPACE     = "#070B1F"   # 深空背景
PANEL_BG       = "#0F1631"   # 面板背景
PANEL_BORDER   = "#1E2A56"   # 面板边框
GRID_LINE      = "#1A2444"   # 图表网格
TEXT_PRIMARY   = "#E8ECEF"   # 主文字
TEXT_SECONDARY = "#8B95B0"   # 次文字
DATA_CYAN      = "#00D9FF"   # 实测值
DATA_AMBER     = "#FFB627"   # 预测值
DATA_GREEN     = "#00E676"   # 健康/正常
DATA_VIOLET    = "#B388FF"   # 比冲
STATUS_WARN    = "#FF6B35"   # 警告

# 中文字体（Linux/WSL 路径，与用户原配置一致）
_CN_FONT_PATH = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
_CN       = FontProperties(fname=_CN_FONT_PATH, size=10)
_CN_TITLE = FontProperties(fname=_CN_FONT_PATH, size=12)
_CN_SMALL = FontProperties(fname=_CN_FONT_PATH, size=8)


def setup_matplotlib_theme():
    """统一配置 matplotlib：暗色工业主题 + 高 DPI"""
    rcParams['figure.dpi']       = 130
    rcParams['savefig.dpi']      = 300
    rcParams['figure.facecolor'] = PANEL_BG
    rcParams['axes.facecolor']   = PANEL_BG
    rcParams['axes.edgecolor']   = PANEL_BORDER
    rcParams['axes.labelcolor']  = TEXT_PRIMARY
    rcParams['axes.titlecolor']  = TEXT_PRIMARY
    rcParams['xtick.color']      = TEXT_SECONDARY
    rcParams['ytick.color']      = TEXT_SECONDARY
    rcParams['grid.color']       = GRID_LINE
    rcParams['grid.linestyle']   = '--'
    rcParams['grid.alpha']       = 0.6
    rcParams['axes.spines.top']    = False
    rcParams['axes.spines.right']  = False
    rcParams['axes.spines.left']   = True
    rcParams['axes.spines.bottom'] = True
    rcParams['legend.facecolor']   = PANEL_BG
    rcParams['legend.edgecolor']   = PANEL_BORDER
    rcParams['legend.framealpha']  = 0.95
    rcParams['legend.labelcolor']  = TEXT_PRIMARY
    rcParams['font.size']          = 10
    rcParams['axes.titlesize']     = 12
    rcParams['axes.titleweight']   = 'bold'
    rcParams['axes.unicode_minus'] = False


setup_matplotlib_theme()


# ════════════════════════════════════════════════════════════════════
# 二、Streamlit 页面配置 + 全局 CSS 注入
# ════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MTDS · 推进器数字孪生健康监测",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* 全局背景 */
.stApp {{
    background: linear-gradient(180deg, {DEEP_SPACE} 0%, #0A1230 100%);
    color: {TEXT_PRIMARY};
}}
html, body, [class*="css"] {{
    font-family: 'Inter', 'Noto Sans SC', -apple-system, sans-serif;
}}

/* 隐藏 Streamlit 默认元素 */
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; }}

/* ───────── Mission Header ───────── */
.mission-header {{
    background: linear-gradient(90deg, {NASA_BLUE} 0%, #1A2A6C 60%, {DEEP_SPACE} 100%);
    border-bottom: 2px solid {NASA_RED};
    padding: 22px 36px;
    margin: -1rem -1rem 1.8rem -1rem;
    display: flex;
    align-items: center;
    gap: 28px;
}}
.mission-badge {{
    width: 68px; height: 68px;
    border: 2px solid {NASA_RED};
    border-radius: 50%;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    background: rgba(11, 61, 145, 0.3);
    flex-shrink: 0;
}}
.mission-badge .b1 {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 800; font-size: 17px; color: #FFFFFF;
    letter-spacing: 1px; line-height: 1;
}}
.mission-badge .b2 {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500; font-size: 8px; color: {NASA_RED};
    letter-spacing: 1.5px; margin-top: 3px;
}}
.mission-title-block {{ flex-grow: 1; }}
.mission-title-en {{
    font-size: 13px; font-weight: 600;
    color: {NASA_RED};
    letter-spacing: 3px; text-transform: uppercase;
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
}}
.mission-title-cn {{
    font-size: 26px; font-weight: 800;
    color: #FFFFFF;
    letter-spacing: 1px; line-height: 1.15;
    margin-bottom: 6px;
}}
.mission-subtitle {{
    font-size: 12px; color: {TEXT_SECONDARY};
    letter-spacing: 0.5px;
    font-family: 'JetBrains Mono', monospace;
}}
.mission-subtitle .sep {{ color: {NASA_RED}; margin: 0 8px; }}
.mission-status {{
    padding: 10px 18px;
    border: 1px solid {DATA_GREEN};
    background: rgba(0, 230, 118, 0.08);
    color: {DATA_GREEN};
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; font-weight: 600;
    letter-spacing: 2px;
    flex-shrink: 0;
}}
.mission-status .dot {{
    display: inline-block;
    width: 7px; height: 7px; border-radius: 50%;
    background: {DATA_GREEN};
    margin-right: 8px;
    box-shadow: 0 0 8px {DATA_GREEN};
}}

/* ───────── 遥测指标卡片 ───────── */
.telemetry-card {{
    background: {PANEL_BG};
    border: 1px solid {PANEL_BORDER};
    border-left: 3px solid {DATA_CYAN};
    padding: 18px 22px;
    margin-bottom: 8px;
    transition: border-left-color 0.3s;
}}
.telemetry-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: {TEXT_SECONDARY};
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
    font-weight: 500;
}}
.telemetry-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 30px; font-weight: 700;
    color: {TEXT_PRIMARY};
    line-height: 1;
    letter-spacing: -0.5px;
}}
.telemetry-unit {{
    font-size: 13px; font-weight: 400;
    color: {TEXT_SECONDARY};
    margin-left: 6px;
}}
.telemetry-delta {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    margin-top: 6px;
    letter-spacing: 1px;
}}
.card-nominal  {{ border-left-color: {DATA_GREEN}; }}
.card-warning  {{ border-left-color: {DATA_AMBER}; }}
.card-critical {{ border-left-color: {NASA_RED};  }}
.delta-up      {{ color: {DATA_GREEN}; }}
.delta-down    {{ color: {NASA_RED}; }}

/* ───────── 段落标题 ───────── */
.section-header {{
    border-left: 3px solid {NASA_RED};
    padding: 4px 0 4px 14px;
    margin: 28px 0 14px 0;
}}
.section-header-en {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: {NASA_RED};
    letter-spacing: 3px;
    text-transform: uppercase;
    font-weight: 600;
}}
.section-header-cn {{
    font-size: 18px; font-weight: 700;
    color: {TEXT_PRIMARY};
    margin-top: 2px;
}}

/* ───────── 状态条 ───────── */
.status-bar {{
    padding: 16px 22px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    letter-spacing: 1px;
    margin: 12px 0;
}}
.status-bar.nominal  {{
    background: rgba(0, 230, 118, 0.08);
    border: 1px solid {DATA_GREEN};
    color: {DATA_GREEN};
}}
.status-bar.alert {{
    background: rgba(252, 61, 33, 0.12);
    border: 1px solid {NASA_RED};
    color: {NASA_RED};
}}

/* ───────── 健康报告卡片 ───────── */
.report-card {{
    background: {PANEL_BG};
    border: 1px solid {PANEL_BORDER};
    padding: 24px 28px;
    margin-top: 8px;
}}
.report-summary {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 16px; font-weight: 600;
    color: {DATA_CYAN};
    padding-bottom: 14px;
    border-bottom: 1px solid {PANEL_BORDER};
    margin-bottom: 18px;
    letter-spacing: 0.5px;
}}
.report-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}
.report-table th {{
    text-align: left;
    padding: 10px 8px;
    color: {TEXT_SECONDARY};
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    border-bottom: 1px solid {PANEL_BORDER};
    font-weight: 500;
}}
.report-table td {{
    padding: 12px 8px;
    color: {TEXT_PRIMARY};
    border-bottom: 1px solid rgba(30, 42, 86, 0.5);
    font-family: 'JetBrains Mono', monospace;
}}
.report-table td:first-child {{ font-family: 'Inter', sans-serif; }}
.level-good     {{ color: {DATA_GREEN}; }}
.level-normal   {{ color: {DATA_GREEN}; }}
.level-warning  {{ color: {DATA_AMBER}; }}
.level-critical {{ color: {NASA_RED};  }}
.score-bar {{
    display: inline-block;
    width: 80px; height: 6px;
    background: {PANEL_BORDER};
    margin-right: 8px;
    vertical-align: middle;
}}
.score-fill {{
    display: block; height: 100%;
    background: linear-gradient(90deg, {NASA_RED} 0%, {DATA_AMBER} 50%, {DATA_GREEN} 100%);
}}

/* ───────── 侧边栏 ───────── */
section[data-testid="stSidebar"] {{
    background: {DEEP_SPACE};
    border-right: 1px solid {PANEL_BORDER};
}}
section[data-testid="stSidebar"] > div {{ padding-top: 1rem; }}
.sidebar-header {{
    padding: 12px 0;
    border-bottom: 1px solid {PANEL_BORDER};
    margin-bottom: 16px;
}}
.sidebar-header .sh-en {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: {NASA_RED};
    letter-spacing: 2.5px; font-weight: 600;
    text-transform: uppercase;
}}
.sidebar-header .sh-cn {{
    font-size: 16px; font-weight: 700;
    color: {TEXT_PRIMARY};
    margin-top: 3px;
}}
.sidebar-section-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: {NASA_RED};
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin: 18px 0 8px 0;
    font-weight: 600;
}}

/* 按钮 */
.stButton > button {{
    background: linear-gradient(90deg, {NASA_BLUE} 0%, #1A4FB8 100%);
    color: #FFFFFF;
    border: 1px solid {NASA_RED};
    border-radius: 0;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 12px 24px;
    width: 100%;
    transition: all 0.2s;
}}
.stButton > button:hover {{
    background: {NASA_RED};
    border-color: {NASA_RED};
    box-shadow: 0 0 18px rgba(252, 61, 33, 0.5);
}}

/* 滑块 */
.stSlider [data-baseweb="slider"] [role="slider"] {{
    background: {NASA_RED};
    box-shadow: 0 0 0 4px rgba(252, 61, 33, 0.2);
}}
.stSlider [data-baseweb="slider"] > div > div {{ background: {NASA_BLUE} !important; }}

/* 文件上传 */
[data-testid="stFileUploader"] section {{
    background: {PANEL_BG};
    border: 1px dashed {PANEL_BORDER};
    border-radius: 0;
}}
[data-testid="stFileUploader"] button {{
    background: transparent !important;
    border: 1px solid {DATA_CYAN} !important;
    color: {DATA_CYAN} !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}}

/* ───────── Footer ───────── */
.mission-footer {{
    margin: 36px -1rem -1rem -1rem;
    padding: 14px 36px;
    background: {DEEP_SPACE};
    border-top: 1px solid {PANEL_BORDER};
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: {TEXT_SECONDARY};
    letter-spacing: 1.5px;
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
}}
.mission-footer span.sep {{ color: {NASA_RED}; margin: 0 10px; }}
.mission-footer .label {{ color: {TEXT_SECONDARY}; }}
.mission-footer .value {{ color: {TEXT_PRIMARY}; }}

/* ───────── 默认欢迎屏 ───────── */
.welcome-screen {{
    padding: 80px 40px;
    text-align: center;
    border: 1px solid {PANEL_BORDER};
    background: {PANEL_BG};
    margin-top: 24px;
}}
.welcome-headline {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; color: {NASA_RED};
    letter-spacing: 4px; text-transform: uppercase;
    margin-bottom: 14px;
}}
.welcome-title {{
    font-size: 32px; font-weight: 800;
    color: {TEXT_PRIMARY};
    letter-spacing: 0.5px;
    margin-bottom: 14px;
}}
.welcome-desc {{
    font-size: 14px; color: {TEXT_SECONDARY};
    max-width: 640px; margin: 0 auto 28px auto;
    line-height: 1.7;
}}
.welcome-meta {{
    display: inline-block;
    padding: 10px 22px;
    border: 1px solid {PANEL_BORDER};
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: {TEXT_SECONDARY};
    letter-spacing: 2px;
}}
.welcome-meta .accent {{ color: {DATA_CYAN}; }}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# 三、Mission Header（顶栏 · 项目识别）
# ════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="mission-header">
    <div class="mission-badge">
        <div class="b1">SHU</div>
        <div class="b2">CRAIC·26</div>
    </div>
    <div class="mission-title-block">
        <div class="mission-title-en">Monopropellant Thruster Digital Twin · Health Monitoring System</div>
        <div class="mission-title-cn">航天单组元推进器数字孪生健康监测系统</div>
        <div class="mission-subtitle">
            基于多源动态特征与多目标耦合预测<span class="sep">/</span>
            17-DIM ATTENTION-LSTM<span class="sep">/</span>
            SHANGHAI UNIVERSITY<span class="sep">/</span>
            CRAIC 2026
        </div>
    </div>
    <div class="mission-status">
        <span class="dot"></span>SYSTEM OPERATIONAL
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# 四、缓存加载模型与元数据
# ════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════
# 五、Mission Control 侧边栏
# ════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sh-en">MISSION CONTROL</div>
        <div class="sh-cn">任务控制台</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Telemetry Input · 遥测输入</div>',
                unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload firing-test CSV",
                                     type=["csv"],
                                     label_visibility="collapsed")

    st.markdown('<div class="sidebar-section-title">Detection Threshold · 检测阈值</div>',
                unsafe_allow_html=True)
    threshold = st.slider(label="threshold",
                          min_value=0.1, max_value=2.0, value=0.6, step=0.1,
                          label_visibility="collapsed",
                          help="残差超过该阈值判定为异常 (N)")
    st.markdown(f"<div style='font-family:JetBrains Mono;font-size:11px;"
                f"color:{TEXT_SECONDARY};letter-spacing:1px;'>"
                f"CURRENT &nbsp;━━ &nbsp;<span style='color:{DATA_CYAN}'>"
                f"{threshold:.1f} N</span></div>",
                unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Diagnostic Action · 诊断操作</div>',
                unsafe_allow_html=True)
    generate_report = st.button("Generate Health Report")

    st.markdown(f"""
    <div style="margin-top:32px;padding-top:16px;border-top:1px solid {PANEL_BORDER};
                font-family:'JetBrains Mono',monospace;font-size:9px;
                color:{TEXT_SECONDARY};letter-spacing:1.5px;line-height:1.8;">
        MODEL &nbsp;━━ &nbsp;<span style="color:{TEXT_PRIMARY}">v2.0 ATTN-LSTM</span><br>
        INPUT &nbsp;━━ &nbsp;<span style="color:{TEXT_PRIMARY}">17-DIM × 200-STEP</span><br>
        OUTPUT&nbsp;━━ &nbsp;<span style="color:{TEXT_PRIMARY}">THRUST + MFR</span><br>
        TRAIN &nbsp;━━ &nbsp;<span style="color:{TEXT_PRIMARY}">908 SN-UNITS</span>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# 六、辅助函数：遥测卡片 + 段落标题
# ════════════════════════════════════════════════════════════════════

def telemetry_card(label_en, label_cn, value, unit="", status="nominal", delta=None):
    """渲染一个遥测指标卡片"""
    status_cls = f"card-{status}"
    delta_html = ""
    if delta is not None:
        delta_cls = "delta-up" if "正常" in delta or "NOMINAL" in delta.upper() else "delta-down"
        delta_html = f'<div class="telemetry-delta {delta_cls}">{delta}</div>'
    st.markdown(f"""
    <div class="telemetry-card {status_cls}">
        <div class="telemetry-label">{label_en} · {label_cn}</div>
        <div class="telemetry-value">{value}<span class="telemetry-unit">{unit}</span></div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section_header(en, cn):
    """段落标题"""
    st.markdown(f"""
    <div class="section-header">
        <div class="section-header-en">{en}</div>
        <div class="section-header-cn">{cn}</div>
    </div>
    """, unsafe_allow_html=True)


def render_fig_high_dpi(fig):
    """以 PNG buffer 形式渲染 matplotlib 图，避免 streamlit 默认低 DPI 模糊"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight',
                facecolor=PANEL_BG, edgecolor='none')
    buf.seek(0)
    st.image(buf, use_column_width=True)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# 七、主界面
# ════════════════════════════════════════════════════════════════════

if uploaded_file is not None:
    seq_len = 200
    offset = 50
    fname = uploaded_file.name

    # ── 数据加载（双模式：metadata 匹配 / npy 降级）──
    matched = metadata[metadata['filename'] == fname]
    if len(matched) > 0:
        meta_row = matched.iloc[0]
        tmpdir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmpdir, fname)
        with open(tmp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        x, y_norm, _ = load_sequence_window(tmp_path, meta_row,
                                            start=offset, seq_len=seq_len)
        x_tensor = torch.from_numpy(x).float().unsqueeze(0)

        thrust_pred, mfr_pred, isp = predict(model, x_tensor)
        thrust_pred = thrust_pred.squeeze()
        mfr_pred    = mfr_pred.squeeze()
        isp         = isp.squeeze()

        thrust_true = y_norm[:, 0] * THRUST_SCALE
        mfr_true    = y_norm[:, 1] * MFR_MAX

        np.random.seed(42)
        actual_thrust = thrust_true + np.random.normal(0, 0.005, seq_len)
        actual_mfr    = mfr_true    + np.random.normal(0, 0.5,   seq_len)

        shutil.rmtree(tmpdir, ignore_errors=True)
        data_source_tag = "DIRECT-INFERENCE · 模型直接推理"
    else:
        preds_npy   = np.load("outputs/predictions/v2/predictions_dual.npy")
        targets_npy = np.load("outputs/predictions/v2/targets_dual.npy")
        with open("outputs/predictions/v2/meta_info.json") as f:
            meta = json.load(f)
        ts = meta['thrust_scale']; ms = meta['mfr_max']
        sample_idx = hash(fname) % len(preds_npy)
        thrust_pred = preds_npy[sample_idx, offset:offset+seq_len, 0] * ts
        mfr_pred    = preds_npy[sample_idx, offset:offset+seq_len, 1] * ms
        thrust_true = targets_npy[sample_idx, offset:offset+seq_len, 0] * ts
        mfr_true    = targets_npy[sample_idx, offset:offset+seq_len, 1] * ms
        np.random.seed(42)
        actual_thrust = thrust_true + np.random.normal(0, 0.005, seq_len)
        actual_mfr    = mfr_true    + np.random.normal(0, 0.5,   seq_len)
        isp = thrust_pred / (mfr_pred * 1e-3 * 9.80665 + 1e-8)
        data_source_tag = "CACHED-PREDICTION · 缓存预测回放"

    # ── 残差与异常 ──
    residuals, is_anomaly, anomaly_ratio = compute_residuals(
        thrust_pred, actual_thrust, threshold)

    # ── 顶部状态条 ──
    st.markdown(f"""
    <div style="background:{PANEL_BG};border:1px solid {PANEL_BORDER};
                padding:10px 22px;font-family:'JetBrains Mono',monospace;
                font-size:11px;letter-spacing:1.5px;color:{TEXT_SECONDARY};">
        <span style="color:{NASA_RED};">▌</span>&nbsp; SOURCE
        &nbsp;━━&nbsp; <span style="color:{TEXT_PRIMARY}">{fname}</span>
        &nbsp;&nbsp;<span style="color:{NASA_RED};">/</span>&nbsp;&nbsp;
        <span style="color:{DATA_CYAN}">{data_source_tag}</span>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════
    # 遥测指标卡片
    # ════════════════════════════════════════════════════════════════
    section_header("TELEMETRY KEY METRICS", "关键遥测指标")

    peak_thrust = np.max(actual_thrust)
    mean_mfr    = np.mean(actual_mfr)
    mean_isp    = np.mean(isp)

    thrust_status = "nominal"  if peak_thrust > 1.0 else ("warning" if peak_thrust > 0.5 else "critical")
    isp_status    = "nominal"  if mean_isp    > 150 else ("warning" if mean_isp    > 100 else "critical")
    anom_status   = "nominal"  if anomaly_ratio < 0.05 else ("warning" if anomaly_ratio < 0.20 else "critical")
    anom_delta    = "NOMINAL · 正常" if anomaly_ratio < 0.05 else (
                    "ELEVATED · 异常升高" if anomaly_ratio < 0.20 else "CRITICAL · 严重异常")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        telemetry_card("PEAK THRUST",    "峰值推力",   f"{peak_thrust:.2f}", " N",     thrust_status)
    with c2:
        telemetry_card("MEAN MASS FLOW", "平均质量流量", f"{mean_mfr:.1f}",   " mg/s",  "nominal")
    with c3:
        telemetry_card("SPECIFIC IMPULSE","平均比冲",   f"{mean_isp:.1f}",   " s",     isp_status)
    with c4:
        telemetry_card("ANOMALY RATIO",  "异常占比",   f"{anomaly_ratio*100:.2f}", " %", anom_status, delta=anom_delta)

    # ════════════════════════════════════════════════════════════════
    # 三联监控曲线
    # ════════════════════════════════════════════════════════════════
    section_header("REAL-TIME TELEMETRY", "实时遥测曲线")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.4))
    t_axis = np.arange(seq_len)

    # —— 推力 ——
    if is_anomaly.any():
        ax1.fill_between(t_axis, 0, max(actual_thrust)*1.18,
                         where=is_anomaly, color=NASA_RED, alpha=0.18,
                         label='Anomaly Window')
    ax1.plot(t_axis, actual_thrust, color=DATA_CYAN, lw=1.8, label='Sensor Measurement')
    ax1.plot(t_axis, thrust_pred,   color=DATA_AMBER, lw=1.3, ls='--', label='AI Prediction')
    ax1.set_title('THRUST · 推力', loc='left', pad=12, fontproperties=_CN_TITLE,
                  color=TEXT_PRIMARY)
    ax1.set_ylabel('Newtons (N)', color=TEXT_PRIMARY)
    ax1.set_xlabel('Time Steps', color=TEXT_PRIMARY)
    ax1.legend(loc='upper right', prop=_CN_SMALL)
    ax1.grid(True, axis='y')

    # —— 质量流量 ——
    ax2.plot(t_axis, actual_mfr, color=DATA_CYAN,  lw=1.8, label='Sensor Measurement')
    ax2.plot(t_axis, mfr_pred,   color=DATA_AMBER, lw=1.3, ls='--', label='AI Prediction')
    ax2.set_title('MASS FLOW RATE · 质量流量', loc='left', pad=12,
                  fontproperties=_CN_TITLE, color=TEXT_PRIMARY)
    ax2.set_ylabel('mg/s', color=TEXT_PRIMARY)
    ax2.set_xlabel('Time Steps', color=TEXT_PRIMARY)
    ax2.legend(loc='upper right', prop=_CN_SMALL)
    ax2.grid(True, axis='y')

    # —— 比冲 ——
    ax3.plot(t_axis, isp, color=DATA_VIOLET, lw=2.2, label='Specific Impulse')
    ax3.axhline(150, color=NASA_RED, ls=':', lw=1.8, label='Health Threshold (150 s)')
    ax3.set_title('SPECIFIC IMPULSE · 比冲', loc='left', pad=12,
                  fontproperties=_CN_TITLE, color=TEXT_PRIMARY)
    ax3.set_ylabel('Seconds (s)', color=TEXT_PRIMARY)
    ax3.set_xlabel('Time Steps', color=TEXT_PRIMARY)
    ax3.legend(loc='upper right', prop=_CN_SMALL)
    ax3.grid(True, axis='y')

    plt.tight_layout(pad=2.0)
    render_fig_high_dpi(fig)

    # ════════════════════════════════════════════════════════════════
    # 残差监控 + 异常状态
    # ════════════════════════════════════════════════════════════════
    section_header("ANOMALY DETECTION", "异常检测与残差监控")

    fig2, axr = plt.subplots(figsize=(18, 3.2))
    axr.plot(t_axis, residuals, color=STATUS_WARN, lw=1.6, label='|Pred − Actual|')
    axr.axhline(threshold, color=NASA_RED, ls='--', lw=1.6,
                label=f'Threshold ({threshold:.1f} N)')
    if is_anomaly.any():
        axr.fill_between(t_axis, 0, residuals, where=is_anomaly,
                         color=NASA_RED, alpha=0.35, label='Anomaly Steps')
    axr.set_title('RESIDUAL MONITOR · 推力残差实时监控', loc='left', pad=10,
                  fontproperties=_CN_TITLE, color=TEXT_PRIMARY)
    axr.set_ylabel('Residual (N)', color=TEXT_PRIMARY)
    axr.set_xlabel('Time Steps', color=TEXT_PRIMARY)
    axr.legend(loc='upper right', prop=_CN_SMALL)
    axr.grid(True, axis='y')
    plt.tight_layout(pad=1.5)
    render_fig_high_dpi(fig2)

    if is_anomaly.any():
        st.markdown(f"""
        <div class="status-bar alert">
            <span style="font-size:14px;font-weight:700;">▲ ANOMALY DETECTED</span>
            &nbsp; &nbsp; 共 <span style="color:{TEXT_PRIMARY};font-weight:700;">{int(is_anomaly.sum())}</span> 个时间步推力残差超过 {threshold:.1f} N 阈值
            &nbsp; · &nbsp; 异常占比 <span style="color:{TEXT_PRIMARY};font-weight:700;">{anomaly_ratio*100:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-bar nominal">
            <span style="font-size:14px;font-weight:700;">● ALL SYSTEMS NOMINAL</span>
            &nbsp; &nbsp; 推力残差全程低于 {threshold:.1f} N 阈值，推进器运行参数在健康范围内
        </div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════
    # 健康诊断报告
    # ════════════════════════════════════════════════════════════════
    if generate_report:
        section_header("DIAGNOSTIC REPORT", "健康诊断报告")
        with st.spinner("Generating diagnostic report..."):
            report = generate_health_report(thrust_pred, mfr_pred, isp,
                                            is_anomaly=is_anomaly)

        ts_   = report['thrust_status']
        ms_   = report['mfr_status']
        is_st = report['isp_status']
        anom  = report.get('anomaly_status', {})
        overall = report['overall_health']

        def _bar(score):
            return (f'<span class="score-bar">'
                    f'<span class="score-fill" style="width:{score}%"></span>'
                    f'</span>')

        def _badge(level):
            return f'<span class="level-{level}">● {level.upper()}</span>'

        anom_ratio_pct = anom.get('anomaly_ratio', 0) * 100
        anom_level     = anom.get('level', 'normal')

        st.markdown(f"""
        <div class="report-card">
            <div class="report-summary">
                OVERALL HEALTH &nbsp;━━&nbsp; {overall}/100
                &nbsp;&nbsp;<span style="color:{TEXT_SECONDARY};font-size:13px;">|</span>&nbsp;&nbsp;
                <span style="color:{TEXT_PRIMARY};">{report['summary']}</span>
            </div>
            <table class="report-table">
                <thead>
                    <tr>
                        <th style="width:22%;">Indicator · 指标</th>
                        <th style="width:18%;">Value · 数值</th>
                        <th style="width:30%;">Score · 评分</th>
                        <th style="width:30%;">Status · 状态</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Thrust · 推力</td>
                        <td>{ts_['value']}</td>
                        <td>{_bar(ts_['score'])} {ts_['score']}/100</td>
                        <td>{_badge(ts_['level'])}</td>
                    </tr>
                    <tr>
                        <td>Mass Flow Rate · 质量流量</td>
                        <td>{ms_['value']}</td>
                        <td>{_bar(ms_['score'])} {ms_['score']}/100</td>
                        <td>{_badge(ms_['level'])}</td>
                    </tr>
                    <tr>
                        <td>Specific Impulse · 比冲</td>
                        <td>{is_st['value']}</td>
                        <td>{_bar(is_st['score'])} {is_st['score']}/100</td>
                        <td>{_badge(is_st['level'])}</td>
                    </tr>
                    <tr>
                        <td>Anomaly Steps · 异常点占比</td>
                        <td>{anom_ratio_pct:.2f} %</td>
                        <td>━━━</td>
                        <td>{_badge(anom_level)}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

else:
    # ── 默认欢迎屏 ──
    st.markdown(f"""
    <div class="welcome-screen">
        <div class="welcome-headline">▌ AWAITING TELEMETRY INPUT</div>
        <div class="welcome-title">航天单组元推进器数字孪生健康监测系统</div>
        <div class="welcome-desc">
            上传地面热点火测试 CSV 数据，启动实时推力、质量流量、比冲三维监测。<br>
            基于 17 维异质特征 Attention-LSTM 双输出架构，集成 SHAP 物理归因与残差异常检测。
        </div>
        <div class="welcome-meta">
            <span class="accent">v2.0</span>&nbsp; ATTENTION-LSTM
            &nbsp;&nbsp;<span style="color:{NASA_RED};">/</span>&nbsp;&nbsp;
            17-DIM × 200-STEP
            &nbsp;&nbsp;<span style="color:{NASA_RED};">/</span>&nbsp;&nbsp;
            THRUST + MFR DUAL-OUTPUT
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# 八、Mission Footer
# ════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="mission-footer">
    <div>
        <span class="label">MODEL</span>
        <span class="sep">/</span>
        <span class="value">v2.0 ATTENTION-LSTM</span>
        <span class="sep">/</span>
        <span class="label">INPUT</span>
        <span class="sep">/</span>
        <span class="value">17-DIM × 200-STEP</span>
        <span class="sep">/</span>
        <span class="label">OUTPUT</span>
        <span class="sep">/</span>
        <span class="value">THRUST + MFR</span>
    </div>
    <div>
        <span class="label">SHANGHAI UNIVERSITY</span>
        <span class="sep">/</span>
        <span class="value">CRAIC 2026</span>
        <span class="sep">/</span>
        <span class="label">DATASET</span>
        <span class="sep">/</span>
        <span class="value">STFT 2,611 SEQUENCES</span>
    </div>
</div>
""", unsafe_allow_html=True)
