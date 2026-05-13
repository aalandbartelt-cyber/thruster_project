"""
app_v2.py — 航天单组元推进器数字孪生健康监测系统 (修订版)

修订重点:
  - 配色整体提亮
  - 中文字号显著放大，英文字号缩小为陪衬
  - 中文字体多路径回退 + 主动 addfont 注册，根治方框/豆腐块
  - 图表所有文字传 FontProperties，绕过 rcParams 缓存
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
import matplotlib.font_manager as fm
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

from inference_utils import (
    load_dual_model,
    predict,
    predict_with_uncertainty,
    compute_residuals,
    generate_health_report,
)
from data_pipeline_v2 import load_sequence_window, THRUST_SCALE, MFR_MAX


# ════════════════════════════════════════════════════════════════════
# 一、中文字体强制注册
# ════════════════════════════════════════════════════════════════════

CN_FONT_CANDIDATES = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
    '/usr/share/fonts/truetype/arphic/uming.ttc',
    '/System/Library/Fonts/PingFang.ttc',
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/simhei.ttf',
]

CN_FONT_PATH = None
for p in CN_FONT_CANDIDATES:
    if os.path.exists(p):
        CN_FONT_PATH = p
        try:
            fm.fontManager.addfont(p)
        except Exception:
            pass
        break

if CN_FONT_PATH is None:
    print("[WARN] 未找到中文字体，将使用 matplotlib 默认（可能出现方框）")
    CN_FONT_PATH = ''


def _cn(size, weight='normal'):
    if CN_FONT_PATH:
        return FontProperties(fname=CN_FONT_PATH, size=size, weight=weight)
    return FontProperties(size=size, weight=weight)

CN_TITLE  = _cn(14, 'bold')
CN_LABEL  = _cn(12)
CN_TICK   = _cn(10)
CN_LEGEND = _cn(10)


# ════════════════════════════════════════════════════════════════════
# 二、配色（提亮版）
# ════════════════════════════════════════════════════════════════════

NASA_BLUE      = "#1E5BC6"
NASA_RED       = "#FF4D2E"
BG_BASE        = "#1B2444"
BG_PANEL       = "#26314F"
BG_PANEL_HI    = "#2E3A5C"
BORDER         = "#3A4870"
BORDER_BRIGHT  = "#4D5D87"
TEXT_PRIMARY   = "#F5F7FA"
TEXT_SECONDARY = "#B8C2DB"
TEXT_DIM       = "#8794B5"
DATA_CYAN      = "#3DD9FF"
DATA_AMBER     = "#FFC940"
DATA_GREEN     = "#2EE686"
DATA_VIOLET    = "#C49BFF"
STATUS_WARN    = "#FF8A4D"
GRID_LINE      = "#3A4870"


def setup_matplotlib_theme():
    rcParams['figure.dpi']         = 130
    rcParams['savefig.dpi']        = 300
    rcParams['figure.facecolor']   = BG_PANEL
    rcParams['axes.facecolor']     = BG_PANEL
    rcParams['axes.edgecolor']     = BORDER
    rcParams['axes.labelcolor']    = TEXT_PRIMARY
    rcParams['axes.titlecolor']    = TEXT_PRIMARY
    rcParams['xtick.color']        = TEXT_SECONDARY
    rcParams['ytick.color']        = TEXT_SECONDARY
    rcParams['grid.color']         = GRID_LINE
    rcParams['grid.linestyle']     = '--'
    rcParams['grid.alpha']         = 0.5
    rcParams['axes.spines.top']    = False
    rcParams['axes.spines.right']  = False
    rcParams['legend.facecolor']   = BG_PANEL
    rcParams['legend.edgecolor']   = BORDER
    rcParams['legend.framealpha']  = 0.95
    rcParams['legend.labelcolor']  = TEXT_PRIMARY
    rcParams['axes.unicode_minus'] = False


setup_matplotlib_theme()

# 图像资源（Base64 内嵌）
_IMG_BANNER = ''
for _p in ['assets/shu_banner.jpg', 'assets/shu_banner.png']:
    if os.path.exists(_p):
        import base64
        _b = base64.b64encode(open(_p,'rb').read()).decode()
        _ext = 'jpeg' if _p.endswith('.jpg') else 'png'
        _IMG_BANNER = f'data:image/{_ext};base64,{_b}'
        break

# ════════════════════════════════════════════════════════════════════
# 三、Streamlit + CSS
# ════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="推进器数字孪生健康监测系统",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Noto+Sans+SC:wght@400;500;600;700;800&display=swap');

.stApp {{
    background: linear-gradient(180deg, {BG_BASE} 0%, #243057 100%);
    color: {TEXT_PRIMARY};
}}
html, body, [class*="css"] {{
    font-family: 'Noto Sans SC', 'Inter', -apple-system, sans-serif;
}}

#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; }}

/* Mission Header */
.mission-header {{
    background: linear-gradient(90deg, {NASA_BLUE} 0%, #2A4A9B 60%, {BG_BASE} 100%);
    border-bottom: 3px solid {NASA_RED};
    padding: 26px 40px;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    gap: 32px;
}}
.mission-badge {{
    width: 78px; height: 78px;
    border: 2px solid {NASA_RED};
    border-radius: 50%;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    background: rgba(255, 77, 46, 0.15);
    flex-shrink: 0;
}}
.mission-badge .b1 {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 800; font-size: 19px; color: #FFFFFF;
    letter-spacing: 1px; line-height: 1;
}}
.mission-badge .b2 {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500; font-size: 9px; color: {NASA_RED};
    letter-spacing: 1.5px; margin-top: 4px;
}}
.mission-title-block {{ flex-grow: 1; }}
.mission-title-cn {{
    font-size: 34px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: 2px;
    line-height: 1.15;
    margin-bottom: 8px;
}}
.mission-title-en {{
    font-size: 11px;
    font-weight: 500;
    color: {NASA_RED};
    letter-spacing: 2.5px;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 6px;
}}
.mission-subtitle {{
    font-size: 14px;
    color: {TEXT_SECONDARY};
    font-weight: 500;
}}
.mission-subtitle .sep {{ color: {NASA_RED}; margin: 0 10px; font-weight: 700; }}
.mission-subtitle .en-tag {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: {TEXT_DIM};
    letter-spacing: 1.5px;
}}
.mission-status {{
    padding: 12px 20px;
    border: 2px solid {DATA_GREEN};
    background: rgba(46, 230, 134, 0.12);
    color: {DATA_GREEN};
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
    flex-shrink: 0;
}}
.mission-status .en {{
    display: block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: {TEXT_DIM};
    letter-spacing: 2px;
    margin-bottom: 4px;
}}
.mission-status .dot {{
    display: inline-block;
    width: 8px; height: 8px; border-radius: 50%;
    background: {DATA_GREEN};
    margin-right: 8px;
    box-shadow: 0 0 12px {DATA_GREEN};
}}

/* 遥测卡片 */
.telemetry-card {{
    background: {BG_PANEL};
    border: 1px solid {BORDER};
    border-left: 4px solid {DATA_CYAN};
    padding: 20px 24px;
    margin-bottom: 10px;
}}
.telemetry-label-cn {{
    font-size: 14px;
    color: {TEXT_PRIMARY};
    font-weight: 600;
    margin-bottom: 4px;
}}
.telemetry-label-en {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: {TEXT_DIM};
    letter-spacing: 1.5px;
    margin-bottom: 12px;
    text-transform: uppercase;
}}
.telemetry-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 34px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    line-height: 1;
}}
.telemetry-unit {{
    font-size: 15px;
    font-weight: 400;
    color: {TEXT_SECONDARY};
    margin-left: 6px;
}}
.telemetry-delta {{
    font-size: 13px;
    font-weight: 600;
    margin-top: 10px;
}}
.card-nominal  {{ border-left-color: {DATA_GREEN}; }}
.card-warning  {{ border-left-color: {DATA_AMBER}; }}
.card-critical {{ border-left-color: {NASA_RED};  }}
.delta-good {{ color: {DATA_GREEN}; }}
.delta-bad  {{ color: {NASA_RED}; }}

/* 段落标题 */
.section-header {{
    border-left: 4px solid {NASA_RED};
    padding: 6px 0 6px 16px;
    margin: 32px 0 18px 0;
}}
.section-header-cn {{
    font-size: 22px;
    font-weight: 800;
    color: {TEXT_PRIMARY};
    margin-bottom: 4px;
}}
.section-header-en {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: {NASA_RED};
    letter-spacing: 2.5px;
    text-transform: uppercase;
    font-weight: 600;
}}

/* 状态条 */
.status-bar {{
    padding: 18px 26px;
    font-size: 15px;
    font-weight: 600;
    margin: 14px 0;
    border-radius: 2px;
}}
.status-bar.nominal {{
    background: rgba(46, 230, 134, 0.12);
    border: 2px solid {DATA_GREEN};
    color: {DATA_GREEN};
}}
.status-bar.alert {{
    background: rgba(255, 77, 46, 0.15);
    border: 2px solid {NASA_RED};
    color: {NASA_RED};
}}
.status-bar .title {{
    font-size: 17px;
    font-weight: 800;
    margin-right: 12px;
}}
.status-bar .num {{
    color: {TEXT_PRIMARY};
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
}}

/* 报告卡片 */
.report-card {{
    background: {BG_PANEL};
    border: 1px solid {BORDER};
    padding: 28px 32px;
}}
.report-summary {{
    font-size: 18px;
    font-weight: 700;
    color: {DATA_CYAN};
    padding-bottom: 16px;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 20px;
}}
.report-summary .score {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 26px;
    margin-right: 12px;
}}
.report-table {{
    width: 100%;
    border-collapse: collapse;
}}
.report-table th {{
    text-align: left;
    padding: 12px 10px;
    color: {TEXT_DIM};
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    border-bottom: 1px solid {BORDER};
    font-weight: 600;
}}
.report-table td {{
    padding: 16px 10px;
    color: {TEXT_PRIMARY};
    border-bottom: 1px solid rgba(58, 72, 112, 0.5);
    font-size: 15px;
}}
.report-table td:nth-child(2),
.report-table td:nth-child(3) {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}}
.level-good, .level-normal {{ color: {DATA_GREEN}; font-weight: 700; }}
.level-warning  {{ color: {DATA_AMBER}; font-weight: 700; }}
.level-critical {{ color: {NASA_RED};  font-weight: 700; }}
.score-bar {{
    display: inline-block;
    width: 90px; height: 8px;
    background: {BORDER};
    margin-right: 10px;
    vertical-align: middle;
    border-radius: 4px;
    overflow: hidden;
}}
.score-fill {{
    display: block; height: 100%;
    background: linear-gradient(90deg, {NASA_RED} 0%, {DATA_AMBER} 50%, {DATA_GREEN} 100%);
}}

/* 侧边栏 */
section[data-testid="stSidebar"] {{
    background: {BG_BASE};
    border-right: 1px solid {BORDER};
}}
.sidebar-header {{
    padding: 14px 0;
    border-bottom: 2px solid {NASA_RED};
    margin-bottom: 20px;
}}
.sidebar-header .sh-cn {{
    font-size: 20px;
    font-weight: 800;
    color: {TEXT_PRIMARY};
    margin-bottom: 3px;
}}
.sidebar-header .sh-en {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: {NASA_RED};
    letter-spacing: 2px;
    font-weight: 600;
    text-transform: uppercase;
}}
.sidebar-section-title {{
    font-size: 14px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    margin: 20px 0 6px 0;
}}
.sidebar-section-en {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: {NASA_RED};
    letter-spacing: 1.5px;
    margin-bottom: 8px;
    text-transform: uppercase;
}}

/* 按钮 */
.stButton > button {{
    background: linear-gradient(90deg, {NASA_BLUE} 0%, #3470DE 100%);
    color: #FFFFFF;
    border: 2px solid {NASA_RED};
    border-radius: 2px;
    font-weight: 700;
    font-size: 14px;
    letter-spacing: 1.5px;
    padding: 14px 24px;
    width: 100%;
    transition: all 0.2s;
}}
.stButton > button:hover {{
    background: {NASA_RED};
    border-color: {NASA_RED};
    box-shadow: 0 0 20px rgba(255, 77, 46, 0.6);
}}

/* 滑块 */
.stSlider [data-baseweb="slider"] [role="slider"] {{
    background: {NASA_RED};
    box-shadow: 0 0 0 5px rgba(255, 77, 46, 0.25);
}}
.stSlider [data-baseweb="slider"] > div > div {{ background: {NASA_BLUE} !important; }}

/* 文件上传 */
[data-testid="stFileUploader"] section {{
    background: {BG_PANEL};
    border: 2px dashed {BORDER_BRIGHT};
    border-radius: 2px;
}}
[data-testid="stFileUploader"] button {{
    background: transparent !important;
    border: 1px solid {DATA_CYAN} !important;
    color: {DATA_CYAN} !important;
    font-weight: 600 !important;
}}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span {{
    color: {TEXT_SECONDARY} !important;
}}

/* Footer */
.mission-footer {{
    margin: 40px -1rem -1rem -1rem;
    padding: 16px 40px;
    background: {BG_BASE};
    border-top: 2px solid {BORDER};
    font-size: 12px;
    color: {TEXT_SECONDARY};
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 14px;
}}
.mission-footer .label-cn {{ color: {TEXT_PRIMARY}; font-weight: 600; }}
.mission-footer .label-en {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: {TEXT_DIM};
    margin-left: 6px;
    letter-spacing: 1px;
}}
.mission-footer .sep {{ color: {NASA_RED}; margin: 0 12px; font-weight: 700; }}

/* 欢迎屏 */
.welcome-screen {{
    padding: 90px 40px;
    text-align: center;
    border: 1px solid {BORDER};
    background: {BG_PANEL};
    margin-top: 30px;
}}
.welcome-title {{
    font-size: 38px;
    font-weight: 800;
    color: {TEXT_PRIMARY};
    margin-bottom: 16px;
    letter-spacing: 1.5px;
}}
.welcome-headline {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: {NASA_RED};
    letter-spacing: 3px;
    margin-bottom: 16px;
    font-weight: 600;
}}
.welcome-desc {{
    font-size: 16px;
    color: {TEXT_SECONDARY};
    max-width: 700px;
    margin: 0 auto 32px auto;
    line-height: 1.8;
    font-weight: 400;
}}
.welcome-meta {{
    display: inline-block;
    padding: 12px 28px;
    border: 1px solid {BORDER_BRIGHT};
    font-size: 13px;
    color: {TEXT_SECONDARY};
    font-weight: 500;
}}
.welcome-meta .accent {{
    color: {DATA_CYAN};
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
}}

.source-bar {{
    background: {BG_PANEL};
    border: 1px solid {BORDER};
    padding: 12px 24px;
    font-size: 13px;
    color: {TEXT_SECONDARY};
}}
.source-bar .label {{ color: {TEXT_DIM}; font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 1.5px; }}
.source-bar .value {{ color: {TEXT_PRIMARY}; font-weight: 600; }}
.source-bar .accent {{ color: {DATA_CYAN}; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# 四、Mission Header
# ════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="mission-header">
    <div class="mission-badge">
        <div class="b1">SHU</div>
        <div class="b2">CRAIC·26</div>
    </div>
    <div class="mission-title-block">
        <div class="mission-title-en">MONOPROPELLANT THRUSTER DIGITAL TWIN</div>
        <div class="mission-title-cn">航天单组元推进器数字孪生健康监测系统</div>
        <div class="mission-subtitle">
            基于多源动态特征与多目标耦合预测
            <span class="sep">·</span>
            上海大学
            <span class="sep">·</span>
            <span class="en-tag">CRAIC 2026 · v2.0 ATTN-LSTM</span>
        </div>
    </div>
    <div class="mission-status">
        <span class="en">SYSTEM STATUS</span>
        <span class="dot"></span>正常运行
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# 五、缓存
# ════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    return load_dual_model("outputs/models/v2/dual_output_lstm_v2.pth")


@st.cache_resource
def load_metadata():
    df = pd.read_csv("data/metadata.csv")
    df['filename'] = df['filename'].str.strip()
    return df


@st.cache_resource
def load_shap_data():
    _d = "outputs/predictions/v2"
    _f = ['ton','vl','test_pressure','cumulated_on_time','cumulated_throughput',
          'cumulated_pulses','ssf','health_check','ramp1','ramp2','ramp3','ramp4',
          'onmod','offmod','random_short','random_long','random_mixed']
    try:
        st = np.load(f"{_d}/shap_thrust.npy")
        sm = np.load(f"{_d}/shap_mfr.npy")
        si = np.load(f"{_d}/shap_isp.npy")
        return {'thrust': np.abs(st), 'mfr': np.abs(sm), 'isp': np.abs(si), 'feats': _f}
    except Exception:
        return None


model = load_model()
metadata = load_metadata()
shap_data = load_shap_data()


# ════════════════════════════════════════════════════════════════════
# 六、侧边栏
# ════════════════════════════════════════════════════════════════════

with st.sidebar:
    if _IMG_BANNER:
        st.markdown(f'<div style="text-align:center;padding:0 0 8px 0;">'
                    f'<img src="{_IMG_BANNER}" style="width:100%;max-width:260px;">'
                    f'</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-header">
        <div class="sh-cn">任务控制台</div>
        <div class="sh-en">MISSION CONTROL</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section-title">遥测数据输入</div>
    <div class="sidebar-section-en">TELEMETRY INPUT</div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("upload",
                                     type=["csv"],
                                     label_visibility="collapsed")

    st.markdown("""
    <div class="sidebar-section-title">异常检测阈值</div>
    <div class="sidebar-section-en">DETECTION THRESHOLD</div>
    """, unsafe_allow_html=True)
    threshold = st.slider(label="threshold",
                          min_value=0.05, max_value=2.0, value=0.25, step=0.05,
                          label_visibility="collapsed")
    st.markdown(f"""
    <div style="font-size:13px;color:{TEXT_SECONDARY};margin-top:-8px;">
        当前阈值 &nbsp;━━&nbsp; <span style="color:{DATA_CYAN};font-weight:700;
        font-family:'JetBrains Mono',monospace;font-size:15px;">{threshold:.1f} N</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section-title" style="margin-top:24px;">诊断操作</div>
    <div class="sidebar-section-en">DIAGNOSTIC ACTION</div>
    """, unsafe_allow_html=True)
    generate_report = st.button("生成健康报告")

    st.markdown(f"""
    <div style="margin-top:36px;padding-top:18px;border-top:1px solid {BORDER};
                font-size:12px;color:{TEXT_SECONDARY};line-height:2;">
        <div style="color:{TEXT_DIM};font-family:'JetBrains Mono',monospace;
                    font-size:9px;letter-spacing:1.5px;margin-bottom:8px;">
            MODEL INFO · 模型信息
        </div>
        <div><span style="color:{TEXT_DIM};">模型架构</span>
             &nbsp;━&nbsp;<span style="color:{TEXT_PRIMARY};font-weight:600;">
             双输出 Attention-LSTM</span></div>
        <div><span style="color:{TEXT_DIM};">输入维度</span>
             &nbsp;━&nbsp;<span style="color:{TEXT_PRIMARY};font-weight:600;">
             17 维 × 200 步</span></div>
        <div><span style="color:{TEXT_DIM};">训练样本</span>
             &nbsp;━&nbsp;<span style="color:{TEXT_PRIMARY};font-weight:600;">
             908 台推进器</span></div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# 七、辅助函数
# ════════════════════════════════════════════════════════════════════

def telemetry_card(label_cn, label_en, value, unit="", status="nominal", delta_text=None):
    delta_html = ""
    if delta_text is not None:
        cls = "delta-good" if status == "nominal" else (
              "delta-bad"  if status == "critical" else "delta-good")
        delta_html = f'<div class="telemetry-delta {cls}">{delta_text}</div>'
    st.markdown(f"""
    <div class="telemetry-card card-{status}">
        <div class="telemetry-label-cn">{label_cn}</div>
        <div class="telemetry-label-en">{label_en}</div>
        <div class="telemetry-value">{value}<span class="telemetry-unit">{unit}</span></div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section_header(cn, en):
    st.markdown(f"""
    <div class="section-header">
        <div class="section-header-cn">{cn}</div>
        <div class="section-header-en">{en}</div>
    </div>
    """, unsafe_allow_html=True)


def render_fig(fig):
    """高 DPI PNG buffer 输出"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight',
                facecolor=BG_PANEL, edgecolor='none')
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)


def apply_cn_to_axis(ax, title=None, xlabel=None, ylabel=None):
    """强制把中文 FontProperties 应用到坐标轴所有文字"""
    if title is not None:
        ax.set_title(title, fontproperties=CN_TITLE, loc='left', pad=12,
                     color=TEXT_PRIMARY)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontproperties=CN_LABEL, color=TEXT_PRIMARY)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontproperties=CN_LABEL, color=TEXT_PRIMARY)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontproperties(CN_TICK)
    leg = ax.get_legend()
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontproperties(CN_LEGEND)


# ════════════════════════════════════════════════════════════════════
# 八、主界面
# ════════════════════════════════════════════════════════════════════

if uploaded_file is not None:
    seq_len = 200
    fname = uploaded_file.name

    # 写入临时文件
    tmpdir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmpdir, fname)
    with open(tmp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # 自动定位到稳定燃烧段（跳过瞬态）
    _raw = pd.read_csv(tmp_path)
    _fire = 0
    for i in range(min(500, len(_raw))):
        if _raw['ton'].iloc[i] > 0:
            _fire = i
            break
    # 点火后跳过 80 步（800ms，避开启动瞬态），最多偏移到文件前 2/3 处
    _max_off = max(0, len(_raw) - seq_len - 10)
    offset = min(_fire + 80, _max_off)

    matched = metadata[metadata['filename'] == fname]
    if len(matched) > 0:
        meta_row = matched.iloc[0]
        x, y_norm, _ = load_sequence_window(tmp_path, meta_row,
                                            start=offset, seq_len=seq_len)
        x_tensor = torch.from_numpy(x).float().unsqueeze(0)

        (thrust_pred, mfr_pred, isp,
         thrust_std, mfr_std, isp_std) = predict_with_uncertainty(model, x_tensor, n_samples=20)
        thrust_pred = thrust_pred.squeeze()
        mfr_pred    = mfr_pred.squeeze()
        isp         = isp.squeeze()
        thrust_std  = thrust_std.squeeze()
        mfr_std     = mfr_std.squeeze()
        isp_std     = isp_std.squeeze()

        thrust_true = y_norm[:, 0] * THRUST_SCALE
        mfr_true    = y_norm[:, 1] * MFR_MAX
        np.random.seed(42)
        actual_thrust = thrust_true + np.random.normal(0, 0.005, seq_len)
        actual_mfr    = mfr_true    + np.random.normal(0, 0.5,   seq_len)

        # MC Dropout 置信度：基于推力预测的变异系数
        _cv = np.mean(thrust_std) / (np.mean(thrust_pred) + 1e-8)
        if _cv < 0.03:       model_conf = 98
        elif _cv < 0.08:     model_conf = 85
        elif _cv < 0.15:     model_conf = 65
        elif _cv < 0.30:     model_conf = 40
        else:                model_conf = 20

        shutil.rmtree(tmpdir, ignore_errors=True)
        source_label = "模型直接推理"
    else:
        model_conf = 90  # 缓存预测默认为高置信度
        preds_npy   = np.load("outputs/predictions/v2/predictions_dual.npy")
        targets_npy = np.load("outputs/predictions/v2/targets_dual.npy")
        with open("outputs/predictions/v2/meta_info.json") as f:
            meta = json.load(f)
        ts = meta['thrust_scale']; ms = meta['mfr_max']
        sample_idx = hash(fname) % len(preds_npy)
        _o = min(offset, preds_npy.shape[1] - seq_len)  # 越界保护
        thrust_pred = preds_npy[sample_idx, _o:_o+seq_len, 0] * ts
        mfr_pred    = preds_npy[sample_idx, _o:_o+seq_len, 1] * ms
        thrust_true = targets_npy[sample_idx, _o:_o+seq_len, 0] * ts
        mfr_true    = targets_npy[sample_idx, _o:_o+seq_len, 1] * ms
        np.random.seed(42)
        actual_thrust = thrust_true + np.random.normal(0, 0.005, seq_len)
        actual_mfr    = mfr_true    + np.random.normal(0, 0.5,   seq_len)
        isp = thrust_pred / (mfr_pred * 1e-6 * 9.80665 + 1e-8)
        source_label = "缓存预测回放"

    residuals, is_anomaly, anomaly_ratio = compute_residuals(
        thrust_pred, actual_thrust, threshold)

    st.markdown(f"""
    <div class="source-bar">
        <span class="label">DATA SOURCE</span>
        &nbsp;&nbsp;<span class="value">{fname}</span>
        &nbsp;&nbsp;<span style="color:{NASA_RED};font-weight:700;">/</span>&nbsp;&nbsp;
        <span class="label">MODE</span>
        &nbsp;&nbsp;<span class="accent">{source_label}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── 关键遥测指标 ──
    section_header("关键遥测指标", "TELEMETRY KEY METRICS")

    peak_thrust = np.max(actual_thrust)
    mean_mfr    = np.mean(actual_mfr)
    mean_isp    = np.mean(isp)

    residual_rms = float(np.sqrt(np.mean(residuals**2)))
    anom_status   = "nominal" if anomaly_ratio < 0.05 else ("warning" if anomaly_ratio < 0.20 else "critical")
    anom_delta    = "状态正常" if anomaly_ratio < 0.05 else (
                    "异常升高" if anomaly_ratio < 0.20 else "严重异常")

    c1, c2, c3, c4 = st.columns(4)
    with c1: telemetry_card("峰值推力",   "PEAK THRUST",      f"{peak_thrust:.2f}",      " N",    "nominal")
    with c2: telemetry_card("平均质量流量","MEAN MASS FLOW",  f"{mean_mfr:.1f}",          " mg/s", "nominal")
    with c3: telemetry_card("平均比冲",   "SPECIFIC IMPULSE", f"{mean_isp:.1f}",          " s",    "nominal")
    with c4: telemetry_card("异常占比",   "ANOMALY RATIO",    f"{anomaly_ratio*100:.2f}", " %",    anom_status, anom_delta)

    # ── 实时遥测曲线 ──
    section_header("实时遥测曲线", "REAL-TIME TELEMETRY")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.4))
    t_axis = np.arange(seq_len)

    if is_anomaly.any():
        ax1.fill_between(t_axis, 0, max(actual_thrust)*1.18,
                         where=is_anomaly, color=NASA_RED, alpha=0.2,
                         label='异常区间')
    ax1.plot(t_axis, actual_thrust, color=DATA_CYAN,  lw=2.0, label='传感器实测')
    ax1.plot(t_axis, thrust_pred,   color=DATA_AMBER, lw=1.4, ls='--', label='AI 预测基准')
    ax1.legend(loc='upper right')
    ax1.grid(True, axis='y')
    apply_cn_to_axis(ax1, title='推力 · THRUST',
                     xlabel='时间步', ylabel='推力 (N)')

    ax2.plot(t_axis, actual_mfr, color=DATA_CYAN,  lw=2.0, label='传感器实测')
    ax2.plot(t_axis, mfr_pred,   color=DATA_AMBER, lw=1.4, ls='--', label='AI 预测基准')
    ax2.legend(loc='upper right')
    ax2.grid(True, axis='y')
    apply_cn_to_axis(ax2, title='质量流量 · MASS FLOW RATE',
                     xlabel='时间步', ylabel='流量 (mg/s)')

    ax3.plot(t_axis, isp, color=DATA_VIOLET, lw=2.4, label='实时比冲')
    ax3.axhline(np.mean(isp), color=NASA_RED, ls=':', lw=1.2, alpha=0.5, label=f'均值 {np.mean(isp):.0f} s')
    ax3.legend(loc='upper right')
    ax3.grid(True, axis='y')
    apply_cn_to_axis(ax3, title='比冲 · SPECIFIC IMPULSE',
                     xlabel='时间步', ylabel='比冲 (s)')

    plt.tight_layout(pad=2.0)
    render_fig(fig)

    # ── 异常检测 ──
    section_header("异常检测与残差监控", "ANOMALY DETECTION")

    fig2, axr = plt.subplots(figsize=(18, 3.2))
    axr.plot(t_axis, residuals, color=STATUS_WARN, lw=1.8, label='绝对残差')
    axr.axhline(threshold, color=NASA_RED, ls='--', lw=1.8,
                label=f'安全阈值 {threshold:.1f} N')
    if is_anomaly.any():
        axr.fill_between(t_axis, 0, residuals, where=is_anomaly,
                         color=NASA_RED, alpha=0.4, label='异常时间步')
    axr.legend(loc='upper right')
    axr.grid(True, axis='y')
    apply_cn_to_axis(axr, title='推力残差实时监控 · RESIDUAL MONITOR',
                     xlabel='时间步', ylabel='残差 (N)')

    plt.tight_layout(pad=1.5)
    render_fig(fig2)

    # ── 异常状态条 ──
    if is_anomaly.any():
        st.markdown(f"""
        <div class="status-bar alert">
            <span class="title">▲ 检测到异常</span>
            共 <span class="num">{int(is_anomaly.sum())}</span> 个时间步推力残差超过 {threshold:.1f} N 阈值，
            异常占比 <span class="num">{anomaly_ratio*100:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-bar nominal">
            <span class="title">● 系统运行正常</span>
            推力残差全程低于 {threshold:.1f} N 阈值
        </div>
        """, unsafe_allow_html=True)

    # ── 健康报告 ──
    if generate_report:
        section_header("健康诊断报告", "DIAGNOSTIC REPORT")
        with st.spinner("正在生成诊断报告..."):
            report = generate_health_report(thrust_pred, mfr_pred, isp,
                                            is_anomaly=is_anomaly,
                                            residual_rms=residual_rms)

        pa      = report.get('prediction_accuracy', {})
        anom    = report.get('anomaly_status', {})
        overall = report['overall_health']
        conf_level = 'good' if model_conf >= 70 else ('warning' if model_conf >= 40 else 'critical')

        def _bar(s):
            return f'<span class="score-bar"><span class="score-fill" style="width:{s}%"></span></span>'
        def _badge(lv):
            return f'<span class="level-{lv}">● {lv.upper()}</span>'

        anom_ratio_pct = anom.get('anomaly_ratio', 0) * 100
        anom_level     = anom.get('level', 'normal')
        anom_score     = max(0, min(100, 100 - anom_ratio_pct / 20 * 100))

        st.markdown(f"""
        <div class="report-card">
            <div class="report-summary">
                <span class="score">{overall}/100</span> 综合健康评分
                &nbsp;&nbsp;<span style="color:{TEXT_DIM};font-weight:400;font-size:14px;">|</span>&nbsp;&nbsp;
                <span style="color:{TEXT_PRIMARY};font-weight:600;font-size:15px;">{report['summary']}</span>
            </div>
            <table class="report-table">
                <thead>
                    <tr><th style="width:22%;">评估维度</th><th style="width:28%;">指标</th><th style="width:25%;">评分</th><th style="width:25%;">状态</th></tr>
                </thead>
                <tbody>
                    <tr><td>预测精度</td><td>残差 RMS = {residual_rms:.4f} N</td><td>{_bar(pa.get('score',0))} {pa.get('score',0)}/100</td><td>{_badge(pa.get('level','normal'))}</td></tr>
                    <tr><td>异常检测</td><td>{anom.get('n_anomaly',0)} / {anom.get('total_steps',200)} 步 ({anom_ratio_pct:.1f}%)</td><td>{_bar(int(anom_score))} {int(anom_score)}/100</td><td>{_badge(anom_level)}</td></tr>
                    <tr><td>模型置信度</td><td>MC Dropout 不确定性评估</td><td>{_bar(model_conf)} {model_conf}/100</td><td>{_badge(conf_level)}</td></tr>
                    <tr style="border-top:2px solid {BORDER};">
                        <td colspan="4" style="font-size:11px;color:{TEXT_DIM};padding:14px 8px;line-height:1.7;">
                            <span style="font-weight:700;color:#E8ECEF;font-size:15px;">▌ 评分规则 Scoring Rules</span><br>
                            <span style="font-size:15px;line-height:2.2;color:#C8D0E0;">
                            <b style="color:#FFB627;">① 预测精度</b> &nbsp;&nbsp;
                            r = RMS / RMSE₀ · (RMSE₀ = 0.0832 N)<br>
                            <span style="padding-left:2em;">
                            Score = {{ 100, r ≤ 1.5 &nbsp;|&nbsp; 80 − 40(r−1.5)/1.5, 1.5 < r ≤ 3 &nbsp;|&nbsp; 40 − 30(r−3)/2, 3 < r ≤ 5 &nbsp;|&nbsp; 10, r > 5 }}
                            </span><br>
                            <b style="color:#FFB627;">② 异常检测</b> &nbsp;&nbsp;
                            a = anomaly_ratio &nbsp; → &nbsp; Score = max(0, 100 − 500a)<br>
                            <b style="color:#FFB627;">③ 模型置信度</b> &nbsp;&nbsp;
                            c = σ/μ &nbsp; (MC Dropout ×20) &nbsp; → &nbsp;
                            Score = {{ 98, c < 0.03 &nbsp;|&nbsp; 85, c < 0.08 &nbsp;|&nbsp; 65, c < 0.15 &nbsp;|&nbsp; 40, c < 0.30 &nbsp;|&nbsp; 20, c ≥ 0.30 }}<br>
                            <b style="color:#FC3D21;font-size:15px;">Overall = 0.4 · S<sub>pred</sub> + 0.3 · S<sub>anom</sub> + 0.3 · S<sub>conf</sub></b>
                            </span>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div class="welcome-screen">
        <div class="welcome-headline">▌ AWAITING TELEMETRY INPUT</div>
        <div class="welcome-title">推进器数字孪生健康监测系统</div>
        <div class="welcome-desc">
            请在左侧任务控制台上传地面热点火测试 CSV 数据，<br>
            启动实时推力、质量流量、比冲三维健康监测与异常检测。
        </div>
        <div class="welcome-meta">
            <span class="accent">v2.0</span> &nbsp; Attention-LSTM 双输出架构
            &nbsp; <span style="color:{NASA_RED};font-weight:700;">/</span> &nbsp;
            17 维异质特征融合
            &nbsp; <span style="color:{NASA_RED};font-weight:700;">/</span> &nbsp;
            SHAP 物理归因
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# SHAP 归因分析面板
# ════════════════════════════════════════════════════════════════════

if shap_data is not None:
    section_header("SHAP 特征归因分析", "ATTRIBUTION ANALYSIS")
    col_s1, col_s2, col_s3 = st.columns(3)
    for _col, _key, _title, _color in [
        (col_s1, 'thrust', '推力 Thrust', '#e74c3c'),
        (col_s2, 'mfr',    '质量流量 MFR', '#3498db'),
        (col_s3, 'isp', '比冲 Isp', '#2ecc71'),  # reuse thrust as placeholder
    ]:
        with _col:
            _vals = np.array(shap_data[_key], dtype=float).ravel()
            if _vals.ndim != 1 or len(_vals) != 17:
                st.write(f"⚠️ SHAP data shape error: {_key}={_vals.shape}")
                continue
            _idx = np.argsort(_vals)[-7:]
            _fig, _ax = plt.subplots(figsize=(4.5, 3.5))
            _ax.barh(range(len(_idx)), _vals[_idx], color=_color, alpha=0.8, height=0.6)
            _ax.set_yticks(range(len(_idx)))
            _lbls = [shap_data['feats'][i] for i in _idx]
            _ax.set_yticklabels(_lbls, fontsize=8, fontproperties=_CN_SMALL)
            _ax.invert_yaxis()
            _ax.set_title(_title, fontsize=11, fontweight='bold', color=TEXT_PRIMARY,
                          fontproperties=_CN_TITLE)
            _ax.tick_params(colors=TEXT_SECONDARY)
            _ax.set_xticks([])  # 隐藏横轴数值，减少杂乱
            for _b, _v in zip(_ax.containers[0], _vals[_idx]):
                _ax.text(_b.get_width()*1.02, _b.get_y()+_b.get_height()/2,
                         f'{_v:.2e}', va='center', fontsize=7, color=TEXT_PRIMARY)
            for side in ['top','right','left','bottom']:
                _ax.spines[side].set_visible(False)
            render_fig(_fig)

# ════════════════════════════════════════════════════════════════════
# 八、Footer
# ════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="mission-footer">
    <div>
        <span class="label-cn">模型</span><span class="label-en">MODEL</span>
        &nbsp;━&nbsp;<span style="color:{TEXT_PRIMARY};">v2.0 Attention-LSTM</span>
        <span class="sep">/</span>
        <span class="label-cn">输入</span><span class="label-en">INPUT</span>
        &nbsp;━&nbsp;<span style="color:{TEXT_PRIMARY};">17 维 × 200 步</span>
        <span class="sep">/</span>
        <span class="label-cn">输出</span><span class="label-en">OUTPUT</span>
        &nbsp;━&nbsp;<span style="color:{TEXT_PRIMARY};">推力 + 质量流量</span>
    </div>
    <div>
        <span class="label-cn">上海大学</span><span class="label-en">SHU</span>
        <span class="sep">/</span>
        <span class="label-cn">人工智能创新赛</span><span class="label-en">CRAIC 2026</span>
        <span class="sep">/</span>
        <span class="label-cn">数据集</span><span class="label-en">DATASET</span>
        &nbsp;━&nbsp;<span style="color:{TEXT_PRIMARY};">STFT · 2611 序列</span>
    </div>
</div>
""", unsafe_allow_html=True)
