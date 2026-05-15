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
    load_adapter_model,
    load_model_for_sn,
    ADAPTER_AVAILABLE_SNS,
    OPTIMAL_FINETUNE_STRATEGY,
    predict,
    predict_with_uncertainty,
    compute_residuals,
    generate_health_report,
    compute_isp,
    G0,
    EPS,
    MG_PER_S_TO_KG_PER_S,
    _health_level,
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
        except (OSError, RuntimeError):
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


# ── 数值常量 ──
THRUST_NOISE_STD        = 0.005
MFR_NOISE_STD           = 0.5
MC_DROPOUT_SAMPLES      = 20
RANDOM_SEED              = 42
MAX_IGNITION_SEARCH     = 500
TRANSIENT_SKIP_STEPS    = 80
SEQ_LEN                  = 200
DEFAULT_THRESHOLD        = 0.25
MC_CV_THRESHOLDS          = (0.03, 0.08, 0.15, 0.30)  # 变异系数分档
MC_CONFIDENCE_VALUES      = (98, 85, 65, 40, 20)       # 对应置信度
ANOMALY_RATIO_WARN        = 0.05
ANOMALY_RATIO_CRIT        = 0.20
ISP_THRESHOLD_SCALE       = 50
CONF_THRESHOLD_GOOD       = 70
CONF_THRESHOLD_WARN       = 40
MODEL_RMSE_REF            = 0.0832  # v2.0 测试 RMSE


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

@st.cache_resource
def _load_banner():
    import base64
    for _p in ['assets/shu_banner.jpg', 'assets/shu_banner.png']:
        try:
            with open(_p, 'rb') as f:
                _b = base64.b64encode(f.read()).decode()
            _ext = 'jpeg' if _p.endswith('.jpg') else 'png'
            return f'data:image/{_ext};base64,{_b}'
        except FileNotFoundError:
            continue
    return ''

_IMG_BANNER = _load_banner()

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

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    background: {BG_PANEL};
    border: 1px solid {BORDER};
    border-bottom: 3px solid {NASA_RED};
    padding: 0;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {TEXT_SECONDARY};
    border: none;
    border-right: 1px solid {BORDER};
    padding: 16px 32px;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 1px;
    transition: all 0.2s;
    border-radius: 0;
}}
.stTabs [aria-selected="true"] {{
    background: {BG_PANEL_HI};
    color: {DATA_CYAN};
    border-bottom: 3px solid {DATA_CYAN};
    margin-bottom: -3px;
}}
.stTabs [data-baseweb="tab"]:hover {{
    background: {BG_PANEL_HI};
    color: {TEXT_PRIMARY};
}}

/* Breathing status indicator */
@keyframes status-pulse {{
    0%, 100% {{
        box-shadow: 0 0 0 0 rgba(46, 230, 134, 0.5),
                    0 0 12px rgba(46, 230, 134, 0.6);
        transform: scale(1);
    }}
    50% {{
        box-shadow: 0 0 0 8px rgba(46, 230, 134, 0),
                    0 0 16px rgba(46, 230, 134, 0.8);
        transform: scale(1.15);
    }}
}}
.mission-status .dot {{
    display: inline-block;
    width: 8px; height: 8px; border-radius: 50%;
    background: {DATA_GREEN};
    margin-right: 8px;
    animation: status-pulse 1.8s ease-in-out infinite;
}}
.mission-status.alert .dot {{
    background: {NASA_RED};
    animation: status-pulse 0.9s ease-in-out infinite;
}}
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
            <span class="en-tag">CRAIC 2026 · v3.0 ATTN-LSTM</span>
        </div>
    </div>
    <div class="mission-status">
        <span class="en">SYSTEM STATUS</span>
        <span class="dot"></span>正常运行
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# 五、模型加载与选择
# ════════════════════════════════════════════════════════════════════

# Session state
if 'model_mode' not in st.session_state:
    st.session_state.model_mode = 'GLOBAL'
if 'active_sn' not in st.session_state:
    st.session_state.active_sn = None
if 'adapter_loaded_sn' not in st.session_state:
    st.session_state.adapter_loaded_sn = None
if 'history_reports' not in st.session_state:
    st.session_state.history_reports = []


@st.cache_resource
def load_global_model_cached():
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
        s_t = np.load(f"{_d}/shap_thrust.npy")
        s_m = np.load(f"{_d}/shap_mfr.npy")
        s_i = np.load(f"{_d}/shap_isp.npy")
        return {'thrust': np.abs(s_t), 'mfr': np.abs(s_m), 'isp': np.abs(s_i), 'feats': _f}
    except (FileNotFoundError, OSError, ValueError):
        return None


def get_active_model(sn=None):
    """根据当前选择的模式返回模型和标签。文件 SN 优先于侧边栏选择。"""
    # GLOBAL 模式：始终返回全局模型
    if st.session_state.model_mode == 'GLOBAL':
        return load_global_model_cached(), "GLOBAL"

    # TUNED 模式：文件 SN 优先，侧边栏选择作 fallback
    target_sn = sn or st.session_state.active_sn
    if target_sn is None:
        return load_global_model_cached(), "GLOBAL"

    # 该 SN 是否有 adapter 可用
    if OPTIMAL_FINETUNE_STRATEGY.get(target_sn) != 'adapter':
        return load_global_model_cached(), f"GLOBAL (SN{target_sn} skip)"

    global_model = load_global_model_cached()
    adapter = load_adapter_model(global_model, target_sn)
    if adapter is not None:
        return adapter, f"SN{target_sn}-TUNED"
    return global_model, "GLOBAL"


global_model_cache = load_global_model_cached()
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
    <div class="sidebar-section-title">模型选择</div>
    <div class="sidebar-section-en">MODEL SELECTION</div>
    """, unsafe_allow_html=True)

    model_options = ['GLOBAL'] + [f'SN{sn}-TUNED' for sn in ADAPTER_AVAILABLE_SNS]
    model_choice = st.selectbox(
        'model_choice',
        model_options,
        index=0,
        label_visibility='collapsed',
        key='model_selector'
    )

    if model_choice == 'GLOBAL':
        st.session_state.model_mode = 'GLOBAL'
        st.session_state.active_sn = None
    else:
        st.session_state.model_mode = 'TUNED'
        st.session_state.active_sn = int(model_choice.replace('SN', '').replace('-TUNED', ''))

    is_tuned = st.session_state.model_mode == 'TUNED'
    model_label = model_choice
    strategy_label = OPTIMAL_FINETUNE_STRATEGY.get(st.session_state.active_sn, 'global') if st.session_state.active_sn else 'global'

    st.markdown(f"""
    <div style="font-size:13px;color:{TEXT_SECONDARY};margin-top:-8px;">
        当前模型 &nbsp;━━&nbsp;
        <span style="color:{DATA_GREEN if is_tuned else DATA_CYAN};font-weight:700;
        font-family:'JetBrains Mono',monospace;font-size:15px;">{model_label}</span>
        {f'<span style="color:{DATA_AMBER};font-size:11px;margin-left:8px;">[微调优化]</span>' if is_tuned else f'<span style="color:{TEXT_DIM};font-size:11px;margin-left:8px;">[全局基座]</span>'}
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
    <div style="margin-top:24px;font-size:12px;color:{TEXT_DIM};">
        健康报告已移至「健康报告」标签页
    </div>
    """, unsafe_allow_html=True)

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
# 八、辅助函数
# ════════════════════════════════════════════════════════════════════

def _compute_actuals(thrust_true, mfr_true, seq_len, seed=RANDOM_SEED):
    """添加高斯噪声生成传感器仿真值，返回 (actual_thrust, actual_mfr, actual_isp)。"""
    rng = np.random.RandomState(seed)
    actual_thrust = thrust_true + rng.normal(0, THRUST_NOISE_STD, seq_len)
    actual_mfr    = mfr_true    + rng.normal(0, MFR_NOISE_STD,    seq_len)
    actual_isp    = compute_isp(actual_thrust, actual_mfr)
    return actual_thrust, actual_mfr, actual_isp


def _cv_to_confidence(cv):
    """变异系数映射到置信度评分。"""
    for threshold, conf in zip(MC_CV_THRESHOLDS, MC_CONFIDENCE_VALUES[:-1]):
        if cv < threshold:
            return conf
    return MC_CONFIDENCE_VALUES[-1]


def telemetry_card(label_cn, label_en, value, unit="", status="nominal", delta_text=None,
                   sparkline_data=None):
    spark_html = ""
    if sparkline_data is not None and len(sparkline_data) > 1:
        mid = len(sparkline_data) // 2
        denom = np.mean(sparkline_data[:mid]) + EPS
        delta_pct = (np.mean(sparkline_data[mid:]) - np.mean(sparkline_data[:mid])) / denom * 100
        arrow = "↑" if delta_pct > 0.5 else "↓" if delta_pct < -0.5 else "→"
        arrow_color = DATA_GREEN if abs(delta_pct) < 2 else DATA_AMBER if abs(delta_pct) < 5 else NASA_RED

        y_min, y_max = sparkline_data.min(), sparkline_data.max()
        rng = y_max - y_min
        if rng < EPS:
            rng = 1.0
        norm = (sparkline_data - y_min) / rng
        n = len(sparkline_data)
        w, h = 80, 18
        points = " ".join(f"{i*w/(n-1):.1f},{h-2-y*(h-4):.1f}" for i, y in enumerate(norm))

        spark_html = f"""
        <div style="display:flex;align-items:center;gap:10px;margin-top:8px;">
            <svg width="80" height="18" viewBox="0 0 80 18"
                 style="flex-shrink:0;">
                <polyline points="{points}" fill="none"
                          stroke="{arrow_color}" stroke-width="1.5"
                          stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <span style="font-family:'JetBrains Mono',monospace;font-size:13px;
                         color:{arrow_color};font-weight:600;">
                {arrow} {abs(delta_pct):.1f}%
            </span>
        </div>
        """
    delta_html = ""
    if delta_text is not None:
        cls = "delta-good" if status == "nominal" else "delta-bad"
        delta_html = f'<div class="telemetry-delta {cls}">{delta_text}</div>'
    st.markdown(f"""
    <div class="telemetry-card card-{status}">
        <div class="telemetry-label-cn">{label_cn}</div>
        <div class="telemetry-label-en">{label_en}</div>
        <div class="telemetry-value">{value}<span class="telemetry-unit">{unit}</span></div>
        {spark_html}
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
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight',
                facecolor=BG_PANEL, edgecolor='none')
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)


def apply_cn_to_axis(ax, title=None, xlabel=None, ylabel=None):
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


def render_health_radar(report, history_report=None):
    """Render 5-axis health radar chart."""
    categories = ['推力综合', '流量综合', '比冲综合', '三维一致性', '模型置信度']
    scores = [
        report['thrust']['dim_score'],
        report['mfr']['dim_score'],
        report['isp']['dim_score'],
        report['consistency']['score'],
        report['model_confidence']['score'],
    ]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_plot = angles + [angles[0]]
    scores_plot = scores + [scores[0]]

    fig, ax = plt.subplots(figsize=(6.2, 6.2), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG_PANEL)
    ax.set_facecolor(BG_PANEL)

    ax.fill(angles_plot, scores_plot, color=DATA_CYAN, alpha=0.22)
    ax.plot(angles_plot, scores_plot, color=DATA_CYAN, lw=2.8, marker='o',
            markersize=9, markerfacecolor=DATA_CYAN, markeredgecolor='white',
            markeredgewidth=1.5, label='当前评估')

    if history_report is not None:
        hist_scores = [
            history_report.get('thrust_dim', 0),
            history_report.get('mfr_dim', 0),
            history_report.get('isp_dim', 0),
            history_report.get('consistency', 0),
            history_report.get('mc', 0),
        ]
        hist_plot = hist_scores + [hist_scores[0]]
        ax.plot(angles_plot, hist_plot, color=DATA_AMBER, lw=1.5, ls='--',
                marker='s', markersize=6, alpha=0.7, label='上次点火')

    for ref, color, ls in [(70, NASA_RED, ':'), (90, DATA_GREEN, ':')]:
        ax.plot(angles_plot, [ref] * len(angles_plot), color=color,
                lw=0.8, ls=ls, alpha=0.35)

    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontproperties=CN_LABEL, color=TEXT_PRIMARY, size=11)
    ax.set_ylim(0, 105)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'],
                       fontproperties=CN_TICK, color=TEXT_SECONDARY)
    ax.tick_params(axis='y', colors=TEXT_SECONDARY)
    ax.grid(color=GRID_LINE, alpha=0.4, lw=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(1.2)
    ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.08),
              prop=CN_LEGEND, framealpha=0.9, edgecolor=BORDER)

    return fig


# ════════════════════════════════════════════════════════════════════
# 八、主界面
# ════════════════════════════════════════════════════════════════════

if uploaded_file is not None:
    fname = uploaded_file.name
    tmpdir = tempfile.mkdtemp()
    try:
        tmp_path = os.path.join(tmpdir, fname)
        with open(tmp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # 自动定位到稳定燃烧段
        _raw = pd.read_csv(tmp_path)
        _fire = 0
        for i in range(min(MAX_IGNITION_SEARCH, len(_raw))):
            if _raw['ton'].iloc[i] > 0:
                _fire = i
                break
        _max_off = max(0, len(_raw) - SEQ_LEN - 10)
        offset = min(_fire + TRANSIENT_SKIP_STEPS, _max_off)

        matched = metadata[metadata['filename'] == fname]
        if len(matched) > 0:
            meta_row = matched.iloc[0]
            file_sn = int(meta_row['sn'])

            # 根据上传文件自动匹配对应 SN 的微调模型
            active_model, model_label = get_active_model(file_sn)

            x, y_norm, _ = load_sequence_window(tmp_path, meta_row,
                                                start=offset, seq_len=SEQ_LEN)
            x_tensor = torch.from_numpy(x).float().unsqueeze(0)

            (thrust_pred, mfr_pred, isp,
             thrust_std, mfr_std, isp_std) = predict_with_uncertainty(
                 active_model, x_tensor, n_samples=MC_DROPOUT_SAMPLES)
            thrust_pred = thrust_pred.squeeze()
            mfr_pred    = mfr_pred.squeeze()
            isp         = isp.squeeze()
            thrust_std  = thrust_std.squeeze()
            mfr_std     = mfr_std.squeeze()
            isp_std     = isp_std.squeeze()

            thrust_true = y_norm[:, 0] * THRUST_SCALE
            mfr_true    = y_norm[:, 1] * MFR_MAX

            model_conf = _cv_to_confidence(
                np.mean(thrust_std) / (np.mean(thrust_pred) + EPS))
            source_label = "模型直接推理"
        else:
            model_conf = 90
            thrust_std = mfr_std = isp_std = None
            preds_npy   = np.load("outputs/predictions/v2/predictions_dual.npy")
            targets_npy = np.load("outputs/predictions/v2/targets_dual.npy")
            sample_idx = hash(fname) % len(preds_npy)
            _o = min(offset, preds_npy.shape[1] - SEQ_LEN)
            thrust_pred = preds_npy[sample_idx, _o:_o+SEQ_LEN, 0] * THRUST_SCALE
            mfr_pred    = preds_npy[sample_idx, _o:_o+SEQ_LEN, 1] * MFR_MAX
            thrust_true = targets_npy[sample_idx, _o:_o+SEQ_LEN, 0] * THRUST_SCALE
            mfr_true    = targets_npy[sample_idx, _o:_o+SEQ_LEN, 1] * MFR_MAX
            isp = compute_isp(thrust_pred, mfr_pred)
            source_label = "缓存预测回放"

        actual_thrust, actual_mfr, actual_isp = _compute_actuals(
            thrust_true, mfr_true, SEQ_LEN)

        # MC Dropout uncertainty for sigma-based anomaly detection
        _t_std = thrust_std
        _m_std = mfr_std
        _i_std = isp_std
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    has_mc = (_t_std is not None)
    res = compute_residuals(
        thrust_pred, actual_thrust, threshold,
        pred_mfr=mfr_pred, actual_mfr=actual_mfr,
        pred_isp=isp, actual_isp=actual_isp,
        thrust_std=_t_std, mfr_std=_m_std, isp_std=_i_std)

    residuals      = res['residuals']
    is_anomaly     = res['is_anomaly']
    anomaly_ratio  = res['anomaly_ratio']
    mfr_residuals  = res.get('mfr_residuals', np.abs(mfr_pred - actual_mfr))
    mfr_is_anomaly = res.get('mfr_is_anomaly', mfr_residuals > threshold * (MFR_MAX / THRUST_SCALE))
    mfr_anom_ratio = res.get('mfr_anomaly_ratio', float(mfr_is_anomaly.mean()))
    isp_residuals  = res.get('isp_residuals', np.abs(isp - actual_isp))
    isp_is_anomaly = res.get('isp_is_anomaly', isp_residuals > threshold * ISP_THRESHOLD_SCALE)
    isp_anom_ratio = res.get('isp_anomaly_ratio', float(isp_is_anomaly.mean()))

    # Combined anomaly ratio (any dimension)
    combined_anom_mask = is_anomaly | mfr_is_anomaly | isp_is_anomaly
    combined_anom_ratio = float(combined_anom_mask.mean())

    is_tuned_model = (model_label != 'GLOBAL')
    tuned_badge = f'<span style="background:{DATA_GREEN};color:#000;font-size:10px;font-weight:700;padding:1px 8px;border-radius:3px;margin-left:10px;">TUNED</span>' if is_tuned_model else ''
    sigma_badge = '<span style="background:#2E3A5C;color:#3DD9FF;font-size:10px;font-weight:600;padding:1px 8px;border-radius:3px;margin-left:6px;">3-SIGMA</span>' if has_mc else ''

    st.markdown(f"""
    <div class="source-bar">
        <span class="label">DATA SOURCE</span>
        &nbsp;&nbsp;<span class="value">{fname}</span>
        &nbsp;&nbsp;<span style="color:{NASA_RED};font-weight:700;">/</span>&nbsp;&nbsp;
        <span class="label">MODEL</span>
        &nbsp;&nbsp;<span class="accent">{model_label}</span>{tuned_badge}{sigma_badge}
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
    if combined_anom_ratio < ANOMALY_RATIO_WARN:
        anom_status, anom_delta = "nominal", "状态正常"
    elif combined_anom_ratio < ANOMALY_RATIO_CRIT:
        anom_status, anom_delta = "warning", "异常升高"
    else:
        anom_status, anom_delta = "critical", "严重异常"

    c1, c2, c3, c4 = st.columns(4)
    with c1: telemetry_card("峰值推力",   "PEAK THRUST",      f"{peak_thrust:.2f}",      " N",    "nominal",
                             sparkline_data=actual_thrust)
    with c2: telemetry_card("平均质量流量","MEAN MASS FLOW",  f"{mean_mfr:.1f}",          " mg/s", "nominal",
                             sparkline_data=actual_mfr)
    with c3: telemetry_card("平均比冲",   "SPECIFIC IMPULSE", f"{mean_isp:.1f}",          " s",    "nominal",
                             sparkline_data=isp)
    with c4: telemetry_card("异常占比(3D)","ANOMALY RATIO 3D", f"{combined_anom_ratio*100:.2f}", " %",    anom_status, anom_delta,
                             sparkline_data=combined_anom_mask.astype(float))

    if is_tuned_model:
        st.markdown(f"""
        <div style="text-align:center;margin:-8px 0 16px 0;">
            <span style="background:{BG_PANEL_HI};color:{DATA_GREEN};font-size:11px;font-weight:600;
                         padding:4px 14px;border-radius:4px;border:1px solid {BORDER};">
                ✓ 个体微调模型 &nbsp;━&nbsp; 针对 SN{st.session_state.active_sn} 优化（零遗忘，base冻结）
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── 4 个标签页 ──
    tab_telem, tab_anom, tab_report, tab_shap = st.tabs([
        "实时遥测",
        "异常检测",
        "健康报告",
        "SHAP 归因",
    ])

    # ═══════════════════ 标签页 1：实时遥测 ═══════════════════
    with tab_telem:
        section_header("实时遥测曲线", "REAL-TIME TELEMETRY")

    t_axis = np.arange(SEQ_LEN)
    # Per-dimension anomaly masks for highlighting
    anom_masks = [is_anomaly, mfr_is_anomaly, isp_is_anomaly]
    fig_tel, axes_tel = plt.subplots(1, 3, figsize=(18, 5.4))
    for ax, actual, pred, amask, title, ylabel in [
        (axes_tel[0], actual_thrust, thrust_pred, is_anomaly,
         '推力 · THRUST', '推力 (N)'),
        (axes_tel[1], actual_mfr, mfr_pred, mfr_is_anomaly,
         '质量流量 · MASS FLOW RATE', '流量 (mg/s)'),
        (axes_tel[2], actual_isp, isp, isp_is_anomaly,
         '比冲 · SPECIFIC IMPULSE', '比冲 (s)'),
    ]:
        max_val = max(np.max(actual), np.max(pred))
        if amask is not None and amask.any():
            ax.fill_between(t_axis, 0, max_val * 1.18,
                            where=amask, color=NASA_RED, alpha=0.2,
                            label='异常区间 (3σ)')
        ax.plot(t_axis, actual, color=DATA_CYAN,  lw=2.0, label='传感器实测')
        ax.plot(t_axis, pred,   color=DATA_AMBER, lw=1.4, ls='--', label='AI 预测基准')
        if ax is axes_tel[2]:
            ax.axhline(np.mean(pred), color=DATA_AMBER, ls=':', lw=1.2, alpha=0.5,
                       label=f'预测均值 {np.mean(pred):.0f} s')
        ax.legend(loc='upper right')
        ax.grid(True, axis='y')
        apply_cn_to_axis(ax, title=title, xlabel='时间步 (0.01s)', ylabel=ylabel)
    plt.tight_layout(pad=2.0)
    render_fig(fig_tel)

    # ═══════════════════ 标签页 2：异常检测 ═══════════════════
    with tab_anom:
        section_header("异常检测与残差监控", "ANOMALY DETECTION")

    # Dynamic thresholds: show sigma-based when available, fallback to hard threshold
    if has_mc:
        from inference_utils import THRUST_NOISE_STD as _TNS, MFR_NOISE_STD as _MNS, ISP_NOISE_STD as _INS
        _ts = np.sqrt(_t_std**2 + _TNS**2) * 3.0
        _ms = np.sqrt(_m_std**2 + _MNS**2) * 3.0
        _is = np.sqrt(_i_std**2 + _INS**2) * 3.0
        thr_label_t = '3σ 动态阈值'
        thr_label_m = '3σ 动态阈值'
        thr_label_i = '3σ 动态阈值'
        thr_line_t = DATA_GREEN
        thr_line_m = DATA_GREEN
        thr_line_i = DATA_GREEN
    else:
        _ts = np.full(SEQ_LEN, threshold)
        _ms = np.full(SEQ_LEN, threshold * (MFR_MAX / THRUST_SCALE))
        _is = np.full(SEQ_LEN, threshold * ISP_THRESHOLD_SCALE)
        thr_label_t = f'固定阈值 {threshold:.2f} N'
        thr_label_m = f'固定阈值 {threshold * (MFR_MAX / THRUST_SCALE):.1f} mg/s'
        thr_label_i = f'固定阈值 {threshold * ISP_THRESHOLD_SCALE:.1f} s'
        thr_line_t = NASA_RED
        thr_line_m = NASA_RED
        thr_line_i = NASA_RED

    data_rows = [
        (residuals,     _ts, is_anomaly,     '推力残差 · THRUST',    'N',     STATUS_WARN, thr_label_t, thr_line_t),
        (mfr_residuals, _ms, mfr_is_anomaly, '质量流量残差 · MASS FLOW RATE', 'mg/s',  DATA_CYAN, thr_label_m, thr_line_m),
        (isp_residuals, _is, isp_is_anomaly, '比冲残差 · SPECIFIC IMPULSE', 's', DATA_VIOLET, thr_label_i, thr_line_i),
    ]
    fig2, axes_res = plt.subplots(3, 1, figsize=(18, 7.5))
    for idx, (_res, _thr, _anom, _title, _yl, _clr, _thr_label, _thr_color) in enumerate(data_rows):
        ax = axes_res[idx]
        ax.plot(t_axis, _res, color=_clr, lw=1.5)
        if _thr.ndim == 0 or _thr.std() < 1e-6:
            ax.axhline(_thr[0] if _thr.ndim > 0 else _thr, color=_thr_color, ls='--', lw=1.2, label=_thr_label)
        else:
            ax.plot(t_axis, _thr, color=_thr_color, ls='--', lw=1.2, alpha=0.8, label=_thr_label)
        if _anom.any():
            ax.fill_between(t_axis, 0, _res, where=_anom, color=NASA_RED, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        xlabel = '时间步 (0.01s)' if idx == 2 else ''
        apply_cn_to_axis(ax, title=_title, ylabel=f'残差 ({_yl})', xlabel=xlabel)
    plt.tight_layout(pad=2.0)
    render_fig(fig2)

    # ── 三维异常状态 ──
    any_anom = is_anomaly.any() or mfr_is_anomaly.any() or isp_is_anomaly.any()

    if any_anom:
        _parts = []
        if is_anomaly.any():     _parts.append(f'推力 {int(is_anomaly.sum())}步')
        if mfr_is_anomaly.any(): _parts.append(f'流量 {int(mfr_is_anomaly.sum())}步')
        if isp_is_anomaly.any(): _parts.append(f'比冲 {int(isp_is_anomaly.sum())}步')
        det_method = '3σ 统计检测' if has_mc else f'硬阈值 ({threshold} N)'
        st.markdown(f"""
        <div class="status-bar alert">
            <span class="title">▲ 检测到异常</span>
            {' / '.join(_parts)} 超过阈值 &nbsp;|&nbsp; 检测方法: {det_method}
        </div>
        """, unsafe_allow_html=True)
    else:
        det_method = '3σ 统计检测' if has_mc else f'硬阈值 ({threshold} N)'
        st.markdown(f"""
        <div class="status-bar nominal">
            <span class="title">● 系统运行正常</span>
            推力 · 流量 · 比冲 三维残差均在阈值范围内 &nbsp;|&nbsp; 检测方法: {det_method}
        </div>
        """, unsafe_allow_html=True)

    # ═══════════════════ 标签页 3：健康报告 ═══════════════════
    with tab_report:
        section_header("健康诊断报告", "DIAGNOSTIC REPORT")
        gen_report = st.button("生成健康报告", key="gen_report_btn")
        if gen_report:
            with st.spinner("正在生成诊断报告..."):
                report = generate_health_report(
                    thrust_pred, mfr_pred, isp,
                    actual_thrust, actual_mfr, actual_isp,
                    thrust_std=_t_std, mfr_std=_m_std, isp_std=_i_std,
                    model_confidence=model_conf)

            overall = report['overall_health']

            # ── Save to history ──
            hist_entry = {
                'filename': fname,
                'thrust_dim': report['thrust']['dim_score'],
                'mfr_dim': report['mfr']['dim_score'],
                'isp_dim': report['isp']['dim_score'],
                'consistency': report['consistency']['score'],
                'mc': report['model_confidence']['score'],
            }
            st.session_state.history_reports.append(hist_entry)
            st.session_state.history_reports = st.session_state.history_reports[-10:]

            # ── Summary big number ──
            lvl_cn = {'good': '正常', 'warning': '需关注', 'critical': '建议检修'}
            lvl_color = {'good': DATA_GREEN, 'warning': DATA_AMBER, 'critical': NASA_RED}
            st.markdown(f"""
            <div class="report-card" style="text-align:center;margin-bottom:18px;">
                <div style="font-size:60px;font-weight:800;font-family:'JetBrains Mono',monospace;
                            color:{lvl_color.get(report['health_level'], DATA_CYAN)};line-height:1;">
                    {overall}<span style="font-size:24px;">/100</span>
                </div>
                <div style="font-size:17px;color:{TEXT_PRIMARY};font-weight:600;margin-top:8px;">
                    {lvl_cn.get(report['health_level'], '未知')} · {report['summary']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Radar chart ──
            prev = st.session_state.history_reports[-2] if len(st.session_state.history_reports) > 1 else None
            fig_radar = render_health_radar(report, history_report=prev)
            col_radar, col_spacer = st.columns([3, 1])
            with col_radar:
                render_fig(fig_radar)

            # ── Dimension details (expander) ──
            with st.expander("查看维度详情 Dimension Details", expanded=False):
                def _bar(s):
                    return f'<span class="score-bar"><span class="score-fill" style="width:{max(0,min(100,s))}%"></span></span>'
                def _badge(lv):
                    cls = 'level-good' if lv in ('good','stable','nominal') else \
                          'level-warning' if lv in ('warning','mild','partial','scattered') else \
                          'level-critical'
                    return f'<span class="{cls}">● {lv.upper()}</span>'
                def _drift_badge(lv):
                    if lv == 'stable': return '<span style="color:#2EE686;">▬ 稳定</span>'
                    if lv == 'mild': return '<span style="color:#FFC940;">↗ 轻微漂移</span>'
                    if lv == 'noticeable': return '<span style="color:#FF8A4D;">↗ 明显漂移</span>'
                    return '<span style="color:#FF4D2E;">↗ 严重漂移</span>'

                dims = ['thrust', 'mfr', 'isp']
                dim_labels_cn = ['推力', '质量流量', '比冲']
                dim_units = ['N', 'mg/s', 's']

                rows_html = ''
                for d, cn, u in zip(dims, dim_labels_cn, dim_units):
                    dd = report[d]
                    rows_html += f"""
                    <tr>
                        <td><b>{cn}</b> <span style="font-size:10px;color:{TEXT_DIM};">{d.upper()}</span></td>
                        <td>RMS={dd['rms']:.4f} {u} | 异常 {dd['n_anomaly']}/{dd['total_steps']}步 ({dd['anomaly_ratio']*100:.1f}%)</td>
                        <td>{_bar(dd['accuracy'])} {dd['accuracy']}/100</td>
                        <td>{_badge(_health_level(dd['accuracy']))}</td>
                    </tr>
                    <tr>
                        <td><span style="padding-left:16px;color:{TEXT_DIM};">↳ 异常评分</span></td>
                        <td><span style="font-size:11px;color:{TEXT_DIM};">检测方法: {'3σ 统计' if has_mc else '硬阈值'} | 漂移: {_drift_badge(dd['drift_level'])}</span></td>
                        <td>{_bar(dd['anomaly'])} {dd['anomaly']}/100</td>
                        <td>{_badge(_health_level(dd['anomaly']))}</td>
                    </tr>
                    <tr>
                        <td><span style="padding-left:16px;color:{TEXT_DIM};">↳ 维度综合</span></td>
                        <td><span style="font-size:11px;color:{TEXT_DIM};">0.5×精度 + 0.5×异常</span></td>
                        <td><b>{_bar(dd['dim_score'])} {dd['dim_score']}/100</b></td>
                        <td>{_badge(_health_level(dd['dim_score']))}</td>
                    </tr>"""

                cons = report['consistency']
                mc = report['model_confidence']

                st.html(f"""
                <div style="background:{BG_PANEL};padding:16px 0;">
                <table class="report-table">
                    <thead>
                        <tr><th style="width:20%;">评估维度</th><th style="width:32%;">指标</th><th style="width:23%;">评分</th><th style="width:25%;">状态</th></tr>
                    </thead>
                    <tbody>
                        {rows_html}
                        <tr style="border-top:1px solid {BORDER};">
                            <td><b>三维一致性</b></td>
                            <td>{cons['detail']}<br><span style="font-size:11px;color:{TEXT_DIM};">Jaccard = 异常交集/并集，衡量多维度异常关联程度</span></td>
                            <td>{_bar(cons['score'])} {cons['score']}/100</td>
                            <td>{_badge(cons['level'])}</td>
                        </tr>
                        <tr>
                            <td><b>模型置信度</b></td>
                            <td>MC Dropout x20 变异系数<br><span style="font-size:11px;color:{TEXT_DIM};">反映模型对当前输入的确定性</span></td>
                            <td>{_bar(mc['score'])} {mc['score']}/100</td>
                            <td>{_badge(mc['level'])}</td>
                        </tr>
                        <tr>
                            <td><b>漂移检测</b></td>
                            <td colspan="2" style="font-size:13px;">{report['drift_summary']}</td>
                            <td>{'<span style="color:#2EE686;">STABLE</span>' if report['drift_summary'] == '无显著漂移' else '<span style="color:#FF8A4D;">DRIFT</span>'}</td>
                        </tr>
                    </tbody>
                </table>
                </div>
                """)

        # ── 评分规则 (LaTeX) ──
        with st.expander("评分规则 Scoring Rules", expanded=False):
            st.latex(r"""
            \text{【1】精度评分（每维）}\\[6pt]
            S_{\rm acc} = f(r), \quad r = \frac{\mathrm{RMS_{dim}}}{\mathrm{RMSE_{ref}}}\\[6pt]
            f(r) = \begin{cases}
            100, & r \leq 1 \\[2pt]
            70 + 30\cdot\frac{1.5-r}{0.5}, & 1 < r \leq 1.5 \\[2pt]
            40 + 30\cdot\frac{2.5-r}{1.0}, & 1.5 < r \leq 2.5 \\[2pt]
            15 + 25\cdot\frac{4.0-r}{1.5}, & 2.5 < r \leq 4.0 \\[2pt]
            5 + 10\cdot\frac{8.0-r}{4.0}, & 4.0 < r \leq 8.0 \\[2pt]
            5, & r > 8.0
            \end{cases}
            """)
            st.latex(r"""
            \text{【2】异常评分（每维）}\\[6pt]
            S_{\rm anom} = 100 \cdot e^{-10a}, \quad
            a = \frac{n_{\rm anom}}{n_{\rm total}}\\[6pt]
            \text{检测：}\ |y_{\rm pred}-y_{\rm actual}| > 3\sqrt{\sigma^2_{\rm mc} + \sigma^2_{\rm noise}}
            """)
            st.latex(r"""
            \text{【3】维度综合}\\[6pt]
            S_{\rm dim} = 0.5 \cdot S_{\rm acc} + 0.5 \cdot S_{\rm anom}
            """)
            st.latex(r"""
            \text{【4】三维一致性}\\[6pt]
            J = \frac{|T \cap M \cap I|}{|T \cup M \cup I|}\\[6pt]
            \text{高 } J \Rightarrow \text{真实故障；低 } J \Rightarrow \text{传感器噪声}
            """)
            st.latex(r"""
            \text{【5】综合评分}\\[6pt]
            S_{\rm overall} = \mathrm{mean}(S_T, S_M, S_I) \times
            \bigl[\,0.85 + 0.15 \times (1 - J)\,\bigr]
            """)

    # ═══════════════════ 标签页 4：SHAP 归因 ═══════════════════
    with tab_shap:
        if shap_data is not None:
            section_header("SHAP 特征归因分析", "ATTRIBUTION ANALYSIS")
            col_s1, col_s2, col_s3 = st.columns(3)
            for _col, _key, _title, _color in [
                (col_s1, 'thrust', '推力 Thrust', '#e74c3c'),
                (col_s2, 'mfr',    '质量流量 MFR', '#3498db'),
                (col_s3, 'isp', '比冲 Isp', '#2ecc71'),
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
                    _ax.set_yticklabels(_lbls, fontsize=8)
                    _ax.invert_yaxis()
                    _ax.set_title(_title, fontsize=11, fontweight='bold', color=TEXT_PRIMARY,
                                  fontproperties=CN_TITLE)
                    _ax.tick_params(colors=TEXT_SECONDARY)
                    _ax.set_xticks([])
                    for _b, _v in zip(_ax.containers[0], _vals[_idx]):
                        _ax.text(_b.get_width()*1.02, _b.get_y()+_b.get_height()/2,
                                 f'{_v:.2e}', va='center', fontsize=7, color=TEXT_PRIMARY)
                    for side in ['top','right','left','bottom']:
                        _ax.spines[side].set_visible(False)
                    render_fig(_fig)
        else:
            st.markdown(f"""
            <div style="text-align:center;padding:60px;color:{TEXT_DIM};font-size:15px;">
                SHAP 特征归因数据未加载。请在 outputs/predictions/v2/ 目录提供 shap_*.npy 文件。
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
            <span class="accent">v3.0</span> &nbsp; Attention-LSTM 双输出架构
            &nbsp; <span style="color:{NASA_RED};font-weight:700;">/</span> &nbsp;
            17 维异质特征融合
            &nbsp; <span style="color:{NASA_RED};font-weight:700;">/</span> &nbsp;
            SHAP 物理归因
        </div>
    </div>
    """, unsafe_allow_html=True)

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
