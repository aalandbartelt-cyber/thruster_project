# 推进器数字孪生健康监测系统 — UI 优化路线图

> **当前版本**：v2.0（app_v2.py）
> **目标版本**：v3.0
> **优化范围**：视觉、信息架构、交互体验
> **代码改动量**：约 30% — 大部分为增量改造，不改动模型推理层

---

## 总览

针对当前 `app_v2.py` 的健康监测界面，整理出 **6 个优化方向**，按"改动成本 / 信息增益"双维度排序如下：

| # | 方向 | 类型 | 优先级 | 改动量 | 信息增益 |
|---|---|---|---|---|---|
| 1 | 遥测卡片加 sparkline 趋势线 | 视觉精修 | P0 | 小 | 高 |
| 2 | 健康报告改用雷达图可视化 | 结构调整 | P0 | 中 | 高 |
| 3 | 主内容区拆分为 4 个标签页 | 信息架构 | P1 | 大 | 中 |
| 4 | 残差图联动布局 + 三维关联热力图 | 信息架构 | P1 | 中 | 高 |
| 5 | 图表时间轴选择器 + Header 呼吸灯 | 视觉精修 | P2 | 小 | 中 |
| 6 | SHAP 归因图改为 Plotly 可交互版本 | 交互增强 | P2 | 中 | 中 |

**落地建议**：先做 P0（1、2），见效最快；P1（3、4）作为版本 v3.0 主线；P2（5、6）后续迭代补全。

---

## 方向 1 — 遥测卡片加 sparkline 趋势线

### 现状问题

当前四个 `telemetry_card` 只显示静态峰值/均值，操作员无法判断参数是上升还是下降。在地面热点火测试场景下，"趋势"和"瞬时值"同样关键 — 推力突然下跌 0.5N 远比一个孤立的均值 18.42N 重要。

### 改进方案

每张卡片新增：
- **迷你 sparkline**（SVG `<polyline>`）显示该指标在 200 步窗口内的轨迹
- **方向箭头 + 变化率**（与窗口前半段均值对比）

### 代码改动点

```python
def telemetry_card(label_cn, label_en, value, unit="", status="nominal",
                   delta_text=None, sparkline_data=None):
    """新增 sparkline_data 参数：一维 numpy array (~200 点)。"""
    spark_html = ""
    if sparkline_data is not None and len(sparkline_data) > 1:
        # 计算趋势方向
        mid = len(sparkline_data) // 2
        delta_pct = (np.mean(sparkline_data[mid:]) - np.mean(sparkline_data[:mid])) \
                    / (np.mean(sparkline_data[:mid]) + EPS) * 100
        arrow = "↑" if delta_pct > 0.5 else "↓" if delta_pct < -0.5 else "→"
        arrow_color = DATA_GREEN if abs(delta_pct) < 2 else DATA_AMBER if abs(delta_pct) < 5 else NASA_RED

        # 归一化生成 polyline 路径
        y_min, y_max = sparkline_data.min(), sparkline_data.max()
        norm = (sparkline_data - y_min) / (y_max - y_min + EPS)
        n = len(sparkline_data)
        points = " ".join(f"{i*80/n:.1f},{18-y*14:.1f}" for i, y in enumerate(norm))

        spark_html = f"""
        <div style="display:flex;align-items:center;gap:10px;margin-top:8px;">
            <svg width="80" height="18" viewBox="0 0 80 18">
                <polyline points="{points}" fill="none"
                          stroke="{arrow_color}" stroke-width="1.5"
                          stroke-linecap="round"/>
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
```

### 调用处改动

```python
# 原 ↓
telemetry_card("峰值推力", "PEAK THRUST", f"{peak_thrust:.2f}", " N", "nominal")

# 新 ↓
telemetry_card("峰值推力", "PEAK THRUST", f"{peak_thrust:.2f}", " N", "nominal",
               sparkline_data=actual_thrust)
```

### 预期收益

- 操作员一眼判断走势（升/降/稳）
- 改动量：30 行 Python + 0 行 CSS
- 不依赖模型层，可独立完成

---

## 方向 2 — 健康报告改用雷达图可视化

### 现状问题

`generate_report` 返回的健康评分有 **5 个维度**（推力精度、流量精度、比冲精度、一致性、模型置信度），目前用 HTML 表格 + 进度条展示，需要逐行读取认知负担重，且看不到维度间的"短板"。

### 改进方案

用 **matplotlib 雷达图**（PolarAxes）替换报告主表格，五个轴分别为：
- 推力综合 / 流量综合 / 比冲综合 / 三维一致性 / 模型置信度

同时保留：
- 顶部综合评分大数字
- 底部维度详情展开区（折叠）
- **历史对比线**：上次点火数据用虚线叠加，形成"漂移可视化"

### 代码改动点

```python
def render_health_radar(report, history_report=None):
    """生成五维健康雷达图。"""
    categories = ['推力\n精度+异常', '流量\n精度+异常', '比冲\n精度+异常',
                  '三维\n一致性', '模型\n置信度']
    scores = [
        report['thrust']['dim_score'],
        report['mfr']['dim_score'],
        report['isp']['dim_score'],
        report['consistency']['score'],
        report['model_confidence']['score'],
    ]

    # 闭合多边形
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    scores_plot = scores + [scores[0]]
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG_PANEL)
    ax.set_facecolor(BG_PANEL)

    # 主线：本次评估
    ax.fill(angles_plot, scores_plot, color=DATA_CYAN, alpha=0.25)
    ax.plot(angles_plot, scores_plot, color=DATA_CYAN, lw=2.5, marker='o',
            markersize=8, label='本次点火')

    # 历史对比
    if history_report is not None:
        hist_scores = [history_report[k] for k in
                       ['thrust_dim', 'mfr_dim', 'isp_dim', 'consistency', 'mc']]
        hist_plot = hist_scores + [hist_scores[0]]
        ax.plot(angles_plot, hist_plot, color=DATA_AMBER, lw=1.5, ls='--',
                marker='s', markersize=5, alpha=0.7, label='上次点火')

    # 评分阈值参考圈（70/90 分线）
    for ref, color in [(70, NASA_RED), (90, DATA_GREEN)]:
        ax.plot(angles_plot, [ref]*len(angles_plot), color=color,
                lw=0.8, ls=':', alpha=0.4)

    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontproperties=CN_LABEL, color=TEXT_PRIMARY)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'],
                       fontproperties=CN_TICK, color=TEXT_SECONDARY)
    ax.grid(color=GRID_LINE, alpha=0.4)
    ax.spines['polar'].set_color(BORDER)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), prop=CN_LEGEND)

    return fig
```

### 持久化历史评分（可选）

```python
# 在 generate_report 后保存到 session_state
if 'history_reports' not in st.session_state:
    st.session_state.history_reports = []
st.session_state.history_reports.append({
    'filename': fname,
    'timestamp': datetime.now().isoformat(),
    'thrust_dim': report['thrust']['dim_score'],
    'mfr_dim': report['mfr']['dim_score'],
    'isp_dim': report['isp']['dim_score'],
    'consistency': report['consistency']['score'],
    'mc': report['model_confidence']['score'],
})
# 只保留最近 10 次
st.session_state.history_reports = st.session_state.history_reports[-10:]
```

### 预期收益

- 五个维度的"短板"立即可见（雷达图凹陷处）
- 历史对比让"漂移"从抽象数字变成视觉对比
- 配合现有 `_health_level` 配色，无需新增设计 token

---

## 方向 3 — 主内容区拆分为 4 个标签页

### 现状问题

`app_v2.py` 当前是 **单一长页面**，从上到下依次为：
1. Mission Header
2. 数据源条
3. 关键遥测指标（4 卡片）
4. 实时遥测曲线（3 子图）
5. 异常检测残差（3 子图）
6. 状态条
7. 健康报告（点击后展开，含 LaTeX 公式）
8. SHAP 归因（3 子图）
9. Footer

**问题**：
- 首屏滚动 4–5 屏才能看完
- 用户做"专项分析"时（如只看 SHAP），需要反复滚动
- LaTeX 公式渲染慢，影响其他模块加载

### 改进方案

用 `st.tabs()` 拆分主内容区为 **4 个标签页**：

```
┌─────────────────────────────────────────────────────┐
│  Mission Header                                      │
│  Data Source Bar                                     │
│  Key Telemetry Cards (4 cards)        ← 保持常驻      │
├─────────────────────────────────────────────────────┤
│  [遥测曲线] [异常检测] [健康报告] [SHAP 归因]          │  ← 标签栏
├─────────────────────────────────────────────────────┤
│                                                      │
│  对应标签页内容                                       │
│                                                      │
└─────────────────────────────────────────────────────┘
```

侧边栏（任务控制台）保持不变 — 始终可见。

### 代码改动点

```python
if uploaded_file is not None:
    # ... 数据加载、模型推理 ...

    # ── 数据源条 + 关键指标（常驻区，不在标签页里） ──
    render_source_bar(...)
    render_telemetry_cards(...)

    # ── 4 个标签页 ──
    tab_telem, tab_anom, tab_report, tab_shap = st.tabs([
        "📡 实时遥测",
        "⚠ 异常检测",
        "📋 健康报告",
        "🔬 SHAP 归因"
    ])

    with tab_telem:
        section_header("实时遥测曲线", "REAL-TIME TELEMETRY")
        # 推力/流量/比冲 3 子图
        render_telemetry_curves(...)

    with tab_anom:
        section_header("异常检测与残差监控", "ANOMALY DETECTION")
        render_residual_charts(...)
        render_anomaly_status_bar(...)

    with tab_report:
        section_header("健康诊断报告", "DIAGNOSTIC REPORT")
        if st.button("生成报告", key="gen_in_tab"):
            with st.spinner("正在生成..."):
                report = generate_health_report(...)
            render_health_radar(report)
            render_report_table(report)
            with st.expander("评分规则", expanded=False):
                render_latex_rules()

    with tab_shap:
        section_header("SHAP 特征归因分析", "ATTRIBUTION ANALYSIS")
        render_shap_panels(shap_data)
```

### 标签栏 CSS 美化

Streamlit 默认 `st.tabs` 样式偏弱，建议自定义：

```css
/* 标签栏整体 */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-bottom: 3px solid var(--nasa-red);
    padding: 0;
}

/* 单个标签 */
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-secondary);
    border: none;
    border-right: 1px solid var(--border);
    padding: 16px 32px;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 1px;
    transition: all 0.2s;
}

/* 选中态：底部红条 + 高亮背景 */
.stTabs [aria-selected="true"] {
    background: var(--bg-panel-hi);
    color: var(--data-cyan);
    border-bottom: 3px solid var(--data-cyan);
    margin-bottom: -3px;
}

.stTabs [data-baseweb="tab"]:hover {
    background: var(--bg-panel-hi);
    color: var(--text-primary);
}
```

### 注意事项

- **状态管理**：原 `generate_report` 按钮在侧边栏，改成放进"健康报告"标签页内更合理（在哪用、在哪触发）；或保留侧边栏按钮，但配合 `st.session_state.active_tab` 自动跳转到报告页。
- **数据计算复用**：所有标签页共享同一份 `thrust_pred / mfr_pred / isp / residuals` 计算结果，**必须在 tabs 块之前完成**，否则切换标签会重算。
- **首屏默认**：用户上传后期望先看遥测曲线，所以 "实时遥测" 应为首个标签（默认选中）。

### 预期收益

- 首屏加载从 4 屏缩到 1 屏
- 用户做专项分析时无需滚动
- LaTeX 公式只在"健康报告"页渲染，不拖累其他页

---

## 方向 4 — 残差图联动布局 + 三维关联热力图

### 现状问题

当前 `axes_res = plt.subplots(3, 1, figsize=(18, 7.5))` 三个残差子图竖排，导致：
- 总高度 ~750px，需要滚动
- **看不出三维异常的时间关联性** — 比如"推力残差在 t=120 飙升时，流量是否同步异常？"

### 改进方案

A. **2+1 布局**：上方一行两个（推力 + 流量），下方一行一个（比冲全宽）
B. **新增三维关联热力图** — 横轴为时间步，纵轴为三个维度（推力/流量/比冲），颜色编码异常强度（残差/阈值的比值）

```
┌─────────────────────┬─────────────────────┐
│  推力残差            │  质量流量残差         │
│  (上方)              │  (上方)              │
├─────────────────────┴─────────────────────┤
│  比冲残差（全宽）                            │
├───────────────────────────────────────────┤
│  三维异常关联热力图（全宽）                    │  ← 新增
│  ┌─────────────────────────────────┐       │
│  │ 推力 ████░░░░██████░░░░░░░░░░    │       │
│  │ 流量 ░░██████████░░░░░░░░░░░░    │       │
│  │ 比冲 ░░░░██░░░░░░░░░░░░██░░░░    │       │
│  └─────────────────────────────────┘       │
└───────────────────────────────────────────┘
```

### 代码改动点

```python
# A. 残差图改为 2+1 布局
fig = plt.figure(figsize=(18, 9))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.6], hspace=0.45)
ax_t = fig.add_subplot(gs[0, 0])  # 推力
ax_m = fig.add_subplot(gs[0, 1])  # 流量
ax_i = fig.add_subplot(gs[1, :])  # 比冲（横跨两列）
ax_heat = fig.add_subplot(gs[2, :])  # 关联热力图

# ... 前 3 个 ax 按原逻辑画 ...

# B. 三维关联热力图
heat_data = np.stack([
    residuals / (_ts + EPS),         # 推力归一化残差
    mfr_residuals / (_ms + EPS),     # 流量归一化残差
    isp_residuals / (_is + EPS),     # 比冲归一化残差
])
# 截断到 [0, 3] 范围（3σ 为最大显示值）
heat_data = np.clip(heat_data, 0, 3)

im = ax_heat.imshow(heat_data, aspect='auto', cmap='RdYlGn_r',
                    vmin=0, vmax=3,
                    extent=[0, SEQ_LEN, 0, 3])
ax_heat.set_yticks([0.5, 1.5, 2.5])
ax_heat.set_yticklabels(['比冲', '流量', '推力'], fontproperties=CN_TICK)
apply_cn_to_axis(ax_heat, title='三维异常关联热力图',
                 xlabel='时间步 (0.01s)', ylabel='')

# 颜色条：< 1 绿（正常），1-2 黄（轻微），> 2 红（严重）
cbar = plt.colorbar(im, ax=ax_heat, orientation='horizontal',
                    pad=0.15, shrink=0.5)
cbar.set_label('残差/阈值 比值', fontproperties=CN_LABEL)
cbar.ax.tick_params(colors=TEXT_SECONDARY)
```

### 配色注意

- 热力图用 `RdYlGn_r`（绿→黄→红）符合"红色 = 危险"直觉
- vmin/vmax 固定在 [0, 3]，避免不同样本之间色阶不一致

### 预期收益

- 同一时刻的三维异常状态一图直观对比
- 可识别"传感器噪声"（单维度异常）vs"真实故障"（多维度同步异常）

---

## 方向 5 — 图表时间轴选择器 + Header 呼吸灯

### 5A — 时间轴 brush 选择器

#### 现状问题

遥测曲线固定显示 SEQ_LEN=200 步，无法聚焦异常区段。当 100 步附近有异常时，用户希望放大查看，但目前做不到。

#### 改进方案

侧边栏新增一个 `st.slider` 双端滑块，控制图表 x 轴显示范围：

```python
with st.sidebar:
    # ... 已有控件 ...

    st.markdown("""
    <div class="sidebar-section-title">时间窗口</div>
    <div class="sidebar-section-en">TIME WINDOW</div>
    """, unsafe_allow_html=True)

    time_range = st.slider(
        "time_range",
        min_value=0, max_value=SEQ_LEN, value=(0, SEQ_LEN), step=5,
        label_visibility="collapsed"
    )
    st.markdown(f"""
    <div style="font-size:13px;color:{TEXT_SECONDARY};margin-top:-8px;">
        显示区段 &nbsp;━━&nbsp;
        <span style="color:{DATA_CYAN};font-weight:700;
        font-family:'JetBrains Mono',monospace;">{time_range[0]}—{time_range[1]}</span>
    </div>
    """, unsafe_allow_html=True)
```

在所有绘图前切片数据：

```python
t0, t1 = time_range
t_axis_view = np.arange(t0, t1)
thrust_view = actual_thrust[t0:t1]
# ... 其他数据同样切片 ...

# 绘图时使用 _view 版本
ax.plot(t_axis_view, thrust_view, ...)
```

**注意**：异常统计（如 `combined_anom_ratio`）应基于全量数据，**不要**基于切片 — 否则用户的视图操作会"误改"指标。

### 5B — Header 呼吸灯动画

#### 现状问题

`.mission-status .dot` 是静态绿点，没有"系统正在运行"的活力感。

#### 改进方案

CSS keyframe 呼吸动画：

```css
@keyframes status-pulse {
    0%, 100% {
        box-shadow: 0 0 0 0 rgba(46, 230, 134, 0.5),
                    0 0 12px rgba(46, 230, 134, 0.6);
        transform: scale(1);
    }
    50% {
        box-shadow: 0 0 0 8px rgba(46, 230, 134, 0),
                    0 0 16px rgba(46, 230, 134, 0.8);
        transform: scale(1.15);
    }
}

.mission-status .dot {
    display: inline-block;
    width: 8px; height: 8px; border-radius: 50%;
    background: #2EE686;
    margin-right: 8px;
    animation: status-pulse 1.8s ease-in-out infinite;
}

/* 异常状态：变红 + 加快脉动 */
.mission-status.alert .dot {
    background: #FF4D2E;
    animation: status-pulse 0.9s ease-in-out infinite;
    --status-color: rgba(255, 77, 46, 0.5);
}
```

并且根据 `combined_anom_ratio` 动态切换 `.mission-status` 的 class：

```python
status_cls = "alert" if combined_anom_ratio > ANOMALY_RATIO_CRIT else ""
status_text = "异常运行" if status_cls else "正常运行"
# 注入到 Header HTML
```

### 预期收益

- 时间窗口选择器：操作员可放大异常段，提高诊断效率
- 呼吸灯：传达"实时监控中"的状态感，符合 NASA / SpaceX 监控大屏视觉语言

---

## 方向 6 — SHAP 归因图改为 Plotly 可交互版本

### 现状问题

当前 SHAP 用 matplotlib 画三个独立的 `barh`，三个子图并排显示。
- 三维对比要左右扫视，无法直接看出"哪些特征同时对三个目标贡献最大"
- 静态图无法点击/筛选/排序

### 改进方案

用 **Plotly 横向条形图**（带下拉切换 + 点击交互）：

```python
import plotly.graph_objects as go

def render_shap_interactive(shap_data):
    feats = shap_data['feats']
    targets = {'推力': shap_data['thrust'],
               '质量流量': shap_data['mfr'],
               '比冲': shap_data['isp']}

    fig = go.Figure()
    for tgt_name, vals in targets.items():
        idx = np.argsort(vals)[-10:]  # Top 10
        fig.add_trace(go.Bar(
            y=[feats[i] for i in idx],
            x=vals[idx],
            orientation='h',
            name=tgt_name,
            visible=(tgt_name == '推力'),  # 默认显示推力
            marker_color=DATA_CYAN if tgt_name == '推力'
                         else DATA_AMBER if tgt_name == '质量流量'
                         else DATA_VIOLET,
            hovertemplate='<b>%{y}</b><br>SHAP 值: %{x:.3e}<extra></extra>',
        ))

    # 下拉切换
    fig.update_layout(
        updatemenus=[dict(
            buttons=[
                dict(label='推力',
                     method='update',
                     args=[{'visible': [True, False, False]}]),
                dict(label='质量流量',
                     method='update',
                     args=[{'visible': [False, True, False]}]),
                dict(label='比冲',
                     method='update',
                     args=[{'visible': [False, False, True]}]),
            ],
            direction='down',
            x=0.5, y=1.15,
            bgcolor=BG_PANEL,
            font=dict(color=TEXT_PRIMARY),
        )],
        plot_bgcolor=BG_PANEL,
        paper_bgcolor=BG_PANEL,
        font=dict(family='Noto Sans SC', color=TEXT_PRIMARY),
        height=450,
        margin=dict(l=120, r=40, t=80, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
```

### 物理含义补充

在条形图下方加一个特征字典展开区：

```python
FEATURE_PHYSICS = {
    'ton': '开机时长 — 单次脉冲点火时长',
    'vl': '阀门电压 — 控制推进剂供给',
    'test_pressure': '测试压力 — 推进剂罐当前压力',
    'cumulated_on_time': '累计开机时长 — 总寿命指标',
    'cumulated_throughput': '累计消耗 — 推进剂消耗总量',
    'cumulated_pulses': '累计脉冲数 — 总点火次数',
    # ...
}

with st.expander("特征物理含义"):
    for feat in feats:
        st.markdown(f"- **`{feat}`** &nbsp;━&nbsp; {FEATURE_PHYSICS.get(feat, '—')}")
```

### 预期收益

- 单图切换三个目标，节省垂直空间
- 鼠标悬停显示精确 SHAP 值
- 配合物理字典，提升模型可解释性

---

## 落地建议时序

```
Week 1
  ├─ Day 1-2  方向 1（sparkline）        ← 立竿见影
  └─ Day 3-5  方向 2（雷达图）            ← 报告界面焕新

Week 2
  ├─ Day 1-3  方向 3（4 标签页）          ← 结构重构主线
  └─ Day 4-5  方向 4（残差联动 + 热力图）  ← 异常分析升级

Week 3（可选）
  ├─ Day 1-2  方向 5（时间轴 + 呼吸灯）   ← 视觉精修
  └─ Day 3-4  方向 6（Plotly SHAP）       ← 交互增强
```

---

## 风险与注意点

1. **Streamlit 重渲染开销**：方向 3 的 `st.tabs` 切换不会重跑模型推理（只要数据计算在 tabs 块之前），但要确保 `predict_with_uncertainty` 的结果被复用，不要写在某个 `with tab_x:` 内。

2. **matplotlib 中文字体缓存**：方向 2 的雷达图新增图表，仍需通过 `apply_cn_to_axis` 注入 `FontProperties`，否则会出现方框。

3. **历史评分持久化**：方向 2 的"历史对比线"若用 `st.session_state`，会在用户刷新后丢失。如需持久化，建议写入 `data/history_reports.json` 或 SQLite。

4. **Plotly 主题适配**：方向 6 改用 Plotly 后，配色需要手动对齐当前 NASA 风格（深色背景、青/琥珀/紫强调色），否则会与 matplotlib 图风格断裂。

5. **响应式断点**：4 标签页 + 雷达图在窄屏（< 1200px）会拥挤，建议在小屏切换为纵向布局，可用 `st.columns([1, 2])` 控制。

---

*文档版本：v1.0 · 生成时间：2026-05-16*
