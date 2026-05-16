# APP_V3 数字孪生大屏 — 完整逻辑与设计文档

> 版本: V3 UI + V4 健康评分系统  
> 日期: 2026-05-16  
> 启动: `streamlit run app_v3.py`

---

## 一、整体架构

```
CSV上传 → 数据预处理 → 17维特征构造 → 模型推理 (MC Dropout)
  → 4标签页展示:
     Tab 1: 遥测监控 (TELEMETRY)      — 4卡片 + sparkline
     Tab 2: 异常检测 (ANOMALY)        — 2+1残差图 + 热力图
     Tab 3: 健康报告 (DIAGNOSTIC)     — 雷达图 + 维度详情
     Tab 4: 特征归因 (ATTRIBUTION)    — Plotly SHAP
```

**核心模块**:
- `app_v3.py`: Streamlit UI 层 (~1700 行)
- `inference_utils.py`: 推理 + MC Dropout + 健康评分引擎 (~600 行)
- `data_pipeline_v2.py`: 17 维特征工程

---

## 二、评分系统 (V4) — 完整设计逻辑

### 2.1 为什么需要重新设计评分？

V3 评分存在**循环标定**问题:
- 用测试集 RMSE 作为 baseline 去评分测试集本身
- 导致 E[χ²] = 1 → 所有人的 stat_score ≈ 100
- 零区分度, 无法识别好坏样本

V4 必须打破循环, 使用**独立于测试集的参考值**.

### 2.2 为什么选 p50 而不是 p25 或 RMSE？

| 选项 | 值 (推力) | 效果 |
|------|----------|------|
| **测试集 RMSE** | 0.0832 N | 循环标定, 全员 100 分 |
| **p25** (最佳 25%) | 0.0175 N | 太紧 — 测试 RMSE 是 9 倍, 平均 χ²≈76, 全员 6 分 |
| **p50** (中位数) | **0.0531 N** | 平均样本 χ²≈9 → 70/100, 好坏样本自然分开 |

p50 的物理含义: "模型在 50% 的预测点上能做到比这个误差更小" — 这是模型能力的**中位水平**, 不是最优水平也不是最差水平.

标定脚本: `scripts/calibrate_baseline.py` (自包含, 无 torch 依赖)

### 2.3 为什么用几何平均而不是算术平均？

```
例子: thrust=100, mfr=100, isp=30
  算术平均: (100+100+30)/3 = 77  ← 看起来还行
  几何平均: (100×100×30)^(1/3) = 66  ← 暴露短板
```

**木桶原理**: 任何一个维度出问题, 推进器都可能失效. 几何平均天然惩罚"偏科"样本, 逼出短板.

同理, 每个维度的 `dim_score` 也是 stat_score、eng_score、anom_score 三者的几何平均.

### 2.4 双轨评分 — 为什么需要两条轨道？

#### 轨道 1: 统计精度 (Statistical Score)

```
sigma_stat = sqrt(mc_var + noise² + p50²)
z_stat = |pred - actual| / sigma_stat
χ² = mean(z_stat²)
score = f(χ²)  # 映射曲线
```

**σ 的三项来源**:
- `mc_var`: MC Dropout 模型认知不确定性 (这次预测有多不确定?)
- `noise²`: 传感器测量噪声 (仪器本身有多准?)
- `p50²`: 模型能力基线 (这个模型总体什么水平?)

三项加权合成一个"合理的期望误差". 如果实际误差正好等于这个 σ, z=1, χ²=1, 得 100 分.

**χ² → 分数映射曲线**:
```
χ² ≤ 1   → 100     (达到/超出预期)
χ² = 4   → 85      (正常的 2σ)
χ² = 9   → 70      (平均样本水平)
χ² = 16  → 55
χ² = 25  → 45      (明显偏差)
χ² > 25  → exp(-χ²/25) 渐近衰减
```

#### 轨道 2: 工程合规 (Engineering Compliance)

```
eng_tol_eff = max(物理指标, 模型能力下限)
           = max(eng_tol_spec, 3*baseline/|actual|)
sigma_eng = eng_tol_eff * |actual| / 3
z_eng = |pred - actual| / sigma_eng
compliance_rate = mean(z_eng < 3)
```

**为什么需要自适应容差?**

固定 ±5% 推力指标对低推力样本不公平:
- 0.31N 推力 × 5% = 0.0155N
- 但模型 p25 误差已经 0.0175N
- 模型做不到物理指标 → eng_score = 0, 不合理

自适应方案: `max(物理指标, 3×baseline/|actual|)`. 当模型能力不如物理要求时, 取模型能力为底线.

#### 轨道 3: 异常评分 (Anomaly Score)

```
z_stat > 3.0 → is_anomaly (3σ 规则)
anomaly_rate = n_anomaly / n_total

# 频率分: 指数衰减, k=6
freq_score = 100 * exp(-6 * anomaly_rate)

# 严重程度乘子: 极端 z 值加重处罚
severity = max(0.3, 1 - (max_z - 5) / 10)

anom_score = freq_score * severity
```

k=6 的含义:
```
异常率 0%   → 100
异常率 5%   → 74
异常率 10%  → 55
异常率 20%  → 30
异常率 30%  → 17
```

比 k=4 (更宽松) 和 k=10 (更严苛) 的折中选择 — 对小比例异常敏感但对大比例不致过度惩罚.

### 2.5 综合评分 — 为什么有 consistency_factor？

```
dim_avg = (T_score × M_score × I_score)^(1/3)
overall = dim_avg × (0.85 + 0.15 × consistency/100)
```

**consistency_factor 的角色**: 调节器 (0.85~1.0), 不是主导因子.

一致性高 (三维同时异常) → factor 接近 1.0 → 不加干预  
一致性低 (仅单维异常) → factor 接近 0.85 → 轻微减轻 (可能只是传感器噪声)

为什么是 0.85~1.0 而不是更大范围? 因为一致性只是"辅助判断" — 真实故障也可能只在一维先现, 一致性低不能大幅改变评分.

### 2.6 2-of-3 投票 — 为什么不是 OR 或 AND？

```
OR  (至少1维): 误报率高 — 传感器噪声常触发单维
AND (全部3维): 漏报率高 — MFR边缘点ISP常无效
2-of-3:       折中 — 真故障通常联动 ≥2维, 噪声通常只触发1维
```

物理逻辑: 推进器故障 (堵塞/泄漏/老化) → 推力+流量+比冲中至少两个同时响应. 单维跳变大概率是传感器噪声.

### 2.7 Isp 评分为什么要 MFR > 10 mg/s？

```
Isp = Thrust / (MFR × 1e-6 × G0)
```

当 MFR → 0 (点火/熄火边缘):
- 分母 → 0 → Isp → ∞
- 被 clip 到 [1, 5000]s
- 真值 ~3000s, 预测 ~1s (或反过来) → 残差 ~3000s

这不是模型的问题, 是数值计算在物理边缘失效. 所以在 `generate_health_report()` 中:
```python
isp_valid = mfr_actual > 10.0  # mg/s
# Isp 评分仅使用 isp_valid 为 True 的点
```

### 2.8 模型置信度 (Model Confidence)

```
mc_norm = clip(mc_std / |pred|, 0, 1)  # 归一化相对不确定性
avg_uncertainty = mean(mc_norm across 3 dims)
score = 100 * (1 - avg_uncertainty)
# 也受 SR (Success Rate) 影响
```

GLOBAL 模型在低推力样本上不确定性高 → 20/100, 通过 conf_factor 传递到总分.

---

## 三、UI 设计逻辑

### 3.1 4 标签页布局

| 标签 | 内容 | 为什么这样设计 |
|------|------|--------------|
| **遥测监控** | 4 张遥测卡片 + sparkline + 任务 header | 首屏给操作员最关键信息: 推力/流量/比冲/异常占比 |
| **异常检测** | 2+1 残差图 + 三维关联热力图 | 技术深度视图: 每个时间步的残差与阈值对比 |
| **健康报告** | 雷达图 + 维度详情 expander | 综合诊断: 五星图 + 分维度评分 + 漂移检测 |
| **特征归因** | Plotly SHAP 下拉切换 | 模型解释: 展示 17 个特征对预测的贡献 |

### 3.2 残差图 2+1 布局

```
┌─────────────────┬─────────────────┐
│  推力残差 (左)   │  流量残差 (右)   │  ← 顶行并排
├─────────────────┴─────────────────┤
│       比冲残差 (全宽)              │  ← 中行全宽
├───────────────────────────────────┤
│     三维关联热力图 (全宽)          │  ← 底行全宽
```

设计意图: 推力+流量是最核心的遥测双元, 并排对比直观; 比冲是导出量, 单独一行展示; 热力图三维叠加, 一眼看全.

### 3.3 Sparkline 趋势线

每张遥测卡片底部嵌入 80×18px SVG polyline:
- 前半段 vs 后半段均值对比 → delta 百分比
- 上升 (↑) 绿色, 下降 (↓) 琥珀色
- 异常占比卡片用滚动窗口 (window=20) 平滑 → 趋势比单点值更有信息量
- 近零保护: 两半段均 < 1e-5 → delta=0%

### 3.4 呼吸灯 (Breathing Status)

Header 状态指示点 CSS 动画:
- 异常占比 > 20%: 红色, 0.9s 脉动周期 (快速 — 紧急)
- 异常占比 > 5%:  琥珀色, 0.9s 脉动周期
- 正常: 绿色, 1.8s 脉动周期 (慢速 — 平稳)

使用 `combined_anom_mask` (2-of-3 投票) 计算异常占比, 与遥测卡片口径一致.

### 3.5 雷达图 (5 轴健康画像)

```
         推力综合 (dim_score)
            /\
           /  \
  模型置信度    流量综合
          |    |
  三维一致性 — 比冲综合
```

- 蓝色填充: 当前评估
- 琥珀虚线: 上次点火 (session 内历史, 保留 10 次)
- 红色虚线圆: 70 分 (警戒)
- 绿色虚线圆: 90 分 (优秀)

居中显示: `st.columns([1, 2, 1])`, 雷达在中间列.

### 3.6 Plotly SHAP (交互式)

下拉菜单切换推力/MFR/Isp 的 SHAP 条形图:
- 17 维特征, 按 |SHAP| 降序排列
- 颜色映射: 特征值高→红, 低→蓝
- FEATURE_PHYSICS 词典: 17 个特征的推断物理含义 (基于命名约定)
- 下拉菜单位于右上角 (x=0.85), 不遮挡条形

---

## 四、数据流

```
CSV 文件
  ↓ data_pipeline_v2.process_csv_file()
17维特征 (n_timesteps × 17)
  ↓ model.forward()
thrust_pred, mfr_pred (n_timesteps × 2)
  ↓ compute_isp()
isp (n_timesteps × 1)
  ↓ predict_with_uncertainty() [MC Dropout 20次]
thrust_std, mfr_std, isp_std (逐点不确定性)
  ↓ compute_residuals() + generate_health_report()
残差 + 异常掩码 + 健康报告
```

---

## 五、关键参数一览

```python
# 模型能力基线 (p50, 来自 calibrate_baseline.py)
BASELINE_THRUST = 0.0531    # N
BASELINE_MFR    = 58.45     # mg/s
BASELINE_ISP    = 86.0      # s

# 传感器噪声 (物理测量误差)
NOISE_THRUST = 0.005   # N
NOISE_MFR    = 0.5     # mg/s

# 工程容差 (3σ 等效, 推进器物理指标)
ENG_TOL_THRUST = 0.05   # ±5%
ENG_TOL_MFR    = 0.10   # ±10%
ENG_TOL_ISP    = 0.05   # ±5%

# 异常检测
ANOMALY_RATIO_WARN  = 0.05   # 琥珀呼吸灯
ANOMALY_RATIO_CRIT  = 0.20   # 红色呼吸灯

# Isp 有效范围
MFR_ISP_MIN = 10.0   # mg/s, 低于此 Isp 不可靠

# 传感器噪声 (用于 UI 层 _compute_actuals)
THRUST_NOISE_STD = 0.005
MFR_NOISE_STD    = 0.5
```

---

## 六、文件清单

| 文件 | 说明 |
|------|------|
| `app_v3.py` | Streamlit 主应用 (~1700行) |
| `inference_utils.py` | 推理封装 + MC Dropout + 健康评分 |
| `data_pipeline_v2.py` | 17维特征工程 |
| `scripts/calibrate_baseline.py` | p50 baseline 标定脚本 |
| `app_v2.py` | V2 回滚基线 (只读) |
| `NIGHT_LOG.md` | 开发日志 + 决策记录 |
| `docs/APP_V3_GUIDE.md` | 本文档 |

---

## 七、已知限制与待改进

1. **GLOBAL 模型置信度低** (20/100): 在低推力样本上 MC Dropout 不确定性高, 拖低总分. 可能需 per-SN 微调模型或校准 uncertainty.
2. **Isp baseline 用固定值**: 误差传播公式在低 MFR 时爆炸, 暂用 p50 固定值代替动态误差传播.
3. **一致性因子范围保守** (0.85~1.0): 对真实故障出现只在一维先现的场景可能不够敏感.
4. **p50 baseline 需要持续更新**: 随着模型改进 (微调/新架构), baseline 应重新标定.
