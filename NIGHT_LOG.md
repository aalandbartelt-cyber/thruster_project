# 通宵优化日志

**开始时间**：2026-05-15 17:22 UTC
**完成时间**：2026-05-15 20:20 UTC
**总耗时**：约 3 小时
**目标**：完成 optimization_roadmap.md 中 6 个方向

## 决策记录与进度

- [17:22] [INFO] Git 仓库初始化，创建 feature/v3-night-optimization 分支
- [17:22] [INFO] 复制 app_v2.py → app_v3.py 作为起点
- [17:25] [DONE] 方向 1：sparkline 趋势线 — commit 7d2a2ca
- [17:35] [DECISION] sparkline 使用 SVG polyline (80x18px)，无额外依赖 — 理由：保持零依赖，纯 HTML 内联渲染
- [17:35] [DONE] 方向 2：雷达图健康报告 — commit c58ff3e
- [17:40] [DECISION] 雷达图历史对比保留 10 次 — 理由：session_state 内存友好，10 次足够对比
- [17:40] [DECISION] 雷达图维度详情表格移到 expander 内 — 理由：减少首屏信息密度
- [18:10] [DONE] 方向 3：4 标签页布局 — commit 549f556
- [18:10] [DECISION] 标签不加 emoji — 理由：与 v2 克制风格一致
- [18:10] [DECISION] 生成报告按钮移到健康报告标签页 — 理由：减少侧边栏负担，报告在报告页触发
- [18:10] [DECISION] 呼吸灯 CSS 预埋到方向 3 的样式块 — 理由：减少重复样式注入
- [18:20] [DONE] 方向 4：残差 2+1 布局 + 热力图 — commit acf39ce
- [18:20] [DECISION] 热力图 cmap 用 RdYlGn_r, vmin=0 vmax=3 — 理由：3σ 为上限，绿→红符合直觉
- [18:35] [DONE] 方向 5：时间窗口 + 呼吸灯 — commit 9cbd080
- [18:35] [DECISION] render_mission_header() 封装为函数，在 if/else 两个分支分别调用 — 理由：Streamlit 无动态更新已渲染元素的能力
- [18:35] [DECISION] 异常占比 > 0.20 红色/0.9s 脉动, > 0.05 琥珀/0.9s, 正常绿色/1.8s — 理由：三档区分度高
- [18:45] [DONE] 方向 6：Plotly 交互式 SHAP — commit 130d837
- [18:45] [DECISION] FEATURE_PHYSICS 词典：17 个特征基于特征名常识推断物理含义 — 理由：无需查询论文，通过命名约定推断
- [18:45] [DECISION] Plotly 下拉菜单在右上角 (x=0.85) — 理由：不遮挡条形图

## 晨间总结

**完成时间**：2026-05-15 20:20 UTC
**总耗时**：约 3 小时

### 已完成方向
- ✅ 方向 1 (P0)：sparkline 趋势线 — commit 7d2a2ca
- ✅ 方向 2 (P0)：雷达图健康报告 + 历史对比 — commit c58ff3e
- ✅ 方向 3 (P1)：4 标签页信息架构 — commit 549f556
- ✅ 方向 4 (P1)：残差 2+1 布局 + 三维关联热力图 — commit acf39ce
- ✅ 方向 5 (P2)：时间窗口选择器 + Header 呼吸灯 — commit 9cbd080
- ✅ 方向 6 (P2)：Plotly 交互式 SHAP + 物理含义词典 — commit 130d837

### 关键决策回顾
1. **标签不加 emoji** — 与 v2 的克制 NASA 风格保持一致
2. **呼吸灯 CSS 预埋** — 方向 3 写样式时一次性加入，避免后续重复注入
3. **FEATURE_PHYSICS AI 推断** — 基于特征名 convention 推断（ramp1-4=斜坡特征, onmod/offmod=调制特征等），非权威定义
4. **Plotly 替代 matplotlib SHAP** — 交互性显著提升，下拉切换减少垂直空间占用
5. **render_mission_header() 封装** — 因 Streamlit 无动态元素更新，在文件加载/未加载两分支分别调用

### 额外修改
- 版本号 v2.0 → v3.0（header + welcome screen）
- 新增依赖：plotly (pip install)
- 侧边栏新增：时间窗口滑块
- app_v2.py 完全未动，可作为回滚基线

### 给用户的建议
- 启动方式：`streamlit run app_v3.py`
- 重点验收：
  1. 4 张遥测卡片的 sparkline 趋势线（方向箭头 + 变化率）
  2. 健康报告雷达图（5 轴 + 历史对比虚线）
  3. 4 个标签页切换流畅度（数据预计算，无重推理）
  4. 异常检测标签页的 2+1 残差布局 + 热力图
  5. 侧边栏时间窗口滑块对图表的切片效果
  6. Header 状态点动画（异常时变红加速脉动）
  7. SHAP 标签页的 Plotly 交互下拉切换
- 已知问题：无

---

# V4 健康评分系统迭代日志

**日期**：2026-05-16
**背景**：V3 评分系统存在严重循环标定问题 → 全员 100 分，需要从零重建评分体系

## 问题 1：V3 循环标定（全员 100 分）

**根因**：V3 用测试集 RMSE 作为 σ_modelref 去评分同一个测试集 → E[χ²]=1 → 所有人 100 分，零区分度

**修复**：用 p25（最佳 25%）替代 RMSE 作为 baseline，通过独立标定脚本 `scripts/calibrate_baseline.py` 计算

## 问题 2：p25 baseline 太紧（全员 6 分）

**现象**：p25=0.0175N，但测试 RMSE=0.1574N（9 倍）→ 平均样本 χ²≈76 → stat_score≈0

**修复**：p25 → p50（中位数），使平均样本映射到 χ²≈9 → 70/100。标定结果：
```
thrust: p50=0.0531N, mfr: p50=58.45mg/s, isp: p50=86.0s
```

## 问题 3：工程合规全部为零

**根因**：固定 ±5% 工程容差对低推力样本（0.31N）过紧 — 5%×0.31=0.0155N，但 p25 误差已达 0.0175N

**修复**：自适应工程容差 `eng_tol_eff = max(eng_tol_spec, 3*baseline/|actual|)` — 模型能力不如物理指标时取模型能力为下限

## 问题 4：Isp 数值爆炸主导评分

**根因**：Isp = T/(mfr·1e-6·G0)，点火边缘 mfr→0 时 Isp→∞，clip 到 [1,5000]s 产生极端残差（4986s）

**修复**：Isp 评分仅使用 MFR > 10 mg/s 的有效点，边缘点不参与评分但保留在图表中

## 问题 5：Sparkline delta 显示 "↑ 100000000.0%"

**根因**：异常率接近 0 时两个半段均值都 ≈ 0，除以 EPS 产生荒谬 delta

**修复**：两半段均 < 1e-5 时显示 0%，其余上限 ±99%

## 问题 6：GLOBAL 模型置信度 20/100

**现象**：GLOBAL 模型在低推力样本上 MC Dropout 不确定性高 → model_confidence=20 → 通过 conf_factor=0.88 拖低总分

**状态**：已知但未修复（模型能力限制，非评分 bug）

## 问题 7：异常状态栏与遥测卡片口径不一致

**现象**：状态栏显示"推力 1 步 / 流量 1 步"（红色），但卡片异常占比 0%

**根因**：状态栏用单维度 mask，遥测卡片用 2-of-3 投票 mask。两个维度各异常 1 步但不在同一时间步 → 投不出 2 票

**修复**：状态栏统一使用 combined_anom_mask（2-of-3 投票），单维度信号降级为"孤立信号"（绿色正常）

## 问题 8：残差图布局 / 雷达图尺寸

- 残差图推力/MFR 并排间距过大 → 图宽 18→14、wspace 0.28→0.06
- 雷达图尺寸过大且未居中 → 6.2→4.8 英寸、columns([1,2,1]) 居中

## V4 评分体系核心设计

### 双轨评分 (per dimension)
- **统计精度**：`sigma_stat = sqrt(mc_var + noise² + p50²)`，χ² → score 映射曲线（χ²≤1→100, χ²≈9→70, χ²≈25→45）
- **工程合规**：自适应容差，合规比例 → 分数（99%→100, 95%→80, 80%→50）
- **异常评分**：频率指数衰减 k=6 + 严重程度乘子 max(0.3, 1-(max_z-5)/10)
- **维度综合**：三维度的几何平均（木桶效应，任一项弱则总分低）

### 综合评分
```
dim_avg = (S_thrust · S_mfr · S_isp)^(1/3)
consistency_factor = 0.85 + 0.15 * consistency/100
overall = dim_avg * consistency_factor
```

### 2-of-3 投票
综合异常掩码：至少 2/3 维度同时报警才算真异常。物理逻辑：推进器故障 → 多物理量联动异常；单一维度跳变 → 传感器噪声

### 三维一致性 (Jaccard)
三个维度异常 mask 的交集/并集，评估异常是否物理关联（高分=三维同时异常=真实故障，低分=单维异常=传感器噪声）

## V4 最终参数

```python
# p50 baseline
BASELINE_THRUST = 0.0531    # N
BASELINE_MFR    = 58.45     # mg/s
BASELINE_ISP    = 86.0      # s (固定值)

# Sensor noise
NOISE_THRUST = 0.005   # N
NOISE_MFR    = 0.5     # mg/s

# Engineering tolerance (3σ equivalent)
ENG_TOL_THRUST = 0.05   # ±5%
ENG_TOL_MFR    = 0.10   # ±10%
ENG_TOL_ISP    = 0.05   # ±5%
```

## 决策记录

- [2026-05-16] [DECISION] p25→p50 baseline — 理由：p25 太紧（9x gap），p50 让平均样本 ~70 分
- [2026-05-16] [DECISION] 自适应工程容差 — 理由：固定 ±5% 对低推力不公平，取 max(物理指标, 模型能力下限)
- [2026-05-16] [DECISION] Isp 评分仅用 MFR>10mg/s — 理由：低流量时误差传播爆炸，Isp 数值无物理意义
- [2026-05-16] [DECISION] 2-of-3 投票 — 理由：减少单传感器误报，多维度联动才是真异常
- [2026-05-16] [DECISION] 异常状态栏与卡片统一使用 combined mask — 理由：消除口径不一致
- [2026-05-16] [DECISION] 残差图 wspace=0.06 消除中部空隙 — 理由：用户反馈并排图太小且不占满
