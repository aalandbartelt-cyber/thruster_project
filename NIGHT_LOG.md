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
