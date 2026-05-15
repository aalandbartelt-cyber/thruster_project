# 个体推进器微调策略文档

## Per-Thruster Fine-Tuning Strategy Documentation

**项目**: CRAIC 2026 · 航天单组元推进器数字孪生健康监测系统  
**分支**: dev-m  
**负责人**: M  
**日期**: 2026-05-15  

---

## 1. 问题背景

### 1.1 全局模型的局限

全局 DualOutputLSTM 模型在 908 台推进器的混合数据上训练，对大部分 SN 表现良好（测试 RMSE 0.0832 N），但对个别推进器的预测存在系统性偏差。原因包括：

- 制造公差导致的个体差异
- 不同 SN 的工况分布不均（训练数据中某些推力区间样本不足）
- 传感器标定差异

### 1.2 灾难性遗忘问题

直接对目标 SN 进行微调（freeze LSTM + Attention，训练 FC+LN）虽然能改善目标 SN 的预测，但会导致**灾难性遗忘**——模型在其他 11 台 SN 上的表现严重退化。

**基线实验** (`finetune_20260514_200630.log`):
- 12 台 SN 微调后，平均推力改善 +5.5%
- **全部 12 台出现严重遗忘**，遗忘比范围 1.33-2.93
- SN15 极端案例：遗忘比达 2.93（微调后在其余 SN 上的 RMSE 翻近 3 倍）

---

## 2. 方法演进

### 2.1 第一阶段：EWC (Elastic Weight Consolidation)

**原理**: 在背景数据上计算 Fisher 信息矩阵对角 F_i，训练时添加二次惩罚项：

```
L_total = L_task + (λ/2) * Σ F_i * (θ_i - θ*_i)²
```

重要参数（高 Fisher 值）移动被重罚，不重要参数可自由适配。

**关键实现细节**:
- **逐层 Fisher 归一化**: 全局归一化会使 bias Fisher 值（~8000）主导 weight Fisher 值（~0.001），导致 EWC 只保护 bias。改为每参数组独立归一化到 mean=1.0
- **分层混合回放**: 30% 背景数据，按 11 种 test_mode 分层抽样（max 12 文件）
- **Fisher 计算**: 200 批次，背景数据 11 文件 / 7529 窗口

**λ=25 实验结果** (`finetune_20260515_101311.log`):

| SN | Thrust 改善 | 遗忘比 | 评价 |
|----|-----------|--------|------|
| SN13 | +3.1% | 1.40 | 遗忘减轻但仍严重 |
| SN14 | +0.2% | 1.36 | 改善被抑制 |
| SN15 | +11.7% | 1.70 | 极端遗忘从 2.93 降至 1.70 |
| SN16 | +0.4% | 1.35 | 改善微弱 |
| SN23 | **+0.0%** | 1.40 | **改善完全被 EWC 杀死** |

**EWC 结论**: 
- 能降低遗忘比（2.93→1.70），但无法突破 ~1.35 的遗忘地板
- 高 λ 值会抑制改善（SN23: +14%→0%）
- EWC 本质是"以改善换遗忘减轻"的折中，无法根本解决

### 2.2 第二阶段：Adapter 微调

**核心思想**: 冻结全局模型全部 1.1M 参数，插入小型瓶颈适配器（~41K params, 3.6%），仅训练适配器。

**架构**:

```
AdapterDualOutputLSTM:
  base (frozen)              adapter (trainable)
  ┌─────────────┐
  │ LSTM (2L,256)│
  ├─────────────┤
  │ SelfAttention│ ──→ attn_adapter: Linear(256→64)→ReLU→Linear(64→256) [zero init]
  │   (4-head)   │
  ├─────────────┤
  │ LayerNorm    │
  ├─────────────┤
  │ FC(256→128)  │ ──→ fc_adapter: Linear(128→32)→ReLU→Linear(32→128) [zero init]
  │ ReLU, ReLU   │
  │ FC(128→2)    │
  └─────────────┘
```

**关键设计**:
- 适配器末层零初始化：训练从全局模型行为起步（near-identity）
- `model.train()` 重写：保持 base 在 eval 模式，防止 LSTM dropout 干扰训练
- 每台 SN 独立训练一个适配器，推理时按 SN 加载

**Adapter-only 结果** (`finetune_20260515_142140.log`):

| SN | Thrust | MFR | Isp |
|----|--------|-----|-----|
| SN13 | +15.1% | +36.7% | +13.2% |
| SN15 | +21.8% | +21.8% | **-17.9%** |
| SN23 | +9.8% | +15.8% | +0.3% |
| 平均 | +4.4% | +8.3% | +1.5% |

**发现**: 
- 遗忘问题根本解决（base model 冻结）
- 但 Isp 出现退化（SN15: -17.9%），因为 dual_loss 只优化 thrust+MFR，不约束比值
- 适配器过度优化个别指标时，thrust/mfr 比值失真

### 2.3 第三阶段：Triple Loss（推力 + 质量流量 + 比冲）

**问题根因**: `dual_loss = 0.7×MSE(thrust) + 0.3×MSE(mfr)` 不约束 Isp = thrust/(mfr×G0)，adapter 可独立调整 thrust 和 mfr 而不保持比值。

**解决方案**: 添加归一化 Isp 损失：

```python
def triple_loss(pred, target, w_thrust=0.6, w_mfr=0.25, w_isp=0.05):
    loss_t = MSE(pred_thrust, target_thrust)
    loss_m = MSE(pred_mfr, target_mfr)
    # 着火段 Isp，除以 100 使量级与归一化 thrust/mfr 对齐
    isp_pred = (pred_t * THRUST_SCALE) / (pred_m * MFR_MAX * 1e-6 * G0 + 1e-8)
    loss_i = MSE(isp_pred / 100.0, isp_true / 100.0)
    return 0.6*loss_t + 0.25*loss_m + 0.05*loss_i
```

**Isp 归一化是关键的工程技巧**: Isp 原始值 10-20 秒，MSE 量级 1-100，若不加 /100，Isp 损失会完全主导训练（loss 从 0.0005 跳到 38.8）。

**Triple Loss 结果** (`finetune_20260515_174930.log`):

| SN | Thrust | MFR | Isp |
|----|--------|-----|-----|
| SN13 | +11.9% | +32.0% | **+19.3%** |
| SN15 | +16.1% | +16.8% | **+12.5%** |
| SN23 | +7.7% | +11.9% | +3.9% |
| 平均 | +3.6% | +7.6% | **+5.3%** |

**对比**:

| 指标 | Adapter-only | Adapter + Isp Loss |
|------|-------------|-------------------|
| Thrust 平均 | +4.4% | +3.6% |
| Isp 平均 | +1.5% (混合) | **+5.3% (全部正向)** |
| Isp 退化台数 | 3台 | **0台** |
| 统计显著性 | p=0.060 | **p=0.039*** |

推力改善从 +4.4% 下降到 +3.6%（~0.8pp），换来了 +3.8pp 的 Isp 全面提升和零退化。这是合理的多目标折中。

---

## 3. 最终 Per-SN 策略

### 3.1 策略规则

基于全量实验结果，对每台 SN 制定最优策略：

- **推力改善 > 0.5%** → 使用 Adapter + Isp Loss
- **推力改善 ≤ 0.5%** → 使用全局模型（跳过 adapter，全局模型已足够好）

### 3.2 Per-SN 策略表

| SN | Thrust Base | Thrust FT | ΔThrust | ΔMFR | ΔIsp | 策略 |
|----|------------|----------|---------|------|------|------|
| SN13 | 0.0909 N | 0.0801 N | **+11.9%** | +32.0% | +19.3% | Adapter + Isp Loss |
| SN14 | 0.0876 N | 0.0865 N | +1.3% | +9.2% | +3.0% | Adapter + Isp Loss |
| SN15 | 0.0848 N | 0.0712 N | **+16.0%** | +16.8% | +12.5% | Adapter + Isp Loss |
| SN16 | 0.0992 N | 0.0988 N | +0.4% | -2.0% | +2.7% | **全局模型** |
| SN17 | 0.0908 N | 0.0902 N | +0.7% | +7.4% | +3.5% | Adapter + Isp Loss |
| SN18 | 0.0998 N | 0.0984 N | +1.4% | +3.4% | +2.8% | Adapter + Isp Loss |
| SN19 | 0.0982 N | 0.0970 N | +1.2% | +2.8% | +2.6% | Adapter + Isp Loss |
| SN20 | 0.0871 N | 0.0866 N | +0.6% | +6.5% | +3.3% | Adapter + Isp Loss |
| SN21 | 0.1033 N | 0.1017 N | +1.5% | +3.5% | +3.2% | Adapter + Isp Loss |
| SN22 | 0.0904 N | 0.0899 N | +0.6% | +0.5% | +2.8% | Adapter + Isp Loss |
| SN23 | 0.1447 N | 0.1336 N | +7.7% | +11.9% | +3.9% | Adapter + Isp Loss |
| SN24 | 0.0903 N | 0.0901 N | +0.2% | -1.5% | +2.7% | **全局模型** |

### 3.3 汇总统计

| 指标 | 数值 |
|------|------|
| Adapter 覆盖 | 10/12 台 (83%) |
| 全局模型覆盖 | 2/12 台 (SN16, SN24) |
| 有效推力改善均值 | **+3.6%** |
| 统计显著性 | **p = 0.039*** |
| 灾难性遗忘 | **零**（全局模型冻结） |
| Isp 退化 | **零**（全部 12 台正向） |
| 额外参数量 | 10 × 41K = 410K (37% of base) |

---

## 4. 实验配置

### 4.1 最佳超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| adapter 架构 | attn: 64→256, fc: 32→128 | 瓶颈维度 |
| 学习率 | 2e-4 | Adam, ReduceLROnPlateau |
| Epochs | 10 (early stop patience=8) | 通常 5-9 epoch 收敛 |
| 损失函数 | triple_loss | w_thrust=0.6, w_mfr=0.25, w_isp=0.05 |
| Isp 归一化 | /100.0 | 将 Isp MSE 量级对齐到 thrust/mfr |
| 训练文件数 | 12/SN | 验证 3 文件，测试余下全部 |
| Batch size | 32 | |
| 梯度裁剪 | max_norm=1.0 | |

### 4.2 使用方式

```bash
# 全量 12 台微调
python model_finetune_v2.py --use_adapters --use_isp_loss --epochs 10

# 单台微调
python model_finetune_v2.py --sn 15 --use_adapters --use_isp_loss --epochs 10

# 自定义 adapter 容量（对困难 SN 可尝试增大）
python model_finetune_v2.py --sn 16 --use_adapters --use_isp_loss \
    --attn_adapter_dim 128 --fc_adapter_dim 64 --epochs 10
```

---

## 5. 关键技术细节

### 5.1 为什么 EWC 不够用

EWC 对 Attention 层的约束是均匀的。但 Attention 的 263K 参数中，真正对全局特征提取关键的只是少数方向（对应高 Fisher 值的特征方向）。EWC 用对角 Fisher 近似，无法捕捉参数间的协方差结构。当目标 SN 需要的特征方向恰好与全局关键方向重叠时，EWC 的惩罚会"锁死"改善空间。

Adapter 方案绕过了这个问题：不修改 Attention 参数，而是在其输出上叠加一个小型校正。校正由 SN 特定的瓶颈层学习，瓶颈维度（64/32）限制了校正的表达能力，天然防止过拟合。

### 5.2 Isp 归一化为什么是 /100

Isp 原始值 10-20 秒，MSE 量级 1-100。而归一化后的 thrust/mfr MSE 量级 0.0001-0.003。若不加缩放：
- w_isp=0.15 × MSE(isp)≈1-10 → 贡献 0.15-1.5
- w_thrust=0.6 × MSE(thrust)≈0.0001 → 贡献 0.00006

Isp 损失比 thrust 大 2500-25000 倍，完全主导训练。除以 100（Isp/100 量级 0.1-0.2）后，MSE 降至 0.0001-0.01，与 thrust/mfr 对齐。

### 5.3 SN16/SN24 为什么不改善

SN16 和 SN24 的全局模型 RMSE 分别为 0.0992 N 和 0.0903 N，接近全局平均（0.0832 N）。这些 SN 的行为已被全局模型充分捕捉，adapter 的 near-identity 初始化意味着"找不到改进方向就保持原样"。

尝试增大 adapter 容量（128/64, 82K params）或提高学习率（5e-4）均无帮助——更大容量反而导致过拟合（SN16: +0.3%→-1.1%）。结论：这些 SN 不需要微调，全局模型已是最优。

---

## 6. App 集成

### 6.1 模型加载

`app_v2.py` 通过 `inference_utils.load_model_for_sn(sn)` 自动选择最优模型：

- SN13-15, SN17-23 → 加载对应 AdapterDualOutputLSTM
- SN16, SN24 → 加载全局 DualOutputLSTM

### 6.2 UI 功能

- **侧边栏模型选择器**: 手动切换 GLOBAL / SN##-TUNED
- **顶部状态栏**: 实时显示当前模型标签
- **遥测卡片**: 使用微调模型时显示 "TUNED" 标识
- **自动匹配**: 上传文件后根据 SN 自动选择合适的模型

---

## 7. 文件清单

| 文件 | 说明 |
|------|------|
| `model_finetune_v2.py` | 微调主脚本，包含 AdapterDualOutputLSTM、triple_loss、per-SN 训练 |
| `inference_utils.py` | 推理工具，包含 load_model_for_sn()、OPTIMAL_FINETUNE_STRATEGY |
| `app_v2.py` | Streamlit 大屏，集成模型选择与微调标识 |
| `outputs/models/v2/dual_output_lstm_v2.pth` | 全局模型权重 (1.1M params) |
| `outputs/models/v2/dual_output_lstm_v2_sn*_adapter.pth` | 各 SN 的 adapter 权重 (41K params × 12) |
| `outputs/finetune_logs/finetune_20260515_174930.log` | 最终 triple_loss 全量日志 |
| `docs/finetune_strategy.md` | 本文档 |

---

## 8. 结论

1. **EWC 能减轻但不能消除遗忘**，遗忘地板 ~1.35
2. **Adapter 从根本上解决遗忘**（base model 冻结），代价是推力改善从 +5.5% 降至 +4.4%
3. **Triple Loss 消除 Isp 退化**，进一步将推力降至 +3.6% 但换来 Isp +5.3% 全面正向
4. **Per-SN 策略**：10 台用 adapter，2 台用全局模型（SN16/SN24）
5. **统计显著** p=0.039*，方法链条完整可复现
