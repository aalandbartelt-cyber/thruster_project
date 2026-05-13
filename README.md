# Thruster Project · 航天器推进器数字孪生与异常检测

> 第二十八届中国机器人及人工智能大赛 · 人工智能创新赛 · 上海赛区  
> 上海大学  
> 项目全称：基于多源动态特征与多目标耦合预测的航天器推进器数字孪生及健康监测系统

基于 Attention-LSTM 与 SHAP 归因的航天器推进器数字孪生系统。已完成 v2 全流程开发（CUDA），同步在**华为云昇腾 CANN** 上进行国产化算力验证。

> **项目定位**：使用 Kaggle 合成数据集（STFT）构建完整的 AI 工程流水线。核心交付是技术方案的工程可行性验证，而非依赖特定数据集的"自导自演"。所有结论在合成数据上成立，反映推进器的物理结构和数据规律，具备迁移到真实数据的条件。

### 技术闭环一览

```
数据清洗 → 17维特征工程 → 5轮架构对比 → 训练 + 可复现性验证
  → 11种 mode 分项测试 → SHAP 归因 + Permutation 交叉验证
    → MC Dropout 不确定性量化 → CANN 国产化算力验证
      → NASA 风格部署大屏 → 个体推进器微调方案
```

---

## v2 最终指标（CUDA 基准）

| 指标 | v1（课程论文） | v2.0（参赛） | 提升 |
|------|-------------|------------|------|
| 输入维度 | 3维（2个常数广播） | **17维**（6动态 + 11 one-hot） | 维度质变 |
| 输出 | 1维（thrust） | **2维（thrust + mfr）+ Isp** | 全新能力 |
| Thrust RMSE | 0.1247 N | **0.0832 N** | **-33.3%** |
| Thrust MAE | — | **0.0637 N** | — |
| MFR RMSE | 无 | **49.1 mg/s** | 全新能力 |
| best val_loss | — | **0.000257**（Attention-LSTM） | — |

### SHAP 关键发现

| 排名 | Thrust | MFR | Isp |
|:----:|:-------|:----|:----|
| 1 | cumulated_throughput | cumulated_throughput | cumulated_throughput |
| 2 | ton | ton | cumulated_on_time |
| 3 | cumulated_on_time | cumulated_on_time | **vl** |

三大老化因子（throughput + on_time + pulses）**占 thrust SHAP 的 49%**，v1 仅占 2%。

---

## 国产化算力验证（🔥 核心创新点）

CUDA 闭源，中国航天军工场景须用国产算力。本项目在**华为云昇腾 CANN 9.0** 上重跑 v1/v2 全流程，验证算法在国产架构上的数值一致性。

| 对比项 | CUDA（NVIDIA） | CANN（华为昇腾） |
|:-------|:--------------:|:--------------:|
| Thrust RMSE | 0.0832 N | （S 验证中） |
| MFR RMSE | 49.1 mg/s | （S 验证中） |
| SHAP 排名 | throughput > ton > on_time | （S 验证中） |

**分支**：`dev-s`

---

## 目录结构

```
thruster_project/
│
├── v1_baseline/                # v1原始代码归档（只读不改）
│
├── data_pipeline_v2.py         # [M] 17维预处理 + 双输出标签
├── model_train_v2.py           # [M] Attention-LSTM 训练（17→2）
├── model_test_v2.py            # [M] 测试评估 + 接口npy导出
├── analyze_ssf.py              # [M] SSF诊断分析
├── model_shap_v2.py            # [M] 17维 SHAP + Permutation
├── inference_utils.py          # [M] 推理封装 + MC Dropout
├── model_anomaly_eval.py       # [Z] 真实标签异常评估
├── model_onnx_export.py        # [Z] ONNX导出 + INT8量化
├── app_v2.py                   # [Z主导·M联调] Streamlit大屏
│
├── outputs/
│   ├── models/v2/              # 模型权重
│   ├── predictions/v2/         # 接口npy + SHAP npy
│   └── figures/v2/             # 图表
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 环境配置（CUDA）

```bash
conda create -n thruster_ai python=3.10
conda activate thruster_ai
pip install -r requirements.txt
```

## 环境配置（CANN 华为云昇腾）

详见 `docs/2026-05-12_致S同学_华为云昇腾CANN部署指南.md`：
- 镜像：CANN 9.0 + PyTorch 2.4.0（官方预装）
- 仅需修改 3 行代码（`cuda` → `npu`）
- SHAP 在 CPU 模式运行，无需改动

---

## 使用方法

### 训练

```bash
python model_train_v2.py
# 模型 → outputs/models/v2/dual_output_lstm_v2.pth
```

### 测试评估

```bash
python model_test_v2.py     # 性能对比图表 + 接口npy导出
python model_shap_v2.py     # 17维 SHAP 归因 + Permutation 交叉验证
```

### 推理（供 Streamlit 调用）

```python
from inference_utils import load_dual_model, predict, predict_with_uncertainty

model = load_dual_model("outputs/models/v2/dual_output_lstm_v2.pth")
thrust, mfr, isp = predict(model, x_tensor)
t_mean, m_mean, i_mean, t_std, m_std, i_std = \
    predict_with_uncertainty(model, x_tensor, n_samples=30)  # MC Dropout
```

### 异常检测（Z负责）

```bash
python model_anomaly_eval.py     # → P/R/F1/IoU
python model_onnx_export.py      # → FP32/FP16/INT8 三档对比
```

### 启动数字孪生大屏

```bash
streamlit run app_v2.py
```

---

## Git 分支规范

```
main          ← 5/14 集成节点合并
├── dev-m     ← M（模型层：pipeline → 训练 → SHAP → 推理）
├── dev-z     ← Z（评估/部署：异常检测 → ONNX → UI）
└── dev-s     ← S（CANN验证：代码迁移 → NPU重跑 → 对比报告）
```

---

## 团队分工

| 角色 | 承担 |
|------|------|
| **M** | 模型层（CUDA）：数据pipeline → 训练 → SHAP → 推理封装 |
| **Z** | 评估部署层：异常检测 → ONNX量化 → Streamlit UI |
| **S**（5/12加入） | **CANN验证层**：华为云昇腾全流程重跑 + CUDA vs CANN 对比报告 |

---

## 数据集

[Spacecraft Thruster Firing Tests Dataset](https://www.kaggle.com/datasets/patrickfleith/spacecraft-thruster-firing-tests-dataset) — Kaggle，合成数据。

## License & 学术诚信

学科竞赛参赛作品，独立开发。
