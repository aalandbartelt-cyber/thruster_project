# Thruster Project · 航天器推进器数字孪生与异常检测

> 第二十八届中国机器人及人工智能大赛 · 人工智能创新赛 · 上海赛区  
> 上海大学  
> 项目全称：基于多源动态特征与多目标耦合预测的航天器推进器数字孪生及健康监测系统

基于 Attention-LSTM 与 SHAP 归因的航天器推进器数字孪生系统。v2 在课程论文 v1 基础上完成五维升级：17维异质特征建模、双输出耦合预测（thrust + mfr + Isp）、基于真实标签的可量化异常检测、17维 SHAP 博弈论归因、MC Dropout 不确定性量化。

---

## v2 最终指标

| 指标 | v1（课程论文） | v2.0（参赛） | 提升 |
|------|-------------|------------|------|
| 输入维度 | 3维（2个常数广播） | **17维**（6动态 + 11 one-hot） | 维度质变 |
| 输出 | 1维（thrust） | **2维（thrust + mfr）+ Isp** | 全新能力 |
| Thrust RMSE | 0.1247 N | **0.0832 N** | **-33.3%** |
| Thrust MAE | — | **0.0637 N** | — |
| MFR RMSE | 无 | **49.1 mg/s** | 全新能力 |
| best val_loss | — | **0.000257**（Attention-LSTM） | — |
| 训练可复现性 | — | 前后两轮差异 < 2% | ✅ |

### SHAP 关键发现

| 排名 | Thrust | MFR | Isp |
|:----:|:-------|:----|:----|
| 1 | cumulated_throughput | cumulated_throughput | cumulated_throughput |
| 2 | ton | ton | cumulated_on_time |
| 3 | cumulated_on_time | cumulated_on_time | **vl** |

三大老化因子（throughput + on_time + pulses）**占 thrust SHAP 的 49%**，v1 仅占 2%。Permutation 交叉验证排名一致。

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
│   └── figures/v2/             # 图表（11张预测图 + 11张SHAP图）
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 环境配置

```bash
conda create -n thruster_ai python=3.10
conda activate thruster_ai
pip install -r requirements.txt
```

---

## 使用方法

### 训练

```bash
python model_train_v2.py
# 模型 → outputs/models/v2/dual_output_lstm_v2.pth
```

### 测试评估

```bash
python model_test_v2.py
# 性能对比图表 + 接口npy文件导出
python model_shap_v2.py
# 17维 SHAP 归因 + Permutation 交叉验证
```

### 推理（供 Streamlit 调用）

```python
from inference_utils import load_dual_model, predict, predict_with_uncertainty

model = load_dual_model("outputs/models/v2/dual_output_lstm_v2.pth")
thrust, mfr, isp = predict(model, x_tensor)  # 确定性推理
t_mean, m_mean, i_mean, t_std, m_std, i_std = \
    predict_with_uncertainty(model, x_tensor, n_samples=30)  # MC Dropout
```

### 异常检测评估（Z负责）

```bash
python model_anomaly_eval.py
# 读取 outputs/predictions/v2/*.npy → P/R/F1/IoU
```

### ONNX 量化（Z负责）

```bash
python model_onnx_export.py
# FP32/FP16/INT8 三档对比
```

### 启动数字孪生大屏

```bash
streamlit run app_v2.py
```

---

## Git 分支规范

```
main          ← 5/14 集成节点合并
├── dev-m     ← M（模型层）
└── dev-z     ← Z（评估/部署/UI）
```

- 每人只在自己分支提交，禁止直接 push 到 main
- 文件归属严格隔离

---

## 团队分工

| 角色 | 承担 |
|------|------|
| **M** | 数据 pipeline → 双输出训练 → 测试评估 → SHAP 归因 → 推理封装 |
| **Z** | 异常检测评估 → ONNX 量化 → Streamlit UI |

---

## 数据集

[Spacecraft Thruster Firing Tests Dataset](https://www.kaggle.com/datasets/patrickfleith/spacecraft-thruster-firing-tests-dataset) — Kaggle，合成数据。

## License & 学术诚信

学科竞赛参赛作品，独立开发。
