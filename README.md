# Thruster Project · 航天器推进器数字孪生与异常检测

> 第二十八届中国机器人及人工智能大赛 · 人工智能创新赛 · 上海赛区  
> 上海大学

基于多架构深度时序网络与SHAP归因的航天器推进器数字孪生系统。当前版本（v2）在课程论文版本（v1）基础上做了三项核心升级：6维异质特征建模、双输出耦合预测（thrust + mfr）、基于真实标签的可量化异常检测评估。

---

## 目录结构

```
thruster_project/
│
├── data/                       # 原始数据（不进git）
│   ├── dataset/                # STFT数据集训练/测试CSV
│   └── metadata.csv
│
├── docs/                       # 项目文档
│   ├── STFT Dataset Description.pdf
│   └── (后续：项目研究报告、查新报告、答辩PPT)
│
├── outputs/                    # 模型、预测结果、图表（不进git）
│   ├── figures/
│   │   ├── v1/                 # v1版本产出图表
│   │   └── v2/                 # v2版本产出图表
│   ├── models/
│   │   ├── v1/                 # v1模型权重（thruster_lstm_v1.pth）
│   │   └── v2/                 # v2模型权重（dual_output_lstm_v2.pth等）
│   └── predictions/
│       ├── v1/
│       └── v2/                 # M→Z接口npy文件
│
├── v1_baseline/                # v1原始代码归档（只读不改，用于答辩对比）
│   ├── app.py                  # Streamlit大屏（v1）
│   ├── model_train.py          # 3维单输出训练
│   ├── model_test.py
│   ├── model_anomaly_detect.py
│   ├── model_finetune.py
│   ├── model_shap_analysis.py
│   ├── model_compare.py
│   ├── model_arch_attention_lstm.py
│   ├── model_arch_cnn_lstm.py
│   ├── model_arch_tcn.py
│   ├── model_arch_transformer.py
│   ├── CUDAcs.py
│   └── 123.py
│
├── data_pipeline_v2.py         # [M] 6维特征预处理（共享import）
├── model_train_v2.py           # [M] 双输出LSTM训练
├── model_test_v2.py            # [M] 新模型测试与对比
├── model_shap_v2.py            # [M] 6维SHAP归因
├── inference_utils.py          # [M] 推理封装（供app_v2.py调用）
├── model_anomaly_eval.py       # [Z] 真实标签定量评估
├── model_onnx_export.py        # [Z] ONNX导出与量化对比
├── app_v2.py                   # [Z主导UI · M接后端] 新版Streamlit大屏
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## v1 vs v2 改进对比

| 维度 | v1（课程论文版） | v2（参赛版） |
|---|---|---|
| 输入特征 | 3维（ton + 2个常数广播） | 6维（ton + pressure + 3个老化因子 + test_mode） |
| 输出维度 | 1维（thrust） | 2维（thrust + mfr）+ 推导Isp |
| 异常检测 | 经验阈值（残差>0.6N） | 真实标签 + P/R/F1/IoU四指标 |
| 模型部署 | 仅PyTorch | ONNX + FP32/FP16/INT8量化对比 |
| 可解释性 | 3维SHAP归因 | 3维vs6维SHAP对比图 |

---

## 环境配置

```bash
# Python 3.10+
pip install -r requirements.txt
```

主要依赖：PyTorch、Streamlit、SHAP、ONNXRuntime、scikit-learn、matplotlib、seaborn、pandas。

---

## 使用方法

### 训练v2模型（M负责）

```bash
python model_train_v2.py
# 模型保存至 outputs/models/v2/dual_output_lstm_v2.pth
```

### 评估v2模型

```bash
python model_test_v2.py     # 性能对比图表
python model_shap_v2.py     # 6维SHAP归因
```

### 异常检测定量评估（Z负责）

```bash
python model_anomaly_eval.py
# 读取 outputs/predictions/v2/*.npy，输出P/R/F1/IoU指标
```

### 模型轻量化（Z负责）

```bash
python model_onnx_export.py
# 输出ONNX模型 + 三档量化对比表
```

### 启动数字孪生大屏

```bash
streamlit run app_v2.py
```

---

## 团队分工

| 角色 | 主要承担 |
|---|---|
| M（队长） | 核心模型层：6维pipeline、双输出训练、SHAP重跑、推理封装 |
| Z | 评估部署层：真实标签评估、ONNX量化、Streamlit UI主导 |

详见 `docs/参赛任务分工计划.md`。

---

## Git协作规范

```
main          ← 稳定版本，仅在5/14集成节点合并
  ├── dev-m   ← M工作分支
  └── dev-z   ← Z工作分支
```

- 每人只在自己分支提交，禁止直接push到main
- 每日提交前先 `git pull origin 自己的分支`
- commit message格式：`[M/Z] 简短描述`
- 文件归属严格隔离，避免修改对方负责的文件

---

## 数据集来源

[Spacecraft Thruster Firing Tests Dataset](https://www.kaggle.com/datasets/patrickfleith/spacecraft-thruster-firing-tests-dataset) by Patrick Fleith on Kaggle，合成数据，部分基于单组元化学推进器的真实物理特性。

数据集详细字段说明见 `docs/STFT Dataset Description.pdf`。

---

## License & 学术诚信声明

本项目为学科竞赛参赛作品，所有代码与文档由团队成员独立开发，符合大赛"严禁一稿多投、严禁抄袭"的诚信要求。