"""
inference_utils.py — 推理封装 + MC Dropout 不确定性量化
M 负责，dev-m 分支
供 app_v2.py（Streamlit 大屏）调用

用法：
    model = load_dual_model("outputs/models/v2/dual_output_lstm_v2.pth")
    thrust, mfr, isp = predict(model, x_tensor)
    thrust, mfr, isp, std = predict_with_uncertainty(model, x_tensor, n_samples=30)
    residuals, is_anomaly = compute_residuals(pred_thrust, actual_thrust)
    report = generate_health_report(isp, thrust, mfr, is_anomaly)
"""
import torch
import numpy as np

from data_pipeline_v2 import THRUST_SCALE, MFR_MAX
from model_train_v2 import DualOutputLSTM, INPUT_DIM

G0 = 9.80665
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ═══════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════

def load_dual_model(model_path="outputs/models/v2/dual_output_lstm_v2.pth",
                    device=None):
    """
    加载训练好的双输出 Attention-LSTM 模型。

    参数
    ----
    model_path : str
        .pth 权重文件路径
    device : torch.device, optional
        默认自动选择 CUDA/CPU

    返回
    ----
    model : nn.Module
        已加载权重的模型（eval 模式）
    """
    if device is None:
        device = DEVICE
    model = DualOutputLSTM(input_dim=INPUT_DIM).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    model.eval()
    return model


# ═══════════════════════════════════════════
# 单次推理
# ═══════════════════════════════════════════

@torch.no_grad()
def predict(model, x_tensor, device=None):
    """
    单次确定性推理。

    参数
    ----
    model : nn.Module
        已加载的模型
    x_tensor : torch.Tensor
        输入张量，shape [batch, 200, 17] 或 [200, 17]
    device : torch.device, optional

    返回
    ----
    thrust : np.ndarray
        推力 (N)，shape [batch, 200] 或 [200]
    mfr : np.ndarray
        质量流量 (mg/s)
    isp : np.ndarray
        比冲 (s)，Isp = thrust / (mfr × G0)
    """
    if device is None:
        device = DEVICE
    single = (x_tensor.dim() == 2)
    if single:
        x_tensor = x_tensor.unsqueeze(0)

    model = model.to(device)
    x_tensor = x_tensor.to(device)
    out = model(x_tensor)  # [B, 200, 2]

    thrust = out[:, :, 0].cpu().numpy() * THRUST_SCALE
    mfr    = out[:, :, 1].cpu().numpy() * MFR_MAX
    eps    = 1e-8
    isp    = thrust / (mfr * G0 * 1e-6 + eps)

    if single:
        thrust, mfr, isp = thrust[0], mfr[0], isp[0]
    return thrust, mfr, isp


# ═══════════════════════════════════════════
# MC Dropout 不确定性量化
# ═══════════════════════════════════════════

@torch.no_grad()
def predict_with_uncertainty(model, x_tensor, n_samples=30, device=None):
    """
    MC Dropout 推理：开启 dropout 多次前向传播 → 均值 ± 标准差。

    参数
    ----
    model : nn.Module
    x_tensor : torch.Tensor
        [batch, 200, 17] 或 [200, 17]
    n_samples : int
        采样次数，默认 30（推荐 20-50）

    返回
    ----
    thrust_mean, mfr_mean, isp_mean : np.ndarray
        各指标的均值
    thrust_std, mfr_std, isp_std : np.ndarray
        各指标的标准差（不确定性估计）
    """
    if device is None:
        device = DEVICE
    single = (x_tensor.dim() == 2)
    if single:
        x_tensor = x_tensor.unsqueeze(0)

    model = model.to(device)
    x_tensor = x_tensor.to(device)
    model.train()  # 开启 dropout

    all_thrust, all_mfr = [], []
    for _ in range(n_samples):
        out = model(x_tensor)  # [B, 200, 2]
        all_thrust.append(out[:, :, 0].cpu().numpy() * THRUST_SCALE)
        all_mfr.append(out[:, :, 1].cpu().numpy() * MFR_MAX)

    model.eval()

    thrust_arr = np.stack(all_thrust, axis=0)  # [n_samples, B, 200]
    mfr_arr    = np.stack(all_mfr,    axis=0)

    thrust_mean = thrust_arr.mean(axis=0)
    thrust_std  = thrust_arr.std(axis=0)
    mfr_mean    = mfr_arr.mean(axis=0)
    mfr_std     = mfr_arr.std(axis=0)

    eps = 1e-8
    isp_mean = thrust_mean / (mfr_mean * G0 * 1e-6 + eps)
    isp_std  = isp_mean * np.sqrt(
        (thrust_std / (thrust_mean + eps))**2 +
        (mfr_std   / (mfr_mean + eps))**2
    )  # 误差传播

    if single:
        return (thrust_mean[0], mfr_mean[0], isp_mean[0],
                thrust_std[0],  mfr_std[0],  isp_std[0])
    return (thrust_mean, mfr_mean, isp_mean,
            thrust_std,  mfr_std,  isp_std)


# ═══════════════════════════════════════════
# 残差计算与异常检测
# ═══════════════════════════════════════════

def compute_residuals(pred_thrust, actual_thrust, threshold=0.6):
    """
    计算推力残差，根据阈值判断异常步。

    参数
    ----
    pred_thrust : np.ndarray
        预测推力 (N)
    actual_thrust : np.ndarray
        实际推力 (N)
    threshold : float
        残差阈值，默认 0.6 N

    返回
    ----
    residuals : np.ndarray
        绝对残差 |pred - actual|
    is_anomaly : np.ndarray (bool)
        True 表示该时间步异常
    anomaly_ratio : float
        异常步占比
    """
    residuals = np.abs(pred_thrust - actual_thrust)
    is_anomaly = residuals > threshold
    anomaly_ratio = is_anomaly.mean()
    return residuals, is_anomaly, anomaly_ratio


# ═══════════════════════════════════════════
# 健康诊断报告
# ═══════════════════════════════════════════

def _health_score(value, good, warn, bad, lower_is_better=False):
    """根据阈值返回健康评分 0-100 和等级（结果钳制在 [0, 100]）"""
    if lower_is_better:
        if value <= good:   score = 100
        elif value <= warn: score = 60 - 40*(value-good)/(warn-good)
        else:               score = 20 - 20*(value-warn)/(bad-warn)
    else:
        if value >= good:   score = 100
        elif value >= warn: score = 60 - 40*(good-value)/(good-warn)
        else:               score = 20 - 20*(warn-value)/(warn-bad)
    score = max(0, min(100, score))
    level = 'good' if score >= 70 else ('warning' if score >= 40 else 'critical')
    return int(score), level


def generate_health_report(thrust, mfr, isp, is_anomaly=None,
                           residual_rms=None):
    """
    生成推进器健康诊断报告。
    评分基于**预测精度（残差）**和**异常占比**，而非绝对值阈值。

    参数
    ----
    thrust : np.ndarray
        推力预测序列 (N)
    mfr : np.ndarray
        质量流量预测序列 (mg/s)
    isp : np.ndarray
        比冲序列 (s)
    is_anomaly : np.ndarray (bool), optional
        异常步标签（残差 > 阈值）
    residual_rms : float, optional
        残差均方根（预测精度指标）

    返回
    ----
    report : dict
    """
    t_mean = float(np.mean(thrust)) if isinstance(thrust, np.ndarray) else thrust
    m_mean = float(np.mean(mfr))    if isinstance(mfr, np.ndarray)    else mfr
    i_mean = float(np.mean(isp))    if isinstance(isp, np.ndarray)    else isp

    # —— 评分逻辑：基于预测精度，用模型测试 RMSE（0.0832N）校准 ——
    MODEL_RMSE = 0.0832
    if residual_rms is not None:
        ratio = residual_rms / MODEL_RMSE
        if ratio <= 1.5:    acc_score = 100
        elif ratio <= 3.0:  acc_score = 80 - (ratio - 1.5) / 1.5 * 40
        elif ratio <= 5.0:  acc_score = 40 - (ratio - 3.0) / 2.0 * 30
        else:               acc_score = 10
        acc_score = max(0, min(100, acc_score))
        acc_level = 'good' if acc_score >= 70 else ('warning' if acc_score >= 40 else 'critical')
    else:
        acc_score = 100
        acc_level = 'good'

    # —— 异常占比 ——
    anomaly_ratio = 0.0
    anom_level = 'normal'
    if is_anomaly is not None:
        anomaly_ratio = float(np.mean(is_anomaly))
        anom_level = 'critical' if anomaly_ratio > 0.2 else (
                     'warning' if anomaly_ratio > 0.05 else 'normal')

    # —— 综合评分：精度权重 0.6 + 异常权重 0.4 ——
    anom_score = max(0, 100 - anomaly_ratio / 0.2 * 100)
    overall = int(0.6 * acc_score + 0.4 * anom_score)

    summary_text = '运行正常' if overall >= 70 else '需关注' if overall >= 40 else '建议检修'

    # —— 各指标状态（基于运行数据的统计描述，非评分）——
    def _desc(v, u):
        return f'{v:.2f}'.rstrip('0').rstrip('.') if abs(v) < 1 else f'{v:.1f}'.rstrip('0').rstrip('.') if abs(v) < 100 else f'{v:.0f}'

    return {
        'summary': f"综合健康评分 {overall}/100 —— {summary_text}",
        'prediction_accuracy': {
            'score': int(acc_score),
            'level': acc_level,
            'detail': f'预测残差 {residual_rms:.4f} N' if residual_rms else '正常'
        },
        'anomaly_status': {
            'anomaly_ratio': anomaly_ratio,
            'n_anomaly': int(is_anomaly.sum()) if is_anomaly is not None else 0,
            'total_steps': len(is_anomaly) if is_anomaly is not None else 0,
            'level': anom_level,
        },
        'telemetry': {
            'thrust': f'{_desc(t_mean, "N")} N',
            'mfr':    f'{_desc(m_mean, "mg/s")} mg/s',
            'isp':    f'{_desc(i_mean, "s")} s',
        },
        'overall_health': overall,
    }
