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
EPS = 1e-8
MG_PER_S_TO_KG_PER_S = 1e-6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_isp(thrust, mfr):
    """比冲 Isp = thrust / (mfr * G0)，mfr 单位为 mg/s，thrust 为 N"""
    return thrust / (mfr * MG_PER_S_TO_KG_PER_S * G0 + EPS)


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
# Adapter 模型加载（零遗忘微调）
# ═══════════════════════════════════════════

# 最优 per-SN 策略（基于全量实验结果）
OPTIMAL_FINETUNE_STRATEGY = {
    13: 'adapter', 14: 'adapter', 15: 'adapter',
    16: 'global',  17: 'adapter', 18: 'adapter',
    19: 'adapter', 20: 'adapter', 21: 'adapter',
    22: 'adapter', 23: 'adapter', 24: 'global',
}

# SN13-24 中哪些有 adapter 模型可用
ADAPTER_AVAILABLE_SNS = [sn for sn, s in OPTIMAL_FINETUNE_STRATEGY.items() if s == 'adapter']


def load_adapter_model(global_model, sn, device=None):
    """
    加载指定 SN 的 Adapter 微调模型。

    参数
    ----
    global_model : nn.Module
        已加载的全局 DualOutputLSTM 模型
    sn : int
        推进器序列号 (13-24)
    device : torch.device, optional

    返回
    ----
    model : AdapterDualOutputLSTM or None
        若该 SN 有 adapter 模型则返回，否则返回 None（应使用 global model）
    """
    import os as _os
    from model_finetune_v2 import AdapterDualOutputLSTM

    if device is None:
        device = DEVICE
    if sn not in ADAPTER_AVAILABLE_SNS:
        return None

    adapter_path = f"outputs/models/v2/dual_output_lstm_v2_sn{sn}_adapter.pth"
    if not _os.path.exists(adapter_path):
        return None

    model = AdapterDualOutputLSTM(global_model)
    model.load_state_dict(torch.load(adapter_path, map_location=device,
                                     weights_only=True))
    model.to(device)
    model.eval()
    return model


def load_model_for_sn(sn, device=None):
    """
    为指定 SN 加载最优模型（adapter 或 global）。

    返回 (model, model_label)
    """
    global_model = load_dual_model(device=device)
    strategy = OPTIMAL_FINETUNE_STRATEGY.get(sn, 'global')

    if strategy == 'adapter':
        adapter = load_adapter_model(global_model, sn, device=device)
        if adapter is not None:
            return adapter, f"SN{sn}-TUNED"
    return global_model, "GLOBAL"
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
    isp    = compute_isp(thrust, mfr)

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

    isp_mean = compute_isp(thrust_mean, mfr_mean)
    isp_std  = isp_mean * np.sqrt(
        (thrust_std / (thrust_mean + EPS))**2 +
        (mfr_std   / (mfr_mean + EPS))**2
    )

    if single:
        return (thrust_mean[0], mfr_mean[0], isp_mean[0],
                thrust_std[0],  mfr_std[0],  isp_std[0])
    return (thrust_mean, mfr_mean, isp_mean,
            thrust_std,  mfr_std,  isp_std)


# ═══════════════════════════════════════════
# 残差计算与异常检测
# ═══════════════════════════════════════════

# 传感器测量噪声标准差
THRUST_NOISE_STD = 0.005   # N
MFR_NOISE_STD    = 0.5     # mg/s
ISP_NOISE_STD    = 13.5    # s — error-propagated: sqrt((dIsp/dT*0.005)^2+(dIsp/dM*0.5)^2) at nominal 1.5N/80mg/s

# 模型基准 RMSE（v2.0 测试集）
REF_RMSE_THRUST = 0.0832   # N
REF_RMSE_MFR    = 20.8     # mg/s
# REF_RMSE_ISP computed dynamically via error propagation at data mean operating point


def _sigma_anomaly_mask(pred, actual, mc_std, noise_std, k=3.0, ref_rmse=None):
    """Sigma-based anomaly detection: |pred - actual| > k * sqrt(mc_std^2 + noise_std^2 + (ref_rmse/k)^2).

    The ref_rmse term floors sigma at the model's known average error, preventing
    unrealistically tight thresholds when MC uncertainty is very low.
    """
    _var = mc_std ** 2 + noise_std ** 2
    if ref_rmse is not None:
        _var = _var + (ref_rmse / k) ** 2
    sigma = np.sqrt(_var)
    z_score = np.abs(pred - actual) / (sigma + EPS)
    return z_score > k, z_score


def _compute_drift(residuals, ref_rmse, seq_len=200):
    """Linear drift detection on residual series.
    Returns (normalized_slope, pvalue, level)."""
    t = np.arange(len(residuals))
    slope, intercept = np.polyfit(t, residuals, 1)
    # Normalize: slope * seq_len gives total drift over window, divide by ref_rmse
    drift = slope * seq_len / (ref_rmse + EPS)
    # Rough significance: correlation coefficient
    corr = np.corrcoef(t, residuals)[0, 1]
    abs_drift = abs(drift)
    if abs_drift < 0.5:
        level = 'stable'
    elif abs_drift < 1.0:
        level = 'mild'
    elif abs_drift < 2.0:
        level = 'noticeable'
    else:
        level = 'severe'
    return drift, corr, level


def _accuracy_score(rms, ref_rmse):
    """Map RMS/REF ratio to 0-100 score with nonlinear decay."""
    ratio = rms / (ref_rmse + EPS)
    if ratio <= 1.0:
        return 100
    elif ratio <= 1.5:
        return int(100 - 30 * (ratio - 1.0) / 0.5)
    elif ratio <= 2.5:
        return int(70 - 30 * (ratio - 1.5) / 1.0)
    elif ratio <= 4.0:
        return int(40 - 25 * (ratio - 2.5) / 1.5)
    elif ratio <= 8.0:
        return int(15 - 10 * (ratio - 4.0) / 4.0)
    else:
        return 5


def _anomaly_score(anomaly_ratio):
    """Exponential decay: 0% -> 100, 5% -> 61, 10% -> 37, 20% -> 14."""
    return int(100 * np.exp(-10.0 * anomaly_ratio))


def _health_level(score):
    if score >= 70:
        return 'good'
    elif score >= 40:
        return 'warning'
    else:
        return 'critical'


def compute_residuals(pred_thrust, actual_thrust, threshold=0.6,
                      pred_mfr=None, actual_mfr=None,
                      pred_isp=None, actual_isp=None,
                      thrust_std=None, mfr_std=None, isp_std=None):
    """
    Multi-dimensional residual analysis.

    When MC uncertainty (thrust_std/mfr_std/isp_std) is provided, uses sigma-based
    statistical anomaly detection (3-sigma rule). Otherwise falls back to hard threshold.

    Parameters
    ----------
    pred_thrust, actual_thrust : np.ndarray
    threshold : float
        Fallback hard threshold when MC uncertainty unavailable.
    pred_mfr, actual_mfr : np.ndarray, optional
    pred_isp, actual_isp : np.ndarray, optional
    thrust_std, mfr_std, isp_std : np.ndarray, optional
        MC Dropout per-point standard deviations.

    Returns
    -------
    dict with keys: residuals, is_anomaly, anomaly_ratio (for thrust),
    plus mfr_*, isp_* if optional inputs provided.
    All masks use sigma-based detection when uncertainty is available.
    """
    has_uncertainty = (thrust_std is not None)

    if has_uncertainty:
        t_mask, t_z = _sigma_anomaly_mask(pred_thrust, actual_thrust,
                                           thrust_std, THRUST_NOISE_STD,
                                           ref_rmse=REF_RMSE_THRUST)
    else:
        t_res = np.abs(pred_thrust - actual_thrust)
        t_mask = t_res > threshold

    result = {
        'residuals': np.abs(pred_thrust - actual_thrust),
        'is_anomaly': t_mask,
        'anomaly_ratio': float(t_mask.mean()),
    }

    if pred_mfr is not None and actual_mfr is not None:
        if has_uncertainty and mfr_std is not None:
            m_mask, _ = _sigma_anomaly_mask(pred_mfr, actual_mfr,
                                             mfr_std, MFR_NOISE_STD,
                                             ref_rmse=REF_RMSE_MFR)
        else:
            m_res = np.abs(pred_mfr - actual_mfr)
            mfr_thr = threshold * (2000.0 / 8.0)  # scale proportional to MFR_MAX/THRUST_SCALE
            m_mask = m_res > mfr_thr
        result['mfr_residuals'] = np.abs(pred_mfr - actual_mfr)
        result['mfr_is_anomaly'] = m_mask
        result['mfr_anomaly_ratio'] = float(m_mask.mean())

    if pred_isp is not None and actual_isp is not None:
        if has_uncertainty and isp_std is not None:
            # Dynamic Isp reference RMSE via error propagation
            _tm = float(np.mean(pred_thrust))
            _mm = float(np.mean(pred_mfr))
            _C = 1e-6 * G0
            _dT = 1.0 / (_mm * _C + EPS)
            _dM = _tm / (_mm**2 * _C + EPS)
            _ref_isp = np.sqrt((_dT * REF_RMSE_THRUST)**2 + (_dM * REF_RMSE_MFR)**2)
            i_mask, _ = _sigma_anomaly_mask(pred_isp, actual_isp,
                                             isp_std, ISP_NOISE_STD,
                                             ref_rmse=_ref_isp)
        else:
            i_res = np.abs(pred_isp - actual_isp)
            i_mask = i_res > threshold * 50
        result['isp_residuals'] = np.abs(pred_isp - actual_isp)
        result['isp_is_anomaly'] = i_mask
        result['isp_anomaly_ratio'] = float(i_mask.mean())

    return result


def _compute_consistency(masks):
    """Cross-dimension anomaly consistency via Jaccard index.
    High consistency (all dims anomalous together) suggests real fault.
    Low consistency (isolated anomalies) suggests sensor noise.

    Args:
        masks: dict with keys 'thrust','mfr','isp' -> bool arrays
    Returns:
        (jaccard_score_0_to_100, level, detail_string)
    """
    t = masks['thrust'].astype(int)
    m = masks['mfr'].astype(int)
    i = masks['isp'].astype(int)

    intersection = (t & m & i).sum()
    union = (t | m | i).sum()

    total_anomalies = union
    total_points = len(t)

    if total_anomalies == 0:
        return 100, 'nominal', '三维均无异常'

    # When anomalies are very sparse, Jaccard is unreliable
    if total_anomalies < 0.05 * total_points:
        return 100, 'nominal', '异常点极少（<' + str(int(total_anomalies)) + '步），一致性分析不适用'

    jaccard = intersection / total_anomalies  # 0 to 1
    # High Jaccard = anomalies overlap across dims = likely real fault = LOW score
    # Low Jaccard = scattered, likely sensor noise = HIGH score (leniency)
    score = int(100 * (1.0 - jaccard))

    if jaccard >= 0.5:
        level = 'consistent'
        detail = '三维异常高度一致，可能存在真实推进器故障'
    elif jaccard >= 0.2:
        level = 'partial'
        detail = '部分维度异常重合，建议关注'
    else:
        level = 'scattered'
        detail = '异常分散在各维度，倾向于传感器噪声'

    return score, level, detail


def generate_health_report(thrust_pred, mfr_pred, isp_pred,
                           thrust_actual, mfr_actual, isp_actual,
                           thrust_std=None, mfr_std=None, isp_std=None,
                           model_confidence=None):
    """
    Multi-dimensional scientific health scoring.

    Six dimensions evaluated:
      Per-metric (thrust/MFR/Isp):
        - Accuracy Score: RMS residual vs reference RMSE
        - Anomaly Score: sigma-based anomaly ratio (exponential decay)
      Cross-dimension:
        - Drift Detection: linear trend on residuals
        - Consistency: Jaccard overlap of anomaly masks across dimensions

    Composite:
      per_dim = 0.5*accuracy + 0.5*anomaly
      dim_avg = mean(thrust, mfr, isp)
      overall = dim_avg * (0.85 + 0.15*consistency/100)

    Returns
    -------
    report : dict with overall_health, summary, per-dimension scores, and diagnostics
    """
    seq_len = len(thrust_pred)
    has_uncertainty = (thrust_std is not None)

    # --- Reference RMS (Isp computed via error propagation at data mean) ---
    t_mean = float(np.mean(thrust_pred))
    m_mean = float(np.mean(mfr_pred))
    C = 1e-6 * G0
    dIsp_dT = 1.0 / (m_mean * C + EPS)
    dIsp_dM = t_mean / (m_mean**2 * C + EPS)  # abs value
    ref_isp_rmse = np.sqrt((dIsp_dT * REF_RMSE_THRUST)**2 + (dIsp_dM * REF_RMSE_MFR)**2)
    # Dynamic Isp noise: same error propagation with sensor noise instead of model RMSE
    isp_noise_dyn = np.sqrt((dIsp_dT * THRUST_NOISE_STD)**2 + (dIsp_dM * MFR_NOISE_STD)**2)

    refs = {
        'thrust': REF_RMSE_THRUST,
        'mfr': REF_RMSE_MFR,
        'isp': ref_isp_rmse,
    }
    noise_stds = {
        'thrust': THRUST_NOISE_STD,
        'mfr': MFR_NOISE_STD,
        'isp': isp_noise_dyn,
    }

    dims = {}
    all_anomaly_masks = {}
    drift_summaries = []

    for dim_name, pred, actual, std in [
        ('thrust', thrust_pred, thrust_actual, thrust_std),
        ('mfr', mfr_pred, mfr_actual, mfr_std),
        ('isp', isp_pred, isp_actual, isp_std),
    ]:
        residuals = np.abs(pred - actual)
        rms = float(np.sqrt(np.mean(residuals ** 2)))

        # Accuracy
        acc = _accuracy_score(rms, refs[dim_name])

        # Anomaly detection
        if has_uncertainty and std is not None:
            mask, z_score = _sigma_anomaly_mask(pred, actual, std, noise_stds[dim_name],
                                                 ref_rmse=refs[dim_name])
        else:
            mask = residuals > 0.25  # fallback
        anom_ratio = float(mask.mean())
        anom = _anomaly_score(anom_ratio)
        n_anom = int(mask.sum())

        # Drift
        drift, drift_corr, drift_level = _compute_drift(residuals, refs[dim_name], seq_len)

        dim_score = int(0.5 * acc + 0.5 * anom)

        dims[dim_name] = {
            'accuracy': acc,
            'anomaly': anom,
            'dim_score': dim_score,
            'rms': rms,
            'anomaly_ratio': anom_ratio,
            'n_anomaly': n_anom,
            'total_steps': seq_len,
            'drift': drift,
            'drift_corr': drift_corr,
            'drift_level': drift_level,
        }

        all_anomaly_masks[dim_name] = mask

        if drift_level != 'stable':
            direction = '上升' if drift > 0 else '下降'
            drift_summaries.append(
                f'{dim_name_map_cn(dim_name)}残差{direction}趋势 (drift={drift:.2f}σ)')

    # --- Consistency ---
    cons_score, cons_level, cons_detail = _compute_consistency(all_anomaly_masks)

    # --- Composite ---
    dim_avg = np.mean([dims[d]['dim_score'] for d in ['thrust', 'mfr', 'isp']])
    consistency_factor = 0.85 + 0.15 * cons_score / 100.0
    overall = int(dim_avg * consistency_factor)
    overall = max(0, min(100, overall))
    level = _health_level(overall)

    # --- Summary text ---
    if overall >= 70:
        summary = '运行正常'
    elif overall >= 40:
        reasons = []
        for d in ['thrust', 'mfr', 'isp']:
            if dims[d]['dim_score'] < 60:
                reasons.append(f'{dim_name_map_cn(d)}评分偏低({dims[d]["dim_score"]})')
        if cons_level != 'nominal':
            reasons.append(cons_detail)
        if drift_summaries:
            reasons.append('; '.join(drift_summaries))
        summary = '需关注 —— ' + ('; '.join(reasons) if reasons else '指标轻度偏离')
    else:
        summary = '建议检修 —— 多项指标显著偏离基准'

    # --- Telemetry ---
    def _desc(v):
        if abs(v) < 1: return f'{v:.3f}'.rstrip('0').rstrip('.')
        elif abs(v) < 100: return f'{v:.1f}'.rstrip('0').rstrip('.')
        else: return f'{v:.0f}'

    # --- Model confidence ---
    conf_score = model_confidence if model_confidence is not None else 90
    conf_level = _health_level(conf_score)

    return {
        'overall_health': overall,
        'health_level': level,
        'summary': f'综合健康评分 {overall}/100 —— {summary}',
        'thrust': dims['thrust'],
        'mfr': dims['mfr'],
        'isp': dims['isp'],
        'consistency': {
            'score': cons_score,
            'level': cons_level,
            'detail': cons_detail,
        },
        'drift_summary': '; '.join(drift_summaries) if drift_summaries else '无显著漂移',
        'model_confidence': {
            'score': conf_score,
            'level': conf_level,
        },
        'telemetry': {
            'thrust': f'{_desc(np.mean(thrust_pred))} N',
            'mfr':    f'{_desc(np.mean(mfr_pred))} mg/s',
            'isp':    f'{_desc(np.mean(isp_pred))} s',
        },
    }


def dim_name_map_cn(name):
    _map = {'thrust': '推力', 'mfr': '质量流量', 'isp': '比冲'}
    return _map.get(name, name)
