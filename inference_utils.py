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
    """比冲 Isp = thrust / (mfr * G0)，mfr 单位为 mg/s，thrust 为 N.
    Clipped to [1, 5000] s to prevent numerical explosion when MFR ≈ 0."""
    raw = thrust / (mfr * MG_PER_S_TO_KG_PER_S * G0 + EPS)
    return np.clip(raw, 1.0, 5000.0)


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
# 残差计算与异常检测 (v4 — p25 baseline + 工程容差)
# ═══════════════════════════════════════════

# 传感器测量噪声标准差
NOISE_THRUST = 0.005   # N
NOISE_MFR    = 0.5     # mg/s
# Isp noise computed dynamically via error propagation

# p50 baseline（scripts/calibrate_baseline.py，2026-05-16）
# 用残差幅值中位数（p50）作为"正常"基线 → 平均测试样本 ≈ 70 分
BASELINE_THRUST = 0.0531    # N   — p50 of |residual_thrust|
BASELINE_MFR    = 58.45     # mg/s — p50 of |residual_mfr|
BASELINE_ISP    = 86.0      # s   — p50 of |residual_isp| (固定值，误差传播在低流量炸)

# 工程容差（相对值，3σ 等价）
ENG_TOL_THRUST = 0.05   # ±5%
ENG_TOL_MFR    = 0.10   # ±10%
ENG_TOL_ISP    = 0.05   # ±5%


def _isp_uncertainty(thrust_ref, mfr_ref, thrust_sigma, mfr_sigma):
    """Isp 误差传播: σ_Isp² = (∂Isp/∂T)²·σ_T² + (∂Isp/∂M)²·σ_M²."""
    k = MG_PER_S_TO_KG_PER_S * G0
    t_mean = float(np.mean(thrust_ref))
    m_mean = float(np.mean(mfr_ref))
    dT = 1.0 / (m_mean * k + EPS)
    dM = t_mean / (m_mean**2 * k + EPS)
    return float(np.sqrt((dT * thrust_sigma)**2 + (dM * mfr_sigma)**2))


def _score_from_chi2_v4(chi2):
    """χ² → 统计判别分（p50 基线校准）。
    平均测试样本 χ²≈9 → 70 分；好样本 χ²≈3 → 85；差样本 χ²≈25 → 45."""
    if chi2 <= 1.0:
        return 100.0
    if chi2 <= 3.0:
        return 100.0 - (chi2 - 1.0) / 2.0 * 15.0   # 100 → 85
    if chi2 <= 9.0:
        return 85.0 - (chi2 - 3.0) / 6.0 * 15.0      # 85 → 70
    if chi2 <= 25.0:
        return 70.0 - (chi2 - 9.0) / 16.0 * 25.0     # 70 → 45
    return max(0.0, 45.0 * np.exp(-(chi2 - 25.0) / 20.0))


def _score_from_compliance(compliance_rate):
    """工程合格率 → 0-100 分."""
    if compliance_rate >= 0.99:
        return 100.0
    if compliance_rate >= 0.95:
        return 80.0 + (compliance_rate - 0.95) / 0.04 * 20.0
    if compliance_rate >= 0.80:
        return 50.0 + (compliance_rate - 0.80) / 0.15 * 30.0
    if compliance_rate >= 0.50:
        return 20.0 + (compliance_rate - 0.50) / 0.30 * 30.0
    return compliance_rate * 40.0


def _score_from_anomaly_v4(z_use, is_anomaly):
    """异常评分：频率衰减 (k=6) + 乘法严重度惩罚."""
    n_anom = int(is_anomaly.sum())
    if n_anom == 0:
        return 100.0
    anom_ratio = n_anom / len(z_use)
    max_z = float(np.max(z_use))
    score = 100.0 * np.exp(-6.0 * anom_ratio)
    if max_z > 5.0:
        severity_factor = max(0.3, 1.0 - (max_z - 5.0) / 10.0)
        score *= severity_factor
    return max(0.0, score)


def _detect_drift(pred, actual):
    """量化漂移: drift_rate = |slope| * n / std(residual)."""
    residual = np.asarray(pred).ravel() - np.asarray(actual).ravel()
    n = len(residual)
    if n < 20:
        return 'stable', 0.0
    t = np.arange(n)
    slope = np.polyfit(t, residual, 1)[0]
    res_std = np.std(residual) + EPS
    drift_rate = abs(slope) * n / res_std
    if drift_rate < 0.5:
        level = 'stable'
    elif drift_rate < 1.5:
        level = 'mild'
    elif drift_rate < 3.0:
        level = 'noticeable'
    else:
        level = 'severe'
    return level, float(drift_rate)


def _health_level(score):
    if score >= 80:
        return 'good'
    elif score >= 60:
        return 'warning'
    else:
        return 'critical'


def _dim_health_v4(pred, actual, mc_std, noise_std, baseline_rmse, eng_tol_rel):
    """V4 双轨维度评分：统计判别 + 工程合格 + 异常检测 → 几何平均."""
    pred = np.asarray(pred).ravel()
    actual = np.asarray(actual).ravel()
    n = len(pred)

    if mc_std is None:
        mc_var = np.zeros_like(pred)
    else:
        mc_var = np.asarray(mc_std).ravel()**2

    # 统计 σ：p25 baseline + MC + noise
    sigma_stat = np.sqrt(mc_var + noise_std**2 + baseline_rmse**2)
    # 工程 σ：自适应容差 = max(物理规范, 模型 p25 能力兜底)
    abs_signal = np.maximum(np.abs(actual), 1e-3)
    eng_tol_eff = np.maximum(eng_tol_rel, 3.0 * baseline_rmse / abs_signal)
    sigma_eng_3sigma = eng_tol_eff * abs_signal
    sigma_eng = sigma_eng_3sigma / 3.0

    err = np.abs(pred - actual)
    z_stat = err / (sigma_stat + EPS)
    z_eng  = err / (sigma_eng + EPS)
    is_anomaly = z_stat > 3.0  # 统计基线判定异常（p25 calibrated）

    chi2 = float(np.mean(z_stat**2))
    stat_score = _score_from_chi2_v4(chi2)

    compliance_rate = float(np.mean(z_eng < 3.0))
    eng_score = _score_from_compliance(compliance_rate)

    anom_score = _score_from_anomaly_v4(z_stat, is_anomaly)

    # 几何平均（短板效应）
    s_geo = (max(stat_score, 1) * max(eng_score, 1) * max(anom_score, 1)) ** (1/3)
    dim_score = int(round(s_geo))

    drift_level, drift_rate = _detect_drift(pred, actual)
    rms = float(np.sqrt(np.mean((pred - actual)**2)))

    return {
        'dim_score':       dim_score,
        'stat_score':      int(round(stat_score)),
        'eng_score':       int(round(eng_score)),
        'anomaly':         int(round(anom_score)),
        'accuracy':        int(round(stat_score)),  # 兼容旧字段名
        'rms':             rms,
        'chi2':            round(chi2, 3),
        'mean_z':          round(float(np.mean(z_stat)), 3),
        'max_z':           round(float(np.max(z_eng)), 3),
        'p95_z':           round(float(np.percentile(z_eng, 95)), 3),
        'compliance_rate': round(compliance_rate, 4),
        'is_anomaly':      is_anomaly,
        'n_anomaly':       int(is_anomaly.sum()),
        'total_steps':     n,
        'anomaly_ratio':   float(is_anomaly.mean()),
        'drift_level':     drift_level,
        'drift_rate':      round(drift_rate, 3),
    }


def compute_residuals(pred_thrust, actual_thrust, threshold=0.6,
                      pred_mfr=None, actual_mfr=None,
                      pred_isp=None, actual_isp=None,
                      thrust_std=None, mfr_std=None, isp_std=None):
    """V4 残差分析：MC 可用时用统计 p25 基线判定异常，否则回退硬阈值."""
    has_uncertainty = (thrust_std is not None)

    if has_uncertainty:
        _mc_var_t = np.asarray(thrust_std).ravel()**2 if thrust_std is not None else 0
        _sigma_t = np.sqrt(_mc_var_t + NOISE_THRUST**2 + BASELINE_THRUST**2)
        t_mask = np.abs(pred_thrust - actual_thrust) > 3.0 * _sigma_t
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
            _mc_var_m = np.asarray(mfr_std).ravel()**2
            _sigma_m = np.sqrt(_mc_var_m + NOISE_MFR**2 + BASELINE_MFR**2)
            m_mask = np.abs(pred_mfr - actual_mfr) > 3.0 * _sigma_m
        else:
            m_res = np.abs(pred_mfr - actual_mfr)
            mfr_thr = threshold * (2000.0 / 8.0)
            m_mask = m_res > mfr_thr
        result['mfr_residuals'] = np.abs(pred_mfr - actual_mfr)
        result['mfr_is_anomaly'] = m_mask
        result['mfr_anomaly_ratio'] = float(m_mask.mean())

    if pred_isp is not None and actual_isp is not None:
        if has_uncertainty and isp_std is not None:
            _isp_ns = _isp_uncertainty(actual_thrust, actual_mfr, NOISE_THRUST, NOISE_MFR)
            _mc_var_i = np.asarray(isp_std).ravel()**2
            _sigma_i = np.sqrt(_mc_var_i + _isp_ns**2 + BASELINE_ISP**2)
            i_mask = np.abs(pred_isp - actual_isp) > 3.0 * _sigma_i
        else:
            i_res = np.abs(pred_isp - actual_isp)
            i_mask = i_res > threshold * 50
        result['isp_residuals'] = np.abs(pred_isp - actual_isp)
        result['isp_is_anomaly'] = i_mask
        result['isp_anomaly_ratio'] = float(i_mask.mean())

    return result


def generate_health_report(thrust_pred, mfr_pred, isp_pred,
                           thrust_actual, mfr_actual, isp_actual,
                           thrust_std=None, mfr_std=None, isp_std=None,
                           model_confidence=None):
    """V4 双轨评分健康报告：统计判别 + 工程合格 + 几何平均."""
    # Isp noise via error propagation; baseline uses fixed p50
    noise_isp = _isp_uncertainty(thrust_pred, mfr_pred, NOISE_THRUST, NOISE_MFR)

    thrust_dim = _dim_health_v4(thrust_pred, thrust_actual, thrust_std,
                                 NOISE_THRUST, BASELINE_THRUST, ENG_TOL_THRUST)
    mfr_dim = _dim_health_v4(mfr_pred, mfr_actual, mfr_std,
                              NOISE_MFR, BASELINE_MFR, ENG_TOL_MFR)

    # Isp: only evaluate where MFR > 10 mg/s (avoids numerical explosion
    # at edge points where Isp = T/(mfr·G0) → ∞ as mfr → 0)
    MFR_ISP_MIN = 10.0  # mg/s — below this Isp is numerically unreliable
    isp_valid = np.asarray(mfr_actual).ravel() > MFR_ISP_MIN
    n_isp_valid = int(isp_valid.sum())

    if n_isp_valid >= 20:
        isp_dim = _dim_health_v4(
            np.asarray(isp_pred).ravel()[isp_valid],
            np.asarray(isp_actual).ravel()[isp_valid],
            np.asarray(isp_std).ravel()[isp_valid] if isp_std is not None else None,
            noise_isp, BASELINE_ISP, ENG_TOL_ISP)
        # Extend anomaly mask to full length (edge points = non-anomalous)
        full_mask = np.zeros(len(isp_pred), dtype=bool)
        full_mask[isp_valid] = isp_dim['is_anomaly']
        isp_dim['is_anomaly'] = full_mask
        isp_dim['n_anomaly'] = int(full_mask.sum())
        isp_dim['total_steps'] = len(isp_pred)
        isp_dim['anomaly_ratio'] = float(full_mask.mean())
    else:
        # Too few valid Isp points — use neutral placeholder
        isp_dim = _dim_health_v4(isp_pred, isp_actual, isp_std,
                                  noise_isp, BASELINE_ISP, ENG_TOL_ISP)

    # --- Cross-dimension consistency ---
    T = thrust_dim['is_anomaly']
    M = mfr_dim['is_anomaly']
    I = isp_dim['is_anomaly']
    any_anom = T | M | I
    all_anom = T & M & I

    if not any_anom.any():
        jaccard = 0.0
        cons_score, cons_level = 100, 'nominal'
        cons_detail = '三维均无异常'
    else:
        jaccard = float(all_anom.sum() / max(1, any_anom.sum()))
        cons_score = int(round(50 + 50 * jaccard))
        if jaccard >= 0.7:
            cons_level = 'consistent'
            cons_detail = f'三维异常高度同步（J={jaccard:.2f}）— 真实故障特征'
        elif jaccard >= 0.3:
            cons_level = 'partial'
            cons_detail = f'三维异常部分重叠（J={jaccard:.2f}）— 混合信号'
        else:
            cons_level = 'scattered'
            cons_detail = f'三维异常分散（J={jaccard:.2f}）— 倾向传感器噪声'

    # --- Composite (几何平均 × 置信度) ---
    s_g = (max(thrust_dim['dim_score'], 1)
           * max(mfr_dim['dim_score'], 1)
           * max(isp_dim['dim_score'], 1)) ** (1/3)
    conf_score = model_confidence if model_confidence is not None else 90
    conf_level = _health_level(conf_score)
    conf_factor = 0.85 + 0.15 * (conf_score / 100.0)
    overall = max(0, min(100, int(round(s_g * conf_factor))))
    level = _health_level(overall)

    # --- Summary ---
    lvl_map = {'good': '运行正常', 'warning': '需关注', 'critical': '建议检修'}
    if overall >= 80:
        summary = '运行正常'
    elif overall >= 60:
        summary = '需关注 —— 指标轻度偏离基准'
    else:
        summary = '建议检修 —— 多项指标显著偏离基准'

    # --- Drift summary ---
    drift_parts = []
    for d, cn in [('thrust', '推力'), ('mfr', '流量'), ('isp', '比冲')]:
        dd = {'thrust': thrust_dim, 'mfr': mfr_dim, 'isp': isp_dim}[d]
        if dd['drift_level'] != 'stable':
            drift_parts.append(f'{cn}: {dd["drift_level"]} (rate={dd["drift_rate"]:.2f})')
    drift_summary = '; '.join(drift_parts) if drift_parts else '无显著漂移'

    def _desc(v):
        if abs(v) < 1: return f'{v:.3f}'.rstrip('0').rstrip('.')
        elif abs(v) < 100: return f'{v:.1f}'.rstrip('0').rstrip('.')
        else: return f'{v:.0f}'

    return {
        'overall_health': overall,
        'health_level': level,
        'summary': f'综合健康评分 {overall}/100 —— {summary}',
        'thrust': thrust_dim,
        'mfr': mfr_dim,
        'isp': isp_dim,
        'consistency': {
            'score': cons_score,
            'level': cons_level,
            'detail': cons_detail,
        },
        'drift_summary': drift_summary,
        'model_confidence': {
            'score': int(conf_score),
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
