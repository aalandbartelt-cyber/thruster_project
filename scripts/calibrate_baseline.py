#!/usr/bin/env python3
"""calibrate_baseline.py — 根据 v2 测试集预测标定每个维度的 p25 baseline。
输出 BASELINE_THRUST, BASELINE_MFR, BASELINE_ISP 的 p25 值。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from data_pipeline_v2 import THRUST_SCALE, MFR_MAX

# Inlined to avoid importing inference_utils (which depends on torch)
G0 = 9.80665
EPS = 1e-8
MG_PER_S_TO_KG_PER_S = 1e-6

def compute_isp(thrust, mfr):
    raw = thrust / (mfr * MG_PER_S_TO_KG_PER_S * G0 + EPS)
    return np.clip(raw, 1.0, 5000.0)

preds   = np.load("outputs/predictions/v2/predictions_dual.npy")
targets = np.load("outputs/predictions/v2/targets_dual.npy")

# Flatten all samples × timesteps into point-level residuals
res_thrust = (preds[..., 0] - targets[..., 0]).ravel() * THRUST_SCALE
res_mfr    = (preds[..., 1] - targets[..., 1]).ravel() * MFR_MAX

pred_thrust   = preds[..., 0].ravel() * THRUST_SCALE
pred_mfr      = preds[..., 1].ravel() * MFR_MAX
target_thrust = targets[..., 0].ravel() * THRUST_SCALE
target_mfr    = targets[..., 1].ravel() * MFR_MAX

isp_pred   = compute_isp(pred_thrust, pred_mfr)
isp_target = compute_isp(target_thrust, target_mfr)
res_isp    = isp_pred - isp_target

print("=" * 65)
print("  V4 Baseline Calibration (p25 instead of RMSE)")
print("=" * 65)
print(f"  Total data points: {len(res_thrust):,}")
print()

for name, res, unit in [('thrust', res_thrust, 'N'),
                          ('mfr',    res_mfr,    'mg/s'),
                          ('isp',    res_isp,    's')]:
    abs_res = np.abs(res)
    # Handle potential NaN in Isp (from near-zero MFR)
    abs_res = abs_res[~np.isnan(abs_res) & ~np.isinf(abs_res)]
    p10 = np.percentile(abs_res, 10)
    p25 = np.percentile(abs_res, 25)
    p50 = np.percentile(abs_res, 50)
    p75 = np.percentile(abs_res, 75)
    p90 = np.percentile(abs_res, 90)
    rms  = np.sqrt(np.mean(res[~np.isnan(res) & ~np.isinf(res)]**2))
    print(f"  {name:8s} ({unit:5s}): "
          f"p10={p10:.4f}  p25={p25:.4f}  p50={p50:.4f}  "
          f"p75={p75:.4f}  p90={p90:.4f}  RMSE={rms:.4f}")
    print(f"           → BASELINE_{name.upper()} = {p25:.4f}  (p25)")

print()
print("  Copy these values into inference_utils.py:")
print(f"  BASELINE_THRUST = {np.percentile(np.abs(res_thrust), 25):.4f}")
print(f"  BASELINE_MFR    = {np.percentile(np.abs(res_mfr[~np.isnan(res_mfr) & ~np.isinf(res_mfr)]), 25):.4f}")
_abs_isp = np.abs(res_isp)
_abs_isp = _abs_isp[~np.isnan(_abs_isp) & ~np.isinf(_abs_isp)]
print(f"  BASELINE_ISP    = {np.percentile(_abs_isp, 25):.4f}")
print("=" * 65)
