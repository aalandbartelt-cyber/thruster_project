"""
baseline_ml.py — XGBoost & LightGBM baseline for thruster digital twin
Compares tree-based models against v2.0 Attention-LSTM.

Key design:
  - Unified test windows: same exact windows for tree models AND LSTM
  - Engineered features (~47 dims) for tree models
  - Pointwise (17 dim) for direct LSTM comparison
"""

import os, sys, time
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_pipeline_v2 import (
    _build_features, THRUST_SCALE, MFR_MAX, INPUT_DIM, TEST_MODE_LIST,
)
import torch
from model_train_v2 import DualOutputLSTM

SEQ_LEN   = 200
STRIDE    = 200
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_TRAIN = 100000
MAX_VAL   = 30000
MAX_TEST  = 150000

# ============================================================
# Engineered features
# ============================================================
def extract_window_features(x_window):
    """~47 statistical features from [200,17] window."""
    feats = []
    for col in range(6):
        d = x_window[:, col]
        feats.extend([np.mean(d), np.std(d), np.min(d), np.max(d), d[-1]])
        feats.append((d[-1] - d[0]) / max(len(d), 1))
    for col in range(6, 17):
        feats.append(np.mean(x_window[:, col]))
    return np.array(feats, dtype=np.float32)

# ============================================================
# Unified test set builder
# ============================================================
def build_unified_test_set(metadata_path, test_dir, max_windows=MAX_TEST):
    """
    Build ONE set of test windows. Returns both flat [N,3400] and
    engineered [N,47] features from the same windows, plus targets and modes.
    """
    df_meta = pd.read_csv(metadata_path)
    df_meta['filename'] = df_meta['filename'].str.strip()
    df_meta = df_meta[(df_meta['anomalous'] == False) & (df_meta['sn'] >= 13)]
    existing = set(os.listdir(test_dir))
    df_meta = df_meta[df_meta['filename'].isin(existing)]

    flat_list, yt_list, ym_list, modes_list = [], [], [], []
    loaded = 0
    for _, row in df_meta.iterrows():
        fpath = os.path.join(test_dir, row['filename'])
        try:
            df = pd.read_csv(fpath)
        except:
            continue
        T = len(df)
        if T < SEQ_LEN:
            continue
        for start in range(0, T - SEQ_LEN + 1, STRIDE):
            x, y, _ = _build_features(df.iloc[start:start + SEQ_LEN], row)
            flat_list.append(x.flatten().astype(np.float32))
            yt_list.append(np.mean(y[:, 0]))
            ym_list.append(np.mean(y[:, 1]))
            modes_list.append(row['test_mode'])
        loaded += 1
        if max_windows and len(flat_list) >= max_windows:
            break

    # Sort deterministically (by file load order, which is already deterministic)
    N = len(flat_list)
    X_flat = np.array(flat_list, dtype=np.float32)
    yt = np.array(yt_list, dtype=np.float32)
    ym = np.array(ym_list, dtype=np.float32)
    modes = np.array(modes_list)

    # Derive engineered features
    X_eng = np.zeros((N, 47), dtype=np.float32)
    for i in range(N):
        X_eng[i] = extract_window_features(X_flat[i].reshape(SEQ_LEN, INPUT_DIM))

    print(f"  Unified test: {N} windows from {loaded} files")
    print(f"    Flat shape: {X_flat.shape}, Eng shape: {X_eng.shape}")
    return X_flat, X_eng, yt, ym, modes


# ============================================================
# Engineered dataset builder (train/val)
# ============================================================
def build_engineered_dataset(metadata_path, data_dir, sn_range,
                             file_filter=None, max_samples=None, seed=42):
    df_meta = pd.read_csv(metadata_path)
    df_meta['filename'] = df_meta['filename'].str.strip()
    df_meta = df_meta[df_meta['anomalous'] == False]
    if sn_range is not None:
        df_meta = df_meta[(df_meta['sn'] >= sn_range[0]) &
                          (df_meta['sn'] <= sn_range[1])]
    if file_filter is not None:
        df_meta = df_meta[df_meta['filename'].isin(file_filter)]
    existing = set(os.listdir(data_dir))
    df_meta = df_meta[df_meta['filename'].isin(existing)]

    X_list, yt_list, ym_list = [], [], []
    loaded = 0
    for _, row in df_meta.iterrows():
        fpath = os.path.join(data_dir, row['filename'])
        try:
            df = pd.read_csv(fpath)
        except:
            continue
        T = len(df)
        if T < SEQ_LEN:
            continue
        for start in range(0, T - SEQ_LEN + 1, STRIDE):
            x, y, _ = _build_features(df.iloc[start:start + SEQ_LEN], row)
            X_list.append(extract_window_features(x))
            yt_list.append(np.mean(y[:, 0]))
            ym_list.append(np.mean(y[:, 1]))
        loaded += 1
        if max_samples and len(X_list) >= max_samples:
            break

    X = np.array(X_list, dtype=np.float32)
    yt = np.array(yt_list, dtype=np.float32)
    ym = np.array(ym_list, dtype=np.float32)
    if max_samples and len(X) > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, yt, ym = X[idx], yt[idx], ym[idx]
    print(f"  Loaded {loaded} files -> {len(X)} windows ({X.shape[1]} feats)")
    return X, yt, ym


# ============================================================
# Pointwise dataset builder
# ============================================================
def build_pointwise_dataset(metadata_path, data_dir, sn_range,
                            file_filter=None, max_samples=60000):
    df_meta = pd.read_csv(metadata_path)
    df_meta['filename'] = df_meta['filename'].str.strip()
    df_meta = df_meta[df_meta['anomalous'] == False]
    if sn_range is not None:
        df_meta = df_meta[(df_meta['sn'] >= sn_range[0]) &
                          (df_meta['sn'] <= sn_range[1])]
    if file_filter is not None:
        df_meta = df_meta[df_meta['filename'].isin(file_filter)]
    existing = set(os.listdir(data_dir))
    df_meta = df_meta[df_meta['filename'].isin(existing)]

    X_list, yt_list, ym_list, modes_list = [], [], [], []
    loaded = 0
    for _, row in df_meta.sample(frac=1, random_state=42).iterrows():
        fpath = os.path.join(data_dir, row['filename'])
        try:
            df = pd.read_csv(fpath)
        except:
            continue
        T = len(df)
        if T < 50:
            continue
        n_sample = min(T, 50)
        indices = np.linspace(0, T - 1, n_sample, dtype=int)
        for idx in indices:
            x, y, _ = _build_features(df.iloc[idx:idx + 1], row)
            X_list.append(x[0])
            yt_list.append(y[0, 0])
            ym_list.append(y[0, 1])
            modes_list.append(row['test_mode'])
        loaded += 1
        if len(X_list) >= max_samples:
            break
    X = np.array(X_list[:max_samples], dtype=np.float32)
    yt = np.array(yt_list[:max_samples], dtype=np.float32)
    ym = np.array(ym_list[:max_samples], dtype=np.float32)
    modes = np.array(modes_list[:max_samples])
    print(f"  Loaded {loaded} files -> {len(X)} pointwise samples")
    return X, yt, ym, modes


# ============================================================
# Helpers
# ============================================================
def train_xgb(X_train, y_train, X_val, y_val, n_est=500, lr=0.05, md=8):
    model = xgb.XGBRegressor(
        n_estimators=n_est, max_depth=md, learning_rate=lr,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, early_stopping_rounds=30,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model

def train_lgb(X_train, y_train, X_val, y_val, n_est=500, lr=0.05, md=10):
    model = lgb.LGBMRegressor(
        n_estimators=n_est, max_depth=md, learning_rate=lr,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    return model

def compute_rmse_mae(pred_norm, actual_norm, scale):
    p = pred_norm * scale
    a = actual_norm * scale
    return np.sqrt(np.mean((p - a) ** 2)), np.mean(np.abs(p - a))

def per_mode_rmse(pred_norm, actual_norm, modes, scale):
    result = {}
    for mode in TEST_MODE_LIST:
        mask = modes == mode
        if mask.sum() > 0:
            rmse, _ = compute_rmse_mae(pred_norm[mask], actual_norm[mask], scale)
            result[mode] = rmse
        else:
            result[mode] = None
    return result

def eval_lstm_window_mean(X_flat, model_path):
    model = DualOutputLSTM().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    N = X_flat.shape[0]
    x_reshaped = X_flat.reshape(N, SEQ_LEN, INPUT_DIM)
    preds_t, preds_m = [], []
    with torch.no_grad():
        for i in range(0, N, 64):
            batch = torch.from_numpy(x_reshaped[i:i + 64]).float().to(DEVICE)
            pred = model(batch).cpu().numpy()
            preds_t.append(np.mean(pred[:, :, 0], axis=1))
            preds_m.append(np.mean(pred[:, :, 1], axis=1))
    return np.concatenate(preds_t), np.concatenate(preds_m)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  Thruster Digital Twin — XGBoost & LightGBM Baseline")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    md = "data/metadata.csv"
    tr = "data/dataset/dataset/train/"
    te = "data/dataset/dataset/test/"
    mp = "outputs/models/v2/dual_output_lstm_v2.pth"

    # --- Train/Val split (same as model_train_v2.py) ---
    df_all = pd.read_csv(md)
    df_all['filename'] = df_all['filename'].str.strip()
    df_all = df_all[(df_all['anomalous'] == False) & (df_all['sn'] <= 12)]
    all_files = df_all['filename'].tolist()
    np.random.seed(42)
    np.random.shuffle(all_files)
    split = int(len(all_files) * 0.8)
    train_files = set(all_files[:split])
    val_files   = set(all_files[split:])
    print(f"\nTrain files: {len(train_files)}, Val files: {len(val_files)}")

    R = {}  # results dict

    # ================================================================
    # 1. Build UNIFIED test set first (same windows for ALL methods)
    # ================================================================
    print("\n" + "=" * 70)
    print("  STEP 1: Building unified test set (same windows for all methods)")
    print("=" * 70)
    X_test_flat, X_test_eng, yt_test, ym_test, modes_test = build_unified_test_set(md, te, MAX_TEST)

    # ================================================================
    # 2. Train engineered models
    # ================================================================
    print("\n" + "=" * 70)
    print("  STEP 2: Engineered features training (~47 dims)")
    print("=" * 70)
    print("\n  Loading train data...")
    X_tr_e, yt_tr_e, ym_tr_e = build_engineered_dataset(
        md, tr, sn_range=(1, 12), file_filter=train_files, max_samples=MAX_TRAIN)
    X_val_e, yt_val_e, ym_val_e = build_engineered_dataset(
        md, tr, sn_range=(1, 12), file_filter=val_files, max_samples=MAX_VAL)
    print(f"  Train: {X_tr_e.shape}, Val: {X_val_e.shape}, Test: {X_test_eng.shape}")

    # Thrust
    print("\n  [Eng] Thrust...")
    t0 = time.time()
    xgb_t = train_xgb(X_tr_e, yt_tr_e, X_val_e, yt_val_e, md=8)
    dt = time.time() - t0
    pred = xgb_t.predict(X_test_eng)
    xgb_et_rmse, xgb_et_mae = compute_rmse_mae(pred, yt_test, THRUST_SCALE)
    xgb_et_modes = per_mode_rmse(pred, yt_test, modes_test, THRUST_SCALE)
    print(f"    XGBoost  thrust RMSE: {xgb_et_rmse:.4f} N  ({dt:.0f}s)")

    t0 = time.time()
    lgb_t = train_lgb(X_tr_e, yt_tr_e, X_val_e, yt_val_e, md=10)
    dt = time.time() - t0
    pred = lgb_t.predict(X_test_eng)
    lgb_et_rmse, lgb_et_mae = compute_rmse_mae(pred, yt_test, THRUST_SCALE)
    lgb_et_modes = per_mode_rmse(pred, yt_test, modes_test, THRUST_SCALE)
    print(f"    LightGBM thrust RMSE: {lgb_et_rmse:.4f} N  ({dt:.0f}s)")

    # MFR
    print("\n  [Eng] MFR...")
    t0 = time.time()
    xgb_m = train_xgb(X_tr_e, ym_tr_e, X_val_e, ym_val_e, md=8)
    dt = time.time() - t0
    pred = xgb_m.predict(X_test_eng)
    xgb_em_rmse, xgb_em_mae = compute_rmse_mae(pred, ym_test, MFR_MAX)
    xgb_em_modes = per_mode_rmse(pred, ym_test, modes_test, MFR_MAX)
    print(f"    XGBoost  mfr    RMSE: {xgb_em_rmse:.1f} mg/s  ({dt:.0f}s)")

    t0 = time.time()
    lgb_m = train_lgb(X_tr_e, ym_tr_e, X_val_e, ym_val_e, md=10)
    dt = time.time() - t0
    pred = lgb_m.predict(X_test_eng)
    lgb_em_rmse, lgb_em_mae = compute_rmse_mae(pred, ym_test, MFR_MAX)
    lgb_em_modes = per_mode_rmse(pred, ym_test, modes_test, MFR_MAX)
    print(f"    LightGBM mfr    RMSE: {lgb_em_rmse:.1f} mg/s  ({dt:.0f}s)")

    R['XGB_eng'] = (xgb_et_rmse, xgb_em_rmse, xgb_et_modes, xgb_em_modes)
    R['LGB_eng'] = (lgb_et_rmse, lgb_em_rmse, lgb_et_modes, lgb_em_modes)

    # ================================================================
    # 3. LSTM window-mean on same test windows
    # ================================================================
    print("\n" + "=" * 70)
    print("  STEP 3: LSTM window-mean on SAME test windows")
    print("=" * 70)
    lstm_t, lstm_m = eval_lstm_window_mean(X_test_flat, mp)
    lstm_t_rmse, lstm_t_mae = compute_rmse_mae(lstm_t, yt_test, THRUST_SCALE)
    lstm_m_rmse, lstm_m_mae = compute_rmse_mae(lstm_m, ym_test, MFR_MAX)
    lstm_t_modes = per_mode_rmse(lstm_t, yt_test, modes_test, THRUST_SCALE)
    lstm_m_modes = per_mode_rmse(lstm_m, ym_test, modes_test, MFR_MAX)
    print(f"  LSTM thrust (window-mean): {lstm_t_rmse:.4f} N")
    print(f"  LSTM mfr    (window-mean): {lstm_m_rmse:.1f} mg/s")
    R['LSTM_wm'] = (lstm_t_rmse, lstm_m_rmse, lstm_t_modes, lstm_m_modes)

    # ================================================================
    # 4. Pointwise training & evaluation
    # ================================================================
    print("\n" + "=" * 70)
    print("  STEP 4: Pointwise (17-dim, directly comparable to LSTM pointwise)")
    print("=" * 70)
    print("\n  Loading pointwise data...")
    X_tr_p, yt_tr_p, ym_tr_p, _ = build_pointwise_dataset(
        md, tr, sn_range=(1, 12), file_filter=train_files, max_samples=60000)
    X_val_p, yt_val_p, ym_val_p, _ = build_pointwise_dataset(
        md, tr, sn_range=(1, 12), file_filter=val_files, max_samples=15000)
    X_test_p, yt_test_p, ym_test_p, modes_test_p = build_pointwise_dataset(
        md, te, sn_range=(13, 24), max_samples=100000)
    print(f"  Train: {X_tr_p.shape}, Val: {X_val_p.shape}, Test: {X_test_p.shape}")

    # Thrust
    print("\n  [Pointwise] Thrust...")
    t0 = time.time()
    xgb_pt = train_xgb(X_tr_p, yt_tr_p, X_val_p, yt_val_p, n_est=300, lr=0.1, md=6)
    dt = time.time() - t0
    pred = xgb_pt.predict(X_test_p)
    xgb_pt_rmse, xgb_pt_mae = compute_rmse_mae(pred, yt_test_p, THRUST_SCALE)
    xgb_pt_modes = per_mode_rmse(pred, yt_test_p, modes_test_p, THRUST_SCALE)
    print(f"    XGBoost  thrust RMSE: {xgb_pt_rmse:.4f} N  ({dt:.0f}s)")

    t0 = time.time()
    lgb_pt = train_lgb(X_tr_p, yt_tr_p, X_val_p, yt_val_p, n_est=300, lr=0.1, md=8)
    dt = time.time() - t0
    pred = lgb_pt.predict(X_test_p)
    lgb_pt_rmse, lgb_pt_mae = compute_rmse_mae(pred, yt_test_p, THRUST_SCALE)
    lgb_pt_modes = per_mode_rmse(pred, yt_test_p, modes_test_p, THRUST_SCALE)
    print(f"    LightGBM thrust RMSE: {lgb_pt_rmse:.4f} N  ({dt:.0f}s)")

    # MFR
    print("\n  [Pointwise] MFR...")
    t0 = time.time()
    xgb_pm = train_xgb(X_tr_p, ym_tr_p, X_val_p, ym_val_p, n_est=300, lr=0.1, md=6)
    dt = time.time() - t0
    pred = xgb_pm.predict(X_test_p)
    xgb_pm_rmse, xgb_pm_mae = compute_rmse_mae(pred, ym_test_p, MFR_MAX)
    xgb_pm_modes = per_mode_rmse(pred, ym_test_p, modes_test_p, MFR_MAX)
    print(f"    XGBoost  mfr    RMSE: {xgb_pm_rmse:.1f} mg/s  ({dt:.0f}s)")

    t0 = time.time()
    lgb_pm = train_lgb(X_tr_p, ym_tr_p, X_val_p, ym_val_p, n_est=300, lr=0.1, md=8)
    dt = time.time() - t0
    pred = lgb_pm.predict(X_test_p)
    lgb_pm_rmse, lgb_pm_mae = compute_rmse_mae(pred, ym_test_p, MFR_MAX)
    lgb_pm_modes = per_mode_rmse(pred, ym_test_p, modes_test_p, MFR_MAX)
    print(f"    LightGBM mfr    RMSE: {lgb_pm_rmse:.1f} mg/s  ({dt:.0f}s)")

    R['XGB_pt'] = (xgb_pt_rmse, xgb_pm_rmse, xgb_pt_modes, xgb_pm_modes)
    R['LGB_pt'] = (lgb_pt_rmse, lgb_pm_rmse, lgb_pt_modes, lgb_pm_modes)

    # Known LSTM pointwise
    lstm_pt_t = 0.0832
    lstm_pt_m = 49.1

    # ================================================================
    # SUMMARY TABLES
    # ================================================================
    print("\n" + "=" * 90)
    print("  GLOBAL COMPARISON: XGBoost vs LightGBM vs v2.0 Attention-LSTM")
    print("=" * 90)
    print(f"  {'Method':<35s} {'Thrust RMSE':>14s} {'MFR RMSE':>14s} {'Metric':>20s}")
    print(f"  {'-'*35} {'-'*14} {'-'*14} {'-'*20}")

    def _t(k): return R[k][0]
    def _m(k): return R[k][1]

    print(f"  {'XGBoost (engineered ~47d)':<35s} {_t('XGB_eng'):>13.4f} N {_m('XGB_eng'):>13.1f} mg/s {'window-mean':>20s}")
    print(f"  {'LightGBM (engineered ~47d)':<35s} {_t('LGB_eng'):>13.4f} N {_m('LGB_eng'):>13.1f} mg/s {'window-mean':>20s}")
    print(f"  {'LSTM (window-mean, same windows)':<35s} {_t('LSTM_wm'):>13.4f} N {_m('LSTM_wm'):>13.1f} mg/s {'window-mean':>20s}")
    print(f"  {'---':<35s} {'---':>14} {'---':>14} {'---':>20}")
    print(f"  {'XGBoost (pointwise 17d)':<35s} {_t('XGB_pt'):>13.4f} N {_m('XGB_pt'):>13.1f} mg/s {'pointwise':>20s}")
    print(f"  {'LightGBM (pointwise 17d)':<35s} {_t('LGB_pt'):>13.4f} N {_m('LGB_pt'):>13.1f} mg/s {'pointwise':>20s}")
    print(f"  {'v2.0 Attention-LSTM (pointwise)':<35s} {lstm_pt_t:>13.4f} N {lstm_pt_m:>13.1f} mg/s {'pointwise':>20s}")
    print("=" * 90)

    # Per-mode window-mean thrust
    print(f"\n{'='*85}")
    print(f"  Per-Mode Thrust RMSE (window-mean, N) — same test windows")
    print(f"  {'Test Mode':15s} {'XGBoost':>10s} {'LightGBM':>10s} {'LSTM':>10s} {'Best':>10s}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for mode in TEST_MODE_LIST:
        x = R['XGB_eng'][2].get(mode)
        l = R['LGB_eng'][2].get(mode)
        s = R['LSTM_wm'][2].get(mode)
        if x is None:
            continue
        vs = {'XGB': x, 'LGB': l, 'LSTM': s}
        best = min(vs, key=lambda k: float(vs[k]))
        print(f"  {mode:15s} {x:>10.4f} {l:>10.4f} {s:>10.4f} {best:>10s}")

    print(f"\n  Per-Mode MFR RMSE (window-mean, mg/s) — same test windows")
    print(f"  {'Test Mode':15s} {'XGBoost':>10s} {'LightGBM':>10s} {'LSTM':>10s} {'Best':>10s}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for mode in TEST_MODE_LIST:
        x = R['XGB_eng'][3].get(mode)
        l = R['LGB_eng'][3].get(mode)
        s = R['LSTM_wm'][3].get(mode)
        if x is None:
            continue
        vs = {'XGB': x, 'LGB': l, 'LSTM': s}
        best = min(vs, key=lambda k: float(vs[k]))
        print(f"  {mode:15s} {x:>10.1f} {l:>10.1f} {s:>10.1f} {best:>10s}")
    print(f"{'='*85}")

    # Per-mode pointwise thrust
    print(f"\n{'='*60}")
    print(f"  Per-Mode Thrust RMSE (pointwise, N)")
    print(f"  {'Test Mode':15s} {'XGBoost':>10s} {'LightGBM':>10s}")
    print(f"  {'-'*15} {'-'*10} {'-'*10}")
    for mode in TEST_MODE_LIST:
        x = R['XGB_pt'][2].get(mode)
        l = R['LGB_pt'][2].get(mode)
        if x is None:
            continue
        print(f"  {mode:15s} {x:>10.4f} {l:>10.4f}")
    print(f"{'='*60}")

    # ================================================================
    # CHART
    # ================================================================
    print("\n[Chart] Saving comparison chart...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    methods = ['XGBoost\n(eng-47d)', 'LightGBM\n(eng-47d)', 'LSTM\n(window-mean)',
               'XGBoost\n(pointwise)', 'LightGBM\n(pointwise)', 'LSTM v2.0\n(pointwise)']
    t_vals = [_t('XGB_eng'), _t('LGB_eng'), _t('LSTM_wm'),
              _t('XGB_pt'), _t('LGB_pt'), lstm_pt_t]
    m_vals = [_m('XGB_eng'), _m('LGB_eng'), _m('LSTM_wm'),
              _m('XGB_pt'), _m('LGB_pt'), lstm_pt_m]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#1565C0', '#2E7D32', '#E65100']

    for ax, vals, ylabel, title in [
        (axes[0], t_vals, 'RMSE (N)', 'Thrust Prediction'),
        (axes[1], m_vals, 'RMSE (mg/s)', 'MFR Prediction'),
    ]:
        bars = ax.bar(methods, vals, color=colors, edgecolor='black', linewidth=0.5)
        mv = max(vals) + 1e-9
        for bar, val in zip(bars, vals):
            fmt = f'{val:.4f}' if 'N' in ylabel else f'{val:.1f}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + mv*0.01,
                    fmt, ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(axis='y', linestyle=':', alpha=0.4)
        ax.set_ylim(0, mv * 1.2)
        ax.tick_params(axis='x', labelsize=8)

    plt.suptitle('XGBoost & LightGBM Baseline vs v2.0 Attention-LSTM\nThruster Digital Twin',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_dir = "outputs/figures/v2"
    os.makedirs(out_dir, exist_ok=True)
    chart_path = os.path.join(out_dir, "baseline_ml_comparison.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {chart_path}")

    # ================================================================
    # ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("  ANALYSIS")
    print("=" * 70)

    btw = min(_t('XGB_eng'), _t('LGB_eng'))
    bmw = min(_m('XGB_eng'), _m('LGB_eng'))
    btp = min(_t('XGB_pt'), _t('LGB_pt'))
    bmp = min(_m('XGB_pt'), _m('LGB_pt'))

    print(f"  1. Window-mean comparison (apples-to-apples, same test windows):")
    print(f"     Best tree thrust: {btw:.4f} N  |  LSTM: {_t('LSTM_wm'):.4f} N  |  Delta: {btw - _t('LSTM_wm'):+.4f} N")
    print(f"     Best tree mfr:    {bmw:.1f} mg/s  |  LSTM: {_m('LSTM_wm'):.1f} mg/s  |  Delta: {bmw - _m('LSTM_wm'):+.1f} mg/s")

    print(f"\n  2. Pointwise comparison (directly comparable to LSTM):")
    print(f"     Best tree thrust: {btp:.4f} N  |  LSTM v2.0: {lstm_pt_t:.4f} N  |  Delta: {btp - lstm_pt_t:+.4f} N")
    print(f"     Best tree mfr:    {bmp:.1f} mg/s  |  LSTM v2.0: {lstm_pt_m:.1f} mg/s  |  Delta: {bmp - lstm_pt_m:+.1f} mg/s")

    print(f"\n  3. Per-mode analysis (window-mean thrust, engineered):")
    for label, modes in [('STEADY-STATE', ['ssf', 'health_check']),
                          ('TRANSIENT', ['offmod', 'random_short', 'random_long', 'random_mixed'])]:
        print(f"     --- {label} ---")
        for mode in modes:
            x = R['XGB_eng'][2].get(mode)
            s = R['LSTM_wm'][2].get(mode)
            if x is not None and s is not None:
                gap = x - s
                tag = '*** TREE MUCH WORSE' if gap > 0.02 else 'TREE WORSE' if gap > 0.005 else '~SAME' if abs(gap) < 0.005 else 'TREE BETTER'
                print(f"     {mode:15s}  Tree={x:.4f}  LSTM={s:.4f}  gap={gap:+.4f} N  {tag}")

    print(f"\n  4. CONCLUSION:")
    print(f"     Thrust: tree models (no temporal context) vs LSTM (200-step memory)")
    if btp <= lstm_pt_t:
        print(f"       -> Tree models MATCH or BEAT LSTM on thrust ({btp:.4f} <= {lstm_pt_t:.4f} N)")
        print(f"       -> Temporal context from LSTM provides NO advantage for thrust prediction")
    else:
        print(f"       -> LSTM outperforms by {lstm_pt_t - btp:.4f} N on thrust (pointwise)")
    if bmp <= lstm_pt_m:
        print(f"     MFR: Tree models MATCH or BEAT LSTM on MFR ({bmp:.1f} <= {lstm_pt_m:.1f} mg/s)")
    else:
        print(f"       -> LSTM outperforms by {lstm_pt_m - bmp:.1f} mg/s on MFR (pointwise)")
        print(f"       -> MFR prediction BENEFITS from temporal context substantially")
    print("=" * 70)
    print("\nDone.")


if __name__ == "__main__":
    main()
