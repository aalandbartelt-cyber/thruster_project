"""
model_test_v2.py — 双输出 Attention-LSTM 测试评估
M 负责，dev-m 分支

评估内容：
  1. 全局 RMSE/MAE（thrust + mfr）
  2. 按 11 种 test_mode 分项 RMSE 表
  3. v1 vs v2 thrust RMSE 对比
  4. Isp = thrust / (mfr × g0) 曲线
  5. 生成接口文件供 Z 使用（predictions_dual.npy, targets_dual.npy, anomaly_labels.npy）
"""

import os, json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_pipeline_v2 import (
    load_sequence, _build_features,
    THRUST_SCALE, MFR_MAX, INPUT_DIM, TEST_MODE_LIST,
    save_meta_info,
)
from model_train_v2 import DualOutputLSTM

G0 = 9.80665  # m/s^2


# ============================================================
# 评估工具
# ============================================================
def compute_metrics(preds, targets, scale):
    """preds/targets 在归一化空间，scale 用于反归一化。返回物理单位的 RMSE/MAE。"""
    p = preds * scale
    t = targets * scale
    rmse = np.sqrt(np.mean((p - t) ** 2))
    mae = np.mean(np.abs(p - t))
    return rmse, mae


# ============================================================
# 主流程
# ============================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 加载 v2 模型 ──
    model_path = "outputs/models/v2/dual_output_lstm_v2.pth"
    model = DualOutputLSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded: {model_path}")

    # ── 路径 ──
    metadata_path = "data/metadata.csv"
    test_dir = "data/dataset/dataset/test/"

    df_meta = pd.read_csv(metadata_path)
    df_meta['filename'] = df_meta['filename'].str.strip()

    # 正常文件（用于 RMSE 评估）
    df_normal = df_meta[(df_meta['anomalous'] == False)].copy()

    # 只保留 test_dir 中实际存在的文件
    existing_files = set(os.listdir(test_dir))
    df_normal = df_normal[df_normal['filename'].isin(existing_files)]
    print(f"\nNormal test files found: {len(df_normal)}")

    # ── 逐文件滑动窗口评估 ──
    seq_len = 200
    stride = 200  # 非重叠

    all_thrust_rmse = []
    all_thrust_mae = []
    all_mfr_rmse = []
    all_mfr_mae = []
    mode_metrics = {m: {'thrust_rmse': [], 'thrust_mae': [], 'mfr_rmse': [], 'mfr_mae': [], 'count': 0}
                    for m in TEST_MODE_LIST}

    for _, row in df_normal.iterrows():
        fpath = os.path.join(test_dir, row['filename'])
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        T = len(df)
        mode = row['test_mode']
        if T < seq_len:
            continue

        file_thrust_rmse = []
        file_mfr_rmse = []
        file_thrust_mae = []
        file_mfr_mae = []

        for start in range(0, T - seq_len + 1, stride):
            x, y, _ = _build_features(df.iloc[start:start + seq_len], row)
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(x_t).squeeze(0).cpu().numpy()  # [200, 2]

            t_rmse, t_mae = compute_metrics(pred[:, 0], y[:, 0], THRUST_SCALE)
            m_rmse, m_mae = compute_metrics(pred[:, 1], y[:, 1], MFR_MAX)

            file_thrust_rmse.append(t_rmse)
            file_thrust_mae.append(t_mae)
            file_mfr_rmse.append(m_rmse)
            file_mfr_mae.append(m_mae)

        if file_thrust_rmse:
            mean_t_rmse = np.mean(file_thrust_rmse)
            mean_t_mae = np.mean(file_thrust_mae)
            mean_m_rmse = np.mean(file_mfr_rmse)
            mean_m_mae = np.mean(file_mfr_mae)

            all_thrust_rmse.append(mean_t_rmse)
            all_thrust_mae.append(mean_t_mae)
            all_mfr_rmse.append(mean_m_rmse)
            all_mfr_mae.append(mean_m_mae)

            if mode in mode_metrics:
                mode_metrics[mode]['thrust_rmse'].append(mean_t_rmse)
                mode_metrics[mode]['thrust_mae'].append(mean_t_mae)
                mode_metrics[mode]['mfr_rmse'].append(mean_m_rmse)
                mode_metrics[mode]['mfr_mae'].append(mean_m_mae)
                mode_metrics[mode]['count'] += 1

    # ── 全局指标 ──
    print(f"\n{'='*60}")
    print(f"  v2 DualOutput Attention-LSTM — 测试集评估")
    print(f"  评估文件数: {len(all_thrust_rmse)}")
    print(f"{'='*60}")
    print(f"  Thrust  RMSE: {np.mean(all_thrust_rmse):.4f} N")
    print(f"  Thrust  MAE : {np.mean(all_thrust_mae):.4f} N")
    print(f"  MFR     RMSE: {np.mean(all_mfr_rmse):.1f} mg/s")
    print(f"  MFR     MAE : {np.mean(all_mfr_mae):.1f} mg/s")
    print(f"{'='*60}")

    # ── 按 test_mode 分项 ──
    print(f"\n{'─'*80}")
    print(f"  {'Test Mode':15s}  {'Files':>5s}  {'Thrust RMSE':>12s}  {'Thrust MAE':>12s}  {'MFR RMSE':>12s}  {'MFR MAE':>12s}")
    print(f"  {'─'*15}  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")
    for mode in TEST_MODE_LIST:
        m = mode_metrics[mode]
        if m['count'] > 0:
            t_rmse = np.mean(m['thrust_rmse'])
            t_mae = np.mean(m['thrust_mae'])
            m_rmse = np.mean(m['mfr_rmse'])
            m_mae = np.mean(m['mfr_mae'])
            print(f"  {mode:15s}  {m['count']:5d}  {t_rmse:12.4f}  {t_mae:12.4f}  {m_rmse:12.1f}  {m_mae:12.1f}")
        else:
            print(f"  {mode:15s}  {m['count']:5d}  {'—':>12}  {'—':>12}  {'—':>12}  {'—':>12}")
    print(f"{'─'*80}")

    # ── v1 vs v2 对比 ──
    v1_model_path = "outputs/models/v1/thruster_lstm_v1.pth"
    v1_rmse = None
    if os.path.exists(v1_model_path):
        print(f"\n{'='*60}")
        print(f"  v1 vs v2 对比")
        print(f"{'='*60}")

        class V1LSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(3, 256, 2, batch_first=True)
                self.fc = nn.Linear(256, 1)
            def forward(self, x):
                self.lstm.flatten_parameters()
                out, _ = self.lstm(x)
                return self.fc(out)

        v1_model = V1LSTM().to(device)
        v1_model.load_state_dict(torch.load(v1_model_path, map_location=device))
        v1_model.eval()

        v1_all_rmse = []
        v2_on_same = []
        for _, row in df_normal.iterrows():
            fpath = os.path.join(test_dir, row['filename'])
            try:
                df = pd.read_csv(fpath)
            except Exception:
                continue
            T = len(df)
            if T < seq_len:
                continue

            for start in range(0, T - seq_len + 1, stride):
                # v2
                x_v2, y_v2, _ = _build_features(df.iloc[start:start + seq_len], row)
                x_t = torch.from_numpy(x_v2).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred_v2 = model(x_t).squeeze(0).cpu().numpy()
                v2_rmse, _ = compute_metrics(pred_v2[:, 0], y_v2[:, 0], THRUST_SCALE)
                v2_on_same.append(v2_rmse)

                # v1
                u = df['ton'].values[start:start + seq_len].astype(np.float32)
                p = np.full_like(u, row['test_pressure'] / 25.0, dtype=np.float32)
                age = np.full_like(u, row['cumulated_on_time'] / 8.0, dtype=np.float32)
                x_v1 = np.stack([u, p, age], axis=1)
                x_v1_t = torch.from_numpy(np.ascontiguousarray(x_v1)).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred_v1 = v1_model(x_v1_t).squeeze().cpu().numpy() * 4.0
                y_v1 = df['thrust'].values[start:start + seq_len]
                v1_rmse_val = np.sqrt(np.mean((y_v1 - pred_v1) ** 2))
                v1_all_rmse.append(v1_rmse_val)

        v1_rmse = np.mean(v1_all_rmse) if v1_all_rmse else 0
        v2_rmse_same = np.mean(v2_on_same) if v2_on_same else 0
        print(f"  v1 (3-dim, single-output): thrust RMSE = {v1_rmse:.4f} N")
        print(f"  v2 (17-dim, dual-output):  thrust RMSE = {v2_rmse_same:.4f} N")
        print(f"  v2 额外输出 MFR RMSE = {np.mean(all_mfr_rmse):.1f} mg/s")
        print(f"{'='*60}")
    else:
        print(f"\n  [INFO] v1 model not found at {v1_model_path}, skipping comparison")

    # ── 逐 mode 出图：thrust + mfr + Isp 三联 ──
    print(f"\n{'='*60}")
    print(f"  逐 test_mode 出图")
    print(f"{'='*60}")
    fig_dir = "outputs/figures/v2"
    os.makedirs(fig_dir, exist_ok=True)

    sample_files = df_normal.groupby('test_mode').first().reset_index()
    for _, row in sample_files.iterrows():
        fpath = os.path.join(test_dir, row['filename'])
        df = pd.read_csv(fpath)
        start = len(df) // 3
        x, y, _ = _build_features(df.iloc[start:start + seq_len], row)
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_t).squeeze(0).cpu().numpy()

        thrust_pred = pred[:, 0] * THRUST_SCALE
        mfr_pred = pred[:, 1] * MFR_MAX
        thrust_true = y[:, 0] * THRUST_SCALE
        mfr_true = y[:, 1] * MFR_MAX
        isp_pred = thrust_pred / (mfr_pred * G0)
        isp_true = thrust_true / (mfr_true * G0)

        mode = row['test_mode']
        sn = row['sn']

        fig, axes = plt.subplots(3, 1, figsize=(16, 10))

        # thrust
        ax = axes[0]
        ax.plot(thrust_true, label='Actual', color='blue', alpha=0.7, linewidth=1.2)
        ax.plot(thrust_pred, label='Predicted', color='red', alpha=0.7, linestyle='--', linewidth=1.2)
        t_rmse = np.sqrt(np.mean((thrust_true - thrust_pred)**2))
        ax.text(0.02, 0.95, f"RMSE={t_rmse:.4f} N", transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_ylabel("Thrust (N)")
        ax.set_title(f"Thrust  —  {mode}  ({sn})", fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.5)

        # mfr
        ax = axes[1]
        ax.plot(mfr_true, label='Actual', color='blue', alpha=0.7, linewidth=1.2)
        ax.plot(mfr_pred, label='Predicted', color='red', alpha=0.7, linestyle='--', linewidth=1.2)
        m_rmse = np.sqrt(np.mean((mfr_true - mfr_pred)**2))
        ax.text(0.02, 0.95, f"RMSE={m_rmse:.1f} mg/s", transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_ylabel("MFR (mg/s)")
        ax.set_title(f"MFR  —  {mode}  ({sn})", fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.5)

        # isp
        ax = axes[2]
        ax.plot(isp_true, label='Actual Isp', color='blue', alpha=0.7, linewidth=1.2)
        ax.plot(isp_pred, label='Predicted Isp', color='red', alpha=0.7, linestyle='--', linewidth=1.2)
        i_rmse = np.sqrt(np.mean((isp_true - isp_pred)**2))
        ax.text(0.02, 0.95, f"RMSE={i_rmse:.2f} s", transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Isp (s)")
        ax.set_title(f"Isp = thrust / (mfr × g0)  —  {mode}  ({sn})", fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.5)

        plt.tight_layout()
        fig_path = os.path.join(fig_dir, f"pred_{mode}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"  {fig_path}")

    # 额外：v1 vs v2 thrust 对比图（各 mode 拼一张汇总）
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()
    for i, (_, row) in enumerate(sample_files.iterrows()):
        if i >= 11:
            break
        fpath = os.path.join(test_dir, row['filename'])
        df = pd.read_csv(fpath)
        start = len(df) // 3
        x, y, _ = _build_features(df.iloc[start:start + seq_len], row)
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_t).squeeze(0).cpu().numpy()
        thrust_pred = pred[:, 0] * THRUST_SCALE
        thrust_true = y[:, 0] * THRUST_SCALE
        ax = axes[i]
        ax.plot(thrust_true, label='Actual', color='blue', alpha=0.5, linewidth=0.8)
        ax.plot(thrust_pred, label='v2 Pred', color='red', alpha=0.7, linestyle='--', linewidth=0.8)
        ax.set_title(f"{row['test_mode']}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle=':', alpha=0.4)
    axes[11].set_visible(False)
    plt.suptitle("v2 Thrust Prediction by Test Mode", fontsize=14, fontweight='bold')
    plt.tight_layout()
    summary_path = os.path.join(fig_dir, "thrust_summary_11mode.png")
    plt.savefig(summary_path, dpi=300)
    plt.close()
    print(f"  {summary_path}")

    # ── 生成接口文件供 Z 使用 ──
    print(f"\n{'='*60}")
    print(f"  生成接口文件")
    print(f"{'='*60}")
    output_dir = "outputs/predictions/v2"
    os.makedirs(output_dir, exist_ok=True)

    all_preds = []
    all_targets = []
    all_labels = []

    # 正常文件
    for _, row in df_normal.iterrows():
        fpath = os.path.join(test_dir, row['filename'])
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        T = len(df)
        if T < seq_len:
            continue
        x, y, labels = _build_features(df.iloc[:seq_len], row)
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_t).squeeze(0).cpu().numpy()
        all_preds.append(pred)
        all_targets.append(y)
        all_labels.append(labels)

    # 异常文件
    df_anomalous = df_meta[df_meta['anomalous'] == True].copy()
    df_anomalous = df_anomalous[df_anomalous['filename'].isin(existing_files)]
    for _, row in df_anomalous.iterrows():
        fpath = os.path.join(test_dir, row['filename'])
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        T = len(df)
        if T < seq_len:
            continue
        x, y, labels = _build_features(df.iloc[:seq_len], row)
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_t).squeeze(0).cpu().numpy()
        all_preds.append(pred)
        all_targets.append(y)
        all_labels.append(labels)

    preds_stack = np.stack(all_preds)    # [N, 200, 2]
    targets_stack = np.stack(all_targets)  # [N, 200, 2]
    labels_stack = np.stack(all_labels)   # [N, 200]

    np.save(os.path.join(output_dir, "predictions_dual.npy"), preds_stack)
    np.save(os.path.join(output_dir, "targets_dual.npy"), targets_stack)
    np.save(os.path.join(output_dir, "anomaly_labels.npy"), labels_stack)
    print(f"  predictions_dual.npy  → {preds_stack.shape}  (normalized)")
    print(f"  targets_dual.npy      → {targets_stack.shape}  (normalized)")
    print(f"  anomaly_labels.npy    → {labels_stack.shape}")
    print(f"  Anomalous files included: {len(df_anomalous)}")

    # meta_info.json
    save_meta_info(os.path.join(output_dir, "meta_info.json"))

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"  评估完成")
    print(f"{'='*60}")
    print(f"  v2 thrust RMSE: {np.mean(all_thrust_rmse):.4f} N")
    print(f"  v2 mfr    RMSE: {np.mean(all_mfr_rmse):.1f} mg/s")
    if v1_rmse is not None:
        print(f"  v1 thrust RMSE: {v1_rmse:.4f} N  (comparison)")
    print(f"  接口文件: {output_dir}/")
    print(f"  图表: {isp_fig_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
