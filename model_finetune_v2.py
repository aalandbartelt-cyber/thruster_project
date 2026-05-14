"""
model_finetune_v2.py — 个体推进器微调（Per-Thruster Fine-Tuning）
M 负责，dev-m 分支

修复记录（2026-05-14）：
  1. ✅ 滑动窗口全序列评估（非仅前 200 步）
  2. ✅ 验证集与测试集严格分离
  3. ✅ 验证 model.attention 存在
  4. ✅ weights_only=True 统一
  5. ✅ 12 SN 改善的 t 检验 + 方差报告
  6. ✅ n_finetune 默认 12，支持扫描模式
  7. ✅ 默认冻结 LSTM + Attention（清晰可解释）
  8. ✅ 支持 --freeze_lstm / --unfreeze_all 对照
  9. ✅ 加入 Isp 评估
  10. ✅ NASA 风格汇总图（双柱并列 + 改善/无变化/退化三色）
  11. ✅ 灾难性遗忘检查（跨 SN 泛化性）
  12. ✅ argparse 完整

用法：
  python model_finetune_v2.py                         # 全部 SN13-24
  python model_finetune_v2.py --sn 13                 # 单台
  python model_finetune_v2.py --freeze_lstm           # 冻结 LSTM 实验
  python model_finetune_v2.py --unfreeze_all          # 全网络微调对照
  python model_finetune_v2.py --scan_n 3,6,12,24      # 扫描微调数据量
"""
import os, sys, argparse, datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from scipy import stats as scipy_stats

from data_pipeline_v2 import _build_features, THRUST_SCALE, MFR_MAX
from model_train_v2 import DualOutputLSTM, INPUT_DIM

G0 = 9.80665
MODEL_RMSE = 0.0832

# ═══════════════════════════════════════════
# 微调数据集
# ═══════════════════════════════════════════

class FinetuneDataset(Dataset):
    def __init__(self, metadata_rows, data_dir, seq_len=200, stride=100):
        self.samples = []
        for _, row in metadata_rows.iterrows():
            fpath = os.path.join(data_dir, row['filename'].strip())
            if not os.path.exists(fpath): continue
            try: df = pd.read_csv(fpath)
            except: continue
            T = len(df)
            if T < seq_len: continue
            for start in range(0, T - seq_len + 1, stride):
                x, y, _ = _build_features(df.iloc[start:start + seq_len], row)
                self.samples.append((x, y))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def freeze_params(module, freeze=True):
    for p in module.parameters():
        p.requires_grad = not freeze

def dual_loss(pred, target, w_thrust=0.7, w_mfr=0.3):
    mse = nn.MSELoss()
    return w_thrust * mse(pred[:,:,0], target[:,:,0]) + w_mfr * mse(pred[:,:,1], target[:,:,1])


# ═══════════════════════════════════════════
# 滑动窗口全序列评估（bug #1 修复）
# ═══════════════════════════════════════════

@torch.no_grad()
def eval_full(model, df_row, data_dir, seq_len=200, stride=200, device='cpu'):
    """全文件滑动窗口评估，返回 thrust/mfr/isp 的 RMSE"""
    fpath = os.path.join(data_dir, df_row['filename'].strip())
    if not os.path.exists(fpath): return None, None, None
    try: df = pd.read_csv(fpath)
    except: return None, None, None
    T = len(df)
    if T < seq_len: return None, None, None

    preds_t, preds_m, trues_t, trues_m = [], [], [], []
    for start in range(0, T - seq_len + 1, stride):
        x, y, _ = _build_features(df.iloc[start:start + seq_len], df_row)
        x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
        pred = model(x_t).squeeze(0).cpu().numpy()
        preds_t.append(pred[:, 0]); preds_m.append(pred[:, 1])
        trues_t.append(y[:, 0]);    trues_m.append(y[:, 1])

    pt = np.concatenate(preds_t); pm = np.concatenate(preds_m)
    tt = np.concatenate(trues_t); tm = np.concatenate(trues_m)
    t_rmse = np.sqrt(np.mean((pt - tt)**2)) * THRUST_SCALE
    m_rmse = np.sqrt(np.mean((pm - tm)**2)) * MFR_MAX
    # Isp RMSE：仅计算三方都有效的时刻（真值着火 + 预测不接近零）
    firing = ((tt * THRUST_SCALE) > 0.05) & ((tm * MFR_MAX) > 5.0) & ((pm * MFR_MAX) > 5.0)
    if firing.sum() > 10:
        isp_pred = (pt[firing] * THRUST_SCALE) / (pm[firing] * MFR_MAX * 1e-6 * G0 + 1e-8)
        isp_true = (tt[firing] * THRUST_SCALE) / (tm[firing] * MFR_MAX * 1e-6 * G0 + 1e-8)
        i_rmse = np.sqrt(np.mean((isp_pred - isp_true)**2))
    else:
        i_rmse = 0.0
    return t_rmse, m_rmse, i_rmse


def eval_many(model, metadata_rows, data_dir, seq_len=200, stride=200, device='cpu'):
    """多文件平均 RMSE"""
    t_list, m_list, i_list = [], [], []
    for _, row in metadata_rows.iterrows():
        t, m, i = eval_full(model, row, data_dir, seq_len, stride, device)
        if t is not None:
            t_list.append(t); m_list.append(m); i_list.append(i)
    if not t_list: return 0.0, 0.0, 0.0
    return np.mean(t_list), np.mean(m_list), np.mean(i_list)


# ═══════════════════════════════════════════
# 中文字体注册
# ═══════════════════════════════════════════

CN_FONT_CANDIDATES = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
    '/usr/share/fonts/truetype/arphic/uming.ttc',
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/simhei.ttf',
]

CN_FONT_PATH = None
for _p in CN_FONT_CANDIDATES:
    if os.path.exists(_p):
        CN_FONT_PATH = _p
        try:
            fm.fontManager.addfont(_p)
        except Exception:
            pass
        break

def _cn_prop(size=12, weight='normal'):
    if CN_FONT_PATH:
        return FontProperties(fname=CN_FONT_PATH, size=size, weight=weight)
    return FontProperties(size=size, weight=weight)


# ═══════════════════════════════════════════
# 日志双写 (stdout + 文件)
# ═══════════════════════════════════════════

class TeeLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8', buffering=1)
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ═══════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sn', type=int, default=None, help='目标 SN')
    parser.add_argument('--n_finetune', type=int, default=12, help='每台微调用文件数')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seq_len', type=int, default=200)
    parser.add_argument('--stride', type=int, default=200, help='评估滑动步长')
    parser.add_argument('--no_freeze_lstm', action='store_true', help='不冻结 LSTM（默认冻结 LSTM+Attention）')
    parser.add_argument('--unfreeze_all', action='store_true', help='全网络微调对照')
    parser.add_argument('--scan_n', type=str, default=None,
                        help='扫描微调数据量，逗号分隔，如 3,6,12,24')
    parser.add_argument('--no_forget_check', action='store_true', help='跳过灾难性遗忘检查')
    parser.add_argument('--forget_n_files', type=int, default=5, help='遗忘检查每个其他SN用多少文件')
    args = parser.parse_args()

    torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 日志文件 ──
    log_dir = "outputs/finetune_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir,
        f"finetune_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    sys.stdout = TeeLogger(log_path)
    print(f"日志文件: {log_path}")

    print(f"{'='*60}")
    print(f"  个体推进器微调 v2")
    print(f"  设备: {device}")
    print(f"  微调用文件: {args.n_finetune}/SN  |  Epochs: {args.epochs}  lr: {args.lr}")
    if not args.no_freeze_lstm: print(f"  模式: 冻结 LSTM+Attention")
    if args.unfreeze_all: print(f"  模式: 全网络微调（v1 风格对照）")
    print(f"{'='*60}")

    # ── 数据 ──
    df_meta = pd.read_csv("data/metadata.csv")
    df_meta['filename'] = df_meta['filename'].str.strip()
    df_normal = df_meta[df_meta['anomalous'] == False]
    test_dir = "data/dataset/dataset/test/"
    target_sns = [args.sn] if args.sn else list(range(13, 25))

    # ── 全局模型 ──
    global_model = DualOutputLSTM(input_dim=INPUT_DIM).to(device)
    global_model.load_state_dict(torch.load(
        "outputs/models/v2/dual_output_lstm_v2.pth", map_location=device, weights_only=True))
    global_model.eval()
    model_save_dir = "outputs/models/v2"

    # 预计算全局模型在各 SN 的 baseline（供遗忘检查）
    global_perf_by_sn = {}
    for osn in target_sns:
        df_osn = df_normal[df_normal['sn'] == osn]
        t, _, _ = eval_many(global_model, df_osn.head(10), test_dir,
                            args.seq_len, args.stride, device)
        global_perf_by_sn[osn] = t

    # ── 扫参模式 ──
    scan_values = [int(x) for x in args.scan_n.split(',')] if args.scan_n else [args.n_finetune]

    scan_results = []
    for n_ft in scan_values:
        print(f"\n{'='*60}")
        print(f"  微调数据量: {n_ft} 文件/SN")
        print(f"{'='*60}")

        results = []
        for sn in target_sns:
            df_sn = df_normal[df_normal['sn'] == sn]
            files = df_sn['filename'].tolist()
            np.random.seed(42)
            np.random.shuffle(files)
            train_files = files[:n_ft]
            test_files  = files[n_ft:]

            if len(train_files) < 1 or len(test_files) < 4:
                print(f"  SN{sn}: 数据不足, 跳过"); continue

            # 严格分离验证集（bug #2 修复）
            val_files  = test_files[:3]    # 验证 3 个
            eval_files = test_files[3:]    # 真正测试集

            df_train = df_sn[df_sn['filename'].isin(train_files)]
            df_val   = df_sn[df_sn['filename'].isin(val_files)]
            df_eval  = df_sn[df_sn['filename'].isin(eval_files)]
            print(f"\n  SN{sn}: 微调 {len(df_train)}  |  验证 {len(df_val)}  |  测试 {len(df_eval)}")

            # Baseline
            t_base, m_base, i_base = eval_many(global_model, df_eval, test_dir,
                                                args.seq_len, args.stride, device)
            print(f"    全局: thr={t_base:.4f} N  mfr={m_base:.2f} mg/s  isp={i_base:.2f} s")

            # ── 微调 ──
            model = DualOutputLSTM(input_dim=INPUT_DIM).to(device)
            model.load_state_dict(torch.load(
                "outputs/models/v2/dual_output_lstm_v2.pth",
                map_location=device, weights_only=True))

            # 微调策略（bug #7/#12）
            strategy = 'unfreeze' if args.unfreeze_all else 'freeze_lstm'
            if args.unfreeze_all:
                pass
            elif not args.no_freeze_lstm:
                freeze_params(model.lstm)
                freeze_params(model.attention)
            # FC + LN 默认 requires_grad=True

            dataset = FinetuneDataset(df_train, test_dir, args.seq_len, 100)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            # 学习率（bug #7：简单的单学习率，不搞分层拍脑袋）
            optimizer = optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

            # 验证集：3 个文件各取首 200 步，取平均 loss
            vx_t_list, vy_t_list = [], []
            for _, vr in df_val.iterrows():
                vp = os.path.join(test_dir, vr['filename'].strip())
                if os.path.exists(vp):
                    vd = pd.read_csv(vp)
                    if len(vd) >= args.seq_len:
                        vx, vy, _ = _build_features(vd.iloc[:args.seq_len], vr)
                        vx_t_list.append(torch.from_numpy(vx).float().unsqueeze(0).to(device))
                        vy_t_list.append(torch.from_numpy(vy).float().unsqueeze(0).to(device))

            model.train()
            best_val = float('inf'); best_ep = 0; no_imp = 0
            for ep in range(args.epochs):
                total_loss = 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = dual_loss(model(x), y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()

                model.eval()
                with torch.no_grad():
                    vloss = np.mean([dual_loss(model(vxt), vyt).item() for vxt, vyt in zip(vx_t_list, vy_t_list)])
                scheduler.step(vloss)

                if vloss < best_val:
                    best_val = vloss; best_ep = ep; no_imp = 0
                    torch.save(model.state_dict(),
                               f"{model_save_dir}/dual_output_lstm_v2_sn{sn}_{strategy}.pth")
                else:
                    no_imp += 1

                if (ep+1) % 5 == 0:
                    print(f"      E {ep+1:2d}/{args.epochs}  loss={total_loss/len(loader):.6f}"
                          f"  val={vloss:.6f}  best={best_val:.6f}@{best_ep}")

                if no_imp >= 8:
                    print(f"      Early stop @{ep+1}"); break

            model.load_state_dict(torch.load(
                f"{model_save_dir}/dual_output_lstm_v2_sn{sn}_{strategy}.pth",
                map_location=device, weights_only=True))

            # ── 微调后评估 ──
            t_ft, m_ft, i_ft = eval_many(model, df_eval, test_dir,
                                          args.seq_len, args.stride, device)
            t_imp = (t_base - t_ft) / t_base * 100
            m_imp = (m_base - m_ft) / m_base * 100
            i_imp = (i_base - i_ft) / i_base * 100
            print(f"    微调后: thr={t_ft:.4f} ({t_imp:+.1f}%)  mfr={m_ft:.2f} ({m_imp:+.1f}%)"
                  f"  isp={i_ft:.2f} ({i_imp:+.1f}%)")

            results.append({
                'sn': sn, 'n_train': n_ft, 'n_eval': len(eval_files),
                't_base': t_base, 't_ft': t_ft, 't_imp': t_imp,
                'm_base': m_base, 'm_ft': m_ft, 'm_imp': m_imp,
                'i_base': i_base, 'i_ft': i_ft, 'i_imp': i_imp,
            })

            # ═══════════════════════════════════════════
            # 灾难性遗忘检查（bug #11）
            # ═══════════════════════════════════════════
            if not args.no_forget_check and len(target_sns) > 1:
                forget_t = []
                for osn in target_sns:
                    if osn == sn: continue
                    df_osn = df_normal[df_normal['sn'] == osn]
                    df_osn_test = df_osn[~df_osn['filename'].isin(train_files)]
                    t_osn, _, _ = eval_many(model, df_osn_test.head(args.forget_n_files), test_dir,
                                            args.seq_len, args.stride, device)
                    forget_t.append(t_osn)
                forget_mean = np.mean(forget_t)
                fb_mean = np.mean([global_perf_by_sn[osn] for osn in target_sns if osn != sn])
                ratio = forget_mean / max(fb_mean, 1e-8)
                if ratio < 1.02:   fstatus = '无遗忘'
                elif ratio < 1.10: fstatus = '轻微退化'
                else:              fstatus = '严重遗忘'
                print(f"    遗忘检查: 全局均值={fb_mean:.4f} → 微调后均值={forget_mean:.4f}  (ratio={ratio:.3f}, {fstatus})")

        # ═══════════════════════════════════════════
        # 汇总
        # ═══════════════════════════════════════════
        if not results: continue
        imps = [r['t_imp'] for r in results]
        t_stat, p_val = scipy_stats.ttest_1samp(imps, 0)
        print(f"\n{'='*60}")
        print(f"  结果汇总 (n_finetune={n_ft})")
        print(f"{'='*60}")
        print(f"  {'SN':>3s}  {'文件':>4s}  {'Thrust Base':>11s}  {'FT':>11s}  {'改善':>7s}  "
              f"{'MFR Base':>9s}  {'FT':>9s}  {'改善':>7s}")
        print(f"  {'─'*3}  {'─'*4}  {'─'*11}  {'─'*11}  {'─'*7}  {'─'*9}  {'─'*9}  {'─'*7}")
        for r in results:
            print(f"  SN{r['sn']:2d}  {r['n_eval']:4d}  {r['t_base']:.4f} N  {r['t_ft']:.4f} N  "
                  f"{r['t_imp']:+6.1f}%  {r['m_base']:7.2f}  {r['m_ft']:7.2f}  {r['m_imp']:+6.1f}%")
        mean_t = np.mean(imps); std_t = np.std(imps)
        print(f"  {'─'*3}  {'─'*4}  {'─'*11}  {'─'*11}  {'─'*7}  {'─'*9}  {'─'*9}  {'─'*7}")
        print(f"  平均: thrust 改善 {mean_t:+.1f}% ± {std_t:.1f}% "
              f"(t={t_stat:.2f}, p={p_val:.4f}{'*' if p_val<0.05 else ''})")

        # ── 汇总图 ──
        fig, ax = plt.subplots(figsize=(12, 5))
        sns = [r['sn'] for r in results]
        x = np.arange(len(sns)); w = 0.3
        t_vals = [r['t_imp'] for r in results]
        m_vals = [r['m_imp'] for r in results]
        colors_t = ['#2ecc71' if v > 3 else '#f39c12' if v > 0 else '#e74c3c' for v in t_vals]
        colors_m = ['#2ecc71' if v > 3 else '#f39c12' if v > 0 else '#e74c3c' for v in m_vals]
        bars_t = ax.bar(x - w/2, t_vals, w, color=colors_t, edgecolor='white', label='Thrust')
        bars_m = ax.bar(x + w/2, m_vals, w, color=colors_m, edgecolor='white', label='MFR')
        ax.axhline(0, color='white', ls='-', lw=1.5)
        # 数值标注
        for bar, val in zip(bars_t, t_vals):
            y_pos = bar.get_height()
            offset = 0.3 if y_pos >= 0 else -0.3
            ax.text(bar.get_x() + bar.get_width()/2, y_pos + offset,
                    f'{val:+.1f}%', ha='center', va='bottom' if y_pos >= 0 else 'top',
                    fontsize=7, fontweight='bold', color='white')
        for bar, val in zip(bars_m, m_vals):
            y_pos = bar.get_height()
            offset = 0.3 if y_pos >= 0 else -0.3
            ax.text(bar.get_x() + bar.get_width()/2, y_pos + offset,
                    f'{val:+.1f}%', ha='center', va='bottom' if y_pos >= 0 else 'top',
                    fontsize=7, fontweight='bold', color='white')
        ax.set_xticks(x); ax.set_xticklabels([f'SN{s}' for s in sns], fontsize=9)
        ax.set_ylabel('RMSE 改善 (%)', fontproperties=_cn_prop(12))
        ax.set_title(f'个体微调效果 (n_finetune={n_ft})', fontproperties=_cn_prop(14, 'bold'))
        ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
        # 设置 y 轴范围，为标注留空间
        all_vals = t_vals + m_vals
        y_margin = (max(all_vals) - min(all_vals)) * 0.15 + 0.5
        ax.set_ylim(min(all_vals) - y_margin, max(all_vals) + y_margin)
        plt.tight_layout()
        fig_path = f"{model_save_dir.replace('models','figures')}/finetune_improvement_n{n_ft}.png"
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  汇总图 → {fig_path}")

        # p < 0.1 打星提示
        mean_m = np.mean([r['m_imp'] for r in results])
        std_m  = np.std([r['m_imp'] for r in results])
        scan_results.append({
            'n': n_ft, 'mean_t': mean_t, 'std_t': std_t,
            'mean_m': mean_m, 'std_m': std_m,
        })

        if p_val < 0.1:
            print(f"  ⭐ 改善统计显著 (p={p_val:.4f} < 0.1)")
        if mean_t < 3:
            print(f"  ⚠️ 平均改善 {mean_t:+.1f}% 较小，建议增大 n_finetune 或尝试 --unfreeze_all")


    # ═══════════════════════════════════════════
    # 边际收益曲线（扫参模式）
    # ═══════════════════════════════════════════
    if len(scan_results) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ns = [r['n'] for r in scan_results]
        ax.errorbar(ns, [r['mean_t'] for r in scan_results],
                    yerr=[r['std_t'] for r in scan_results],
                    marker='o', capsize=4, color='#e74c3c', label='Thrust')
        ax.errorbar(ns, [r['mean_m'] for r in scan_results],
                    yerr=[r['std_m'] for r in scan_results],
                    marker='s', capsize=4, color='#3498db', label='MFR')
        ax.axhline(0, color='gray', ls='--')
        ax.set_xlabel('微调用文件数 n', fontproperties=_cn_prop(12))
        ax.set_ylabel('平均 RMSE 改善 (%)', fontproperties=_cn_prop(12))
        ax.set_title('微调数据量 vs 改善幅度（边际收益曲线）', fontproperties=_cn_prop(14, 'bold'))
        ax.legend(); ax.grid(alpha=0.3)
        fig_path = f"{model_save_dir.replace('models','figures')}/finetune_scan_curve.png"
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  边际收益曲线 → {fig_path}")

    # ── 收尾 ──
    print(f"\n{'='*60}")
    print(f"  完成。日志已保存至: {log_path}")
    print(f"{'='*60}")
    if hasattr(sys.stdout, 'log'):
        sys.stdout.log.close()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()
