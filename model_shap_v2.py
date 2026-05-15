"""
model_shap_v2.py — 17维 SHAP + Permutation 归因分析（v2.0 DualOutput Attention-LSTM）
M 负责，dev-m 分支

输出：thrust / mfr / Isp SHAP + Permutation + v1 vs v2 对比
"""
import torch, torch.nn as nn
import pandas as pd, numpy as np, shap
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os, sys, warnings, time
from collections import defaultdict
warnings.filterwarnings("ignore")

from data_pipeline_v2 import _build_features, THRUST_SCALE, MFR_MAX
from model_train_v2 import DualOutputLSTM, INPUT_DIM

# ─── model wrappers ───
class V1LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 256, 2, batch_first=True)
        self.fc = nn.Linear(256, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

class V1Wrapper(nn.Module):
    def __init__(self, base): super().__init__(); self.base = base
    def forward(self, x):
        return self.base(x).mean(dim=1, keepdim=True)

class V2Wrapper(nn.Module):
    def __init__(self, base, idx=0): super().__init__(); self.base = base; self.idx = idx
    def forward(self, x):
        return self.base(x)[:, :, self.idx].mean(dim=1, keepdim=True)

class V2WrapperIsp(nn.Module):
    def __init__(self, base): super().__init__(); self.base = base; self.g = 9.80665
    def forward(self, x):
        o = self.base(x)
        return (o[:,:,0]*THRUST_SCALE / (o[:,:,1]*MFR_MAX*self.g*1e-3+1e-8)).mean(dim=1, keepdim=True)

# ─── config ───
FN = ['ton','vl','test_pressure','cumulated_on_time','cumulated_throughput',
      'cumulated_pulses','ssf','health_check','ramp1','ramp2','ramp3','ramp4',
      'onmod','offmod','random_short','random_long','random_mixed']
V1F = ['Valve (ton)','Pressure','On-Time (aging)']
GN  = ['Control\n(ton+vl)','Environment\n(pressure)','Aging\n(3 factors)','One-hot\n(11 modes)']
G0, FIG, NP = 9.80665, "outputs/figures/v2", "outputs/predictions/v2"
os.makedirs(FIG, exist_ok=True); os.makedirs(NP, exist_ok=True)
# 清理旧版废弃图表
for old in ['perm_isp.png', 'shap_thrust_vs_mfr.png']:
    p = os.path.join(FIG, old)
    if os.path.exists(p): os.remove(p); print(f"  cleaned: {old}")
N_BG, N_T, N_PERM = 50, 16, 3

# 赛级绘图参数
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.titlesize': 16, 'axes.labelsize': 13,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
})

# ─── data ───
def load_bg(md, dr, n=N_BG, seq=200):
    df = pd.read_csv(md); df['filename'] = df['filename'].str.strip()
    xs = []
    for _, r in df[(df['anomalous']==False) & (df['sn']<=12)].iterrows():
        fp = os.path.join(dr, r['filename'])
        if not os.path.exists(fp): continue
        try: d = pd.read_csv(fp)
        except: continue
        if len(d) < seq: continue
        xs.append(_build_features(d.iloc[:seq], r)[0])
        if len(xs) >= n: break
    return np.stack(xs)

def load_test(md, dr, seq=200):
    df = pd.read_csv(md); df['filename'] = df['filename'].str.strip()
    ex = set(os.listdir(dr))
    sub = df[(df['anomalous']==False) & (df['sn']>12) & df['filename'].isin(ex)]
    xs, modes = [], []
    for m in sub['test_mode'].unique():
        for _, r in sub[sub['test_mode']==m].head(2).iterrows():
            fp = os.path.join(dr, r['filename'])
            if not os.path.exists(fp): continue
            try: d = pd.read_csv(fp)
            except: continue
            if len(d) < seq: continue
            x, _, _ = _build_features(d.iloc[:seq], r); xs.append(x); modes.append(m)
    return np.stack(xs)[:N_T], modes[:N_T]

def load_v1(md, dr, seq=200):
    df = pd.read_csv(md); df['filename'] = df['filename'].str.strip()
    xs = []
    for _, r in df[(df['anomalous']==False) & (df['sn']<=12)].head(10).iterrows():
        fp = os.path.join(dr, r['filename'])
        if not os.path.exists(fp): continue
        try: d = pd.read_csv(fp)
        except: continue
        if len(d) < seq: continue
        u = d['ton'].values[:seq].astype(np.float32)
        xs.append(np.stack([u, np.full_like(u, r['test_pressure']/25),
                            np.full_like(u, r['cumulated_on_time']/8)], axis=1))
    xa = np.stack(xs); n = len(xa)//2
    return xa[:n], xa[n:]

# ─── SHAP ───
def run_shap(wrapper, bg_t, ts_t, lb):
    wrapper.train()
    with torch.no_grad(): p = wrapper(ts_t[:1])
    print(f"  [{lb}] out ~{p[0,0].item():.4f}"); sys.stdout.flush()
    t0 = time.time()
    sv = shap.GradientExplainer(wrapper, bg_t).shap_values(ts_t, nsamples=200)
    if isinstance(sv, list): sv = sv[0]
    s = sv.copy().astype(np.float64)
    while s.ndim > 3: s = s.squeeze(-1)
    if s.ndim == 3: s = np.abs(s).mean(axis=1)
    elif s.ndim == 2: s = np.abs(s)
    else: s = np.abs(s).mean(axis=tuple(range(1, s.ndim)))
    print(f"  [{lb}] done {time.time()-t0:.0f}s  {s.shape}  [{s.min():.6f}, {s.max():.6f}]")
    return s

def perm_imp(model, d_t, t_t, idx, is_isp=False):
    model.eval()
    factor = THRUST_SCALE if idx == 0 else MFR_MAX
    with torch.no_grad():
        bp = model(d_t)
        if is_isp:
            v = (bp[:,:,0]*THRUST_SCALE/(bp[:,:,1]*MFR_MAX*G0*1e-3+1e-8)).mean(dim=1)
            t = (t_t[:,:,0]*THRUST_SCALE/(t_t[:,:,1]*MFR_MAX*G0*1e-3+1e-8)).mean(dim=1)
            be = ((v-t)**2).mean().sqrt().item()
        else:
            be = ((bp[:,:,idx]-t_t[:,:,idx])**2).mean().sqrt().item()*factor
    imp = np.zeros(17); dn = d_t.cpu().numpy().copy()
    for f in range(17):
        es = []
        for _ in range(3):
            s = dn.copy(); np.random.shuffle(s[:,:,f])
            st = torch.from_numpy(s).to(d_t.device)
            with torch.no_grad():
                p = model(st)
                if is_isp:
                    v2 = (p[:,:,0]*THRUST_SCALE/(p[:,:,1]*MFR_MAX*G0*1e-3+1e-8)).mean(dim=1)
                    e = ((v2-t)**2).mean().sqrt().item()
                else:
                    e = ((p[:,:,idx]-t_t[:,:,idx])**2).mean().sqrt().item()*factor
                es.append(e)
        imp[f] = np.mean(es) - be
    return imp, be

# ═══════════════════════════════════════════
# 赛级绘图
# ═══════════════════════════════════════════

# —— 色盘 ——
C_RED  = '#c0392b'
C_BLUE = '#2980b9'
C_GRN  = '#27ae60'
C_YL   = '#f39c12'

def _fmt(v):
    if abs(v) >= 0.001: return f'{v:.4f}'
    elif abs(v) >= 1e-6: return f'{v:.2e}'
    else: return f'{v:.1e}'

def _shap_bar(ax, vals, names, pal='Reds'):
    idx = np.argsort(vals)[::-1]; v, lbl = vals[idx], [names[i] for i in idx]
    vn = v / v.max() * 100 if v.max() > 0 else np.zeros_like(v)
    clr = getattr(plt.cm, pal)(np.linspace(0.3, 0.85, len(v)))
    bars = ax.barh(range(len(v)), vn, color=clr, height=0.7, edgecolor='white', lw=0.5)
    ax.set_yticks(range(len(v))); ax.set_yticklabels(lbl, fontsize=12)
    ax.invert_yaxis(); ax.set_xlim(0, max(vn)*1.4 if max(vn) > 0 else 100)
    ax.grid(axis='x', alpha=0.3, ls='--')
    for b, val in zip(bars, v):
        ax.text(b.get_width()+1.5, b.get_y()+b.get_height()/2, _fmt(val),
                va='center', fontsize=10, color='#333', fontweight='bold')
    return bars

def plot_shap(vals, names, title, path, pal='Reds'):
    fig, ax = plt.subplots(figsize=(12, 7))
    _shap_bar(ax, vals, names, pal)
    ax.set_xlabel('Relative Importance (%)', fontsize=13)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()
    print(f"  → {os.path.basename(path)}")

def plot_perm_abs(vals, names, title, path, unit=''):
    idx = np.argsort(vals)[::-1]; v, lb = vals[idx], [names[i] for i in idx]
    fig, ax = plt.subplots(figsize=(12, 7))
    clr = plt.cm.Oranges(np.linspace(0.3, 0.85, len(v)))
    bars = ax.barh(range(len(v)), v, color=clr, height=0.7, ec='white')
    ax.set_yticks(range(len(v))); ax.set_yticklabels(lb, fontsize=12)
    ax.invert_yaxis(); ax.set_xlabel(f'RMSE Increase ({unit})', fontsize=13)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, ls='--')
    mx = max(v) if max(v) > 0 else 1
    for b, val in zip(bars, v):
        xpos = b.get_width() + mx*0.03 if b.get_width() > 0 else mx*0.03
        ax.text(xpos, b.get_y()+b.get_height()/2, _fmt(val),
                va='center', fontsize=8, fontweight='bold', color='#333')
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()
    print(f"  → {os.path.basename(path)}")

def plot_v1_vs_v2(v1, v2, path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for axx in axes:
        axx.grid(axis='x', alpha=0.3, ls='--')
    c1 = plt.cm.Oranges(np.linspace(0.3, 0.8, 3))
    axes[0].barh(range(3), v1/v1.max()*100, color=c1, height=0.6, ec='white')
    axes[0].set_yticks(range(3)); axes[0].set_yticklabels(V1F, fontsize=13)
    axes[0].invert_yaxis(); axes[0].set_title('v1: 3-dim Input', fontsize=15, fontweight='bold')
    axes[0].set_xlim(0, 115)
    for b, val in zip(axes[0].containers[0], v1):
        axes[0].text(b.get_width()+1, b.get_y()+b.get_height()/2, _fmt(val), va='center', fontsize=10)
    v2c = v2[:6]
    c2 = plt.cm.Reds(np.linspace(0.3, 0.85, 6))
    axes[1].barh(range(6), v2c/v2c.max()*100, color=c2, height=0.6, ec='white')
    axes[1].set_yticks(range(6)); axes[1].set_yticklabels(FN[:6], fontsize=13)
    axes[1].invert_yaxis(); axes[1].set_title('v2: 6-dim Continuous + 11 one-hot', fontsize=15, fontweight='bold')
    axes[1].set_xlim(0, 115)
    for b, val in zip(axes[1].containers[0], v2c):
        axes[1].text(b.get_width()+1, b.get_y()+b.get_height()/2, _fmt(val), va='center', fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()
    print(f"  → {os.path.basename(path)}")

def _normalize_per_group(vals):
    """将每个输出各自归一化到组内 0-100%"""
    a = np.array(vals, dtype=float)
    mx = a.max()
    return a / mx * 100 if mx > 0 else np.zeros_like(a)

def _subplot_bars(ax, vals, names, color, title):
    """单子图的水平柱状图"""
    vals_a = np.array(vals, dtype=float)
    idx = np.argsort(vals_a)[::-1]
    v, lb = vals_a[idx], [names[i] for i in idx]
    bars = ax.barh(range(len(v)), v, color=color, height=0.6, alpha=.85, ec='white')
    ax.set_yticks(range(len(v)))
    ax.set_yticklabels(lb, fontsize=10, rotation='horizontal')
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, ls='--')
    mx = max(v) if max(v) > 0 else 1
    ax.set_xlim(0, mx * 1.7)
    labels = [_fmt(val) for val in v]
    ax.bar_label(bars, labels=labels, padding=mx*0.05, fontsize=9, fontweight='bold')

def plot_groups(mt, mm, mi, path):
    """3个子图：每个输出一行，4个特征组"""
    def g(v): return [v[0]+v[1], v[2], v[3]+v[4]+v[5], v[6:].sum()]
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    for ax, vals, tt, clr in [(axes[0],g(mt),'Thrust',C_RED),
                               (axes[1],g(mm),'MFR',C_BLUE),
                               (axes[2],g(mi),'Isp',C_GRN)]:
        _subplot_bars(ax, vals, GN, clr, tt)
    plt.suptitle('Feature Group SHAP Contribution', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()
    print(f"  → {os.path.basename(path)}")

def plot_continuous(mt, mm, mi, path):
    """3个子图：每个输出一行，6个连续特征"""
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    for ax, vals, tt, clr in [(axes[0],mt[:6],'Thrust',C_RED),
                               (axes[1],mm[:6],'MFR',C_BLUE),
                               (axes[2],mi[:6],'Isp',C_GRN)]:
        _subplot_bars(ax, vals, FN[:6], clr, tt)
    plt.suptitle('SHAP — Continuous Features', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()
    print(f"  → {os.path.basename(path)}")

def plot_compare(ms, pi, nm, lb, path):
    """SHAP vs Permutation：左侧归一化SHAP，右侧绝对Permutation"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, vals, cm, tt, is_shap in [
        (axes[0], ms, 'Reds', f'{lb}: SHAP (relative)', True),
        (axes[1], pi, 'Blues', f'{lb}: Permutation (absolute)', False)]:
        idx = np.argsort(vals)[::-1]; v, l = vals[idx], [nm[i] for i in idx]
        v_show = _normalize_per_group(v) if is_shap else v
        clr = getattr(plt.cm, cm)(np.linspace(0.3, 0.85, len(v)))
        bars = ax.barh(range(len(v)), v_show, color=clr, height=0.7, edgecolor='white')
        ax.set_yticks(range(len(v))); ax.set_yticklabels(l, fontsize=11)
        ax.invert_yaxis(); ax.set_title(tt, fontsize=15, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(0, max(v_show)*1.4 if max(v_show) > 0 else 100)
        for b, val in zip(bars, v):
            ax.text(b.get_width()+max(v_show)*0.03 if is_shap else b.get_width()*1.03,
                    b.get_y()+b.get_height()/2, _fmt(val),
                    va='center', fontsize=9, fontweight='bold')
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()
    print(f"  → {os.path.basename(path)}")

def plot_heatmap(svs, modes, path):
    mg = defaultdict(list)
    for i, m in enumerate(modes): mg[m].append(svs[i])
    ml = list(dict.fromkeys(modes))
    dd = np.array([np.mean(mg[m], axis=0)[:6] for m in ml])
    fig, ax = plt.subplots(figsize=(12, max(5.5, len(ml)*0.55)))
    im = ax.imshow(dd, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(6)); ax.set_xticklabels(FN[:6], rotation=45, ha='right', fontsize=11)
    ax.set_yticks(range(len(ml))); ax.set_yticklabels(ml, fontsize=11)
    ax.set_title('SHAP by Test Mode — Continuous Features', fontsize=16, fontweight='bold', pad=15)
    for i in range(len(ml)):
        for j in range(6):
            ax.text(j, i, f'{dd[i,j]:.5f}', ha='center', va='center', fontsize=8,
                    color='white' if dd[i,j] > dd.max()*0.6 else 'black')
    plt.colorbar(im, ax=ax, label='Mean |SHAP value|', shrink=0.8)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()
    print(f"  → {os.path.basename(path)}")

def plot_continuous(mt, mm, mi, path):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(6); w = 0.25
    b1 = ax.bar(x-w, mt[:6], w, label='Thrust', color=C_RED, alpha=.85)
    b2 = ax.bar(x,   mm[:6], w, label='MFR',    color=C_BLUE, alpha=.85)
    b3 = ax.bar(x+w, mi[:6], w, label='Isp',    color=C_GRN, alpha=.85)
    ax.set_xticks(x); ax.set_xticklabels(FN[:6], fontsize=12)
    ax.set_ylabel('Mean |SHAP value|', fontsize=13)
    ax.set_title('SHAP — Continuous Features (thrust / mfr / Isp)', fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=12); ax.grid(axis='y', alpha=0.3, linestyle='--')
    for bars in [b1, b2, b3]:
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height(), f'{b.get_height():.6f}',
                    ha='center', va='bottom', fontsize=8, rotation=90)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()
    print(f"  → {os.path.basename(path)}")

# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    print(f"{'='*60}\n  SHAP + Permutation — v2.0 Attention-LSTM\n"
          f"  分析: thrust | mfr | Isp + v1对比\n{'='*60}"); sys.stdout.flush()

    device = torch.device('cpu')

    # v2 model
    model = DualOutputLSTM(input_dim=INPUT_DIM).to(device)
    model.load_state_dict(torch.load("outputs/models/v2/dual_output_lstm_v2.pth",
                                     map_location=device, weights_only=True))
    model.train()
    print("v2 loaded")

    bg = load_bg("data/metadata.csv", "data/dataset/dataset/train/")
    ts, modes = load_test("data/metadata.csv", "data/dataset/dataset/test/")
    bg_t = torch.from_numpy(bg).to(device); ts_t = torch.from_numpy(ts).to(device)
    print(f"  bg={bg_t.shape}  test={ts_t.shape}")

    # targets
    df_m = pd.read_csv("data/metadata.csv"); df_m['filename'] = df_m['filename'].str.strip()
    ex = set(os.listdir("data/dataset/dataset/test/"))
    df_te = df_m[(df_m['anomalous']==False) & (df_m['sn']>12) & df_m['filename'].isin(ex)]
    ys = []
    for m in set(modes):
        for _, r in df_te[df_te['test_mode']==m].head(2).iterrows():
            fp = os.path.join("data/dataset/dataset/test/", r['filename'])
            if not os.path.exists(fp): continue
            try:
                d = pd.read_csv(fp)
                if len(d)>=200:
                    _, y, _ = _build_features(d.iloc[:200], r); ys.append(y)
            except: pass
        if len(ys) >= N_T: break
    target_t = torch.from_numpy(np.stack(ys[:N_T])).to(device)

    # ══════════════════════════════════
    # V2 SHAP
    # ══════════════════════════════════
    print(f"\n{'='*50}\n  SHAP (v2)\n{'='*50}")
    st = run_shap(V2Wrapper(model,0), bg_t, ts_t, "thrust")
    sm = run_shap(V2Wrapper(model,1), bg_t, ts_t, "mfr")
    si = run_shap(V2WrapperIsp(model), bg_t, ts_t, "Isp")
    # 截断 Isp SHAP 离群值（mfr 接近 0 时 Isp 爆炸）
    si_flat = np.abs(si).ravel()
    upper = np.percentile(si_flat, 95)
    si = np.clip(si, -upper, upper)
    print(f"  [Isp] clipped at ±{upper:.4f}")
    mt = st.mean(0); mm = sm.mean(0); mi = si.mean(0)
    for a, n in [(mt,'thrust'),(mm,'mfr'),(mi,'isp')]:
        np.save(os.path.join(NP, f"shap_{n}.npy"), a)

    # ══════════════════════════════════
    # V1 SHAP
    # ══════════════════════════════════
    print(f"\n{'='*50}\n  SHAP (v1)\n{'='*50}")
    v1m = V1LSTM().to(device)
    v1m.load_state_dict(torch.load("outputs/models/v1/thruster_lstm_v1.pth",
                                    map_location=device, weights_only=True))
    v1m.train()
    bg_v, ts_v = load_v1("data/metadata.csv", "data/dataset/dataset/train/")
    sv1 = run_shap(V1Wrapper(v1m), torch.from_numpy(bg_v).to(device),
                   torch.from_numpy(ts_v).to(device), "v1 thrust")
    v1m_v = sv1.mean(0)
    np.save(os.path.join(NP, "shap_v1_thrust.npy"), sv1)

    # ══════════════════════════════════
    # Permutation
    # ══════════════════════════════════
    print(f"\n{'='*50}\n  Permutation\n{'='*50}")
    pt, bt = perm_imp(model, ts_t, target_t, 0)
    pm, bm = perm_imp(model, ts_t, target_t, 1)
    print(f"  base RMSE: thrust={bt:.4f}N  mfr={bm:.2f}mg/s")

    # ══════════════════════════════════
    # PLOTS
    # ══════════════════════════════════
    print(f"\n{'='*50}\n  Plots\n{'='*50}")

    # SHAP bars (归一化相对重要性)
    plot_shap(mt, FN, "SHAP Feature Importance — Thrust (17-dim)",
              os.path.join(FIG, "shap_thrust_17dim.png"))
    plot_shap(mm, FN, "SHAP Feature Importance — MFR (17-dim)",
              os.path.join(FIG, "shap_mfr_17dim.png"))
    plot_shap(mi, FN, "SHAP Feature Importance — Isp (17-dim)",
              os.path.join(FIG, "shap_isp_17dim.png"))

    # 连续特征
    plot_continuous(mt, mm, mi, os.path.join(FIG, "shap_continuous.png"))

    # v1 vs v2 核心对比
    plot_v1_vs_v2(v1m_v, mt, os.path.join(FIG, "shap_v1_vs_v2_comparison.png"))

    # 分组
    plot_groups(mt, mm, mi, os.path.join(FIG, "feature_groups.png"))

    # Permutation（thrust 用绝对量级）
    plot_perm_abs(pt, FN, "Permutation Importance — Thrust",
                  os.path.join(FIG, "perm_thrust.png"), unit='N')
    plot_perm_abs(pm, FN, "Permutation Importance — MFR",
                  os.path.join(FIG, "perm_mfr.png"), unit='mg/s')

    # SHAP vs Perm 对比
    plot_compare(mt, pt, FN, "Thrust", os.path.join(FIG, "shap_vs_perm_thrust.png"))
    plot_compare(mm, pm, FN, "MFR",    os.path.join(FIG, "shap_vs_perm_mfr.png"))

    # 热力图
    plot_heatmap(st, modes, os.path.join(FIG, "shap_mode_heatmap.png"))

    # ══════════════════════════════════
    print(f"\n{'='*60}\n  摘要\n{'='*60}")
    for nm, v in [("Thrust",mt),("MFR",mm),("Isp",mi)]:
        top = [FN[i] for i in np.argsort(v)[-1:-4:-1]]
        print(f"  {nm:7s} Top-3: {top}")
    print(f"  v1: ton={v1m_v[0]:.6f}  pressure={v1m_v[1]:.6f}  aging={v1m_v[2]:.6f}")
    print(f"  连续特征占比: thrust={mt[:6].sum()/mt.sum()*100:.0f}%  "
          f"mfr={mm[:6].sum()/mm.sum()*100:.0f}%  Isp={mi[:6].sum()/mi.sum()*100:.0f}%")
    print(f"  图表 → {FIG}/  npy → {NP}/")

if __name__ == "__main__":
    main()
