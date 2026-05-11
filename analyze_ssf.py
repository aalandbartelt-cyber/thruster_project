"""Quick diagnosis of SSF and mode-specific fitting issues."""
import numpy as np
import pandas as pd
import os

df_meta = pd.read_csv("data/metadata.csv")
df_meta['filename'] = df_meta['filename'].str.strip()

test_dir = "data/dataset/dataset/test/"

# Load predictions
preds = np.load("outputs/predictions/v2/predictions_dual.npy")    # [1344, 200, 2]
targets = np.load("outputs/predictions/v2/targets_dual.npy")

# ============ 1. Per-mode fluctuation analysis ============
print("=" * 80)
print("Per-mode pred_std vs true_std (thrust, in N)")
print("=" * 80)

# Get mode for each sample (normal files first, then anomalous)
df_normal = df_meta[df_meta['anomalous'] == False].copy()
existing = set(os.listdir(test_dir))
df_normal = df_normal[df_normal['filename'].isin(existing)]
normal_files = df_normal['filename'].tolist()

df_anom = df_meta[df_meta['anomalous'] == True].copy()
df_anom = df_anom[df_anom['filename'].isin(existing)]
anom_files = df_anom['filename'].tolist()

all_files = normal_files + anom_files
file_to_mode = {}
for _, row in df_meta.iterrows():
    file_to_mode[row['filename']] = row['test_mode']

print(f"{'Mode':<16s} {'#Files':>6s} {'PredStd':>8s} {'TrueStd':>8s} {'Ratio':>8s} {'RMSE(N)':>8s}")
print("-" * 60)

from collections import defaultdict
mode_stats = defaultdict(lambda: {'p_std': [], 't_std': [], 'rmse': []})

for i, fname in enumerate(all_files):
    if i >= len(preds):
        break
    mode = file_to_mode.get(fname, 'unknown')
    p = preds[i, :, 0] * 8.0
    t = targets[i, :, 0] * 8.0
    mode_stats[mode]['p_std'].append(p.std())
    mode_stats[mode]['t_std'].append(t.std())
    mode_stats[mode]['rmse'].append(np.sqrt(np.mean((p - t) ** 2)))

for mode in sorted(mode_stats.keys()):
    s = mode_stats[mode]
    avg_p_std = np.mean(s['p_std'])
    avg_t_std = np.mean(s['t_std'])
    ratio = avg_p_std / avg_t_std if avg_t_std > 0 else 0
    avg_rmse = np.mean(s['rmse'])
    print(f"{mode:<16s} {len(s['p_std']):6d} {avg_p_std:8.4f} {avg_t_std:8.4f} {ratio:8.3f} {avg_rmse:8.4f}")

# ============ 2. SSF deep dive ============
print("\n" + "=" * 80)
print("SSF: per-file thrust mean vs prediction mean")
print("=" * 80)
ssf_files = df_meta[(df_meta['test_mode'] == 'ssf') & (df_meta['anomalous'] == False) & (df_meta['filename'].isin(existing))]

# For SSF, check if model learns the correct thrust LEVEL (bias) vs fluctuations
print(f"{'File':<30s} {'SN':>4s} {'Pressure':>8s} {'TrueMean':>8s} {'PredMean':>8s} {'Bias':>8s} {'TrueStd':>8s} {'PredStd':>8s}")
print("-" * 90)
for _, row in ssf_files.head(10).iterrows():
    fname = row['filename']
    if fname not in all_files:
        continue
    idx = all_files.index(fname)
    if idx >= len(preds):
        continue
    p = preds[idx, :, 0] * 8.0
    t = targets[idx, :, 0] * 8.0
    bias = p.mean() - t.mean()
    print(f"{fname:<30s} {row['sn']:4.0f} {row['test_pressure']:8.1f} {t.mean():8.4f} {p.mean():8.4f} {bias:8.4f} {t.std():8.4f} {p.std():8.4f}")

# ============ 3. Across-SN thrust variation for SSF ============
print("\n" + "=" * 80)
print("SSF: Thrust level variation across different SNs (test set)")
print("=" * 80)
ssf_sn_stats = ssf_files.groupby('sn').agg(
    n_files=('filename', 'count'),
    avg_pressure=('test_pressure', 'mean'),
).reset_index()
print(ssf_sn_stats.to_string())

# Load a few SSF files from different SNs and show actual thrust levels
print("\nActual thrust means for SSF files by SN:")
for sn in sorted(ssf_files['sn'].unique()):
    subset = ssf_files[ssf_files['sn'] == sn]
    thrust_means = []
    for _, row in subset.head(5).iterrows():
        try:
            df = pd.read_csv(os.path.join(test_dir, row['filename']))
            thrust_means.append(df['thrust'].mean())
        except:
            pass
    if thrust_means:
        print(f"  SN{sn:.0f}: n={len(subset)}, thrust_mean={np.mean(thrust_means):.4f} N, range=[{np.min(thrust_means):.4f}, {np.max(thrust_means):.4f}]")
