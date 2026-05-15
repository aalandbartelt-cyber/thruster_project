"""从日志重新生成微调汇总图（深色背景版）"""
import os, re, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties

# ── 中文字体 ──
CN_FONT_CANDIDATES = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    'C:/Windows/Fonts/msyh.ttc',
]
CN_FONT_PATH = None
for p in CN_FONT_CANDIDATES:
    if os.path.exists(p):
        CN_FONT_PATH = p
        try: fm.fontManager.addfont(p)
        except: pass
        break

def _cn(size=12, weight='normal'):
    if CN_FONT_PATH:
        return FontProperties(fname=CN_FONT_PATH, size=size, weight=weight)
    return FontProperties(size=size, weight=weight)

# ── 解析日志 ──
log_path = sys.argv[1] if len(sys.argv) > 1 else max(
    [os.path.join('outputs/finetune_logs', f) for f in os.listdir('outputs/finetune_logs') if f.endswith('.log')],
    key=os.path.getmtime)

with open(log_path, 'r', encoding='utf-8') as f:
    text = f.read()

results = []
# 解析汇总表：SN13    88  0.0909 N  0.0761 N   +16.3%    92.14    56.72   +38.4%
for m in re.finditer(
    r'SN(\d+)\s+\d+\s+([\d.]+) N\s+([\d.]+) N\s+([+\-\d.]+)%\s+([\d.]+)\s+([\d.]+)\s+([+\-\d.]+)%',
    text):
    results.append({
        'sn': int(m.group(1)),
        't_base': float(m.group(2)),
        't_ft': float(m.group(3)),
        't_imp': float(m.group(4)),
        'm_base': float(m.group(5)),
        'm_ft': float(m.group(6)),
        'm_imp': float(m.group(7)),
    })

if not results:
    print("ERROR: 日志中未找到结果数据"); sys.exit(1)

print(f"解析到 {len(results)} 台 SN 数据")

# ── 汇总图 ──
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 5))
sns = [r['sn'] for r in results]
x = np.arange(len(sns)); w = 0.3
t_vals = [r['t_imp'] for r in results]
m_vals = [r['m_imp'] for r in results]
colors_t = ['#2ecc71' if v > 3 else '#f39c12' if v > 0 else '#e74c3c' for v in t_vals]
colors_m = ['#2ecc71' if v > 3 else '#f39c12' if v > 0 else '#e74c3c' for v in m_vals]

bars_t = ax.bar(x - w/2, t_vals, w, color=colors_t, edgecolor='#1a1a2e', label='Thrust')
bars_m = ax.bar(x + w/2, m_vals, w, color=colors_m, edgecolor='#1a1a2e', label='MFR')
ax.axhline(0, color='#aaaaaa', ls='-', lw=1.5)

for bar, val in zip(bars_t, t_vals):
    y_pos = bar.get_height()
    offset = 0.3 if y_pos >= 0 else -0.3
    ax.text(bar.get_x() + bar.get_width()/2, y_pos + offset,
            f'{val:+.1f}%', ha='center', va='bottom' if y_pos >= 0 else 'top',
            fontsize=7, fontweight='bold', color='#ffffff')
for bar, val in zip(bars_m, m_vals):
    y_pos = bar.get_height()
    offset = 0.3 if y_pos >= 0 else -0.3
    ax.text(bar.get_x() + bar.get_width()/2, y_pos + offset,
            f'{val:+.1f}%', ha='center', va='bottom' if y_pos >= 0 else 'top',
            fontsize=7, fontweight='bold', color='#ffffff')

ax.set_xticks(x)
ax.set_xticklabels([f'SN{s}' for s in sns], fontsize=9, color='#cccccc')
ax.set_ylabel('RMSE 改善 (%)', fontproperties=_cn(12), color='#cccccc')
ax.set_title(f'个体微调效果 (n_finetune=12)', fontproperties=_cn(14, 'bold'), color='#ffffff')
ax.legend(fontsize=10, facecolor='#2d2d2d', edgecolor='#555555', labelcolor='#cccccc')
ax.grid(axis='y', alpha=0.2, color='#888888')
ax.tick_params(colors='#aaaaaa')

all_vals = t_vals + m_vals
y_margin = (max(all_vals) - min(all_vals)) * 0.15 + 0.5
ax.set_ylim(min(all_vals) - y_margin, max(all_vals) + y_margin)

os.makedirs('outputs/figures/v2', exist_ok=True)
fig_path = 'outputs/figures/v2/finetune_improvement_n12.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
plt.style.use('default')
print(f"图表已生成 → {fig_path}")
