"""生成 Figure 4: 11种工况分项RMSE柱状图"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans SC", "Arial", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

# ── 数据：来自 v2.0 测试结果 ──
modes = [
    "ssf", "health_check", "onmod",
    "ramp1", "ramp2", "ramp3", "ramp4",
    "offmod",
    "random_short", "random_long", "random_mixed",
]
mode_labels = [
    "稳态\nssf", "健康检查\nhealth_check", "开启态\nonmod",
    "渐变1\nramp1", "渐变2\nramp2", "渐变3\nramp3", "渐变4\nramp4",
    "关闭态\noffmod",
    "随机短\nrnd_short", "随机长\nrnd_long", "随机混合\nrnd_mixed",
]
thrust_rmse = [0.1390, 0.0395, 0.0212, 0.0820, 0.0793, 0.0840, 0.0895, 0.0967, 0.0937, 0.0791, 0.0863]
mfr_rmse    = [47.6,   33.1,   22.7,   56.5,   62.2,   44.7,   49.9,   73.5,   71.4,   44.6,   57.6]

# 按 thrust_rmse 排序
order = np.argsort(thrust_rmse)
modes_sorted = [modes[i] for i in order]
labels_sorted = [mode_labels[i] for i in order]
thrust_sorted = [thrust_rmse[i] for i in order]
mfr_sorted = [mfr_rmse[i] for i in order]

# ── 颜色编码：稳态/渐变/开关/随机 ──
cat_colors = {
    "ssf": "#5B9BD5", "health_check": "#5B9BD5",  # 稳态 blue
    "ramp1": "#ED7D31", "ramp2": "#ED7D31", "ramp3": "#ED7D31", "ramp4": "#ED7D31",  # 渐变 orange
    "onmod": "#A5A5A5", "offmod": "#A5A5A5",  # 开关 grey
    "random_short": "#70AD47", "random_long": "#70AD47", "random_mixed": "#70AD47",  # 随机 green
}
colors = [cat_colors[m] for m in modes_sorted]

# ── 画图 ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1, 1]})

x = np.arange(len(modes_sorted))
w = 0.55

# --- Thrust RMSE ---
bars1 = ax1.bar(x, thrust_sorted, w, color=colors, edgecolor="white", linewidth=0.5)
ax1.set_title("Thrust RMSE by Operating Mode", fontsize=11, fontweight="bold")
ax1.set_ylabel("Thrust RMSE (N)")
ax1.set_xticks(x)
ax1.set_xticklabels(labels_sorted, fontsize=5)
ax1.set_ylim(0, max(thrust_sorted) * 1.18)
for bar, val in zip(bars1, thrust_sorted):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
             f"{val:.4f}", ha="center", va="bottom", fontsize=7, fontweight="bold", color="#333")

# highlight best/worst
best1 = np.argmin(thrust_sorted)
worst1 = np.argmax(thrust_sorted)
bars1[best1].set_edgecolor("#2E7D32")
bars1[best1].set_linewidth(1.5)
bars1[worst1].set_edgecolor("#C62828")
bars1[worst1].set_linewidth(1.5)

# --- MFR RMSE ---
bars2 = ax2.bar(x, mfr_sorted, w, color=colors, edgecolor="white", linewidth=0.5)
ax2.set_title("MFR RMSE by Operating Mode", fontsize=11, fontweight="bold")
ax2.set_ylabel("MFR RMSE (mg/s)")
ax2.set_xticks(x)
ax2.set_xticklabels(labels_sorted, fontsize=5)
ax2.set_ylim(0, max(mfr_sorted) * 1.18)
for bar, val in zip(bars2, mfr_sorted):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
             f"{val:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold", color="#333")

best2 = np.argmin(mfr_sorted)
worst2 = np.argmax(mfr_sorted)
bars2[best2].set_edgecolor("#2E7D32")
bars2[best2].set_linewidth(1.5)
bars2[worst2].set_edgecolor("#C62828")
bars2[worst2].set_linewidth(1.5)

# ── 图例 ──
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#5B9BD5", label="稳态 (ssf, health_check)"),
    Patch(facecolor="#ED7D31", label="渐变 (ramp1–4)"),
    Patch(facecolor="#A5A5A5", label="开关 (onmod, offmod)"),
    Patch(facecolor="#70AD47", label="随机 (random_*)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=7.5, frameon=False)

# suptitle removed per user request

plt.tight_layout(rect=[0, 0.06, 1, 0.97])

out_dir = r"D:\上海大学__\26CRAICai创新赛\备份代码\outputs\figures\v2"
out_base = f"{out_dir}\\rmse_by_mode_11"

fig.savefig(f"{out_base}.svg", bbox_inches="tight")
fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
print(f"Saved: {out_base}.svg / .pdf / .png")
