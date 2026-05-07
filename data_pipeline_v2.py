"""
data_pipeline_v2.py — 16维输入（5连续 + 11 one-hot） + 双输出预处理
M 负责，dev-m 分支
接口约定：Z 通过 load_sequence() 和 get_test_mode_list() 获取数据，不重复写预处理逻辑。
"""

import os
import numpy as np
import pandas as pd

# ============================================================
# 归一化系数（基于全量 metadata 探索确认，2026-05-08）
# ============================================================
THRUST_SCALE   = 4.0       # thrust / 4.0 → [~0, ~2.5]  (v1 兼容)
MFR_MAX        = 2000.0    # mfr   / 2000 → [0, ~0.81]  (实测 max=1612.72, 向上取整留 headroom)
PRESSURE_MAX   = 25.0      # test_pressure / 25.0 → [0.2, 0.96]  (v1 兼容)
ON_TIME_MAX    = 8.0       # cumulated_on_time / 8.0 → [0, ~0.92]  (v1 兼容)
THROUGHPUT_MAX = 25.0      # cumulated_throughput / 25.0 → [0, ~0.90]
PULSES_MAX     = 25000.0   # cumulated_pulses / 25000 → [0, ~0.95]

# ============================================================
# test_mode → 11 维 one-hot 编码
# 不使用标量编码，避免引入伪序关系
# ============================================================
TEST_MODE_LIST = [
    'ssf', 'health_check',
    'ramp1', 'ramp2', 'ramp3', 'ramp4',
    'onmod', 'offmod',
    'random_short', 'random_long', 'random_mixed',
]


def get_test_mode_list():
    """返回 test_mode 列表（供 Z 的脚本查询可用模式）"""
    return TEST_MODE_LIST


def encode_test_mode(mode_str):
    """单行 one-hot 编码 → np.ndarray shape (11,)"""
    arr = np.zeros(len(TEST_MODE_LIST), dtype=np.float32)
    idx = TEST_MODE_LIST.index(mode_str)
    arr[idx] = 1.0
    return arr


# ============================================================
# 核心 API：load_sequence()
# ============================================================
def load_sequence(filepath, meta_row, seq_len=200):
    """
    加载单个 CSV 文件，构建 16 维输入 + 2 维输出。

    参数
    ----
    filepath : str          CSV 文件的完整路径
    meta_row : pd.Series    metadata.csv 中该文件对应的行
    seq_len  : int          截断/填充长度

    返回
    ----
    x_tensor : np.ndarray  [seq_len, 16]  5 连续 + 11 one-hot
    y_tensor : np.ndarray  [seq_len, 2]   [thrust/THRUST_SCALE, mfr/MFR_MAX]
    """
    df = pd.read_csv(filepath)

    # -- 5 个连续特征 --
    ton        = df['ton'].values.astype(np.float32)
    pressure   = np.full_like(ton, meta_row['test_pressure']        / PRESSURE_MAX)
    on_time    = np.full_like(ton, meta_row['cumulated_on_time']    / ON_TIME_MAX)
    throughput = np.full_like(ton, meta_row['cumulated_throughput'] / THROUGHPUT_MAX)
    pulses     = np.full_like(ton, meta_row['cumulated_pulses']     / PULSES_MAX)

    # -- 11 维 one-hot (每时间步重复) --
    mode_onehot = encode_test_mode(meta_row['test_mode'])          # (11,)
    mode_tiled  = np.tile(mode_onehot, (len(ton), 1))              # (T, 11)

    # -- 拼接 16 维输入 --
    x = np.column_stack([
        ton, pressure, on_time, throughput, pulses,                 # 5 维
        mode_tiled,                                                 # 11 维
    ]).astype(np.float32)

    # -- 双输出标签 --
    thrust_norm = df['thrust'].values.astype(np.float32) / THRUST_SCALE
    mfr_norm    = df['mfr'].values.astype(np.float32)    / MFR_MAX
    y = np.column_stack([thrust_norm, mfr_norm]).astype(np.float32)

    # -- 截断 / 填充到固定长度 --
    if len(x) > seq_len:
        x, y = x[:seq_len], y[:seq_len]
    else:
        pad_len = seq_len - len(x)
        x = np.pad(x, ((0, pad_len), (0, 0)), mode='constant')
        y = np.pad(y, ((0, pad_len), (0, 0)), mode='constant')

    return x, y


# ============================================================
# 快速验证脚本
# ============================================================
def validate():
    """加载 5 个不同 test_mode 的文件，打印 shape 和数值范围，确认无 NaN"""
    metadata_path = "data/metadata.csv"
    train_dir     = "data/dataset/dataset/train/"

    if not os.path.exists(metadata_path):
        print("[WARN] data/metadata.csv not found, check data directory")
        return

    df_meta = pd.read_csv(metadata_path)
    df_meta['filename'] = df_meta['filename'].str.strip()

    # 每种 test_mode 取 1 个文件
    seen_modes = set()
    test_files = []
    for _, row in df_meta.iterrows():
        mode = row['test_mode']
        if mode not in seen_modes:
            fpath = os.path.join(train_dir, row['filename'])
            if os.path.exists(fpath):
                test_files.append((row, fpath))
                seen_modes.add(mode)
        if len(test_files) >= 5:
            break

    print(f"=== data_pipeline_v2 validation: {len(test_files)} files ===\n")
    all_ok = True
    for meta_row, fpath in test_files:
        x, y = load_sequence(fpath, meta_row)
        has_nan = np.isnan(x).any() or np.isnan(y).any()
        status = "OK" if not has_nan else "NaN!"
        if has_nan:
            all_ok = False

        print(f"  {meta_row['filename']}")
        print(f"    mode={meta_row['test_mode']:15s}  "
              f"x={str(x.shape):12s}  y={str(y.shape):12s}  "
              f"x_range=[{x.min():.3f}, {x.max():.3f}]  "
              f"y_range=[{y.min():.3f}, {y.max():.3f}]  {status}")

    print(f"\n{'All passed' if all_ok else 'NaN detected - check data'}")


if __name__ == "__main__":
    validate()
