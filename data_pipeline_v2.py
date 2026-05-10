"""
data_pipeline_v2.py — 17维输入（5连续 + vl + 11 one-hot） + 双输出预处理
M 负责，dev-m 分支
接口约定：Z 通过 load_sequence() 和 get_test_mode_list() 获取数据，不重复写预处理逻辑。
"""

import os
import numpy as np
import pandas as pd

# ============================================================
# 归一化系数（基于全量 metadata 探索确认，2026-05-08）
# ============================================================
THRUST_SCALE   = 8.0       # thrust / 8.0 → [0, ~0.88]  (实测正常 max≈7N, 全局 outlier=23N→2.9, 安全)
MFR_MAX        = 2000.0    # mfr   / 2000 → [0, ~0.81]  (实测 max=1612.72, 向上取整留 headroom)
INPUT_DIM      = 17        # 5 continuous + vl + 11 one-hot (不含 SN embedding)
PRESSURE_MAX   = 25.0      # test_pressure / 25.0 → [0.2, 0.96]  (实测 max=24bar)
ON_TIME_MAX    = 8.0       # cumulated_on_time / 8.0 → [0, ~0.92]  (实测 max=7.37h)
THROUGHPUT_MAX = 25.0      # cumulated_throughput / 25.0 → [0, ~0.90]  (实测 max=22.54)
PULSES_MAX     = 25000.0   # cumulated_pulses / 25000 → [0, ~0.95]  (实测 max=23653)
SN_COUNT       = 24        # SN01–SN24 → index 0–23
UNKNOWN_SN_ID  = 24        # unseen thruster token (for test-set generalization)
SN_EMB_DIM     = 8         # SN embedding 维度，拼接到每个时间步

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
# 内部：从 DataFrame 切片构建特征（load_sequence 和 load_sequence_window 共用）
# ============================================================
def _build_features(df_slice, meta_row):
    """给定 DataFrame 切片（已按窗口截取），返回 x, y, labels, sn_id"""
    ton        = df_slice['ton'].values.astype(np.float32)
    vl         = df_slice['vl'].values.astype(np.float32)           # [0, 2]
    pressure   = np.full_like(ton, meta_row['test_pressure']        / PRESSURE_MAX)
    on_time    = np.full_like(ton, meta_row['cumulated_on_time']    / ON_TIME_MAX)
    throughput = np.full_like(ton, meta_row['cumulated_throughput'] / THROUGHPUT_MAX)
    pulses     = np.full_like(ton, meta_row['cumulated_pulses']     / PULSES_MAX)

    mode_onehot = encode_test_mode(meta_row['test_mode'])          # (11,)
    mode_tiled  = np.tile(mode_onehot, (len(ton), 1))              # (T, 11)

    x = np.column_stack([
        ton, vl, pressure, on_time, throughput, pulses,             # 6 维
        mode_tiled,                                                 # 11 维
    ]).astype(np.float32)

    thrust_norm = df_slice['thrust'].values.astype(np.float32) / THRUST_SCALE
    mfr_norm    = df_slice['mfr'].values.astype(np.float32)    / MFR_MAX
    y = np.column_stack([thrust_norm, mfr_norm]).astype(np.float32)

    raw_labels = df_slice['anomaly_code'].values
    labels = np.nan_to_num(raw_labels, nan=0.0).astype(np.int32)

    sn_id = int(meta_row['sn']) - 1  # SN01→0, SN24→23

    return x, y, labels, sn_id


# ============================================================
# 核心 API
# ============================================================
def load_sequence_window(filepath, meta_row, start=0, seq_len=200):
    """
    从指定起点 start 读取 CSV 窗口，不做截断/填充。
    供滑动窗口 Dataset 使用，Dataset 保证 T >= seq_len。

    返回
    ----
    x, y, labels, sn_id  各自长度 = min(seq_len, T - start)
    """
    df = pd.read_csv(filepath)
    end = min(start + seq_len, len(df))
    return _build_features(df.iloc[start:end], meta_row)


def load_sequence(filepath, meta_row, seq_len=200):
    """
    加载完整 CSV 文件前 seq_len 步，不足则零填充。

    返回
    ----
    x      : np.ndarray  [seq_len, 17]
    y      : np.ndarray  [seq_len, 2]
    labels : np.ndarray  [seq_len]
    sn_id  : int         0–23 (SN01–SN24)
    """
    x, y, labels, sn_id = load_sequence_window(filepath, meta_row, start=0, seq_len=seq_len)

    if len(x) < seq_len:
        pad_len = seq_len - len(x)
        x      = np.pad(x,      ((0, pad_len), (0, 0)), mode='constant')
        y      = np.pad(y,      ((0, pad_len), (0, 0)), mode='constant')
        labels = np.pad(labels, (0, pad_len),            mode='constant')
    elif len(x) > seq_len:
        x, y, labels = x[:seq_len], y[:seq_len], labels[:seq_len]

    return x, y, labels, sn_id


# ============================================================
# 元信息导出（5/10 训练完成后跑一次，生成 outputs/predictions/v2/meta_info.json 供 Z 使用）
# ============================================================
def save_meta_info(output_path="outputs/predictions/v2/meta_info.json"):
    """把归一化系数写成 JSON 供 Z 的评估脚本反归一化使用"""
    import json
    info = {
        "thrust_scale":   THRUST_SCALE,
        "mfr_max":        MFR_MAX,
        "pressure_max":   PRESSURE_MAX,
        "on_time_max":    ON_TIME_MAX,
        "throughput_max": THROUGHPUT_MAX,
        "pulses_max":     PULSES_MAX,
        "test_mode_list": TEST_MODE_LIST,
        "input_dim":      17,
        "output_dim":     2,
        "seq_len":        200,
        "sn_count":       SN_COUNT,
        "unknown_sn_id":  UNKNOWN_SN_ID,
        "sn_emb_dim":     SN_EMB_DIM,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"meta_info saved → {output_path}")


# ============================================================
# 快速验证脚本
# ============================================================
def validate():
    """加载全部 11 种 test_mode 的文件，打印 shape、数值范围和异常标签统计"""
    metadata_path = "data/metadata.csv"
    train_dir     = "data/dataset/dataset/train/"

    if not os.path.exists(metadata_path):
        print("[WARN] data/metadata.csv not found, check data directory")
        return

    df_meta = pd.read_csv(metadata_path)
    df_meta['filename'] = df_meta['filename'].str.strip()

    # 每种 test_mode 取 1 个文件，覆盖全部 11 种
    seen_modes = set()
    test_files = []
    for _, row in df_meta.iterrows():
        mode = row['test_mode']
        if mode not in seen_modes:
            fpath = os.path.join(train_dir, row['filename'])
            if os.path.exists(fpath):
                test_files.append((row, fpath))
                seen_modes.add(mode)
        if len(test_files) >= len(TEST_MODE_LIST):
            break

    print(f"=== data_pipeline_v2 validation: {len(test_files)} files ({len(seen_modes)} modes) ===\n")
    all_ok = True
    for meta_row, fpath in test_files:
        x, y, labels, sn_id = load_sequence(fpath, meta_row)
        has_nan = np.isnan(x).any() or np.isnan(y).any()
        n_anomaly = (labels != 0).sum()
        status = "OK" if not has_nan else "NaN!"
        if has_nan:
            all_ok = False

        print(f"  {meta_row['filename']}")
        print(f"    mode={meta_row['test_mode']:15s}  sn_id={sn_id}  "
              f"x={str(x.shape):12s}  y={str(y.shape):12s}  "
              f"x_range=[{x.min():.3f}, {x.max():.3f}]  "
              f"y_range=[{y.min():.3f}, {y.max():.3f}]  "
              f"anomaly_steps={n_anomaly}/{len(labels)}  {status}")

    print(f"\n{'All passed' if all_ok else 'NaN detected - check data'}")

    # 额外验证：找异常在前 200 步以内的文件，确认 labels 提取正确
    df_anom = df_meta[df_meta['anomalous'] == True]
    found_with_labels = False
    for _, row in df_anom.iterrows():
        fpath = os.path.join(train_dir, row['filename'])
        if not os.path.exists(fpath):
            fpath = os.path.join("data/dataset/dataset/test/", row['filename'])
        if not os.path.exists(fpath):
            continue
        _, _, labels_anom, _ = load_sequence(fpath, row)
        n = (labels_anom != 0).sum()
        if n > 0:
            print(f"\n  [anomaly check] {row['filename']}")
            print(f"    anomaly_steps={n}/{len(labels_anom)}  OK - labels detected")
            found_with_labels = True
            break
        else:
            print(f"\n  [anomaly check] {row['filename']}")
            print(f"    anomaly_steps=0/{len(labels_anom)}  "
                  f"skipped (anomaly beyond step 200, trying next file...)")

    if not found_with_labels:
        print("\n  [anomaly check] WARN: no file with anomaly in first 200 steps found "
              "- check anomaly_code column")


if __name__ == "__main__":
    validate()
