"""
评估模块最终结论：
- 单 thrust 残差阈值法：0.6N 阈值下 Recall=0.008，无法检测异常
- 融合 thrust+mfr 残差（取最大值）：Recall=0.992，但 Precision≈0
- 结论：正常点与异常点在残差幅值空间高度重叠，固定阈值法不适用。
  必须引入时序突变检测、MC Dropout 不确定性量化等方法，
  此为未来改进方向。
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc
)
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

############################################
# 1. 加载接口文件
############################################
def load_predictions(output_dir="outputs/predictions/v2"):
    """
    读取M交付的预测、真实值、异常标签和元信息
    """
    output_path = Path(output_dir)
    preds = np.load(output_path / "predictions_dual.npy")    # [N, seq_len, 2]
    targets = np.load(output_path / "targets_dual.npy")      # [N, seq_len, 2]
    labels = np.load(output_path / "anomaly_labels.npy")     # [N, seq_len]
    with open(output_path / "meta_info.json", "r") as f:
        meta = json.load(f)
    return preds, targets, labels, meta

############################################
# 2. 反归一化
############################################
def denormalize(preds, targets, meta):
    """
    根据元信息里的缩放系数，将归一化值转回物理单位
    """
    thrust_scale = meta["thrust_scale"]   # 8.0
    mfr_max = meta["mfr_max"]             # 2000.0

    preds_denorm = np.zeros_like(preds)
    targets_denorm = np.zeros_like(targets)
    preds_denorm[:, :, 0] = preds[:, :, 0] * thrust_scale    # thrust: N
    preds_denorm[:, :, 1] = preds[:, :, 1] * mfr_max         # mfr: mg/s
    targets_denorm[:, :, 0] = targets[:, :, 0] * thrust_scale
    targets_denorm[:, :, 1] = targets[:, :, 1] * mfr_max
    return preds_denorm, targets_denorm

############################################
# 3. 残差计算与阈值检测
############################################
def compute_residuals(preds, targets):
    """
    返回各点的绝对残差，只基于thrust（主要指标）
    你也可以同时计算mfr的残差
    """
    residual = np.abs(preds[:, :, 0] - targets[:, :, 0])  # thrust残差
    return residual

def threshold_detection(residual, threshold=0.6):
    """
    将残差超过阈值的点标记为预测异常
    """
    return (residual > threshold).astype(int)

############################################
# 4. 评估指标计算
############################################
def compute_metrics(y_pred, y_true):
    """
    计算Precision, Recall, F1（逐点），以及IoU（逐序列平均）
    y_pred, y_true: 形状 [N, seq_len]
    """
    # 展平计算逐点指标
    yp = y_pred.flatten()
    yt = y_true.flatten()
    precision, recall, f1, _ = precision_recall_fscore_support(yt, yp, average='binary', zero_division=0)

    # 计算IoU（逐序列）
    iou_list = []
    for i in range(y_pred.shape[0]):
        # 将连续异常段合并为区间
        def get_intervals(arr):
            intervals = []
            start = None
            for j, val in enumerate(arr):
                if val == 1 and start is None:
                    start = j
                elif val == 0 and start is not None:
                    intervals.append((start, j-1))
                    start = None
            if start is not None:
                intervals.append((start, len(arr)-1))
            return intervals

        pred_intervals = get_intervals(y_pred[i])
        true_intervals = get_intervals(y_true[i])

        # 计算并集和交集长度（简化版：计算所有异常点的重叠）
        pred_set = set(np.where(y_pred[i]==1)[0])
        true_set = set(np.where(y_true[i]==1)[0])
        intersection = len(pred_set & true_set)
        union = len(pred_set | true_set)
        iou = intersection / union if union > 0 else 1.0  # 如果都无异常，IoU=1
        iou_list.append(iou)

    avg_iou = np.mean(iou_list)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": avg_iou
    }

############################################
# 5. 按异常类型统计
############################################
def analyze_by_anomaly_type(preds_bin, true_labels, raw_labels, meta):
    """
    根据原始anomaly_code分类统计检测效果
    raw_labels: 形状 [N, seq_len]，中保留了0~31的原始编码
    """
    # 获取所有出现过的异常类型
    unique_codes = np.unique(raw_labels)
    results = {}
    for code in unique_codes:
        if code == 0:  # 正常点跳过
            continue
        # 找出该类型的所有位置
        mask = (raw_labels == code)
        if mask.sum() == 0:
            continue
        y_true_t = np.ones(mask.sum())   # 这些位置都是真异常
        y_pred_t = preds_bin[mask]
        prec, rec, f1, _ = precision_recall_fscore_support(y_true_t, y_pred_t, average='binary', zero_division=0)
        results[code] = {
            "count": int(mask.sum()),
            "precision": prec,
            "recall": rec,
            "f1": f1
        }
    return results

############################################
# 6. 绘图函数
############################################
def plot_confusion_matrix(y_pred, y_true, save_path=None):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = [0,1]
    plt.xticks(tick_marks, ['Normal', 'Anomaly'])
    plt.yticks(tick_marks, ['Normal', 'Anomaly'])
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_threshold_sensitivity(residuals, labels_bin, thresholds=[0.3,0.4,0.5,0.6,0.8], save_path=None):
    precs, recs, f1s = [], [], []
    for th in thresholds:
        pred = threshold_detection(residuals, threshold=th)
        metrics = compute_metrics(pred, labels_bin)
        precs.append(metrics['precision'])
        recs.append(metrics['recall'])
        f1s.append(metrics['f1'])
    plt.figure(figsize=(8,5))
    plt.plot(thresholds, precs, 'o-', label='Precision')
    plt.plot(thresholds, recs, 'o-', label='Recall')
    plt.plot(thresholds, f1s, 'o-', label='F1')
    plt.xlabel('Threshold (N)')
    plt.ylabel('Score')
    plt.title('Anomaly Detection Performance vs. Threshold')
    plt.legend()
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_residual_distribution(residuals, labels_bin, save_path=None):
    normal_res = residuals[labels_bin==0]
    anomaly_res = residuals[labels_bin==1]
    plt.figure(figsize=(8,5))
    plt.hist(normal_res, bins=50, alpha=0.6, label='Normal', color='green')
    plt.hist(anomaly_res, bins=50, alpha=0.6, label='Anomaly', color='red')
    plt.xlabel('Absolute Thrust Residual (N)')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

############################################
# 7. 主流程（用于mock与真实数据切换）
############################################
def main():
    # ---- Mock 数据模式（今天先用这个跑通） ----
    USE_MOCK = False  # 今晚拿到真实npy后改为 False
    if USE_MOCK:
        print("使用 Mock 数据验证流程...")
        N, seq_len = 50, 200
        # 生成假预测和真实值（归一化）
        np.random.seed(42)
        preds = np.random.randn(N, seq_len, 2) * 0.1
        targets = np.random.randn(N, seq_len, 2) * 0.1
        # 生成假异常标签（10%概率异常）
        labels = (np.random.rand(N, seq_len) < 0.1).astype(int)
        # 虚构一个meta
        meta = {"THRUST_SCALE": 8.0, "MFR_MAX": 2000.0}
        # 保存一份mock文件供测试
        Path("outputs/mock").mkdir(parents=True, exist_ok=True)
        np.save("outputs/mock/predictions_dual.npy", preds)
        np.save("outputs/mock/targets_dual.npy", targets)
        np.save("outputs/mock/anomaly_labels.npy", labels)
        with open("outputs/mock/meta_info.json", "w") as f:
            json.dump(meta, f)
        preds, targets, labels, meta = load_predictions("outputs/mock")
    else:
        # 真实数据路径
        preds, targets, labels, meta = load_predictions("outputs/predictions/v2")

    # 反归一化
    preds_denorm, targets_denorm = denormalize(preds, targets, meta)

    # 计算残差（只用thrust）
    residuals_thrust = compute_residuals(preds_denorm, targets_denorm)  # thrust 残差
    residuals_mfr = np.abs(preds_denorm[:, :, 1] - targets_denorm[:, :, 1])  # mfr 残差 (mg/s)
    residuals = np.maximum(residuals_thrust, residuals_mfr)  # 取较大值融合

    # 默认阈值检测
 # 默认阈值检测
    threshold = 0.6
    pred_bin = threshold_detection(residuals, threshold)

# 保留原始多类标签（用于后续异常类型分析）
    raw_labels = labels.copy()

# 将真实标签二值化（0=正常，非0=异常）以匹配二值预测
    labels_bin = (labels != 0).astype(int)

    # 计算指标
    metrics = compute_metrics(pred_bin, labels_bin)
    print("===== 评估结果（Thrust） =====")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1 Score:  {metrics['f1']:.3f}")
    print(f"Avg IoU:   {metrics['iou']:.3f}")

    # 按异常类型分析（仅当labels中含有非0原始码时）
    if not USE_MOCK:
        # 真实数据中labels可能是二值化后的，这里需要原始的anomaly_code。
        # M提供的anomaly_labels.npy已经是二值化后的，因此无法直接按31类拆分。
        # 解决方案：M额外提供一个raw_anomaly_codes.npy，或你直接从CSV重新提取。
        # 这里先给出代码框架，具体调用时确认。
        pass
    else:
        # Mock数据中labels就是二值，无法分类，跳过
        print("\nMock模式，跳过异常类型分析。")

    # 绘图
    fig_dir = Path("outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(pred_bin, labels, save_path=fig_dir/"06_confusion_matrix.png")
    plot_threshold_sensitivity(residuals, labels_bin, save_path=fig_dir/"05_threshold_analysis.png")
    plot_residual_distribution(residuals, labels_bin, save_path=fig_dir/"residual_dist.png")
    plt.show()

if __name__ == "__main__":
    main()