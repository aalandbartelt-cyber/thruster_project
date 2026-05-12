import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import onnxruntime as ort
from model_train_v2 import DualOutputLSTM  # M 的模型类，你只读不修改

# 全局设置
DEVICE = 'cpu'
MODEL_PATH = "outputs/models/v2/dual_output_lstm_v2.pth"
ONNX_PATH = "dual_output_lstm_v2.onnx"

def export_onnx(model, save_path=ONNX_PATH):
    """导出 ONNX 格式"""
    dummy = torch.randn(1, 200, 17).to(DEVICE)
    torch.onnx.export(
        model,
        dummy,
        save_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=14
    )
    print(f"✅ ONNX 导出完成：{save_path}")

def benchmark_onnx(onnx_path, n_warmup=10, n_test=100):
    """测试 ONNX FP32 推理延迟"""
    sess = ort.InferenceSession(onnx_path)
    dummy = np.random.randn(1, 200, 17).astype(np.float32)

    # 预热
    for _ in range(n_warmup):
        sess.run(None, {'input': dummy})

    # 计时
    start = time.time()
    for _ in range(n_test):
        sess.run(None, {'input': dummy})
    lat_ms = (time.time() - start) / n_test * 1000
    print(f"⏱️ ONNX FP32 推理延迟：{lat_ms:.2f} ms")
    return lat_ms

def pytorch_fp16_benchmark(model, n_warmup=10, n_test=100):
    """测试 PyTorch FP16 推理延迟（CPU 上模拟，实际无加速，仅供对比）"""
    model = model.half().eval()
    dummy = torch.randn(1, 200, 17).half().to(DEVICE)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
        start = time.time()
        for _ in range(n_test):
            model(dummy)
        lat_ms = (time.time() - start) / n_test * 1000
    print(f"⏱️ PyTorch FP16 推理延迟：{lat_ms:.2f} ms")
    return lat_ms

def main():
    print("===== ONNX 导出与性能对比 =====")

    # 1. 加载 M 训练好的模型（只读）
    model = DualOutputLSTM().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # 2. 导出 ONNX
    export_onnx(model)

    # 3. 基准测试
    lat_fp32 = benchmark_onnx(ONNX_PATH)
    lat_fp16 = pytorch_fp16_benchmark(model)  # FP16 在 CPU 上延迟可能更高，可注明

    # 4. 量化对比（动态量化，简化版）
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantized_path = "dual_output_lstm_int8.onnx"
        print("正在进行 INT8 动态量化...")
        quantize_dynamic(ONNX_PATH, quantized_path, weight_type=QuantType.QInt8)
        lat_int8 = benchmark_onnx(quantized_path)
        size_int8 = Path(quantized_path).stat().st_size / 1024  # KB
    except Exception as e:
        print(f"INT8 量化失败（可能是 onnxruntime 版本问题）：{e}")
        lat_int8 = None
        size_int8 = None
        quantized_path = None

    # 5. 模型大小
    size_fp32 = Path(ONNX_PATH).stat().st_size / 1024  # KB

    # 6. 汇总表格
    print("\n===== 量化对比表格 =====")
    print(f"{'格式':<10} {'大小(KB)':<10} {'延迟(ms)':<10}")
    print(f"{'FP32':<10} {size_fp32:<10.1f} {lat_fp32:<10.2f}")
    print(f"{'FP16':<10} {'-':<10} {lat_fp16:<10.2f} (PyTorch)")
    if lat_int8:
        print(f"{'INT8':<10} {size_int8:<10.1f} {lat_int8:<10.2f}")

    # 7. 绘制并保存对比图
    formats = ['FP32', 'FP16', 'INT8']
    lats = [lat_fp32, lat_fp16, lat_int8 if lat_int8 else 0]
    sizes = [size_fp32, 0, size_int8 if size_int8 else 0]

    fig, ax1 = plt.subplots(figsize=(8,5))
    color1 = 'tab:blue'
    ax1.bar(formats, lats, color=color1, alpha=0.6, label='Inference Latency (ms)')
    ax1.set_ylabel('Latency (ms)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.plot(formats, sizes, 'o-', color=color2, linewidth=2, label='Model Size (KB)')
    ax2.set_ylabel('Size (KB)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('ONNX Model Performance Comparison')
    fig.tight_layout()
    fig.savefig('outputs/figures/07_onnx_quantization_table.png', dpi=300, bbox_inches='tight')
    print("📊 量化对比图已保存：outputs/figures/07_onnx_quantization_table.png")
    plt.close()

if __name__ == "__main__":
    main()