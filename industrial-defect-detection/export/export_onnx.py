"""
模型导出脚本：PyTorch (.pt) → ONNX (.onnx)
支持后续 ONNX Runtime 推理或 TensorRT 加速。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import PT_MODEL_PATH, ONNX_MODEL_PATH, MODEL_INPUT_SIZE


def export_to_onnx(pt_path: str = None, onnx_path: str = None):
    """将 YOLOv8 .pt 模型导出为 .onnx 格式"""
    from ultralytics import YOLO

    pt_path = pt_path or PT_MODEL_PATH
    onnx_path = onnx_path or ONNX_MODEL_PATH

    if not os.path.exists(pt_path):
        print(f"错误：模型文件不存在 {pt_path}")
        print("请先运行训练脚本: python train/train.py")
        return None

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    print(f"加载模型: {pt_path}")
    model = YOLO(pt_path)

    print(f"导出 ONNX (imgsz={MODEL_INPUT_SIZE})...")
    # ultralytics 内置 ONNX 导出
    model.export(
        format="onnx",
        imgsz=MODEL_INPUT_SIZE,
        half=False,       # FP32 精度（CPU 推理更稳定）
        simplify=True,    # 简化模型图
        opset=12,
        workspace=4.0,
    )

    # 导出的 onnx 默认保存在 pt 同目录
    # ultralytics 导出路径: models/defect_detector.onnx
    default_onnx = os.path.join(os.path.dirname(pt_path),
                                os.path.splitext(os.path.basename(pt_path))[0] + ".onnx")
    if os.path.exists(default_onnx) and default_onnx != onnx_path:
        import shutil
        shutil.move(default_onnx, onnx_path)

    # 验证 ONNX 模型
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX 模型验证通过")
    except ImportError:
        print("(安装 onnx 可进行模型验证: pip install onnx)")
    except Exception as e:
        print(f"ONNX 验证警告: {e}")

    file_size = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"导出完成: {onnx_path} ({file_size:.1f} MB)")
    return onnx_path


if __name__ == "__main__":
    export_to_onnx()
