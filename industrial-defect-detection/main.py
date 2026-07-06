"""
工业表面缺陷检测系统 — 全流程入口
顺序执行：合成数据 → 数据增强 → 训练 → 导出 ONNX → 推理验证
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def step1_generate_data():
    print("\n" + "=" * 60)
    print("  STEP 1/6: 生成合成缺陷数据")
    print("=" * 60)
    from data.synthetic_generator import generate_dataset
    from config.settings import DATA_DIR, SYNTHETIC_NUM_TRAIN, SYNTHETIC_NUM_VAL

    os.makedirs(os.path.join(DATA_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "val"), exist_ok=True)

    generate_dataset(os.path.join(DATA_DIR, "train"), SYNTHETIC_NUM_TRAIN)
    generate_dataset(os.path.join(DATA_DIR, "val"), SYNTHETIC_NUM_VAL)


def step2_augment_data():
    print("\n" + "=" * 60)
    print("  STEP 2/6: 数据增强 (Mosaic + Copy-Paste)")
    print("=" * 60)
    from data.augmentation import run_augmentation
    from config.settings import DATA_DIR, AUG_DIR

    train_img = os.path.join(DATA_DIR, "train", "images")
    train_lbl = os.path.join(DATA_DIR, "train", "labels")
    out_img = os.path.join(AUG_DIR, "train", "images")
    out_lbl = os.path.join(AUG_DIR, "train", "labels")

    run_augmentation(train_img, train_lbl, out_img, out_lbl)


def step3_train():
    print("\n" + "=" * 60)
    print("  STEP 3/6: 训练 YOLOv8n")
    print("=" * 60)
    from train.train import train
    train()


def step4_export():
    print("\n" + "=" * 60)
    print("  STEP 4/6: 导出 ONNX 模型")
    print("=" * 60)
    from export.export_onnx import export_to_onnx
    export_to_onnx()


def step5_inference():
    print("\n" + "=" * 60)
    print("  STEP 5/6: ONNX 推理验证")
    print("=" * 60)
    from inference.inference import infer_onnx
    from config.settings import DATA_DIR

    # 从验证集随机选一张
    val_dir = os.path.join(DATA_DIR, "val", "images")
    test_img = ""
    if os.path.exists(val_dir):
        imgs = [f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.png'))]
        if imgs:
            test_img = os.path.join(val_dir, imgs[0])

    result = infer_onnx(test_img)
    if result:
        print(f"  检测到 {result['detections']} 个缺陷, 推理耗时 {result['time_ms']:.2f} ms")


def step6_done():
    print("\n" + "=" * 60)
    print("  STEP 6/6: 完成！")
    print("=" * 60)
    print(f"""
  项目输出:
  ├── models/defect_detector.pt      ← PyTorch 训练模型
  ├── models/defect_detector.onnx    ← ONNX 推理模型
  ├── output/inference_result.jpg    ← 推理可视化结果
  └── output/train_results/          ← 训练日志与权重

  快速测试:
    python inference/inference.py <图片路径>

  容器化部署:
    docker build -t defect-detector .
    docker run defect-detector
""")


def main():
    parser = argparse.ArgumentParser(description="工业表面缺陷检测系统 Demo")
    parser.add_argument("--skip-generate", action="store_true", help="跳过合成数据生成")
    parser.add_argument("--skip-augment", action="store_true", help="跳过数据增强")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练")
    parser.add_argument("--skip-export", action="store_true", help="跳过模型导出")
    parser.add_argument("--inference-only", type=str, metavar="IMG",
                        help="仅运行推理，指定图像路径")
    args = parser.parse_args()

    if args.inference_only:
        from inference.inference import infer_onnx
        infer_onnx(args.inference_only)
        return

    if not args.skip_generate:
        step1_generate_data()
    else:
        print("[SKIP] 合成数据生成")

    if not args.skip_augment:
        step2_augment_data()
    else:
        print("[SKIP] 数据增强")

    if not args.skip_train:
        step3_train()
    else:
        print("[SKIP] 模型训练")

    if not args.skip_export:
        step4_export()
    else:
        print("[SKIP] 模型导出")

    step5_inference()
    step6_done()


if __name__ == "__main__":
    main()
