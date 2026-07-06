"""
YOLOv8n 训练脚本
使用合成缺陷数据训练 6 类缺陷检测模型。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    YAML_PATH, MODEL_DIR, BATCH_SIZE, EPOCHS, IMG_SIZE_TRAIN,
    LEARNING_RATE, WORKERS, DEFECT_CLASSES, DATA_DIR, AUG_DIR,
)


def write_dataset_yaml():
    """生成 YOLOv8 数据集配置文件"""
    import yaml

    # 优先使用增强后的数据，否则用原始合成数据
    train_img = os.path.join(AUG_DIR, "train", "images")
    if not os.path.exists(train_img) or len(os.listdir(train_img)) == 0:
        train_img = os.path.join(DATA_DIR, "train", "images")

    val_img = os.path.join(DATA_DIR, "val", "images")

    config = {
        "path": os.path.abspath("."),
        "train": os.path.abspath(train_img),
        "val": os.path.abspath(val_img),
        "nc": len(DEFECT_CLASSES),
        "names": DEFECT_CLASSES,
    }

    os.makedirs(os.path.dirname(YAML_PATH), exist_ok=True)
    with open(YAML_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"数据集配置已写入 {YAML_PATH}")
    return YAML_PATH


def train():
    """主训练流程"""
    from ultralytics import YOLO

    # 生成数据集配置
    yaml_path = write_dataset_yaml()

    # 加载预训练 YOLOv8n 模型
    print("加载 YOLOv8n 预训练模型...")
    model = YOLO("yolov8n.pt")

    # 训练
    print(f"开始训练：{EPOCHS} epochs, batch={BATCH_SIZE}, lr={LEARNING_RATE}")
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE_TRAIN,
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        workers=WORKERS,
        device="cpu",   # CPU 训练，无 GPU 也能跑
        patience=10,    # 早停
        save=True,
        save_period=5,
        project="output",
        name="train_results",
        exist_ok=True,
        verbose=True,
    )

    # 保存最终模型
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_DIR + "/defect_detector.pt")
    print(f"模型已保存至 {MODEL_DIR}/defect_detector.pt")

    # 验证
    print("验证模型...")
    metrics = model.val()
    print(f"验证结果: mAP@0.5={metrics.box.map50:.4f}, mAP@0.5:0.95={metrics.box.map:.4f}")

    return results


if __name__ == "__main__":
    train()
