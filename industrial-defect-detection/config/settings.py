"""全局配置：6 种缺陷类型、路径、超参数"""

# 6 种金属表面典型缺陷
DEFECT_CLASSES = [
    "crack",       # 裂纹
    "scratch",     # 划痕
    "pit",         # 凹坑
    "inclusion",   # 夹杂
    "oxidation",   # 氧化皮
    "stain",       # 斑痕
]

NUM_CLASSES = len(DEFECT_CLASSES)

# ---- 路径 ----
DATA_DIR = "data/raw"                # 原始合成图像存放
AUG_DIR = "data/augmented"           # 增强后图像存放
MODEL_DIR = "models"
OUTPUT_DIR = "output"

# YOLOv8 数据集配置
YAML_PATH = "config/defect_dataset.yaml"

# ONNX 导出
ONNX_MODEL_PATH = f"{MODEL_DIR}/defect_detector.onnx"
PT_MODEL_PATH = f"{MODEL_DIR}/defect_detector.pt"

# ---- 合成数据参数 ----
SYNTHETIC_NUM_TRAIN = 300   # 训练集合成图片数
SYNTHETIC_NUM_VAL = 60      # 验证集合成图片数
IMAGE_SIZE = 640             # 合成图像的原始尺寸
MODEL_INPUT_SIZE = 160        # 模型输入尺寸（训练/导出/推理统一）

# ---- 训练参数 ----
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.001
IMG_SIZE_TRAIN = 160  # 需与 MODEL_INPUT_SIZE 一致
WORKERS = 2

# ---- 推理参数 ----
CONF_THRESH = 0.25   # 置信度阈值
IOU_THRESH = 0.45    # NMS IoU 阈值
