"""YOLOv8 姿态检测封装"""

import numpy as np
from ultralytics import YOLO
from .config import MODEL_PATH, IMGSZ, CONF, KP

model = None


def init():
    global model
    model = YOLO(MODEL_PATH)
    return model


def detect(frame: np.ndarray):
    """返回 (17, 2) 关键点数组，无人时返回 None"""
    results = model(frame, imgsz=IMGSZ, conf=CONF, verbose=False)
    if results[0].keypoints is None or results[0].keypoints.xy is None:
        return None
    kp = results[0].keypoints.xy.cpu().numpy()
    if len(kp) == 0:
        return None
    # 返回第一个人的关键点 (17, 2)
    return kp[0]


def has_full_body(kp: np.ndarray, conf: np.ndarray = None) -> bool:
    """检查是否检测到完整身体（双肩+双髋可见）"""
    required = [KP["left_shoulder"], KP["right_shoulder"],
                KP["left_hip"], KP["right_hip"]]
    return all(kp[i][0] > 0 and kp[i][1] > 0 for i in required)
