"""所有可调参数和配置"""

import numpy as np
import threading

# ---- YOLO COCO 关键点索引 ----
KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

# ---- 相机 ----
CAMERA_ID = 0

# ---- YOLO ----
MODEL_PATH = "yolov8n-pose.pt"
IMGSZ = 320
CONF = 0.5

# ---- MediaPipe Face Mesh ----
FACE_MODEL_PATH = None  # None = 自动下载内置模型
FACE_CONF = 0.5
# 虹膜关键点索引
IRIS_LEFT = [468, 469, 470, 471, 472]
IRIS_RIGHT = [473, 474, 475, 476, 477]
# 眼角索引（用于计算虹膜相对位置）
EYE_LEFT_INNER = 133
EYE_LEFT_OUTER = 33
EYE_RIGHT_INNER = 362
EYE_RIGHT_OUTER = 263
# 走神阈值
GAZE_AWAY_SEC = 3.0          # 连续不看屏幕 N 秒 → 提醒
GAZE_IRIS_RATIO_THRESH = 0.25  # 虹膜位置偏离中心 > 25% 判为走神

# ---- UI 颜色 (B, G, R) ----
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_ORANGE = (0, 165, 255)

# ---- Web 推流 ----
WEB_PORT = 8080
latest_frame = None          # 共享帧（主循环写入，API 读取）
frame_lock = threading.Lock()

# ---- 角度计算 ----
def angle_3pts(a, b, c):
    """计算 ∠ABC，以 b 为顶点，返回角度（度）"""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
