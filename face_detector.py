"""MediaPipe Face Mesh 面部检测 + 虹膜追踪"""

import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from .config import FACE_MODEL_PATH, FACE_CONF, IRIS_LEFT, IRIS_RIGHT

_landmarker = None


def init():
    global _landmarker
    opts = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=FACE_CONF,
        min_tracking_confidence=FACE_CONF,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    _landmarker = FaceLandmarker.create_from_options(opts)


def detect(rgb_frame: np.ndarray):
    """检测面部关键点，返回 (478, 3) landmarks 或 None"""
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
    result = _landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    pts = result.face_landmarks[0]
    return np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)


def get_iris_ratio(landmarks, iris_idx):
    """虹膜中心相对眼角的水平偏移比（-1~1，0=正中）"""
    pts = landmarks[iris_idx]
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    return cx, cy


def gaze_direction(landmarks):
    """返回视线方向: center/left/right/up/down，和水平偏移比"""
    from .config import EYE_LEFT_INNER, EYE_LEFT_OUTER, EYE_RIGHT_INNER, EYE_RIGHT_OUTER

    # 左眼虹膜中心
    lx, ly = get_iris_ratio(landmarks, IRIS_LEFT)
    # 右眼虹膜中心
    rx, ry = get_iris_ratio(landmarks, IRIS_RIGHT)

    # 眼角范围
    left_inner = landmarks[EYE_LEFT_INNER][0]
    left_outer = landmarks[EYE_LEFT_OUTER][0]
    right_inner = landmarks[EYE_RIGHT_INNER][0]
    right_outer = landmarks[EYE_RIGHT_OUTER][0]

    # 左眼虹膜相对位置（-1=内眼角，1=外眼角）
    left_range = left_outer - left_inner
    left_ratio = (lx - left_inner) / (left_range + 1e-8) * 2 - 1 if left_range > 0 else 0

    # 右眼虹膜相对位置
    right_range = right_outer - right_inner
    right_ratio = (rx - right_inner) / (right_range + 1e-8) * 2 - 1 if right_range > 0 else 0

    # 平均水平偏移
    h_ratio = (left_ratio + right_ratio) / 2

    if abs(h_ratio) < 0.25:
        return "center", h_ratio
    elif h_ratio < -0.25:
        return "left", h_ratio
    elif h_ratio > 0.25:
        return "right", h_ratio
    return "center", h_ratio
