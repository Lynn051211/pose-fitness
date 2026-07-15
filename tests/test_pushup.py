"""测试俯卧撑检测逻辑"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
import pytest
from pose_fitness.config import KP, angle_3pts
from pose_fitness.exercises.pushup import Pushup


def make_pushup_kp(elbow_angle_deg, body_deviation_deg=10):
    """
    创建模拟关键点，精确控制肘关节角度和身体角度。
    elbow_angle_deg: 肘关节角度 (shoulder-elbow-wrist)
    body_deviation_deg: 身体与完全直线的偏差（越小越水平，0=完全直线）
    """
    kp = np.zeros((17, 2), dtype=np.float32)
    kp[:, 0] = 200
    kp[:, 1] = 200

    # 身体：shoulder-hip-ankle 形成近似直线
    shoulder = np.array([0.0, 200.0])
    hip = np.array([150.0, 200.0])
    # ankle 微小偏移 → body_angle ≈ 180 - body_deviation
    ankle = np.array([300.0, 200.0 + math.tan(math.radians(body_deviation_deg)) * 300])

    # 肘部：构建指定角度
    # 固定 shoulder 和 elbow，计算对应 wrist
    elbow = np.array([50.0, 260.0])  # 肘在肩下方偏右
    shoulder_elbow = elbow - shoulder  # 方向向量

    # 从 elbow 到 wrist 的向量，与 shoulder_elbow 形成指定角度
    target_rad = math.radians(elbow_angle_deg)
    # 旋转 shoulder→elbow 的方向来得到 elbow→wrist
    # shoulder→elbow 的方向角 + π - target_rad = elbow→wrist 的方向角
    base_angle = math.atan2(shoulder_elbow[1], shoulder_elbow[0])
    wrist_angle = base_angle + math.pi - target_rad
    wrist_dir = np.array([math.cos(wrist_angle), math.sin(wrist_angle)])
    wrist = elbow + wrist_dir * 60.0  # 前臂长度60

    kp[KP["right_shoulder"]] = shoulder
    kp[KP["right_elbow"]] = elbow
    kp[KP["right_wrist"]] = wrist
    kp[KP["right_hip"]] = hip
    kp[KP["right_ankle"]] = ankle

    return kp


class TestPushupDetection:
    """俯卧撑检测测试"""

    def test_straight_arms_is_ready(self):
        """直臂(>140°) + 身体水平 → READY"""
        kp = make_pushup_kp(elbow_angle_deg=170, body_deviation_deg=10)
        pushup = Pushup()
        assert pushup.is_ready(kp)

    def test_bent_arms_is_down(self):
        """弯臂(<90°) + 身体水平 → DOWN"""
        kp = make_pushup_kp(elbow_angle_deg=80, body_deviation_deg=5)
        pushup = Pushup()
        assert pushup.is_down(kp)

    def test_bent_arms_bad_body_not_down(self):
        """弯臂但身体塌腰 → 不是 DOWN"""
        kp = make_pushup_kp(elbow_angle_deg=80, body_deviation_deg=35)
        pushup = Pushup()
        assert not pushup.is_down(kp)

    def test_mid_angle_not_ready_nor_down(self):
        """中间角度 → 既不是 ready 也不是 down"""
        kp = make_pushup_kp(elbow_angle_deg=120, body_deviation_deg=10)
        pushup = Pushup()
        assert not pushup.is_ready(kp)
        assert not pushup.is_down(kp)

    def test_none_keypoints(self):
        """None 不崩溃"""
        pushup = Pushup()
        assert not pushup.is_ready(None)
        assert not pushup.is_down(None)


class TestPushupFormHints:
    """俯卧撑姿势提示测试"""

    def test_body_sagging_hint(self):
        """身体塌腰 > 40度 → 提示"""
        kp = make_pushup_kp(elbow_angle_deg=80, body_deviation_deg=50)
        pushup = Pushup()
        hints = pushup.get_form_hints(kp)
        assert "身体塌腰" in hints

    def test_no_hint_good_form(self):
        """标准姿势无提示"""
        kp = make_pushup_kp(elbow_angle_deg=80, body_deviation_deg=10)
        pushup = Pushup()
        hints = pushup.get_form_hints(kp)
        assert "身体塌腰" not in hints

    def test_none_keypoints_hints(self):
        pushup = Pushup()
        assert pushup.get_form_hints(None) == []
