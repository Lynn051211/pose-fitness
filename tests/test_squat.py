"""测试深蹲检测逻辑"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from pose_fitness.config import KP
from pose_fitness.exercises.squat import Squat


def make_kp_with_knee_angle(angle_deg):
    """
    创建模拟关键点，控制膝关节角度。
    右膝角度 = angle(hip, knee, ankle)
    """
    kp = np.zeros((17, 2), dtype=np.float32)
    # 设置基本坐标使所有关键点"可见"（非零）
    kp[:, 0] = 200
    kp[:, 1] = 200

    hip = np.array([0.0, 100.0])
    knee = np.array([0.0, 200.0])

    # 根据角度计算脚踝位置
    import math
    rad = math.radians(angle_deg)
    # 在 knee 处，hip-knee-ankle 夹角 = angle_deg
    # 设定 ankle 在 knee 下方且偏移
    upper_dir = hip - knee  # 指向 hip 的方向，即向上
    upper_dir = upper_dir / np.linalg.norm(upper_dir)
    # 旋转向量得到 ankle 方向
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    # 逆时针旋转 upper_dir
    ankle_dir = np.array([
        upper_dir[0] * cos_r - upper_dir[1] * sin_r,
        upper_dir[0] * sin_r + upper_dir[1] * cos_r
    ])
    ankle = knee + ankle_dir * 100.0

    kp[KP["right_hip"]] = hip
    kp[KP["right_knee"]] = knee
    kp[KP["right_ankle"]] = ankle

    return kp


class TestSquatDetection:
    """深蹲检测测试"""

    def test_standing_is_ready(self):
        """站立姿势(>150°)判为 READY"""
        kp = make_kp_with_knee_angle(170)  # 接近直立
        squat = Squat()
        assert squat.is_ready(kp)

    def test_deep_squat_is_down(self):
        """深蹲姿势(<90°)判为 DOWN"""
        kp = make_kp_with_knee_angle(80)
        squat = Squat()
        assert squat.is_down(kp)

    def test_mid_angle_not_down(self):
        """中间角度(120°)既不是 ready 也不是 down"""
        kp = make_kp_with_knee_angle(120)
        squat = Squat()
        assert not squat.is_ready(kp)
        assert not squat.is_down(kp)

    def test_none_keypoints(self):
        """None 关键点不应崩溃"""
        squat = Squat()
        assert not squat.is_ready(None)
        assert not squat.is_down(None)
        assert not squat.is_up(None)

    def test_exactly_90_degrees(self):
        """恰好90度 - 边界值"""
        kp = make_kp_with_knee_angle(90)
        squat = Squat()
        # is_down 条件是 < 90, 所以 90 度不算 down（刚好90不算下蹲到底）
        assert not squat.is_down(kp)


class TestSquatFormHints:
    """深蹲姿势提示测试"""

    def test_knee_over_toes(self):
        """膝盖超过脚尖应提示"""
        kp = np.zeros((17, 2), dtype=np.float32)
        kp[KP["right_knee"]] = [100, 200]
        kp[KP["right_ankle"]] = [50, 250]  # ankle x < knee x → knee[0] > ankle[0] + 30

        squat = Squat()
        hints = squat.get_form_hints(kp)
        assert "膝盖过脚尖" in hints

    def test_no_hints_when_good_form(self):
        """姿势正确时不提示"""
        kp = np.zeros((17, 2), dtype=np.float32)
        kp[KP["right_knee"]] = [100, 200]
        kp[KP["right_ankle"]] = [110, 250]  # ankle 在膝盖前方

        squat = Squat()
        hints = squat.get_form_hints(kp)
        assert "膝盖过脚尖" not in hints

    def test_none_keypoints_hints(self):
        """None关键点不抛异常"""
        squat = Squat()
        hints = squat.get_form_hints(None)
        assert hints == []
