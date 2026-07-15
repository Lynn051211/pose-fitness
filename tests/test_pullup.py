"""测试引体向上检测逻辑"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from pose_fitness.config import KP
from pose_fitness.exercises.pullup import Pullup


def make_pullup_kp(wrist_y, shoulder_y):
    """创建关键点，控制右手腕 y 和右肩 y"""
    kp = np.zeros((17, 2), dtype=np.float32)
    kp[:, 0] = 200
    kp[:, 1] = 300

    kp[KP["right_shoulder"]] = [200.0, shoulder_y]
    kp[KP["right_wrist"]] = [200.0, wrist_y]
    kp[KP["right_hip"]] = [200.0, 400.0]
    kp[KP["left_hip"]] = [200.0, 400.0]

    return kp


class TestPullupDetection:
    """引体向上检测测试"""

    def test_wrist_below_shoulder_is_ready(self):
        """腕低于肩（悬挂状态）→ READY"""
        kp = make_pullup_kp(wrist_y=350, shoulder_y=300)
        pullup = Pullup()
        assert pullup.is_ready(kp)

    def test_wrist_above_shoulder_is_down(self):
        """腕高于肩（拉起状态）→ DOWN"""
        kp = make_pullup_kp(wrist_y=250, shoulder_y=300)
        pullup = Pullup()
        assert pullup.is_down(kp)

    def test_wrist_near_shoulder_neither(self):
        """腕接近肩（15像素内）→ 既不是ready也不是down"""
        kp = make_pullup_kp(wrist_y=305, shoulder_y=300)
        pullup = Pullup()
        assert not pullup.is_ready(kp)
        assert not pullup.is_down(kp)

    def test_none_keypoints(self):
        """None 不崩溃"""
        pullup = Pullup()
        assert not pullup.is_ready(None)
        assert not pullup.is_down(None)


class TestPullupFormHints:
    """引体向上姿势提示测试"""

    def test_body_swing_hint(self):
        """身体大幅摆动提示"""
        kp = np.zeros((17, 2), dtype=np.float32)
        kp[KP["right_hip"]] = [100, 400]
        kp[KP["left_hip"]] = [200, 400]  # x 差距 100 > 50

        pullup = Pullup()
        hints = pullup.get_form_hints(kp)
        assert "减少摆动" in hints

    def test_no_swing_hint(self):
        """身体稳定不提示"""
        kp = np.zeros((17, 2), dtype=np.float32)
        kp[KP["right_hip"]] = [100, 400]
        kp[KP["left_hip"]] = [130, 400]  # x 差距 30 < 50

        pullup = Pullup()
        hints = pullup.get_form_hints(kp)
        assert "减少摆动" not in hints

    def test_none_keypoints_hints(self):
        pullup = Pullup()
        assert pullup.get_form_hints(None) == []


class TestPullupSymmetry:
    """双侧对称检测测试"""

    def test_bilateral_averaging(self):
        """验证双侧平均检测 — 左右手取平均位置判断"""
        kp = make_pullup_kp(wrist_y=250, shoulder_y=300)
        # 左侧腕在很低的位置，拉起右侧无法抵消
        kp[KP["left_wrist"]] = [200.0, 500.0]   # 左腕很低
        kp[KP["left_shoulder"]] = [200.0, 300.0]  # 左肩同高

        pullup = Pullup()
        # 左腕-左肩 = 200, 右腕-右肩 = -50, 平均 = 75 > 15 → READY（悬挂）
        # 不是 DOWN（因为左侧拉高了平均值）
        assert pullup.is_ready(kp)

    def test_both_arms_up_is_down(self):
        """双手都拉起 → DOWN"""
        kp = np.zeros((17, 2), dtype=np.float32)
        kp[:, 0] = 200
        kp[:, 1] = 300
        # 双手腕都高于肩
        kp[KP["right_wrist"]] = [200.0, 250.0]
        kp[KP["left_wrist"]] = [200.0, 250.0]
        kp[KP["right_shoulder"]] = [200.0, 300.0]
        kp[KP["left_shoulder"]] = [200.0, 300.0]

        pullup = Pullup()
        assert pullup.is_down(kp)
