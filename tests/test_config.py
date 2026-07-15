"""测试 config.py 中的 angle_3pts 和关键点索引"""
import sys
import os
import numpy as np
import pytest

# 确保可以从 tests/ 目录导入 pose_fitness
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pose_fitness.config import angle_3pts, KP


class TestAngle3pts:
    """角度计算函数测试"""

    def test_right_angle(self):
        """直角 (90度)"""
        a = np.array([1, 0])
        b = np.array([0, 0])
        c = np.array([0, 1])
        result = angle_3pts(a, b, c)
        assert abs(result - 90.0) < 0.01, f"期望 90°, 得到 {result}°"

    def test_straight_line(self):
        """共线 (180度)"""
        a = np.array([1, 0])
        b = np.array([0, 0])
        c = np.array([-1, 0])
        result = angle_3pts(a, b, c)
        assert abs(result - 180.0) < 0.01, f"期望 180°, 得到 {result}°"

    def test_acute_angle(self):
        """锐角 (45度)"""
        a = np.array([1, 0])
        b = np.array([0, 0])
        c = np.array([1, 1])
        result = angle_3pts(a, b, c)
        assert abs(result - 45.0) < 0.01, f"期望 45°, 得到 {result}°"

    def test_obtuse_angle(self):
        """钝角 (135度)"""
        a = np.array([1, 0])
        b = np.array([0, 0])
        c = np.array([-1, 1])
        result = angle_3pts(a, b, c)
        assert abs(result - 135.0) < 0.01, f"期望 135°, 得到 {result}°"

    def test_zero_angle(self):
        """零度角 (同向点)"""
        a = np.array([1, 0])
        b = np.array([0, 0])
        c = np.array([1, 0])
        result = angle_3pts(a, b, c)
        assert abs(result) < 0.01, f"期望 0°, 得到 {result}°"

    def test_same_points(self):
        """两个点相同 - 应该不崩溃"""
        a = np.array([0, 0])
        b = np.array([0, 0])
        c = np.array([1, 1])
        result = angle_3pts(a, b, c)
        # len(ba) = 0, 除以 (0 + 1e-8), 所以 arccos(0/1e-8) → 取决于浮点
        # 两个相同点导致零向量，除以极小值 → cos 接近0 → 约90度
        # 这是正常行为（不会崩溃）
        assert isinstance(result, float)

    def test_2d_vs_3d(self):
        """2D坐标处理"""
        a = np.array([0, 1, 0])
        b = np.array([0, 0, 0])
        c = np.array([1, 0, 0])
        result = angle_3pts(a, b, c)
        assert abs(result - 90.0) < 0.01, f"期望 90°, 得到 {result}°"

    def test_negative_coordinates(self):
        """负坐标"""
        a = np.array([-5, 0])
        b = np.array([-2, 0])
        c = np.array([-2, 3])
        result = angle_3pts(a, b, c)
        assert abs(result - 90.0) < 0.01, f"期望 90°, 得到 {result}°"


class TestKeypointIndices:
    """关键点索引测试"""

    def test_kp_has_17_points(self):
        """应该恰好17个关键点"""
        assert len(KP) == 17, f"期望 17 个关键点, 得到 {len(KP)}"

    def test_kp_indices_contiguous(self):
        """关键点索引应连续 0-16"""
        values = sorted(KP.values())
        assert values == list(range(17)), f"关键点索引不连续: {values}"

    def test_required_keypoints_exist(self):
        """必需的关键点存在"""
        required = ["nose", "left_shoulder", "right_shoulder",
                     "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                     "left_hip", "right_hip", "left_knee", "right_knee",
                     "left_ankle", "right_ankle", "left_eye", "right_eye",
                     "left_ear", "right_ear"]
        for k in required:
            assert k in KP, f"缺少关键点: {k}"

    def test_left_right_symmetry(self):
        """左右关键点应成对出现"""
        left_keys = [k for k in KP if k.startswith("left_")]
        right_keys = [k for k in KP if k.startswith("right_")]
        assert len(left_keys) == len(right_keys), \
            f"左右关键点数量不一致: left={len(left_keys)}, right={len(right_keys)}"
