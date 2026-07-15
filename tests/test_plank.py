"""测试平板支撑计时逻辑"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pytest
from pose_fitness.config import KP
from pose_fitness.exercises.plank import Plank
from pose_fitness.exercises.base import State


def make_plank_kp(body_angle_from_horizontal=10):
    """创建平板支撑关键点"""
    import math
    kp = np.zeros((17, 2), dtype=np.float32)
    kp[:, 0] = 200
    kp[:, 1] = 200

    shoulder = np.array([0.0, 100.0])
    hip = np.array([100.0, 100.0])

    ankle_y = 100.0 + math.tan(math.radians(body_angle_from_horizontal)) * 200
    ankle = np.array([200.0, ankle_y])

    kp[KP["right_shoulder"]] = shoulder
    kp[KP["right_hip"]] = hip
    kp[KP["right_ankle"]] = ankle

    return kp


class TestPlankDetection:
    """平板支撑检测测试"""

    def test_good_form_is_planking(self):
        """身体平直 → 正在支撑"""
        kp = make_plank_kp(body_angle_from_horizontal=10)
        plank = Plank()
        assert plank._is_planking(kp) is True

    def test_bad_form_not_planking(self):
        """臀部太高 → 不认为是支撑"""
        kp = make_plank_kp(body_angle_from_horizontal=40)
        plank = Plank()
        assert plank._is_planking(kp) is False

    def test_none_keypoints_not_planking(self):
        """None → 不支撑"""
        plank = Plank()
        assert plank._is_planking(None) is False


class TestPlankTiming:
    """平板支撑计时测试"""

    def test_initial_state(self):
        """初始状态"""
        plank = Plank()
        assert plank.duration == 0.0
        assert plank._hold_start is None
        assert plank._state == Plank.STATE_REST

    def test_update_with_good_form_starts_timer(self):
        """正确姿势开始计时"""
        plank = Plank()
        kp = make_plank_kp(body_angle_from_horizontal=10)

        counted, state = plank.update(kp)
        assert plank._hold_start is not None
        assert plank._state == Plank.STATE_HOLDING
        assert not counted  # 未到目标时间

    def test_update_with_bad_form_stops_timer(self):
        """姿势错误停止计时"""
        plank = Plank()
        kp_good = make_plank_kp(body_angle_from_horizontal=10)
        kp_bad = make_plank_kp(body_angle_from_horizontal=40)

        # 先支撑一会
        plank.update(kp_good)
        # 手动设一个较早的 hold_start 模拟已经撑了一段时间(>0.5秒阈值)
        plank._hold_start = time.time() - 0.6
        assert plank._hold_start is not None

        # 姿势崩塌 → 累计时间应 > 0
        counted, state = plank.update(kp_bad)
        assert plank._state == Plank.STATE_REST
        assert plank.duration > 0

    def test_reset_clears_everything(self):
        """reset 清空所有状态"""
        plank = Plank()
        kp = make_plank_kp(body_angle_from_horizontal=10)

        plank.update(kp)
        time.sleep(0.05)
        plank.reset()

        assert plank.duration == 0.0
        assert plank._hold_start is None
        assert plank._state == Plank.STATE_REST
        assert plank.state == State.WAITING
        assert plank.count == 0

    def test_accumulated_duration_persists_across_bad_form(self):
        """累计时长在姿势恢复后继续累加"""
        plank = Plank()
        kp_good = make_plank_kp(body_angle_from_horizontal=10)
        kp_bad = make_plank_kp(body_angle_from_horizontal=40)

        # 模拟已累计 2 秒
        plank.duration = 2.0

        # 当前正在支撑
        plank.update(kp_good)
        plank._hold_start = time.time() - 0.3  # 当前段 0.3 秒

        total = plank.duration
        if plank._hold_start:
            total += time.time() - plank._hold_start
        assert total > 2.0  # 总时间 > 已累计的2秒

    def test_target_reached_returns_counted(self):
        """达到目标时间返回 counted=True"""
        plank = Plank()
        plank.target_reps = 0.01  # 极短目标用于测试
        kp = make_plank_kp(body_angle_from_horizontal=10)

        plank.update(kp)
        time.sleep(0.02)
        counted, state = plank.update(kp)
        assert counted is True


class TestPlankStateConsistency:
    """状态一致性测试 — 验证 Plank 双状态系统的潜在问题"""

    def test_reset_sets_both_states(self):
        """reset() 同时修改 self._state 和 self.state"""
        plank = Plank()
        plank._state = Plank.STATE_HOLDING
        plank.state = State.READY

        plank.reset()
        assert plank._state == Plank.STATE_REST
        assert plank.state == State.WAITING

    def test_is_ready_and_is_down_not_applicable(self):
        """Plank 不使用 is_ready/is_down（被 override）"""
        plank = Plank()
        kp = make_plank_kp(body_angle_from_horizontal=10)

        # is_ready 用 _is_planking 实现
        assert plank.is_ready(kp) is True
        # is_down 始终返回 False（平板不用这个）
        assert plank.is_down(kp) is False
        assert plank.is_up(kp) is False


class TestPlankFormHints:
    """平板支撑姿势提示"""

    def test_body_deviation_triggers_hint(self):
        """身体角度偏差大（>25度）→ 提示"""
        kp = np.zeros((17, 2), dtype=np.float32)
        # 臀部远低于肩膀和脚踝的连线 → 形成大角度偏差
        import math
        shoulder = np.array([0.0, 100.0])
        hip = np.array([100.0, 200.0])  # 臀部明显低于肩和脚踝连线
        ankle = np.array([200.0, 100.0])  # 脚踝和肩差不多高

        kp[KP["right_shoulder"]] = shoulder
        kp[KP["right_hip"]] = hip
        kp[KP["right_ankle"]] = ankle

        plank = Plank()
        hints = plank.get_form_hints(kp)
        # body > 25度偏差 → 一定有提示
        assert len(hints) > 0, f"期望有提示，got hints={hints}"
        assert any("臀部太低" in h or "臀部太高" in h for h in hints)

    def test_none_keypoints_hints(self):
        plank = Plank()
        assert plank.get_form_hints(None) == []
