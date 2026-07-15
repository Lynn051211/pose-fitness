"""测试动作基类状态机"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from pose_fitness.exercises.base import BaseExercise, State


# ---- 一个具体的简单测试实现 ----
class MockExercise(BaseExercise):
    """用简单条件测试状态机：手举高了=READY, 手放下=DOWN, 手再举高=UP"""
    def is_ready(self, kp) -> bool:
        return kp is not None and kp[0][1] < 100  # y < 100 表示手举高

    def is_down(self, kp) -> bool:
        return kp is not None and kp[0][1] > 200  # y > 200 表示手放下

    def is_up(self, kp) -> bool:
        return self.is_ready(kp)


# ---- 辅助函数：创建模拟关键点 ----
def make_kp(hand_y):
    """创建 (17,2) 关键点，只关心第一个点的 y 坐标"""
    kp = np.zeros((17, 2), dtype=np.float32)
    kp[:, 0] = 100  # 随便给 x
    kp[:, 1] = hand_y
    return kp


class TestStateMachineBasics:
    """状态机基础测试"""

    def test_initial_state(self):
        """初始状态应为 WAITING"""
        ex = MockExercise("测试动作")
        assert ex.state == State.WAITING
        assert ex.count == 0

    def test_waiting_to_ready(self):
        """WAITING → READY：手举高持续4帧"""
        ex = MockExercise("测试动作")
        kp_ready = make_kp(50)  # y < 100 → is_ready = True

        for _ in range(3):
            counted, state = ex.update(kp_ready)
            assert state == State.WAITING, f"第{_+1}帧不应切换到READY"
            assert not counted

        # 第4帧
        counted, state = ex.update(kp_ready)
        assert state == State.READY, "第4帧应切换到READY"
        assert not counted

    def test_waiting_keep_waiting(self):
        """WAITING 保持：条件不满足时停留"""
        ex = MockExercise("测试动作")
        kp_bad = make_kp(150)  # 不满足 is_ready

        for _ in range(10):
            counted, state = ex.update(kp_bad)
            assert state == State.WAITING
            assert not counted

    def test_ready_to_down(self):
        """READY → DOWN：手放下持续4帧"""
        ex = MockExercise("测试动作")
        kp_ready = make_kp(50)
        kp_down = make_kp(250)

        # 先进入 READY
        for _ in range(4):
            ex.update(kp_ready)
        assert ex.state == State.READY

        # 进入 DOWN
        for _ in range(3):
            counted, state = ex.update(kp_down)
            assert state == State.READY, f"第{_+1}帧不应切换到DOWN"

        counted, state = ex.update(kp_down)
        assert state == State.DOWN
        assert not counted

    def test_down_to_up_counting(self):
        """DOWN → UP：计数+1"""
        ex = MockExercise("测试动作")
        kp_ready = make_kp(50)
        kp_down = make_kp(250)

        # WAITING → READY
        for _ in range(4): ex.update(kp_ready)
        # READY → DOWN
        for _ in range(4): ex.update(kp_down)

        # DOWN → UP（手再次举高）
        for _ in range(3):
            counted, state = ex.update(kp_ready)
            assert not counted

        counted, state = ex.update(kp_ready)
        assert counted, "第4帧举高应该计数"
        assert state == State.UP
        assert ex.count == 1

    def test_up_auto_to_ready(self):
        """UP 状态自动跳转到 READY"""
        ex = MockExercise("测试动作")
        kp_ready = make_kp(50)
        kp_down = make_kp(250)

        # 完成一个完整循环
        for _ in range(4): ex.update(kp_ready)  # → READY
        for _ in range(4): ex.update(kp_down)   # → DOWN
        for _ in range(4): ex.update(kp_ready)  # → UP (count=1)

        # UP 下一帧自动回到 READY
        counted, state = ex.update(kp_ready)
        assert state == State.READY
        assert not counted


class TestDebounce:
    """4帧防抖测试"""

    def test_debounce_resets_on_condition_change(self):
        """条件变化时帧计时器重置"""
        ex = MockExercise("测试动作")
        kp_ready = make_kp(50)
        kp_bad = make_kp(150)

        # 2帧 ready, 然后变回不满足
        ex.update(kp_ready)
        ex.update(kp_ready)
        assert ex._state_timer == 2

        ex.update(kp_bad)  # 不满足 → 重置
        assert ex._state_timer == 0
        assert ex.state == State.WAITING

    def test_debounce_partial_interruption(self):
        """中途打断后需要重新累积4帧"""
        ex = MockExercise("测试动作")
        kp_ready = make_kp(50)

        # 先进入 READY
        for _ in range(4): ex.update(kp_ready)
        assert ex.state == State.READY

        kp_down = make_kp(250)
        kp_mid = make_kp(150)  # 不满足任何条件

        # 2帧 down，然后干扰
        ex.update(kp_down)  # timer=1
        ex.update(kp_down)  # timer=2
        ex.update(kp_mid)   # 不满足 is_down 也不满足 is_ready → 回 WAITING
        assert ex.state == State.WAITING


class TestNullKeypoints:
    """关键点为 None 的边界测试"""

    def test_none_keypoints(self):
        """关键点为 None 时不应崩溃"""
        ex = MockExercise("测试动作")
        counted, state = ex.update(None)
        assert state == State.WAITING
        assert not counted

    def test_none_in_ready_state(self):
        """READY 状态下突然丢失关键点"""
        ex = MockExercise("测试动作")
        kp_ready = make_kp(50)

        for _ in range(4): ex.update(kp_ready)
        assert ex.state == State.READY

        # 突然丢失
        counted, state = ex.update(None)
        assert state == State.WAITING  # 应回到 WAITING


class TestReset:
    """reset() 方法测试"""

    def test_reset_count(self):
        """reset 后计数归零"""
        ex = MockExercise("测试动作")
        kp_ready = make_kp(50)
        kp_down = make_kp(250)

        for _ in range(4): ex.update(kp_ready)
        for _ in range(4): ex.update(kp_down)
        for _ in range(4): ex.update(kp_ready)
        assert ex.count == 1

        ex.reset()
        assert ex.count == 0
        assert ex.state == State.WAITING
        assert ex._state_timer == 0

    def test_reset_from_mid_state(self):
        """中途 reset 后状态正确"""
        ex = MockExercise("测试动作")
        kp_ready = make_kp(50)

        for _ in range(2): ex.update(kp_ready)
        ex.reset()
        assert ex.state == State.WAITING
        assert ex._state_timer == 0


class TestMultipleReps:
    """多组重复计数测试"""

    def test_three_reps(self):
        """连续3次完整动作"""
        ex = MockExercise("测试动作")
        kp_ready = make_kp(50)
        kp_down = make_kp(250)

        for _ in range(3):
            for _ in range(4): ex.update(kp_ready)  # → READY
            for _ in range(4): ex.update(kp_down)   # → DOWN
            for _ in range(4): ex.update(kp_ready)  # → UP(count+1)
            ex.update(kp_ready)                      # → READY

        assert ex.count == 3
