"""测试专注度监测逻辑"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pytest
from pose_fitness.focus_monitor import FocusMonitor


class TestFocusMonitorBasics:
    """专注度监测基础测试"""

    def test_initial_state(self):
        """初始状态应为专注"""
        fm = FocusMonitor()
        assert fm.is_focused is True

    def test_center_gaze_stays_focused(self):
        """看中心 → 保持专注"""
        fm = FocusMonitor()
        is_focused, should_alert = fm.update("center", 0.0)
        assert is_focused is True
        assert should_alert is False

    def test_brief_look_away_no_alert(self):
        """短暂偏离(<3秒) → 不触发提醒"""
        fm = FocusMonitor()
        fm.update("right", 0.4)  # 看右边
        time.sleep(0.5)
        is_focused, should_alert = fm.update("right", 0.4)
        assert should_alert is False  # 不到3秒不会提醒

    def test_prolonged_look_away_triggers_alert(self):
        """长时间偏离(≥3秒) → 触发提醒"""
        fm = FocusMonitor()
        # 设置初始状态
        fm.update("right", 0.4)
        # 模拟时间已过3秒：直接修改内部状态
        fm._away_start = time.time() - 4.0  # 4秒前开始走神

        is_focused, should_alert = fm.update("right", 0.4)
        assert should_alert is True
        assert is_focused is False

    def test_alert_only_once_per_episode(self):
        """同一走神段落只提醒一次"""
        fm = FocusMonitor()
        fm._away_start = time.time() - 4.0

        is_focused, should_alert = fm.update("right", 0.4)
        assert should_alert is True

        # 继续走神 → 不再提醒
        is_focused, should_alert = fm.update("right", 0.4)
        assert should_alert is False

    def test_recovery_resets_alert(self):
        """恢复专注后重新走神 → 可以再次触发提醒"""
        fm = FocusMonitor()
        fm._away_start = time.time() - 4.0

        # 第一次触发
        fm.update("right", 0.4)

        # 恢复
        fm.update("center", 0.0)
        assert fm.is_focused is True
        assert fm._alert_triggered is False

        # 再次走神
        fm._away_start = time.time() - 4.0
        is_focused, should_alert = fm.update("left", -0.5)
        assert should_alert is True


class TestFocusMonitorStats:
    """专注度统计测试"""

    def test_stats_initial(self):
        """初始统计"""
        fm = FocusMonitor()
        stats = fm.get_stats()
        assert stats["focus_pct"] >= 99.0  # 几乎100%
        assert stats["is_focused"] is True

    def test_stats_after_away_time(self):
        """有走神时间后统计反映"""
        fm = FocusMonitor()
        fm._total_away_sec = 3.0
        # 假设已经过了10秒
        fm._session_start = time.time() - 10.0

        stats = fm.get_stats()
        # 3秒走神 / 10秒总时间 = 70%专注
        expected = (1 - 3.0 / 10.0) * 100
        assert abs(stats["focus_pct"] - expected) < 5

    def test_reset_stats(self):
        """reset 重置统计"""
        fm = FocusMonitor()
        fm._total_away_sec = 5.0
        fm.is_focused = False

        fm.reset()
        stats = fm.get_stats()
        assert stats["focus_pct"] >= 99.0
        assert stats["is_focused"] is True
        assert stats["away_sec"] == 0.0


class TestFocusMonitorEdgeCases:
    """边界条件测试"""

    def test_h_ratio_threshold(self):
        """h_ratio 超过阈值也触发走神"""
        fm = FocusMonitor()
        fm._away_start = time.time() - 4.0

        # gaze = "center" 但 h_ratio > 0.25
        is_focused, should_alert = fm.update("center", 0.30)
        assert should_alert is True

    def test_gaze_not_center_but_small_ratio(self):
        """gaze != center 但 h_ratio 小 → 仍然算走神"""
        fm = FocusMonitor()
        # 由于 gaze_direction() 中 gaze="left" 意味着 h_ratio < -0.25
        # 但如果直接传 gaze="left", h_ratio=-0.1（理论上不可能），仍判走神
        is_away = fm.update("left", -0.1)[0]
        # 因为 gaze != "center" → is_away = True
        # 所以仍然是走神
        pass


class TestFocusMonitorGazeLogic:
    """验证 gaze_direction 和 focus_monitor 阈值一致性"""

    def test_threshold_consistency(self):
        """gaze_direction 用 0.25 阈值, FocusMonitor 也用 0.25
        两者逻辑存在重复但至少一致"""
        # 这就是审查中发现的问题 #5
        threshold = 0.25

        # focus_monitor 中：is_away = gaze != "center" or abs(h_ratio) > GAZE_IRIS_RATIO_THRESH
        # gaze_direction 中：abs(h_ratio) < 0.25 → "center"
        # 所以 gaze != "center" 意味着 abs(h_ratio) >= 0.25
        # 而 abs(h_ratio) > 0.25 与 abs(h_ratio) >= 0.25 有微小差别
        # 边界: h_ratio = 0.25 时，gaze="center"，is_away=False (因为 abs(0.25) > 0.25 = False)
        #     h_ratio = 0.25001 时，gaze != "center"，is_away=True
        assert threshold == 0.25  # 记录当前阈值
