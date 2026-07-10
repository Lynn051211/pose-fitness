"""专注度监测：根据视线方向判断是否走神"""

import time
from .config import GAZE_AWAY_SEC, GAZE_IRIS_RATIO_THRESH


class FocusMonitor:

    def __init__(self):
        self.is_focused = True            # 当前是否专注
        self._away_start = None           # 开始走神的时间
        self._alert_triggered = False     # 本轮走神是否已提醒
        self._total_away_sec = 0.0        # 累计走神时间
        self._session_start = time.time()

    def update(self, gaze: str, h_ratio: float):
        """每帧调用，返回 (is_focused, should_alert)"""
        is_away = gaze != "center" or abs(h_ratio) > GAZE_IRIS_RATIO_THRESH

        if is_away:
            if self._away_start is None:
                self._away_start = time.time()
            else:
                away_dur = time.time() - self._away_start
                if away_dur >= GAZE_AWAY_SEC and not self._alert_triggered:
                    self._alert_triggered = True
                    self.is_focused = False
                    self._total_away_sec += away_dur
                    return False, True  # 触发提醒
            return self.is_focused, False
        else:
            if self._away_start is not None:
                self._total_away_sec += time.time() - self._away_start
                self._away_start = None
            self._alert_triggered = False
            self.is_focused = True
            return True, False

    def get_stats(self):
        """返回当前专注统计"""
        away = self._total_away_sec
        if self._away_start is not None:
            away += time.time() - self._away_start
        total = time.time() - self._session_start
        focus_pct = max(0, (1 - away / max(total, 1))) * 100
        return {
            "focus_pct": round(focus_pct, 1),
            "away_sec": round(away, 1),
            "total_sec": round(total, 1),
            "is_focused": self.is_focused,
        }

    def reset(self):
        self._away_start = None
        self._alert_triggered = False
        self._total_away_sec = 0.0
        self._session_start = time.time()
        self.is_focused = True
