"""平板支撑：计时 + 姿势检测"""

import time
import numpy as np
from .base import BaseExercise, State
from ..config import KP, angle_3pts


class Plank(BaseExercise):

    STATE_HOLDING = "HOLDING"
    STATE_REST = "REST"

    def __init__(self):
        super().__init__("平板支撑", target_reps=60)  # target 秒
        self.duration = 0.0            # 累计支撑时间（秒）
        self._hold_start = None        # 当前段开始时间
        self._rest_start = None
        self._state = self.STATE_REST

    def update(self, keypoints):
        """覆写基类：计时模式"""
        if not self._is_planking(keypoints):
            self._rest_start = self._rest_start or time.time()
            if self._hold_start and time.time() - self._hold_start > 0.5:
                self.duration += time.time() - self._hold_start
                self._hold_start = None
            self._state = self.STATE_REST
            return False, State.DOWN if self._rest_start else State.WAITING
        else:
            self._hold_start = self._hold_start or time.time()
            self._rest_start = None
            self._state = self.STATE_HOLDING

        # 累计当前段
        total = self.duration
        if self._hold_start:
            total += time.time() - self._hold_start
        return total >= self.target_reps, State.READY if total < self.target_reps else State.UP

    def _is_planking(self, kp) -> bool:
        if kp is None:
            return False
        body = abs(angle_3pts(kp[KP["right_shoulder"]], kp[KP["right_hip"]], kp[KP["right_ankle"]]) - 180)
        return body < 35

    def is_ready(self, kp) -> bool:
        return self._is_planking(kp)

    def is_down(self, kp) -> bool:
        return False

    def is_up(self, kp) -> bool:
        return False

    def get_form_hints(self, kp) -> list:
        hints = []
        if kp is None:
            return hints
        body = abs(angle_3pts(kp[KP["right_shoulder"]], kp[KP["right_hip"]], kp[KP["right_ankle"]]) - 180)
        if body > 25:
            hints.append("臀部太低" if kp[KP["right_hip"]][1] > kp[KP["right_shoulder"]][1] else "臀部太高")
        return hints

    def is_holding(self):
        return self._state == self.STATE_HOLDING

    def reset(self):
        self.count = 0
        self.duration = 0.0
        self._hold_start = None
        self._rest_start = None
        self._state = self.STATE_REST
        self._set_state(State.WAITING)
