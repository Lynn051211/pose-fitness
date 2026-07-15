"""俯卧撑：肘关节角度变化检测"""

import numpy as np
from .base import BaseExercise
from ..config import KP, angle_3pts


class Pushup(BaseExercise):

    def __init__(self):
        super().__init__("俯卧撑", target_reps=15)

    def _elbow_angle(self, kp):
        """右肘角度（shoulder-elbow-wrist）"""
        return angle_3pts(kp[KP["right_shoulder"]], kp[KP["right_elbow"]], kp[KP["right_wrist"]])

    def _body_angle(self, kp):
        """身体与水平面夹角（shoulder-hip-ankle），越小越水平"""
        return angle_3pts(kp[KP["right_shoulder"]], kp[KP["right_hip"]], kp[KP["right_ankle"]])

    def is_ready(self, kp) -> bool:
        if kp is None:
            return False
        return bool(self._elbow_angle(kp) > 140 and abs(self._body_angle(kp) - 180) < 30)

    def is_down(self, kp) -> bool:
        if kp is None:
            return False
        return bool(self._elbow_angle(kp) < 90 and abs(self._body_angle(kp) - 180) < 30)

    def is_up(self, kp) -> bool:
        return self.is_ready(kp)

    def get_form_hints(self, kp) -> list:
        hints = []
        if kp is None:
            return hints
        body = abs(self._body_angle(kp) - 180)
        if body > 40:
            hints.append("身体塌腰")
        return hints
