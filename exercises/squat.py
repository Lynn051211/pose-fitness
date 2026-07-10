"""深蹲：膝关节角度变化检测"""

import numpy as np
from .base import BaseExercise
from ..config import KP, angle_3pts


class Squat(BaseExercise):

    def __init__(self):
        super().__init__("深蹲", target_reps=15)

    def _knee_angle(self, kp):
        """右膝角度（hip-knee-ankle）"""
        return angle_3pts(kp[KP["right_hip"]], kp[KP["right_knee"]], kp[KP["right_ankle"]])

    def is_ready(self, kp) -> bool:
        return kp is not None and self._knee_angle(kp) > 150

    def is_down(self, kp) -> bool:
        return kp is not None and self._knee_angle(kp) < 90

    def is_up(self, kp) -> bool:
        return self.is_ready(kp)

    def get_form_hints(self, kp) -> list:
        hints = []
        if kp is None:
            return hints
        # 膝盖不要超过脚尖太多
        knee = kp[KP["right_knee"]]
        ankle = kp[KP["right_ankle"]]
        if knee[0] > ankle[0] + 30:
            hints.append("膝盖过脚尖")
        return hints
