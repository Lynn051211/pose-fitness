"""引体向上：手腕与肩部相对位置检测（左右双侧平均）"""

import numpy as np
from .base import BaseExercise
from ..config import KP


class Pullup(BaseExercise):

    def __init__(self):
        super().__init__("引体向上", target_reps=10)

    def _avg_wrist_shoulder_diff(self, kp) -> float:
        """左右手腕-肩膀 y 坐标差的平均值（正=腕低于肩/悬挂，负=腕高于肩/拉起）"""
        left_diff = kp[KP["left_wrist"]][1] - kp[KP["left_shoulder"]][1]
        right_diff = kp[KP["right_wrist"]][1] - kp[KP["right_shoulder"]][1]
        return (left_diff + right_diff) / 2.0

    def is_ready(self, kp) -> bool:
        if kp is None:
            return False
        return bool(self._avg_wrist_shoulder_diff(kp) > 15)

    def is_down(self, kp) -> bool:
        if kp is None:
            return False
        return bool(self._avg_wrist_shoulder_diff(kp) < -15)

    def is_up(self, kp) -> bool:
        return self.is_ready(kp)

    def get_form_hints(self, kp) -> list:
        hints = []
        if kp is None:
            return hints
        # 检测身体是否大幅摆动（髋部水平位移）
        if abs(kp[KP["right_hip"]][0] - kp[KP["left_hip"]][0]) > 50:
            hints.append("减少摆动")
        return hints
