"""引体向上：手腕与肩部相对位置检测"""

import numpy as np
from .base import BaseExercise
from ..config import KP


class Pullup(BaseExercise):

    def __init__(self):
        super().__init__("引体向上", target_reps=10)

    def _wrist_above_shoulder(self, kp) -> bool:
        """右腕 y 坐标是否高于右肩（画面中 y 越小越靠上）"""
        return kp[KP["right_wrist"]][1] < kp[KP["right_shoulder"]][1] - 15

    def _wrist_below_shoulder(self, kp) -> bool:
        """右腕 y 坐标是否低于右肩（悬挂状态）"""
        return kp[KP["right_wrist"]][1] > kp[KP["right_shoulder"]][1] + 15

    def is_ready(self, kp) -> bool:
        return kp is not None and self._wrist_below_shoulder(kp)

    def is_down(self, kp) -> bool:
        return kp is not None and self._wrist_above_shoulder(kp)

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
