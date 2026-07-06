"""动作基类 + 通用状态机"""

from abc import ABC, abstractmethod
from enum import Enum


class State(Enum):
    WAITING = "WAITING"
    READY = "READY"
    DOWN = "DOWN"
    UP = "UP"


class BaseExercise(ABC):

    def __init__(self, name: str, target_reps: int = 15):
        self.name = name
        self.count = 0
        self.state = State.WAITING
        self.target_reps = target_reps
        self._state_timer = 0         # 当前状态持续帧数
        self._min_frames = 4          # 最少持续帧数（防抖）

    def update(self, keypoints):
        """每帧调用，返回 (just_counted: bool, state: State)"""
        if self.state == State.WAITING:
            return self._handle_waiting(keypoints)
        elif self.state == State.READY:
            return self._handle_ready(keypoints)
        elif self.state == State.DOWN:
            return self._handle_down(keypoints)
        elif self.state == State.UP:
            return self._handle_up(keypoints)

    def _handle_waiting(self, kp):
        if self.is_ready(kp):
            self._state_timer += 1
            if self._state_timer >= self._min_frames:
                self._set_state(State.READY)
                return False, State.READY
        else:
            self._state_timer = 0
        return False, State.WAITING

    def _handle_ready(self, kp):
        if not self.is_ready(kp) and self.is_down(kp):
            self._state_timer += 1
            if self._state_timer >= self._min_frames:
                self._set_state(State.DOWN)
                return False, State.DOWN
        elif not self.is_ready(kp):
            self._state_timer = 0
            self._set_state(State.WAITING)
            return False, State.WAITING
        else:
            self._state_timer = 0
        return False, State.READY

    def _handle_down(self, kp):
        if self.is_up(kp):
            self._state_timer += 1
            if self._state_timer >= self._min_frames:
                self.count += 1
                self._set_state(State.UP)
                return True, State.UP
        elif self.is_down(kp):
            self._state_timer = 0
        else:
            self._state_timer = 0
            self._set_state(State.WAITING)
            return False, State.WAITING
        return False, State.DOWN

    def _handle_up(self, kp):
        self._set_state(State.READY)
        return False, State.READY

    def _set_state(self, state: State):
        self.state = state
        self._state_timer = 0

    # ---- 子类必须实现 ----
    @abstractmethod
    def is_ready(self, kp) -> bool: ...

    @abstractmethod
    def is_down(self, kp) -> bool: ...

    @abstractmethod
    def is_up(self, kp) -> bool: ...

    # ---- 子类可选 ----
    def get_form_hints(self, kp) -> list:
        """返回姿势提示列表，如 ['臀部太低', '背部弯曲']"""
        return []

    def reset(self):
        self.count = 0
        self._set_state(State.WAITING)
