"""OpenCV 界面渲染"""

import cv2
from .config import KP, COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_WHITE, COLOR_ORANGE, COLOR_BLUE
from .exercises.base import State

# 骨架连线定义（哪些关键点之间画线）
SKELETON = [
    (KP["left_eye"], KP["right_eye"]),
    (KP["nose"], KP["left_eye"]), (KP["nose"], KP["right_eye"]),
    (KP["left_shoulder"], KP["right_shoulder"]),
    (KP["left_shoulder"], KP["left_elbow"]), (KP["left_elbow"], KP["left_wrist"]),
    (KP["right_shoulder"], KP["right_elbow"]), (KP["right_elbow"], KP["right_wrist"]),
    (KP["left_shoulder"], KP["left_hip"]), (KP["right_shoulder"], KP["right_hip"]),
    (KP["left_hip"], KP["right_hip"]),
    (KP["left_hip"], KP["left_knee"]), (KP["left_knee"], KP["left_ankle"]),
    (KP["right_hip"], KP["right_knee"]), (KP["right_knee"], KP["right_ankle"]),
]


def draw_skeleton(frame, kp):
    """在画面上画出关键点和骨架连线"""
    for i, j in SKELETON:
        if kp[i][0] > 0 and kp[i][1] > 0 and kp[j][0] > 0 and kp[j][1] > 0:
            pt1 = tuple(kp[i].astype(int))
            pt2 = tuple(kp[j].astype(int))
            cv2.line(frame, pt1, pt2, COLOR_GREEN, 2)
    for pt in kp:
        if pt[0] > 0 and pt[1] > 0:
            p = tuple(pt.astype(int))
            cv2.circle(frame, p, 4, COLOR_RED, -1)


def draw_info_panel(frame, exercise_name: str, count: int,
                    target: int, state, fps: float, hints: list,
                    plank_duration: float = 0,
                    focus_stats: dict = None, gaze: str = ""):
    """顶部信息栏 + 底部状态栏"""
    h, w = frame.shape[:2]

    # ---- 顶部栏 ----
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, exercise_name, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)

    if isinstance(state, str) and state in ("HOLDING", "REST"):
        mins = int(plank_duration // 60)
        secs = int(plank_duration % 60)
        cv2.putText(frame, f"{mins:02d}:{secs:02d} / {target}s", (180, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_YELLOW, 2)
    else:
        cv2.putText(frame, f"{count} / {target}", (180, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_YELLOW, 2)

    # 专注度指示器（右上角）
    if focus_stats is not None:
        fc = COLOR_GREEN if focus_stats["is_focused"] else COLOR_RED
        focus_label = f"{focus_stats['focus_pct']:.0f}%"
        cv2.putText(frame, focus_label, (w - 100, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fc, 1)
        if gaze:
            cv2.putText(frame, gaze, (w - 100, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, fc, 1)

    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 190, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)

    state_color = COLOR_GREEN if state in (State.READY, "HOLDING") else COLOR_YELLOW
    state_str = str(state).split(".")[-1] if "." in str(state) else str(state)
    cv2.putText(frame, state_str, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

    # ---- 底部栏 ----
    bar_y = h - 50
    cv2.rectangle(overlay, (0, bar_y), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if isinstance(state, str) and state in ("HOLDING", "REST"):
        progress = min(plank_duration / max(target, 1), 1.0)
    else:
        progress = min(count / max(target, 1), 1.0)

    bar_w = w - 60
    cv2.rectangle(frame, (30, bar_y + 10), (30 + bar_w, bar_y + 30), COLOR_WHITE, 1)
    cv2.rectangle(frame, (30, bar_y + 10), (30 + int(bar_w * progress), bar_y + 30),
                  COLOR_GREEN, -1)

    if hints:
        hint_text = " | ".join(hints)
        cv2.putText(frame, hint_text, (30, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 1)

    cv2.putText(frame, "[1]深蹲 [2]俯卧撑 [3]引体 [4]平板 [r]重置 [ESC]退出",
                (30, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)


def draw_angle_visual(frame, kp, idx_a, idx_b, idx_c, label=""):
    """在关键点角度处画弧度线"""
    if kp is None:
        return
    a, b, c = kp[idx_a], kp[idx_b], kp[idx_c]
    if any(p[0] <= 0 or p[1] <= 0 for p in (a, b, c)):
        return
    cv2.putText(frame, label, (int(b[0]) + 10, int(b[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 1)
    cv2.circle(frame, tuple(b.astype(int)), 8, COLOR_YELLOW, 1)
