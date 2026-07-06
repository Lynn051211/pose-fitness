"""
运动计数 + 姿态估计 + 专注度检测 + Web 仪表盘

requirements.txt:
-------------------
opencv-python>=4.8.0
ultralytics>=8.0.0
pyttsx3>=2.90
numpy>=1.24.0
mediapipe>=0.10.30
fastapi>=0.100.0
uvicorn>=0.20.0
"""

import cv2
import sys
import time
import threading

from . import config as cfg
from .pose_detector import init as init_pose, detect
from .face_detector import init as init_face, detect as detect_face, gaze_direction
from .focus_monitor import FocusMonitor
from .exercises import EXERCISES
from .exercises.plank import Plank
from . import voice, database, ui, api


def main():
    # 初始化
    database.init()
    init_pose()
    init_face()

    cap = cv2.VideoCapture(cfg.CAMERA_ID)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        sys.exit(1)

    # 启动 Web API 线程
    web_thread = threading.Thread(target=api.run, daemon=True)
    web_thread.start()

    # 状态
    current_key = "1"
    exercise = EXERCISES[current_key]()
    focus_monitor = FocusMonitor()
    prev_time = time.time()
    gaze = "center"
    face_kp = None

    print(f"启动成功！按键切换动作 | Web 仪表盘: http://localhost:8080")
    voice.say("准备开始")

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("错误：读取帧失败")
            break

        # ---- 姿态检测（YOLO） ----
        kp = detect(frame)

        # ---- 面部检测（MediaPipe）- 每3帧跑一次降低负载 ----
        face_kp = None
        frame_idx += 1
        if frame_idx % 3 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_kp = detect_face(rgb)

        # ---- 画骨架 ----
        if kp is not None:
            ui.draw_skeleton(frame, kp)

        # ---- 动作更新 ----
        counted, state = exercise.update(kp)

        plank_duration = 0
        if isinstance(exercise, Plank):
            plank_duration = exercise.duration
            if exercise._hold_start:
                plank_duration += time.time() - exercise._hold_start
            state = exercise._state
            if counted and exercise._hold_start:
                voice.say(f"平板支撑完成，{int(plank_duration)}秒")
                database.save(exercise.name, int(plank_duration), int(plank_duration))
                exercise.reset()
        elif counted:
            voice.say(str(exercise.count))

        # ---- 专注度检测 ----
        h_ratio = 0.0
        if face_kp is not None:
            gaze, h_ratio = gaze_direction(face_kp)
        is_focused, should_alert = focus_monitor.update(gaze, h_ratio)
        focus_stats = focus_monitor.get_stats()

        if should_alert:
            voice.say("请集中注意力")
            print("走神提醒！")

        # ---- 姿势提示 ----
        hints = exercise.get_form_hints(kp) if kp is not None else []

        # ---- FPS ----
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-8)
        prev_time = curr_time

        # ---- UI ----
        ui.draw_info_panel(
            frame, exercise.name, exercise.count,
            exercise.target_reps, state, fps, hints, plank_duration,
            focus_stats, gaze,
        )

        # ---- Web 推流 ----
        with cfg.frame_lock:
            cfg.latest_frame = frame.copy()

        # ---- API 状态 ----
        api.update_status(exercise.name, exercise.count, fps, focus_stats)

        # ---- OpenCV 窗口 ----
        cv2.imshow("Pose Fitness", frame)

        # ---- 按键 ----
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("q"):
            break
        elif chr(key) in EXERCISES:
            if exercise.count > 0:
                database.save(exercise.name, exercise.count)
            if isinstance(exercise, Plank) and exercise.duration > 0:
                database.save(exercise.name, int(exercise.duration), int(exercise.duration))

            current_key = chr(key)
            exercise = EXERCISES[current_key]()
            focus_monitor.reset()
            voice.say(exercise.name)
        elif key == ord("r"):
            if exercise.count > 0 or (isinstance(exercise, Plank) and exercise.duration > 0):
                database.save(exercise.name, exercise.count,
                              int(exercise.duration) if isinstance(exercise, Plank) else 0)
            exercise.reset()
            focus_monitor.reset()
            voice.say("重新开始")

    # ---- 退出 ----
    if exercise.count > 0:
        database.save(exercise.name, exercise.count)
    if isinstance(exercise, Plank) and exercise.duration > 0:
        database.save(exercise.name, int(exercise.duration), int(exercise.duration))

    cap.release()
    cv2.destroyAllWindows()
    print("训练结束，数据已保存")


if __name__ == "__main__":
    main()
