"""语音播报模块"""

import threading
import pyttsx3

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 200)
        _engine.setProperty("volume", 0.8)
    return _engine


def say(text: str):
    """异步播报，不阻塞主线程"""
    threading.Thread(target=_speak, args=(text,), daemon=True).start()


def _speak(text: str):
    try:
        engine = _get_engine()
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass  # 语音不是核心功能，失败静默处理
