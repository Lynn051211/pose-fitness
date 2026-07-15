"""SQLite 训练记录"""

import os
import sqlite3
from datetime import datetime

# 使用模块所在目录作为数据库路径基准，避免 CWD 变化导致数据库散落各处
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fitness.db")


def init():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                exercise TEXT NOT NULL,
                reps INTEGER NOT NULL,
                duration_sec INTEGER DEFAULT 0
            )
        """)


def save(exercise: str, reps: int, duration_sec: int = 0):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO sessions (date, exercise, reps, duration_sec) VALUES (?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), exercise, reps, duration_sec),
        )


def get_history(exercise: str = None, days: int = 7):
    with sqlite3.connect(DB_PATH) as conn:
        if exercise:
            rows = conn.execute(
                "SELECT date, exercise, reps, duration_sec FROM sessions "
                "WHERE exercise = ? AND date >= date('now', ?) ORDER BY date DESC",
                (exercise, f"-{days} days"),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT date, exercise, reps, duration_sec FROM sessions "
                "WHERE date >= date('now', ?) ORDER BY date DESC",
                (f"-{days} days",),
            ).fetchall()
    return rows
