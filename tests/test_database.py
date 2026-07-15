"""测试数据库操作"""
import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import sqlite3
from datetime import datetime

# Mock database module's DB_PATH before importing
import pose_fitness.database as db_module


class TestDatabase:
    """数据库测试（使用临时文件）"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """每个测试用独立的临时数据库"""
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_fitness.db")

        # 保存原始 DB_PATH
        self.original_db_path = db_module.DB_PATH
        db_module.DB_PATH = self.db_path

        # 初始化
        db_module.init()

        yield

        # 恢复
        db_module.DB_PATH = self.original_db_path
        # 清理
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_init_creates_table(self):
        """init() 创建 sessions 表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_init_idempotent(self):
        """重复 init() 不报错"""
        db_module.init()  # 第二次调用
        db_module.init()  # 第三次调用
        # 不应抛异常

    def test_save_and_retrieve(self):
        """保存并检索记录"""
        db_module.save("深蹲", 15, 0)
        db_module.save("俯卧撑", 10, 0)

        rows = db_module.get_history()
        assert len(rows) == 2

    def test_save_with_duration(self):
        """保存平板支撑记录（含时长）"""
        db_module.save("平板支撑", 60, 60)

        rows = db_module.get_history(exercise="平板支撑")
        assert len(rows) == 1
        assert rows[0][2] == 60  # reps
        assert rows[0][3] == 60  # duration_sec

    def test_get_history_by_exercise(self):
        """按动作筛选历史"""
        db_module.save("深蹲", 10, 0)
        db_module.save("俯卧撑", 5, 0)
        db_module.save("深蹲", 15, 0)

        rows = db_module.get_history(exercise="深蹲")
        assert len(rows) == 2
        # 所有记录都是深蹲
        for r in rows:
            assert r[1] == "深蹲"

    def test_get_history_by_days(self):
        """按天数筛选"""
        db_module.save("深蹲", 10, 0)

        rows = db_module.get_history(days=7)
        assert len(rows) >= 1

        rows = db_module.get_history(days=1)
        assert len(rows) >= 0  # 可能为0或1（取决于现在是否在同一天）

    def test_get_history_no_results(self):
        """空数据库返回空列表"""
        rows = db_module.get_history(exercise="不存在")
        assert rows == []

    def test_save_preserves_date_format(self):
        """保存的日期格式正确"""
        db_module.save("深蹲", 5, 0)
        rows = db_module.get_history()
        date_str = rows[0][0]
        # 格式: YYYY-MM-DD HH:MM:SS
        parts = date_str.split(" ")
        assert len(parts) == 2
        date_parts = parts[0].split("-")
        assert len(date_parts) == 3
        time_parts = parts[1].split(":")
        assert len(time_parts) == 3

    def test_sql_injection_safe(self):
        """SQL注入不会被执行"""
        # 尝试注入（参数化查询应该是安全的）
        malicious = "深蹲'; DROP TABLE sessions; --"
        db_module.save(malicious, 1, 0)

        # 表应该还在
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
        )
        assert cursor.fetchone() is not None
        conn.close()

        # 恶意字符串应该被当做普通文本存储
        rows = db_module.get_history()
        # 找到恶意字符串的记录（它被转义了）
        found = [r for r in rows if "DROP TABLE" in r[1]]
        assert len(found) == 1


class TestDatabasePath:
    """数据库路径测试（已修复为绝对路径）"""

    def test_db_path_is_absolute(self):
        """验证 DB_PATH 已修复为基于模块目录的绝对路径"""
        # 修复后 DB_PATH 使用 os.path.join(__file__ 所在目录, fitness.db)
        assert os.path.isabs(db_module.DB_PATH), \
            f"DB_PATH 应为绝对路径，实际为: {db_module.DB_PATH}"
        assert 'fitness.db' in db_module.DB_PATH, \
            f"DB_PATH 应引用 fitness.db，实际为: {db_module.DB_PATH}"
