"""测试 FastAPI 端点（不依赖摄像头）"""
import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import threading
import time
import numpy as np
from fastapi.testclient import TestClient

# Mock pose_fitness.config before importing api
import pose_fitness.config as cfg

# Mock 必要的全局状态
cfg.latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # 空黑帧

# 用临时数据库替换，避免影响真实数据
import pose_fitness.database as db_module


@pytest.fixture(autouse=True)
def setup_database(monkeypatch):
    """每个测试使用独立的临时数据库"""
    tmpdir = tempfile.mkdtemp()
    tmp_db = os.path.join(tmpdir, "test_fitness.db")
    monkeypatch.setattr(db_module, "DB_PATH", tmp_db)
    db_module.init()
    yield
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


from pose_fitness.api import app, update_status

client = TestClient(app)


class TestAPIStatus:
    """测试 /api/status 端点"""

    def test_status_initial(self):
        """初始状态返回"""
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "exercise" in data
        assert "count" in data
        assert "fps" in data
        assert "focus" in data

    def test_status_after_update(self):
        """更新状态后返回正确数据"""
        update_status("深蹲", 10, 30.5, {"is_focused": True, "focus_pct": 95.0})
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data["exercise"] == "深蹲"
        assert data["count"] == 10
        assert data["fps"] == 30.5
        assert data["focus"]["is_focused"] is True
        assert data["focus"]["focus_pct"] == 95.0

    def test_status_returns_json(self):
        """确认返回的是 JSON"""
        response = client.get("/api/status")
        assert response.headers["content-type"] == "application/json"


class TestAPIHistory:
    """测试 /api/history 端点"""

    def test_history_returns_list(self):
        """历史返回列表"""
        response = client.get("/api/history")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_history_with_exercise_filter(self):
        """按动作过滤历史"""
        response = client.get("/api/history?exercise=深蹲&days=30")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_history_with_days_filter(self):
        """按天数过滤"""
        response = client.get("/api/history?days=1")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_history_missing_params(self):
        """缺少参数时使用默认值"""
        response = client.get("/api/history")
        assert response.status_code == 200


class TestAPIHomepage:
    """测试首页"""

    def test_homepage_returns_html(self):
        """首页返回 HTML"""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Pose Fitness" in response.text

    def test_homepage_contains_chart_js(self):
        """首页包含 Chart.js"""
        response = client.get("/")
        assert "chart.js" in response.text.lower() or "Chart" in response.text

    def test_homepage_contains_key_elements(self):
        """首页包含关键元素"""
        response = client.get("/")
        assert "video_feed" in response.text
        assert "api/status" in response.text
        assert "api/history" in response.text


class TestAPIVideoFeed:
    """测试 /video_feed MJPEG 推流"""

    def test_video_feed_route_registered(self):
        """验证 /video_feed 路由已注册"""
        # MJPEG 流是无限生成器，同步 TestClient 会阻塞等待完成
        # 所以只验证路由存在
        routes = [r.path for r in app.routes]
        assert "/video_feed" in routes, "/video_feed 路由未注册"

    def test_video_feed_route_method(self):
        """验证 /video_feed 使用 GET 方法"""
        for r in app.routes:
            if r.path == "/video_feed":
                assert "GET" in r.methods
                break
        else:
            pytest.fail("未找到 /video_feed 路由")


class TestAPINotFound:
    """测试不存在的路由"""

    def test_404(self):
        """不存在的路由返回404"""
        response = client.get("/nonexistent")
        assert response.status_code == 404


class TestUpdateStatusThreadSafety:
    """update_status 线程安全测试"""

    def test_concurrent_updates(self):
        """并发更新不导致崩溃"""
        def update_many():
            for i in range(100):
                update_status(f"动作{i % 4}", i, float(i), {"focus_pct": 50.0})

        threads = []
        for _ in range(5):
            t = threading.Thread(target=update_many)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # 最后读取状态应正常
        response = client.get("/api/status")
        assert response.status_code == 200
