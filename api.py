"""FastAPI 后端 + MJPEG 视频推流"""

import time
import threading
import cv2
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from .config import latest_frame, frame_lock, WEB_PORT
from .database import get_history

app = FastAPI(title="Pose Fitness")

# 共享状态（主循环写入）
_shared_status = {"exercise": "", "count": 0, "fps": 0, "focus": {}, "timestamp": 0}
_status_lock = threading.Lock()


def update_status(exercise: str, count: int, fps: float, focus: dict):
    with _status_lock:
        _shared_status.update({
            "exercise": exercise, "count": count, "fps": round(fps, 1),
            "focus": focus, "timestamp": time.time(),
        })


def _generate_frames():
    """MJPEG 推流生成器"""
    while True:
        with frame_lock:
            if latest_frame is not None:
                _, buf = cv2.imencode(".jpg", latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                frame_bytes = buf.tobytes()
            else:
                frame_bytes = b""
        if frame_bytes:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.03)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(_generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
def api_status():
    with _status_lock:
        return JSONResponse(_shared_status)


@app.get("/api/history")
def api_history(exercise: str = None, days: int = 7):
    rows = get_history(exercise, days)
    return JSONResponse([
        {"date": r[0], "exercise": r[1], "reps": r[2], "duration_sec": r[3]}
        for r in rows
    ])


@app.get("/")
def index():
    return HTMLResponse(_HTML)


def run():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=WEB_PORT, log_level="warning")


# ---- HTML 前端 ----
_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pose Fitness</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0f0f0f;color:#eee;display:flex;height:100vh;overflow:hidden}
#video-panel{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:16px}
#video-panel img{max-width:100%;max-height:70vh;border-radius:8px;border:2px solid #333}
#sidebar{width:340px;background:#1a1a1a;padding:20px;display:flex;flex-direction:column;gap:16px;overflow-y:auto}
h2{font-size:18px;color:#4ade80;margin-bottom:8px}
.stat-box{background:#222;border-radius:8px;padding:14px}
.stat-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #333}
.stat-row:last-child{border-bottom:none}
.stat-label{color:#999}
.stat-value{font-weight:700;font-size:16px}
.stat-value.focused{color:#4ade80}
.stat-value.distracted{color:#ef4444}
.chart-box{background:#222;border-radius:8px;padding:14px}
canvas{width:100%!important;max-height:200px}
#exercise-select{display:flex;gap:8px;flex-wrap:wrap}
#exercise-select button{flex:1;min-width:60px;padding:8px;border:none;border-radius:6px;background:#333;color:#eee;cursor:pointer;font-size:13px}
#exercise-select button.active{background:#4ade80;color:#000;font-weight:700}
#alert-box{background:#7f1d1d;color:#fca5a5;padding:10px;border-radius:6px;display:none;text-align:center;font-weight:bold}
</style>
</head>
<body>
<div id="video-panel">
  <img id="stream" src="/video_feed" alt="Camera Stream">
  <p style="margin-top:8px;color:#666;font-size:12px">OpenCV 窗口按 ESC/q 退出 | Web 仅观看</p>
</div>
<div id="sidebar">
  <h2>Pose Fitness</h2>
  <div id="alert-box">请集中注意力！</div>
  <div class="stat-box">
    <h2>当前状态</h2>
    <div id="status-content">
      <div class="stat-row"><span class="stat-label">动作</span><span class="stat-value" id="s-exercise">-</span></div>
      <div class="stat-row"><span class="stat-label">计数</span><span class="stat-value" id="s-count">0</span></div>
      <div class="stat-row"><span class="stat-label">FPS</span><span class="stat-value" id="s-fps">0</span></div>
      <div class="stat-row"><span class="stat-label">专注度</span><span class="stat-value focused" id="s-focus">-</span></div>
    </div>
  </div>
  <div class="chart-box">
    <h2>专注度趋势</h2>
    <canvas id="focus-chart"></canvas>
  </div>
  <div class="chart-box">
    <h2>训练历史（7天）</h2>
    <canvas id="history-chart"></canvas>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7" crossorigin="anonymous"></script>
<script>
let focusHistory = [];
const focusChart = new Chart(document.getElementById('focus-chart'), {
  type: 'line',
  data: {labels:[],datasets:[{label:'专注度%',data:[],borderColor:'#4ade80',backgroundColor:'rgba(74,222,128,0.1)',fill:true,tension:0.4}]},
  options:{responsive:true,scales:{x:{display:false},y:{min:0,max:100,ticks:{color:'#999'}}},plugins:{legend:{display:false}}}
});
const historyChart = new Chart(document.getElementById('history-chart'), {
  type: 'bar',
  data: {labels:[],datasets:[{label:'次数',data:[],backgroundColor:'#4ade80'}]},
  options:{responsive:true,scales:{x:{ticks:{color:'#999'}},y:{ticks:{color:'#999'}}},plugins:{legend:{display:false}}}
});

async function poll(){
  try{
    const r=await fetch('/api/status');
    const s=await r.json();
    document.getElementById('s-exercise').textContent=s.exercise||'-';
    document.getElementById('s-count').textContent=s.count;
    document.getElementById('s-fps').textContent=s.fps;
    const fv=s.focus||{};
    const fel=document.getElementById('s-focus');
    fel.textContent=(fv.focus_pct||0)+'%';
    fel.className='stat-value '+(fv.is_focused?'focused':'distracted');
    const ab=document.getElementById('alert-box');
    ab.style.display=fv.is_focused===false?'block':'none';
    if(focusHistory.length>60)focusHistory.shift();
    focusHistory.push(fv.focus_pct||0);
    focusChart.data.labels=focusHistory.map((_,i)=>i);
    focusChart.data.datasets[0].data=focusHistory;
    focusChart.update('none');
  }catch(e){}
  try{
    const r2=await fetch('/api/history');
    const h=await r2.json();
    const groups={};
    h.forEach(row=>{const k=row.exercise||'?';groups[k]=(groups[k]||0)+row.reps;});
    historyChart.data.labels=Object.keys(groups);
    historyChart.data.datasets[0].data=Object.values(groups);
    historyChart.update('none');
  }catch(e){}
}
setInterval(poll,1000);
poll();
</script>
</body>
</html>"""
